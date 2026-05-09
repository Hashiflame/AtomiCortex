"""
src/ingestion/binance_downloader.py

Downloads historical USDT-M futures data from the public Binance Data Portal
(https://data.binance.vision).  All I/O is async (aiohttp); concurrency is
capped by an asyncio.Semaphore.

Typical usage
-------------
    async with BinanceDataDownloader() as dl:
        paths = await dl.download_klines(
            symbol="BTCUSDT",
            interval="4h",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
            dest_dir=Path("/data/raw"),
        )
"""

from __future__ import annotations

import asyncio
import hashlib
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import aiohttp
from tqdm.asyncio import tqdm as atqdm

from src.logger import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://data.binance.vision"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0       # seconds; doubled each attempt
RATE_LIMIT_DELAY = 60.0      # seconds to wait on HTTP 429
CHUNK_SIZE = 65_536           # 64 KB streaming chunks

_log = get_logger(__name__)

DownloadStatus = Literal["ok", "skipped", "not_found", "error"]
DayResult = tuple[Path | None, str]   # (csv_path | None, status)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Return True when file's SHA-256 matches expected_hash (hex string)."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash.lower().strip()


def extract_zip(zip_path: Path) -> Path:
    """Extract the first CSV from *zip_path*, delete the zip, return CSV path.

    Raises
    ------
    zipfile.BadZipFile
        If the archive is corrupted.
    ValueError
        If no CSV file is found inside the archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV entry found in {zip_path.name}")
        zf.extract(csv_names[0], zip_path.parent)

    csv_path = zip_path.parent / csv_names[0]
    zip_path.unlink()
    return csv_path


# ---------------------------------------------------------------------------
# Downloader class
# ---------------------------------------------------------------------------

class BinanceDataDownloader:
    """Async downloader for Binance Data Portal (USDT-M futures).

    Must be used as an async context manager so the aiohttp session is
    created and cleanly closed:

        async with BinanceDataDownloader(max_concurrent=5) as dl:
            ...
    """

    def __init__(self, max_concurrent: int = 5) -> None:
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BinanceDataDownloader":
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "AtomiCortex/1.0 (data-ingestion)"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def _s(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError(
                "BinanceDataDownloader must be used as an async context manager."
            )
        return self._session

    # ------------------------------------------------------------------
    # Private: checksum fetch
    # ------------------------------------------------------------------

    async def _fetch_checksum(self, checksum_url: str) -> str | None:
        """Fetch expected SHA-256 from a .CHECKSUM file.  Returns None if unavailable."""
        try:
            async with self._s.get(checksum_url) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                text = await resp.text()
                parts = text.split()
                return parts[0].lower() if parts else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private: core download (semaphore + checksum + retry)
    # ------------------------------------------------------------------

    async def _do_download(self, url: str, dest_path: Path) -> DownloadStatus:
        """Download *url* to *dest_path* with full safety checks.

        Concurrency is limited by self._semaphore.

        Returns
        -------
        "ok"        — file downloaded successfully
        "skipped"   — file already existed with valid checksum
        "not_found" — HTTP 404, date has no data (normal)
        "error"     — all retries exhausted or unrecoverable error
        """
        async with self._semaphore:
            expected_hash = await self._fetch_checksum(url + ".CHECKSUM")

            # Already on disk and valid — skip.
            if dest_path.exists() and dest_path.stat().st_size > 0:
                if expected_hash is None or _verify_sha256(dest_path, expected_hash):
                    _log.debug(f"Skip (valid): {dest_path.name}")
                    return "skipped"

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = dest_path.with_suffix(".tmp")

            for attempt in range(MAX_RETRIES):
                try:
                    async with self._s.get(url) as resp:
                        if resp.status == 404:
                            _log.debug(f"Not found (404): {url}")
                            return "not_found"

                        if resp.status == 429:
                            _log.warning(
                                f"Rate limited — sleeping {RATE_LIMIT_DELAY:.0f}s"
                            )
                            await asyncio.sleep(RATE_LIMIT_DELAY)
                            continue

                        resp.raise_for_status()

                        with tmp_path.open("wb") as fh:
                            async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                                fh.write(chunk)

                    # Checksum verification
                    if expected_hash and not _verify_sha256(tmp_path, expected_hash):
                        tmp_path.unlink(missing_ok=True)
                        _log.warning(
                            f"Checksum mismatch on attempt {attempt + 1}: {dest_path.name}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                            continue
                        return "error"

                    tmp_path.rename(dest_path)
                    _log.debug(f"Downloaded: {dest_path.name}")
                    return "ok"

                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    tmp_path.unlink(missing_ok=True)
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        _log.warning(
                            f"Attempt {attempt + 1}/{MAX_RETRIES} failed "
                            f"({type(exc).__name__}: {exc}), retrying in {delay:.0f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        _log.error(f"All retries exhausted for {url}: {exc}")
                        return "error"

            return "error"

    # ------------------------------------------------------------------
    # Public: download_file (per spec)
    # ------------------------------------------------------------------

    async def download_file(self, url: str, dest_path: Path) -> bool:
        """Download *url* to *dest_path* with checksum verification and retry.

        Returns True on success (downloaded or already valid), False on 404
        or unrecoverable error.
        """
        status = await self._do_download(url, dest_path)
        return status in ("ok", "skipped")

    # ------------------------------------------------------------------
    # Private: single-day download + extraction
    # ------------------------------------------------------------------

    async def _download_day(
        self, url: str, zip_path: Path, csv_path: Path
    ) -> DayResult:
        """Download one day's ZIP and extract to CSV.

        Returns (csv_path, status) where status is one of:
        "downloaded", "skipped", "not_found", "error"
        """
        # Fast-path: CSV already present.
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return csv_path, "skipped"

        status = await self._do_download(url, zip_path)

        if status == "not_found":
            return None, "not_found"
        if status == "error":
            zip_path.unlink(missing_ok=True)
            return None, "error"

        # "ok" or "skipped" (zip exists from a previous interrupted run)
        try:
            csv = extract_zip(zip_path)
            return csv, "downloaded"
        except zipfile.BadZipFile:
            _log.warning(f"Corrupted ZIP, removing: {zip_path.name}")
            zip_path.unlink(missing_ok=True)
            return None, "error"
        except Exception as exc:
            _log.error(f"Extract error for {zip_path.name}: {exc}")
            zip_path.unlink(missing_ok=True)
            return None, "error"

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _date_range(start: date, end: date) -> list[date]:
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]

    @staticmethod
    def _symbol_data_dir(base_dir: Path, symbol: str, data_type: str) -> Path:
        """Return the canonical storage directory for one symbol + data type."""
        return base_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / data_type

    @staticmethod
    def _collect_paths(results: list[Any]) -> list[Path]:
        """Filter asyncio.gather results, returning only successful Path values."""
        paths: list[Path] = []
        for r in results:
            if isinstance(r, Exception):
                _log.error(f"Unexpected gather exception: {r}")
            elif isinstance(r, tuple) and r[0] is not None:
                paths.append(r[0])
        return paths

    # ------------------------------------------------------------------
    # Public: typed downloaders
    # ------------------------------------------------------------------

    async def download_klines(
        self,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        dest_dir: Path,
    ) -> list[Path]:
        """Download daily kline ZIPs for *symbol* / *interval* over the date range.

        Parameters
        ----------
        symbol:    Binance symbol, e.g. ``"BTCUSDT"``
        interval:  Kline interval, e.g. ``"4h"`` or ``"1d"``
        dest_dir:  Base data directory (exchange/symbol sub-dirs created automatically)

        Returns a list of extracted CSV paths (may be shorter than the date range
        if some dates have no data).
        """
        data_dir = self._symbol_data_dir(dest_dir, symbol, f"klines_{interval}")
        data_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for day in self._date_range(start_date, end_date):
            ds = day.strftime("%Y-%m-%d")
            stem = f"{symbol}-{interval}-{ds}"
            url = (
                f"{BASE_URL}/data/futures/um/daily/klines"
                f"/{symbol}/{interval}/{stem}.zip"
            )
            tasks.append(
                self._download_day(url, data_dir / f"{stem}.zip", data_dir / f"{stem}.csv")
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._collect_paths(results)  # type: ignore[arg-type]

    async def download_funding_rate(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        dest_dir: Path,
    ) -> list[Path]:
        """Download daily funding-rate CSVs for *symbol*."""
        data_dir = self._symbol_data_dir(dest_dir, symbol, "funding_rate")
        data_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for day in self._date_range(start_date, end_date):
            ds = day.strftime("%Y-%m-%d")
            stem = f"{symbol}-fundingRate-{ds}"
            url = f"{BASE_URL}/data/futures/um/daily/fundingRate/{symbol}/{stem}.zip"
            tasks.append(
                self._download_day(url, data_dir / f"{stem}.zip", data_dir / f"{stem}.csv")
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._collect_paths(results)  # type: ignore[arg-type]

    async def download_metrics(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        dest_dir: Path,
    ) -> list[Path]:
        """Download daily open-interest / L-S ratio / taker-volume metrics for *symbol*."""
        data_dir = self._symbol_data_dir(dest_dir, symbol, "metrics")
        data_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for day in self._date_range(start_date, end_date):
            ds = day.strftime("%Y-%m-%d")
            stem = f"{symbol}-metrics-{ds}"
            url = f"{BASE_URL}/data/futures/um/daily/metrics/{symbol}/{stem}.zip"
            tasks.append(
                self._download_day(url, data_dir / f"{stem}.zip", data_dir / f"{stem}.csv")
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._collect_paths(results)  # type: ignore[arg-type]

    async def download_agg_trades(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        dest_dir: Path,
    ) -> list[Path]:
        """Download daily aggregated trades CSVs for *symbol*."""
        data_dir = self._symbol_data_dir(dest_dir, symbol, "agg_trades")
        data_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for day in self._date_range(start_date, end_date):
            ds = day.strftime("%Y-%m-%d")
            stem = f"{symbol}-aggTrades-{ds}"
            url = f"{BASE_URL}/data/futures/um/daily/aggTrades/{symbol}/{stem}.zip"
            tasks.append(
                self._download_day(url, data_dir / f"{stem}.zip", data_dir / f"{stem}.csv")
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._collect_paths(results)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public: download_all (orchestrator)
    # ------------------------------------------------------------------

    async def download_all(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        base_dir: Path,
        include_agg_trades: bool = True,
        kline_intervals: list[str] | None = None,
    ) -> dict[str, Any]:
        """Download all data types for all symbols over the date range.

        Runs up to ``max_concurrent`` simultaneous downloads (set in __init__).

        Parameters
        ----------
        kline_intervals:
            Kline intervals to fetch (e.g. ``["4h", "1d", "1h", "15m"]``).
            Defaults to ``["4h", "1d"]`` for backward compatibility.

        Returns
        -------
        dict with keys:
            downloaded      — newly fetched files
            skipped         — files that were already present
            not_found       — dates with no data on Binance (HTTP 404, normal)
            failed          — actual errors (network / checksum / extract)
            errors          — list of descriptive strings for failed tasks
            elapsed_seconds — wall-clock time
        """
        import time

        if kline_intervals is None:
            kline_intervals = ["4h", "1d"]

        t0 = time.monotonic()
        stats: dict[str, Any] = {
            "downloaded": 0,
            "skipped": 0,
            "not_found": 0,
            "failed": 0,
            "errors": [],
            "elapsed_seconds": 0.0,
        }

        days = self._date_range(start_date, end_date)

        # Build the full task list with labels for error reporting.
        task_coros: list[Any] = []
        task_labels: list[str] = []

        for symbol in symbols:
            # klines (configurable intervals — default: 4h + 1d)
            for interval in kline_intervals:
                d = self._symbol_data_dir(base_dir, symbol, f"klines_{interval}")
                d.mkdir(parents=True, exist_ok=True)
                for day in days:
                    ds = day.strftime("%Y-%m-%d")
                    stem = f"{symbol}-{interval}-{ds}"
                    url = (
                        f"{BASE_URL}/data/futures/um/daily/klines"
                        f"/{symbol}/{interval}/{stem}.zip"
                    )
                    task_coros.append(
                        self._download_day(url, d / f"{stem}.zip", d / f"{stem}.csv")
                    )
                    task_labels.append(f"{symbol} klines_{interval} {ds}")

            # funding rate
            fr_dir = self._symbol_data_dir(base_dir, symbol, "funding_rate")
            fr_dir.mkdir(parents=True, exist_ok=True)
            for day in days:
                ds = day.strftime("%Y-%m-%d")
                stem = f"{symbol}-fundingRate-{ds}"
                url = f"{BASE_URL}/data/futures/um/daily/fundingRate/{symbol}/{stem}.zip"
                task_coros.append(
                    self._download_day(url, fr_dir / f"{stem}.zip", fr_dir / f"{stem}.csv")
                )
                task_labels.append(f"{symbol} funding_rate {ds}")

            # metrics
            m_dir = self._symbol_data_dir(base_dir, symbol, "metrics")
            m_dir.mkdir(parents=True, exist_ok=True)
            for day in days:
                ds = day.strftime("%Y-%m-%d")
                stem = f"{symbol}-metrics-{ds}"
                url = f"{BASE_URL}/data/futures/um/daily/metrics/{symbol}/{stem}.zip"
                task_coros.append(
                    self._download_day(url, m_dir / f"{stem}.zip", m_dir / f"{stem}.csv")
                )
                task_labels.append(f"{symbol} metrics {ds}")

            # aggTrades (optional)
            if include_agg_trades:
                at_dir = self._symbol_data_dir(base_dir, symbol, "agg_trades")
                at_dir.mkdir(parents=True, exist_ok=True)
                for day in days:
                    ds = day.strftime("%Y-%m-%d")
                    stem = f"{symbol}-aggTrades-{ds}"
                    url = f"{BASE_URL}/data/futures/um/daily/aggTrades/{symbol}/{stem}.zip"
                    task_coros.append(
                        self._download_day(
                            url, at_dir / f"{stem}.zip", at_dir / f"{stem}.csv"
                        )
                    )
                    task_labels.append(f"{symbol} agg_trades {ds}")

        # Run everything with a tqdm progress bar.
        raw_results: list[Any] = await atqdm.gather(
            *task_coros,
            desc="Downloading",
            total=len(task_coros),
            unit="file",
        )

        for result, label in zip(raw_results, task_labels):
            if isinstance(result, Exception):
                stats["failed"] += 1
                stats["errors"].append(f"{label}: {result}")
                continue

            _path, status = result  # DayResult
            if status == "downloaded":
                stats["downloaded"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "not_found":
                stats["not_found"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(label)

        stats["elapsed_seconds"] = time.monotonic() - t0
        return stats
