#!/usr/bin/env python
"""
Multi-timeframe historical data downloader for AtomiCortex v2.0
Downloads klines from Binance Data Portal for 1m, 5m, 15m, 1h intervals.

Does NOT touch existing 4H/1D data.

Usage:
  # Download one interval:
  python scripts/download_mtf_data.py --interval 1h --symbol BTCUSDT
  python scripts/download_mtf_data.py --interval 15m --symbol BTCUSDT
  python scripts/download_mtf_data.py --interval 5m --symbol BTCUSDT
  python scripts/download_mtf_data.py --interval 1m --symbol BTCUSDT

  # Download all new intervals (1h, 15m, 5m, 1m):
  python scripts/download_mtf_data.py --all --symbol BTCUSDT

  # With explicit date range:
  python scripts/download_mtf_data.py --interval 1h --symbol BTCUSDT \
      --start 2023-01 --end 2025-12

  # Dry run (show what would be downloaded, don't download):
  python scripts/download_mtf_data.py --all --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running the script directly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://data.binance.vision"
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_START = "2023-01"
DEFAULT_END = "2025-12"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds; doubled each attempt
MAX_WORKERS = 4
CHUNK_SIZE = 65_536       # 64 KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_month_range(start_str: str, end_str: str) -> list[tuple[int, int]]:
    """Generate list of (year, month) tuples from ``YYYY-MM`` strings."""
    start_y, start_m = map(int, start_str.split("-"))
    end_y, end_m = map(int, end_str.split("-"))

    months: list[tuple[int, int]] = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def build_url(symbol: str, interval: str, year: int, month: int) -> str:
    """Build Binance Data Portal URL for a monthly klines ZIP."""
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    return (
        f"{BASE_URL}/data/futures/um/monthly/klines"
        f"/{symbol}/{interval}/{filename}"
    )


def build_checksum_url(url: str) -> str:
    """Derive checksum URL from the data URL."""
    return f"{url}.CHECKSUM"


def dest_dir_for(base_dir: Path, symbol: str, interval: str) -> Path:
    """Return canonical storage directory for one symbol + interval."""
    return base_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / f"interval={interval}"


def fetch_checksum(url: str, session: requests.Session) -> str | None:
    """Fetch expected SHA-256 from a .CHECKSUM file. Returns None if unavailable."""
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        parts = resp.text.strip().split()
        return parts[0].lower() if parts else None
    except Exception:
        return None


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Return True when file's SHA-256 matches *expected_hash*."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash.lower().strip()


# ---------------------------------------------------------------------------
# Core: download + extract one monthly file
# ---------------------------------------------------------------------------

def download_and_extract(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    dest_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download one monthly ZIP, verify checksum, extract CSV, remove ZIP.

    Returns a dict with ``status``, ``file``, ``message``.
    Possible status values: downloaded, skipped, not_found, error, dry_run.
    """
    url = build_url(symbol, interval, year, month)
    stem = f"{symbol}-{interval}-{year}-{month:02d}"
    csv_path = dest_dir / f"{stem}.csv"
    zip_path = dest_dir / f"{stem}.zip"

    result: dict[str, Any] = {
        "year": year, "month": month, "interval": interval,
        "file": str(csv_path),
    }

    if dry_run:
        result["status"] = "dry_run"
        result["message"] = f"Would download: {url}"
        return result

    # Resume: skip if CSV already exists and is non-empty.
    if csv_path.exists() and csv_path.stat().st_size > 0:
        result["status"] = "skipped"
        result["message"] = f"Already exists: {csv_path.name}"
        return result

    dest_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "AtomiCortex/2.0 (mtf-ingestion)"})

    # Fetch expected checksum
    expected_hash = fetch_checksum(build_checksum_url(url), session)

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, stream=True, timeout=120)

            if resp.status_code == 404:
                result["status"] = "not_found"
                result["message"] = f"Not found (404): {stem}"
                return result

            resp.raise_for_status()

            tmp_path = zip_path.with_suffix(".tmp")
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)

            # Checksum verification
            if expected_hash and not verify_sha256(tmp_path, expected_hash):
                tmp_path.unlink(missing_ok=True)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    _log.warning(
                        f"Checksum mismatch {stem}, retry in {delay:.0f}s"
                    )
                    time.sleep(delay)
                    continue
                result["status"] = "error"
                result["message"] = f"Checksum mismatch after {MAX_RETRIES} attempts"
                return result

            tmp_path.rename(zip_path)
            break

        except requests.RequestException as exc:
            zip_path.with_suffix(".tmp").unlink(missing_ok=True)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                _log.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {stem}: "
                    f"{exc}, retry in {delay:.0f}s"
                )
                time.sleep(delay)
            else:
                result["status"] = "error"
                result["message"] = f"All retries exhausted: {exc}"
                return result

    # Extract ZIP → CSV
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found in {zip_path.name}")
            zf.extract(csv_names[0], zip_path.parent)

            extracted = zip_path.parent / csv_names[0]
            if extracted != csv_path:
                extracted.rename(csv_path)

        zip_path.unlink(missing_ok=True)
        result["status"] = "downloaded"
        result["message"] = (
            f"OK: {csv_path.name} ({csv_path.stat().st_size:,} bytes)"
        )
        return result

    except (zipfile.BadZipFile, ValueError) as exc:
        zip_path.unlink(missing_ok=True)
        result["status"] = "error"
        result["message"] = f"Extract failed: {exc}"
        return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_download(
    symbol: str,
    intervals: list[str],
    start: str,
    end: str,
    data_dir: Path,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS,
) -> dict[str, int]:
    """Orchestrate parallel download of all months × intervals."""
    months = build_month_range(start, end)

    tasks: list[tuple[str, str, int, int, Path, bool]] = []
    for interval in intervals:
        dest = dest_dir_for(data_dir, symbol, interval)
        for year, month in months:
            tasks.append((symbol, interval, year, month, dest, dry_run))

    stats: dict[str, int] = {
        "downloaded": 0, "skipped": 0, "not_found": 0, "error": 0, "dry_run": 0,
    }
    errors: list[str] = []

    print(f"\n{'─' * 60}")
    print(f"  Symbol      : {symbol}")
    print(f"  Intervals   : {', '.join(intervals)}")
    print(f"  Period      : {start} → {end} ({len(months)} months)")
    print(f"  Total files : {len(tasks)}")
    print(f"  Workers     : {max_workers}")
    print(f"  Dry run     : {dry_run}")
    print(f"{'─' * 60}\n")

    if dry_run:
        for task in tasks:
            result = download_and_extract(*task)
            print(f"  {result['message']}")
            stats["dry_run"] += 1
        return stats

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_and_extract, *t): t for t in tasks}

        with tqdm(total=len(tasks), desc="Downloading", unit="file") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                status = result["status"]
                stats[status] = stats.get(status, 0) + 1

                if status == "error":
                    errors.append(result["message"])
                    _log.error(result["message"])
                elif status == "downloaded":
                    _log.debug(result["message"])

                pbar.update(1)

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  ✅ Downloaded : {stats['downloaded']:>6}")
    print(f"  ⏭️  Skipped    : {stats['skipped']:>6}")
    print(f"  🔍 Not found  : {stats['not_found']:>6}")
    print(f"  ❌ Failed     : {stats['error']:>6}")
    print(f"{'─' * 60}")

    if errors:
        print("\nErrors (first 10):")
        for e in errors[:10]:
            print(f"  • {e}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the MTF data downloader."""
    parser = argparse.ArgumentParser(
        description="AtomiCortex MTF data downloader (Binance Data Portal)",
    )
    parser.add_argument(
        "--symbol", default=DEFAULT_SYMBOL,
        help=f"Binance symbol (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--interval", choices=MTF_INTERVALS,
        help="Single interval to download",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_intervals",
        help="Download all MTF intervals (1h, 15m, 5m, 1m)",
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start month YYYY-MM (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help=f"End month YYYY-MM (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Base data directory (default: data/raw)",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_WORKERS,
        help=f"Max parallel downloads (default: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )

    args = parser.parse_args()
    setup_logging()

    if not args.interval and not args.all_intervals:
        parser.error("Specify --interval or --all")

    intervals = MTF_INTERVALS if args.all_intervals else [args.interval]
    data_dir = Path(args.data_dir)

    _log.info(
        f"Starting MTF download: symbol={args.symbol} "
        f"intervals={intervals} {args.start}→{args.end}"
    )

    stats = run_download(
        symbol=args.symbol.upper(),
        intervals=intervals,
        start=args.start,
        end=args.end,
        data_dir=data_dir,
        dry_run=args.dry_run,
        max_workers=args.workers,
    )

    sys.exit(1 if stats.get("error", 0) > 0 else 0)


if __name__ == "__main__":
    main()
