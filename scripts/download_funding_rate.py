#!/usr/bin/env python
"""
scripts/download_funding_rate.py

Downloads historical funding rate data from Binance USDT-M Futures REST API.

    GET https://fapi.binance.com/fapi/v1/fundingRate

Paginates in 1000-record chunks.  Saves one CSV file per calendar month:

    {data_dir}/exchange=BINANCE_UM/symbol={SYMBOL}/funding_rate/
    {SYMBOL}-fundingRate-{YYYY-MM}.csv

CSV columns: fundingTime, fundingRate, markPrice, symbol

Usage
-----
    python scripts/download_funding_rate.py \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --start 2024-01-01 \\
        --end   2025-12-31
"""

from __future__ import annotations

import asyncio
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import click

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://fapi.binance.com"
ENDPOINT = "/fapi/v1/fundingRate"
PAGE_LIMIT = 1_000
SLEEP_BETWEEN = 0.5    # seconds — stays well under Binance's 2400 weight/min
MAX_RETRIES = 3
RETRY_BASE = 1.0       # seconds; doubled each attempt

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Low-level fetch — one page, with retry
# ---------------------------------------------------------------------------

async def _fetch_page(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    """Fetch up to PAGE_LIMIT funding rate records.

    Returns an empty list when the range has no data.
    Raises on unrecoverable errors after MAX_RETRIES attempts.
    """
    params: dict[str, Any] = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": PAGE_LIMIT,
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(API_BASE + ENDPOINT, params=params) as resp:
                if resp.status == 429:
                    wait = float(resp.headers.get("Retry-After", 60))
                    _log.warning(f"Rate limited — sleeping {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue

                if resp.status >= 400:
                    body = await resp.text()
                    _log.error(f"HTTP {resp.status} for {symbol}: {body[:200]}")
                    resp.raise_for_status()

                return await resp.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE * (2 ** attempt)
                _log.warning(
                    f"{symbol}: attempt {attempt + 1}/{MAX_RETRIES} failed "
                    f"({type(exc).__name__}: {exc}), retry in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                _log.error(f"{symbol}: all retries exhausted — {exc}")
                raise

    return []


# ---------------------------------------------------------------------------
# Per-symbol download + monthly CSV writer
# ---------------------------------------------------------------------------

async def download_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    data_dir: Path,
) -> int:
    """Download all funding rate records for *symbol* over the date range.

    Writes one CSV file per calendar month under:
        {data_dir}/exchange=BINANCE_UM/symbol={symbol}/funding_rate/

    Returns the total number of records written.
    """
    _log.info(f"[{symbol}] downloading funding rate …")

    all_records: list[dict[str, Any]] = []
    cursor = start_ms

    while cursor <= end_ms:
        page = await _fetch_page(session, symbol, cursor, end_ms)

        if not page:
            break

        all_records.extend(page)
        last_ts = int(page[-1]["fundingTime"])
        _log.debug(
            f"[{symbol}] +{len(page)} records (total {len(all_records)}), "
            f"last={datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).isoformat()}"
        )

        # Stop when we received a partial page or reached the end boundary.
        if len(page) < PAGE_LIMIT or last_ts >= end_ms:
            break

        cursor = last_ts + 1          # advance past the last fetched timestamp
        await asyncio.sleep(SLEEP_BETWEEN)

    if not all_records:
        _log.warning(f"[{symbol}] no data returned")
        return 0

    # ------------------------------------------------------------------
    # Group by calendar month and write CSV files
    # ------------------------------------------------------------------
    monthly: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_records:
        ts_ms = int(rec["fundingTime"])
        month = datetime.fromtimestamp(ts_ms / 1_000, tz=timezone.utc).strftime("%Y-%m")
        monthly[month].append(rec)

    out_dir = data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / "funding_rate"
    out_dir.mkdir(parents=True, exist_ok=True)

    for month, records in sorted(monthly.items()):
        records.sort(key=lambda r: int(r["fundingTime"]))
        csv_path = out_dir / f"{symbol}-fundingRate-{month}.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fundingTime", "fundingRate", "markPrice", "symbol"])
            for r in records:
                writer.writerow([
                    int(r["fundingTime"]),
                    r.get("fundingRate", ""),
                    r.get("markPrice", ""),
                    r.get("symbol", symbol),
                ])

        _log.debug(f"[{symbol}] wrote {len(records)} rows → {csv_path.name}")

    _log.info(
        f"[{symbol}] done — {len(all_records)} records, "
        f"{len(monthly)} monthly files"
    )
    return len(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--symbols",
    default="BTCUSDT,ETHUSDT,SOLUSDT",
    show_default=True,
    help="Comma-separated USDT-M futures symbols.",
)
@click.option(
    "--start",
    default="2024-01-01",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (inclusive), YYYY-MM-DD.",
)
@click.option(
    "--end",
    default="2025-12-31",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (inclusive), YYYY-MM-DD.",
)
@click.option(
    "--data-dir",
    default="/mnt/hdd/AtomiCortex/data/raw",
    show_default=True,
    type=click.Path(),
    help="Base raw data directory.",
)
def main(symbols: str, start: datetime, end: datetime, data_dir: str) -> None:
    """Download funding rate history from Binance Futures REST API."""
    setup_logging()

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1_000)
    # Include the full last day
    end_ms = int(end.replace(tzinfo=timezone.utc).timestamp() * 1_000) + 86_400_000 - 1
    base_dir = Path(data_dir)

    click.echo(
        f"\n{'─'*56}\n"
        f"  Symbols  : {', '.join(symbol_list)}\n"
        f"  Start    : {start.date()}\n"
        f"  End      : {end.date()}\n"
        f"  Data dir : {base_dir}\n"
        f"{'─'*56}\n"
    )

    async def _run() -> dict[str, int]:
        connector = aiohttp.TCPConnector(limit=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        counts: dict[str, int] = {}

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for sym in symbol_list:
                try:
                    n = await download_symbol(session, sym, start_ms, end_ms, base_dir)
                    counts[sym] = n
                except Exception as exc:  # noqa: BLE001
                    _log.error(f"[{sym}] failed: {exc}")
                    counts[sym] = -1
                # Small gap between symbols to be polite to the API
                await asyncio.sleep(SLEEP_BETWEEN)

        return counts

    counts = asyncio.run(_run())

    click.echo(f"\n{'─'*56}")
    total = 0
    for sym, n in counts.items():
        if n >= 0:
            click.echo(f"  ✅  {sym:<10} {n:>6} records")
            total += n
        else:
            click.echo(f"  ❌  {sym:<10} FAILED")
    click.echo(f"{'─'*56}")
    click.echo(f"  Total    : {total} records\n")

    sys.exit(0 if all(n >= 0 for n in counts.values()) else 1)


if __name__ == "__main__":
    main()
