#!/usr/bin/env python
"""
scripts/download_historical.py

CLI for downloading historical USDT-M futures data from Binance Data Portal.

Commands
--------
    download   — fetch klines, funding-rate, metrics (and optionally aggTrades)
    status     — show what is already on disk

Examples
--------
    python scripts/download_historical.py download \\
        --symbols BTCUSDT,ETHUSDT \\
        --start 2024-01-01 \\
        --end   2025-12-31 \\
        --data-dir /mnt/hdd/AtomiCortex/data/raw \\
        --no-agg-trades

    python scripts/download_historical.py status \\
        --data-dir /mnt/hdd/AtomiCortex/data/raw
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running the script directly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.binance_downloader import BinanceDataDownloader
from src.logger import get_logger, setup_logging


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """AtomiCortex — Binance historical data downloader."""
    setup_logging()
    ctx.ensure_object(dict)


# ---------------------------------------------------------------------------
# download command
# ---------------------------------------------------------------------------

@cli.command("download")
@click.option(
    "--symbols",
    required=True,
    metavar="BTCUSDT,ETHUSDT",
    help="Comma-separated Binance symbols.",
)
@click.option(
    "--start",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (inclusive), YYYY-MM-DD.",
)
@click.option(
    "--end",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (inclusive), YYYY-MM-DD.",
)
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(),
    help="Base directory for raw data storage.",
)
@click.option(
    "--no-agg-trades",
    is_flag=True,
    default=False,
    help="Skip aggTrades (large files, ~10× bigger than klines).",
)
@click.option(
    "--concurrent",
    default=5,
    show_default=True,
    help="Max simultaneous downloads.",
)
@click.option(
    "--intervals",
    default="4h,1d",
    show_default=True,
    help="Comma-separated kline intervals, e.g. '4h,1d,1h,15m'.",
)
def cmd_download(
    symbols: str,
    start: datetime,
    end: datetime,
    data_dir: str,
    no_agg_trades: bool,
    concurrent: int,
    intervals: str,
) -> None:
    """Download historical futures data from https://data.binance.vision."""
    log = get_logger(__name__)

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    intervals_list = [i.strip() for i in intervals.split(",") if i.strip()]
    start_date: date = start.date()
    end_date: date = end.date()
    base_dir = Path(data_dir)

    if start_date > end_date:
        click.echo("❌ --start must be before --end", err=True)
        sys.exit(1)

    if not intervals_list:
        click.echo("❌ --intervals must contain at least one interval", err=True)
        sys.exit(1)

    n_days = (end_date - start_date).days + 1
    data_types = [f"klines_{i}" for i in intervals_list] + ["funding_rate", "metrics"]
    if not no_agg_trades:
        data_types.append("agg_trades")

    click.echo(
        f"\n{'─'*56}\n"
        f"  Symbols     : {', '.join(symbol_list)}\n"
        f"  Date range  : {start_date} → {end_date}  ({n_days} days)\n"
        f"  Data types  : {', '.join(data_types)}\n"
        f"  Destination : {base_dir}\n"
        f"  Concurrent  : {concurrent}\n"
        f"{'─'*56}\n"
    )

    log.info(
        f"Starting download: symbols={symbol_list} "
        f"{start_date}→{end_date} agg_trades={not no_agg_trades}"
    )

    async def _run() -> dict:
        async with BinanceDataDownloader(max_concurrent=concurrent) as dl:
            return await dl.download_all(
                symbols=symbol_list,
                start_date=start_date,
                end_date=end_date,
                base_dir=base_dir,
                include_agg_trades=not no_agg_trades,
                kline_intervals=intervals_list,
            )

    stats = asyncio.run(_run())

    click.echo(
        f"\n{'─'*56}\n"
        f"  ✅  Downloaded : {stats['downloaded']:>6}\n"
        f"  ⏭️   Skipped    : {stats['skipped']:>6}\n"
        f"  🔍  Not found  : {stats['not_found']:>6}  (404 — normal for some dates)\n"
        f"  ❌  Failed     : {stats['failed']:>6}\n"
        f"  ⏱️   Elapsed    : {stats['elapsed_seconds']:>6.1f}s\n"
        f"{'─'*56}"
    )

    if stats["errors"]:
        click.echo(f"\nFailed tasks (first 15):")
        for e in stats["errors"][:15]:
            click.echo(f"  • {e}")

    log.info(
        f"Download complete: downloaded={stats['downloaded']} "
        f"skipped={stats['skipped']} failed={stats['failed']}"
    )

    sys.exit(1 if stats["failed"] > 0 else 0)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

@cli.command("status")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(),
    help="Base directory used during download.",
)
def cmd_status(data_dir: str) -> None:
    """Show how much data is already on disk."""
    base_dir = Path(data_dir)
    exchange_dir = base_dir / "exchange=BINANCE_UM"

    if not exchange_dir.exists():
        click.echo(f"\n⚠️  No data found under {exchange_dir}\n")
        return

    click.echo(f"\n{'─'*60}")
    click.echo(f"  Data directory: {base_dir}")
    click.echo(f"{'─'*60}\n")

    total_files = 0
    total_bytes = 0

    for symbol_dir in sorted(exchange_dir.iterdir()):
        if not symbol_dir.is_dir() or not symbol_dir.name.startswith("symbol="):
            continue
        symbol = symbol_dir.name.split("=", 1)[1]
        click.echo(f"  {symbol}")

        for dtype_dir in sorted(symbol_dir.iterdir()):
            if not dtype_dir.is_dir():
                continue
            csv_files = sorted(dtype_dir.glob("*.csv"))
            if not csv_files:
                continue

            size_mb = sum(f.stat().st_size for f in csv_files) / 1_048_576
            total_files += len(csv_files)
            total_bytes += sum(f.stat().st_size for f in csv_files)

            # Date range from filenames: last 10 chars are YYYY-MM-DD
            dates = sorted(f.stem[-10:] for f in csv_files)
            date_range = f"{dates[0]} → {dates[-1]}" if dates else "—"

            click.echo(
                f"    {dtype_dir.name:<22} "
                f"{len(csv_files):>5} files  "
                f"{size_mb:>8.1f} MB  "
                f"[{date_range}]"
            )
        click.echo()

    total_mb = total_bytes / 1_048_576
    total_gb = total_mb / 1024
    click.echo(f"{'─'*60}")
    click.echo(
        f"  Total: {total_files} files  "
        f"{total_mb:.1f} MB  ({total_gb:.2f} GB)\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
