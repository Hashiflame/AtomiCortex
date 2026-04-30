#!/usr/bin/env python
"""
scripts/convert_to_parquet.py

CLI for converting raw Binance CSV files to Parquet and inspecting the store.

Commands
--------
    convert  — CSV → Parquet (ZSTD-3) with parallel workers
    summary  — table of what is already in the Parquet store

Examples
--------
    python scripts/convert_to_parquet.py convert \\
        --raw-dir /mnt/hdd/AtomiCortex/data/raw \\
        --output-dir /mnt/hdd/AtomiCortex/data/features \\
        --symbols BTCUSDT,ETHUSDT \\
        --workers 4

    python scripts/convert_to_parquet.py summary \\
        --data-dir /mnt/hdd/AtomiCortex/data/features
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.data_store import DataStore
from src.ingestion.parquet_converter import ParquetConverter, _CONFIGS
from src.logger import get_logger, setup_logging


@click.group()
def cli() -> None:
    """AtomiCortex — Parquet conversion utilities."""
    setup_logging()


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

@cli.command("convert")
@click.option("--raw-dir", required=True, type=click.Path(), help="Raw CSV data directory.")
@click.option("--output-dir", required=True, type=click.Path(), help="Parquet output directory.")
@click.option(
    "--symbols",
    required=True,
    metavar="BTCUSDT,ETHUSDT",
    help="Comma-separated symbols.",
)
@click.option(
    "--data-types",
    default=None,
    metavar="klines_4h,klines_1d,metrics",
    help=f"Comma-separated data types to convert. Default: all ({', '.join(_CONFIGS)}).",
)
@click.option("--workers", default=4, show_default=True, help="Parallel worker processes.")
@click.option(
    "--compression-level",
    default=3,
    show_default=True,
    help="ZSTD compression level (1-22).",
)
def cmd_convert(
    raw_dir: str,
    output_dir: str,
    symbols: str,
    data_types: str | None,
    workers: int,
    compression_level: int,
) -> None:
    """Convert raw CSV files to ZSTD-compressed Parquet."""
    log = get_logger(__name__)

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    dt_list = (
        [d.strip() for d in data_types.split(",") if d.strip()]
        if data_types
        else None
    )

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    click.echo(
        f"\n{'─'*60}\n"
        f"  Raw dir    : {raw_path}\n"
        f"  Output dir : {out_path}\n"
        f"  Symbols    : {', '.join(symbol_list)}\n"
        f"  Data types : {', '.join(dt_list) if dt_list else 'all'}\n"
        f"  Workers    : {workers}\n"
        f"  ZSTD level : {compression_level}\n"
        f"{'─'*60}\n"
    )

    converter = ParquetConverter()
    stats = converter.convert_directory(
        raw_dir=raw_path,
        output_dir=out_path,
        symbols=symbol_list,
        data_types=dt_list,
        workers=workers,
        compression_level=compression_level,
    )

    click.echo(
        f"\n{'─'*60}\n"
        f"  ✅  Converted      : {stats['converted']:>6}\n"
        f"  ⏭️   Skipped (ok)   : {stats['skipped']:>6}\n"
        f"  🔍  Skipped (empty): {stats['skipped_empty']:>6}\n"
        f"  ❌  Failed         : {stats['failed']:>6}\n"
        f"  ⏱️   Elapsed        : {stats['elapsed_seconds']:>6.1f}s\n"
        f"{'─'*60}"
    )

    if stats["errors"]:
        click.echo("\nFailed files (first 10):")
        for e in stats["errors"][:10]:
            click.echo(f"  • {e}")

    log.info(
        f"Conversion complete: "
        f"converted={stats['converted']} skipped={stats['skipped']} failed={stats['failed']}"
    )
    sys.exit(1 if stats["failed"] > 0 else 0)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

@cli.command("summary")
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(),
    help="Parquet feature store directory.",
)
def cmd_summary(data_dir: str) -> None:
    """Show a summary table of all available Parquet data."""
    feat_dir = Path(data_dir)
    if not feat_dir.exists():
        click.echo(f"\n⚠️  Directory not found: {feat_dir}\n")
        sys.exit(1)

    store = DataStore(feat_dir)
    summary = store.get_data_summary()
    store.close()

    if not summary:
        click.echo(f"\n⚠️  No Parquet data found under {feat_dir}\n")
        return

    # Print table
    HDR = f"  {'Symbol':<10} {'Data type':<22} {'Rows':>10} {'Files':>6} {'Size MB':>9}  Date range"
    SEP = "  " + "─" * (len(HDR) - 2)

    click.echo(f"\n{'─'*70}")
    click.echo(f"  Parquet store: {feat_dir}")
    click.echo(f"{'─'*70}")
    click.echo(HDR)
    click.echo(SEP)

    total_rows = 0
    total_mb = 0.0

    for entry in sorted(summary.values(), key=lambda e: (e["symbol"], e["data_type"])):
        rows = entry["row_count"]
        mb = entry["size_mb"]
        total_rows += max(rows, 0)
        total_mb += mb
        rows_str = f"{rows:>10,}" if rows >= 0 else "       ?"
        click.echo(
            f"  {entry['symbol']:<10} {entry['data_type']:<22} "
            f"{rows_str} {entry['file_count']:>6} {mb:>9.2f}  {entry['date_range']}"
        )

    click.echo(SEP)
    click.echo(
        f"  {'TOTAL':<10} {'':<22} {total_rows:>10,} {'':>6} {total_mb:>9.2f}\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
