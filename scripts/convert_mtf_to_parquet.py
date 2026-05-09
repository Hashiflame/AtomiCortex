#!/usr/bin/env python
"""
Converts downloaded MTF CSV klines to Parquet (ZSTD compression).
For 1m, 5m, 15m, 1h intervals.

Does NOT touch existing 4H/1D data or parquet files.

Usage:
  python scripts/convert_mtf_to_parquet.py --interval 1h
  python scripts/convert_mtf_to_parquet.py --interval 15m
  python scripts/convert_mtf_to_parquet.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MTF_INTERVALS = ["1m", "5m", "15m", "1h"]
DEFAULT_SYMBOL = "BTCUSDT"
ROW_GROUP_SIZE = 131_072   # 128K rows per row-group

# Binance monthly klines CSV columns (no header).
# Files may have 11 or 12 columns; the 12th ("ignore") is always 0.
KLINE_COLUMNS_12 = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]
KLINE_COLUMNS_11 = KLINE_COLUMNS_12[:-1]

KLINE_DTYPES: dict[str, type[pl.DataType]] = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Int64,
    "quote_volume": pl.Float64,
    "n_trades": pl.Int64,
    "taker_buy_base_vol": pl.Float64,
    "taker_buy_quote_vol": pl.Float64,
}

# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


def read_kline_csv(csv_path: Path) -> pl.DataFrame:
    """Read a Binance monthly klines CSV (no header, 11 or 12 columns)."""
    # Peek at first line to determine column count.
    with csv_path.open("r") as fh:
        first_line = fh.readline().strip()
    n_cols = len(first_line.split(","))

    columns = KLINE_COLUMNS_12 if n_cols >= 12 else KLINE_COLUMNS_11

    df = pl.read_csv(
        csv_path,
        has_header=False,
        new_columns=columns,
        try_parse_dates=False,
        null_values=["", "NA"],
        ignore_errors=True,
    )

    # Drop ignore column if present.
    if "ignore" in df.columns:
        df = df.drop("ignore")

    # Cast to target types.
    cast_exprs = [
        pl.col(col).cast(dtype, strict=False)
        for col, dtype in KLINE_DTYPES.items()
        if col in df.columns
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


# ---------------------------------------------------------------------------
# Single CSV → daily Parquet files
# ---------------------------------------------------------------------------


def convert_csv_to_parquet(
    csv_path: Path,
    symbol: str,
    interval: str,
    output_base: Path,
    compression: str = "zstd",
    compression_level: int = 3,
) -> list[Path]:
    """Convert one monthly CSV to per-day Parquet files.

    Returns the list of created Parquet paths.
    """
    df = read_kline_csv(csv_path)

    if df.is_empty():
        _log.debug(f"Empty CSV, skipping: {csv_path.name}")
        return []

    # Add derived columns.
    df = df.with_columns([
        pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp"),
        pl.lit(symbol).alias("symbol"),
        pl.lit(interval).alias("interval"),
        pl.lit("BINANCE_UM").alias("exchange"),
    ])

    df = df.sort("open_time")

    # Partition key: date string derived from timestamp.
    df = df.with_columns(
        pl.col("timestamp").dt.date().cast(pl.Utf8).alias("_date")
    )

    created: list[Path] = []
    for group in df.partition_by("_date", maintain_order=True):
        date_val = group["_date"][0]
        day_df = group.drop("_date").sort("open_time")

        parquet_dir = (
            output_base
            / "exchange=BINANCE_UM"
            / f"symbol={symbol}"
            / f"interval={interval}"
            / f"date={date_val}"
        )
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = parquet_dir / "klines.parquet"

        day_df.write_parquet(
            parquet_path,
            compression=compression,
            compression_level=compression_level,
            row_group_size=ROW_GROUP_SIZE,
            statistics=True,
        )
        created.append(parquet_path)

    _log.debug(
        f"Converted {csv_path.name} → {len(created)} daily parquet files "
        f"({len(df)} rows total)"
    )
    return created


# ---------------------------------------------------------------------------
# Batch converter for one interval
# ---------------------------------------------------------------------------


def convert_interval(
    symbol: str,
    interval: str,
    data_dir: Path,
    output_dir: Path | None = None,
    compression: str = "zstd",
    compression_level: int = 3,
) -> dict[str, Any]:
    """Convert all CSV files for one interval.

    Returns a stats dict with converted / skipped / failed counts.
    """
    t0 = time.monotonic()

    csv_dir = (
        data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / f"interval={interval}"
    )
    output_base = output_dir or data_dir

    stats: dict[str, Any] = {
        "interval": interval,
        "converted": 0,
        "skipped": 0,
        "failed": 0,
        "parquet_files": 0,
        "total_rows": 0,
        "errors": [],
    }

    if not csv_dir.exists():
        stats["errors"].append(f"Directory not found: {csv_dir}")
        return stats

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        stats["errors"].append(f"No CSV files in {csv_dir}")
        return stats

    for csv_path in tqdm(csv_files, desc=f"Converting {interval}", unit="file"):
        try:
            created = convert_csv_to_parquet(
                csv_path=csv_path,
                symbol=symbol,
                interval=interval,
                output_base=output_base,
                compression=compression,
                compression_level=compression_level,
            )
            if created:
                stats["converted"] += 1
                stats["parquet_files"] += len(created)
                # Count rows from created files directly to avoid re-reading.
                for pf in created:
                    stats["total_rows"] += pl.scan_parquet(pf).select(
                        pl.len()
                    ).collect().item()
            else:
                stats["skipped"] += 1
        except Exception as exc:
            stats["failed"] += 1
            stats["errors"].append(f"{csv_path.name}: {exc}")
            _log.error(f"Failed to convert {csv_path.name}: {exc}")

    stats["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the MTF CSV → Parquet converter."""
    parser = argparse.ArgumentParser(
        description="AtomiCortex MTF CSV → Parquet converter",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", choices=MTF_INTERVALS)
    parser.add_argument(
        "--all", action="store_true", dest="all_intervals",
        help="Convert all MTF intervals",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output base dir (default: same as data-dir)",
    )

    args = parser.parse_args()
    setup_logging()

    if not args.interval and not args.all_intervals:
        parser.error("Specify --interval or --all")

    intervals = MTF_INTERVALS if args.all_intervals else [args.interval]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"\n{'─' * 60}")
    print(f"  AtomiCortex MTF CSV → Parquet Converter")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Intervals : {', '.join(intervals)}")
    print(f"  Data dir  : {data_dir}")
    print(f"{'─' * 60}\n")

    for interval in intervals:
        stats = convert_interval(
            symbol=args.symbol.upper(),
            interval=interval,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        print(
            f"\n  {interval}: {stats['converted']} CSVs → "
            f"{stats['parquet_files']} parquet files "
            f"({stats['total_rows']:,} rows) "
            f"in {stats.get('elapsed_seconds', 0):.1f}s"
        )
        if stats["errors"]:
            for e in stats["errors"][:5]:
                print(f"    ⚠ {e}")

    print(f"\n{'─' * 60}")
    print("  ✅ Conversion complete")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
