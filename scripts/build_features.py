#!/usr/bin/env python3
"""
scripts/build_features.py

Build the ML feature matrix for one symbol and date range.

Usage
-----
    python scripts/build_features.py \
        --symbol BTCUSDT \
        --start 2024-01-01 \
        --end 2025-12-31 \
        --data-dir /mnt/hdd/AtomiCortex/data/features \
        --output-dir /mnt/hdd/AtomiCortex/data/features/ml_features
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.data_store import DataStore
from src.features.feature_pipeline import FeaturePipeline, FEATURE_GROUPS
from src.logger import get_logger

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ML feature matrix")
    p.add_argument("--symbol", required=True, help="Binance symbol, e.g. BTCUSDT")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Root Parquet data directory",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for output Parquet file",
    )
    p.add_argument(
        "--interval",
        default="4h",
        help="Kline interval (default: 4h)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    output_path = args.output_dir / f"{args.symbol}_{args.interval}_features.parquet"

    print(f"\n{'='*60}")
    print(f"  AtomiCortex Feature Builder — Phase 3")
    print(f"{'='*60}")
    print(f"  Symbol   : {args.symbol}")
    print(f"  Interval : {args.interval}")
    print(f"  Range    : {args.start} → {args.end}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Output   : {output_path}")
    print(f"{'='*60}\n")

    with DataStore(args.data_dir) as store:
        pipeline = FeaturePipeline(store, args.symbol, args.interval)
        df = pipeline.build(start, end, save_to=output_path)
        feature_names = pipeline.get_feature_names()

    if df.is_empty():
        print("ERROR: No data produced. Check data-dir and date range.")
        sys.exit(1)

    present = [f for f in feature_names if f in df.columns]

    # --- NaN check ---
    nan_report: dict[str, int] = {}
    for col in present:
        null_n = df[col].null_count()
        nan_n = df[col].is_nan().sum() if df[col].dtype.is_float() else 0
        total = null_n + nan_n
        if total > 0:
            nan_report[col] = total

    print(f"\n{'─'*60}")
    print(f"  Features created : {len(present)}")
    print(f"  Rows             : {len(df):,}")
    print(f"  NaN/null count   : {sum(nan_report.values()) if nan_report else 0}")
    if nan_report:
        print(f"  NaN columns      : {nan_report}")
    else:
        print(f"  NaN check        : ✓ clean")
    print(f"{'─'*60}")

    print("\nFeature groups:")
    for group, names in FEATURE_GROUPS.items():
        in_df = [n for n in names if n in df.columns]
        print(f"  {group:15s}: {len(in_df)} features")

    print(f"\nTop-5 rows (selected feature columns):\n")
    display_cols = ["open_time"] + present[:8]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df.head(5).select(display_cols))

    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB\n")


if __name__ == "__main__":
    main()
