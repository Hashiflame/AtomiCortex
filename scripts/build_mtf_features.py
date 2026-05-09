#!/usr/bin/env python
"""
Builds feature matrices for 1H and 15m timeframes.

Reads raw OHLCV Parquet data (from Phase 1 download), applies
microstructure, derivatives, regime, session, ORB, and MTF context
features, and writes the result to data/features/.

Usage:
  python scripts/build_mtf_features.py --interval 1h --symbol BTCUSDT
  python scripts/build_mtf_features.py --interval 15m --symbol BTCUSDT
  python scripts/build_mtf_features.py --all

Output:
  data/features/symbol=BTCUSDT/interval=1h/features.parquet
  data/features/symbol=BTCUSDT/interval=15m/features.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Project root on sys.path.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.feature_pipeline import FeaturePipeline
from src.features.microstructure import (
    add_cvd_features,
    add_price_features,
    add_volume_features,
)
from src.features.regime_detector import (
    RegimeDetector,
    RegimeDetector1H,
    RegimeDetector15M,
)
from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

MTF_INTERVALS = ["1h", "15m"]
DEFAULT_SYMBOL = "BTCUSDT"
_WARMUP = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ohlcv(data_dir: Path, symbol: str, interval: str) -> pl.DataFrame:
    """Load all Parquet klines for one interval."""
    base = data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / f"interval={interval}"
    if not base.exists():
        _log.error(f"Data directory not found: {base}")
        return pl.DataFrame()

    files = sorted(base.rglob("*.parquet"))
    if not files:
        _log.error(f"No parquet files in {base}")
        return pl.DataFrame()

    dfs = [pl.read_parquet(f, hive_partitioning=False) for f in files]
    df = pl.concat(dfs).sort("open_time").unique(subset=["open_time"])
    _log.info(f"Loaded {len(df):,} bars for {symbol}/{interval}")
    return df


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def build_features(
    data_dir: Path,
    symbol: str,
    interval: str,
    output_dir: Path,
) -> Path | None:
    """Build full feature matrix for one interval."""
    t0 = time.monotonic()

    df = load_ohlcv(data_dir, symbol, interval)
    if df.is_empty():
        return None

    # Ensure required columns exist for microstructure.
    if "taker_buy_volume" not in df.columns:
        if "taker_buy_base_vol" in df.columns:
            df = df.rename({"taker_buy_base_vol": "taker_buy_volume"})
        else:
            df = df.with_columns(
                (pl.col("volume") * 0.5).alias("taker_buy_volume")
            )

    # Basic microstructure + price features.
    df = add_cvd_features(df)
    df = add_volume_features(df)
    df = add_price_features(df)

    # Regime detection with TF-appropriate parameters.
    detector_map = {
        "1h": RegimeDetector1H,
        "15m": RegimeDetector15M,
    }
    det_cls = detector_map.get(interval, RegimeDetector)
    detector = det_cls()
    min_bars = detector.hurst_window
    df = detector.detect_all(df, min_bars=min_bars)

    # MTF-specific features (session, ORB, HTF context).
    pipeline = FeaturePipeline(
        data_store=None,  # type: ignore[arg-type]
        symbol=symbol,
        interval=interval,
    )

    # Load HTF data for context.
    df_htf_4h = None
    df_htf_1h = None

    if interval in ("1h", "15m"):
        df_4h_raw = load_ohlcv(data_dir, symbol, "4h") if (
            data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / "interval=4h"
        ).exists() else pl.DataFrame()

        if not df_4h_raw.is_empty():
            # Add minimal regime columns for HTF context.
            if "taker_buy_volume" not in df_4h_raw.columns:
                if "taker_buy_base_vol" in df_4h_raw.columns:
                    df_4h_raw = df_4h_raw.rename({"taker_buy_base_vol": "taker_buy_volume"})
                else:
                    df_4h_raw = df_4h_raw.with_columns(
                        (pl.col("volume") * 0.5).alias("taker_buy_volume")
                    )
            df_4h_det = RegimeDetector()
            df_htf_4h = df_4h_det.detect_all(df_4h_raw)

    if interval == "15m":
        df_1h_raw = load_ohlcv(data_dir, symbol, "1h") if (
            data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / "interval=1h"
        ).exists() else pl.DataFrame()

        if not df_1h_raw.is_empty():
            if "taker_buy_volume" not in df_1h_raw.columns:
                if "taker_buy_base_vol" in df_1h_raw.columns:
                    df_1h_raw = df_1h_raw.rename({"taker_buy_base_vol": "taker_buy_volume"})
                else:
                    df_1h_raw = df_1h_raw.with_columns(
                        (pl.col("volume") * 0.5).alias("taker_buy_volume")
                    )
            df_1h_det = RegimeDetector1H()
            df_htf_1h = df_1h_det.detect_all(df_1h_raw, min_bars=100)

    df = pipeline.build_mtf(df, df_htf_4h=df_htf_4h, df_htf_1h=df_htf_1h)

    # Drop warmup rows.
    df = df.slice(_WARMUP)

    # Save.
    out_dir = output_dir / f"symbol={symbol}" / f"interval={interval}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    df.write_parquet(out_path, compression="zstd", compression_level=3)

    elapsed = time.monotonic() - t0
    _log.info(
        f"Built {interval} features: {len(df):,} rows, "
        f"{len(df.columns)} cols, {out_path.stat().st_size / 1024:.1f} KB "
        f"in {elapsed:.1f}s"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="AtomiCortex MTF feature builder",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", choices=MTF_INTERVALS)
    parser.add_argument("--all", action="store_true", dest="all_intervals")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/features")

    args = parser.parse_args()
    setup_logging()

    if not args.interval and not args.all_intervals:
        parser.error("Specify --interval or --all")

    intervals = MTF_INTERVALS if args.all_intervals else [args.interval]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"\n{'─' * 60}")
    print(f"  AtomiCortex MTF Feature Builder")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Intervals : {', '.join(intervals)}")
    print(f"{'─' * 60}\n")

    for interval in intervals:
        result = build_features(data_dir, args.symbol.upper(), interval, output_dir)
        if result:
            print(f"  ✅ {interval}: {result}")
        else:
            print(f"  ⚠ {interval}: no data available")

    print(f"\n{'─' * 60}")
    print("  Feature building complete")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
