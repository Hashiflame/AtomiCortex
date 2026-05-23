#!/usr/bin/env python3
"""
scripts/build_features_15m_agg.py

Augment existing 4H feature parquets with the 4 agg_15m_* columns from
Block-3 Step 3. Writes to a sibling directory so the baseline 4H feature
set in ml_features/ stays bit-for-bit identical (clean A/B comparison
against high_vol_model_v3.pkl).

Source : data/features/ml_features/{SYMBOL}_4h_features.parquet
Output : data/features/ml_features_15m/{SYMBOL}_4h_features.parquet

Usage
-----
    python3 scripts/build_features_15m_agg.py
    python3 scripts/build_features_15m_agg.py --symbols BTCUSDT
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.agg_15m import (
    AGG_15M_FEATURE_NAMES,
    add_15m_aggregated_features,
)
from src.ingestion.data_store import DataStore
from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SRC_DIR = Path("data/features/ml_features")
OUT_DIR = Path("data/features/ml_features_15m")
DATA_DIR = Path("data/features")  # DataStore root


def _augment_symbol(symbol: str, ds: DataStore) -> None:
    src_path = SRC_DIR / f"{symbol}_4h_features.parquet"
    out_path = OUT_DIR / f"{symbol}_4h_features.parquet"

    if not src_path.exists():
        _log.error(f"Missing baseline parquet: {src_path}")
        return

    df_4h = pl.read_parquet(src_path)
    if df_4h.is_empty():
        _log.warning(f"{symbol}: empty source parquet, skipping")
        return

    # Bound the 15m fetch to the 4H frame's time range, plus a small pad
    # on either side so floor() bucket alignment never drops edge rows.
    t_min = int(df_4h["open_time"].min())
    t_max = int(df_4h["open_time"].max())
    pad = 4 * 60 * 60 * 1000  # 4h in ms
    start = datetime.fromtimestamp((t_min - pad) / 1000)
    end = datetime.fromtimestamp((t_max + pad) / 1000)

    _log.info(f"{symbol}: loading 15m klines {start.date()} → {end.date()}")
    df_15m = ds.get_klines(symbol, "15m", start, end)
    _log.info(f"{symbol}: got {len(df_15m)} 15m bars, augmenting {len(df_4h)} 4H bars")

    out = add_15m_aggregated_features(df_4h, df_15m)

    # NaN audit on the new columns
    for c in AGG_15M_FEATURE_NAMES:
        n_bad = out[c].null_count() + (
            out[c].is_nan().sum() if out[c].dtype.is_float() else 0
        )
        if n_bad:
            _log.warning(f"{symbol}: {c} has {n_bad} NaN/null after join")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="zstd", compression_level=3)
    _log.info(
        f"{symbol}: wrote {out_path}  ({out.shape[0]} rows × {out.shape[1]} cols, "
        f"{out_path.stat().st_size/1024:.1f} KB)"
    )

    # Quick stats so we can sanity-check the new columns
    desc = out.select(AGG_15M_FEATURE_NAMES).describe()
    print(f"\n[{symbol}] agg_15m_* stats:")
    print(desc)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = p.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    with DataStore(DATA_DIR) as ds:
        for sym in symbols:
            _augment_symbol(sym, ds)


if __name__ == "__main__":
    main()
