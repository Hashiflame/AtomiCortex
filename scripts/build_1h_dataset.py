#!/usr/bin/env python3
"""
scripts/build_1h_dataset.py

Builds feature + target dataset for 1H LightGBM training.

Reads klines_1h Parquet, applies feature pipeline with timeframe='1h',
constructs target variable, splits by regime.

Output:
  data/features/symbol=BTCUSDT/interval=1h/dataset_trend.parquet
  data/features/symbol=BTCUSDT/interval=1h/dataset_high_vol.parquet

Usage:
  python scripts/build_1h_dataset.py --symbol BTCUSDT
  python scripts/build_1h_dataset.py --symbol BTCUSDT --start 2023-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.strategy_1h import MLStrategyConfig1H
from src.features.derivatives import add_funding_features, add_oi_features
from src.features.feature_pipeline import FeaturePipeline
from src.features.microstructure import (
    add_cvd_features,
    add_price_features,
    add_volume_features,
)
from src.features.regime_detector import RegimeDetector, RegimeDetector1H
from src.ingestion.data_store import DataStore
from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

# Default config
_CFG = MLStrategyConfig1H()

# Warmup rows to trim (NaN from rolling windows)
_WARMUP = 200

# Regime labels for each dataset split
_TREND_REGIMES = {"trend_up", "trend_down"}
_HIGH_VOL_REGIMES = {"high_vol"}
_SKIP_REGIMES = {"range", "unknown"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_klines(
    data_dir: Path,
    symbol: str,
    interval: str,
) -> pl.DataFrame:
    """Load all Parquet klines for one interval.

    Searches two directory conventions:
    1. ``data_dir/exchange=BINANCE_UM/symbol={symbol}/klines_{interval}/``
       (DataStore convention, used for 4H data)
    2. ``data_dir/exchange=BINANCE_UM/symbol={symbol}/interval={interval}/``
       (MTF convention, used for downloaded 1H/15m data)
    """
    candidates = [
        data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / f"klines_{interval}",
        data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / f"interval={interval}",
    ]

    for base in candidates:
        if not base.exists():
            continue
        files = sorted(base.rglob("*.parquet"))
        if not files:
            continue

        dfs = [pl.read_parquet(f, hive_partitioning=False) for f in files]
        df = pl.concat(dfs, how="diagonal").sort("open_time").unique(subset=["open_time"], maintain_order=True)
        _log.info(f"Loaded {len(df):,} bars from {base} ({symbol}/{interval})")
        return df

    _log.error(f"No klines data found for {symbol}/{interval} in {data_dir}")
    return pl.DataFrame()


def _load_derivatives(
    data_dir: Path,
    raw_dir: Path,
    symbol: str,
    start: datetime,
    end: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load funding_rate + metrics via DataStore (same logic as 4H build).

    Tries ``data_dir`` first, then ``raw_dir`` (DataStore convention:
    ``<root>/exchange=BINANCE_UM/symbol={symbol}/{funding_rate,metrics}``).
    Returns ``(funding_df, metrics_df)`` — empty frames when absent;
    add_funding_features/add_oi_features zero-fill gracefully downstream.
    """
    funding_df, metrics_df = pl.DataFrame(), pl.DataFrame()
    for root in (data_dir, raw_dir):
        store = DataStore(root)
        if funding_df.is_empty():
            try:
                funding_df = store.get_funding_rate(symbol, start, end)
            except Exception as exc:
                _log.warning(f"get_funding_rate failed for {root}: {exc}")
        if metrics_df.is_empty():
            try:
                metrics_df = store.get_metrics(symbol, start, end)
            except Exception as exc:
                _log.warning(f"get_metrics failed for {root}: {exc}")
    _log.info(
        f"Derivatives: funding={len(funding_df):,} rows, "
        f"metrics={len(metrics_df):,} rows"
    )
    return funding_df, metrics_df


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df_1h: pl.DataFrame,
    df_4h: pl.DataFrame,
    symbol: str,
    *,
    funding_df: pl.DataFrame | None = None,
    metrics_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build full feature matrix for 1H data.

    Steps:
    1. Ensure taker_buy_volume column exists
    2. Microstructure features (CVD, volume, price)
    3. Regime detection via RegimeDetector1H
    4. MTF features (session, 4H HTF context)
    5. Drop warmup rows
    """
    # 1. Ensure taker_buy_volume
    if "taker_buy_volume" not in df_1h.columns:
        if "taker_buy_base_vol" in df_1h.columns:
            df_1h = df_1h.rename({"taker_buy_base_vol": "taker_buy_volume"})
        else:
            df_1h = df_1h.with_columns(
                (pl.col("volume") * 0.5).alias("taker_buy_volume")
            )

    # 2. Microstructure
    df_1h = add_cvd_features(df_1h)
    df_1h = add_volume_features(df_1h)
    df_1h = add_price_features(df_1h)

    # 2b. Derivatives (funding + OI) — base columns for MTF momentum
    # features. add_* zero-fill when data is empty (fail-soft).
    df_1h = add_funding_features(
        df_1h, funding_df if funding_df is not None else pl.DataFrame()
    )
    df_1h = add_oi_features(
        df_1h, metrics_df if metrics_df is not None else pl.DataFrame()
    )

    # 3. Regime detection (1H-tuned parameters)
    detector = RegimeDetector1H()
    df_1h = detector.detect_all(df_1h, min_bars=detector.hurst_window)

    # 4. Prepare 4H HTF data (regime detection on 4H)
    df_htf_4h = None
    if not df_4h.is_empty():
        if "taker_buy_volume" not in df_4h.columns:
            if "taker_buy_base_vol" in df_4h.columns:
                df_4h = df_4h.rename({"taker_buy_base_vol": "taker_buy_volume"})
            else:
                df_4h = df_4h.with_columns(
                    (pl.col("volume") * 0.5).alias("taker_buy_volume")
                )
        det_4h = RegimeDetector()
        df_htf_4h = det_4h.detect_all(df_4h)

    # 5. MTF features (session + 4H HTF context)
    pipeline = FeaturePipeline(
        data_store=None,  # type: ignore[arg-type]
        symbol=symbol,
        interval="1h",
    )
    df_1h = pipeline.build_mtf(df_1h, df_htf_4h=df_htf_4h)

    # 6. Drop warmup rows
    if len(df_1h) > _WARMUP:
        df_1h = df_1h.slice(_WARMUP)
    _log.info(f"After warmup trim: {len(df_1h):,} rows, {len(df_1h.columns)} columns")

    return df_1h


# ---------------------------------------------------------------------------
# Regime split
# ---------------------------------------------------------------------------

def split_by_regime(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split dataset by regime into trend and high_vol subsets.

    - trend:    regime in {trend_up, trend_down}
    - high_vol: regime == high_vol
    - Excluded: range, unknown (insufficient signal)
    - FLAT (target=0) rows are excluded from all datasets.
    """
    if "regime" not in df.columns:
        raise ValueError("DataFrame must have 'regime' column")

    # Exclude FLAT target (not useful for training directional models)
    df_directional = df.filter(pl.col("target") != 0)
    _log.info(
        f"After FLAT exclusion: {len(df_directional):,} rows "
        f"(dropped {len(df) - len(df_directional):,} flat bars)"
    )

    # Trend dataset
    df_trend = df_directional.filter(
        pl.col("regime").is_in(list(_TREND_REGIMES))
    )

    # High-vol dataset
    df_high_vol = df_directional.filter(
        pl.col("regime").is_in(list(_HIGH_VOL_REGIMES))
    )

    _log.info(
        f"Regime split: trend={len(df_trend):,}, "
        f"high_vol={len(df_high_vol):,}, "
        f"excluded={len(df_directional) - len(df_trend) - len(df_high_vol):,} "
        f"(range/unknown)"
    )

    return df_trend, df_high_vol


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_dataset_summary(
    df_trend: pl.DataFrame,
    df_high_vol: pl.DataFrame,
    df_full: pl.DataFrame,
    n_features: int,
) -> None:
    """Print formatted dataset summary."""
    print(f"\n{'═'*60}")
    print(f"  1H Dataset Summary")
    print(f"{'═'*60}")

    for name, ds in [("trend", df_trend), ("high_vol", df_high_vol)]:
        n = len(ds)
        if n == 0:
            print(f"\n  {name:10s}: 0 rows (empty)")
            continue
        n_up = int(ds["target"].eq(1).sum())
        n_down = int(ds["target"].eq(-1).sum())
        n_flat = int(ds["target"].eq(0).sum())
        pct_up = 100 * n_up / n if n else 0
        pct_down = 100 * n_down / n if n else 0
        pct_flat = 100 * n_flat / n if n else 0
        print(
            f"\n  {name:10s}: {n:>7,} rows | "
            f"+1: {pct_up:>5.1f}% | -1: {pct_down:>5.1f}% | "
            f"skip: {pct_flat:>5.1f}%"
        )

    # Date range
    ts_col = "open_time"
    if ts_col in df_full.columns:
        min_ts = df_full[ts_col].min()
        max_ts = df_full[ts_col].max()
        if min_ts is not None and max_ts is not None:
            from datetime import datetime as dt
            start_dt = dt.fromtimestamp(min_ts / 1000, tz=timezone.utc)
            end_dt = dt.fromtimestamp(max_ts / 1000, tz=timezone.utc)
            print(f"\n  Date range : {start_dt.date()} → {end_dt.date()}")

    print(f"  Features   : {n_features} columns")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 1H ML dataset")
    p.add_argument(
        "--symbol",
        default="BTCUSDT",
        help=(
            "Binance symbol, or several comma-separated "
            "(e.g. BTCUSDT,ETHUSDT,SOLUSDT). Multiple symbols are "
            "each built independently then concatenated."
        ),
    )
    p.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    p.add_argument(
        "--data-dir",
        default="/mnt/hdd/AtomiCortex/data/features",
        type=Path,
        help="Root Parquet data directory (also checks data/raw structure)",
    )
    p.add_argument(
        "--raw-dir",
        default="/mnt/hdd/AtomiCortex/data/raw",
        type=Path,
        help="Raw MTF data directory (interval=1h format)",
    )
    p.add_argument(
        "--output-dir",
        default="data/features",
        type=Path,
        help="Output directory for dataset parquets",
    )
    return p.parse_args()


def _build_symbol_df(
    symbol: str,
    args: argparse.Namespace,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Load + feature + triple-barrier target for ONE symbol.

    Features (rolling microstructure, regime detection, MTF) and the
    triple-barrier target are computed strictly per-symbol, so a later
    ``pl.concat`` of several symbols cannot leak across the boundary
    (no rolling-window / forward-return bleed between symbols).

    Returns the full feature+target frame with a ``symbol`` column,
    or an empty frame when klines are missing (caller skips it).
    """
    # 1. Load 1H klines — try multiple locations
    _log.info(f"[{symbol}] Loading 1H klines...")
    df_1h = _load_klines(args.raw_dir, symbol, "1h")
    if df_1h.is_empty():
        df_1h = _load_klines(args.data_dir, symbol, "1h")
    if df_1h.is_empty():
        _log.error(f"[{symbol}] No 1H klines data found — skipping")
        return pl.DataFrame()

    # Filter by date range
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    df_1h = df_1h.filter(
        (pl.col("open_time") >= start_ms) & (pl.col("open_time") <= end_ms)
    )
    _log.info(f"[{symbol}] After date filter: {len(df_1h):,} 1H bars")

    # 2. Load 4H klines for HTF context
    _log.info(f"[{symbol}] Loading 4H klines for HTF context...")
    df_4h = _load_klines(args.data_dir, symbol, "4h")
    if df_4h.is_empty():
        df_4h = _load_klines(args.raw_dir, symbol, "4h")
    if df_4h.is_empty():
        _log.warning(f"[{symbol}] No 4H data — MTF context features absent")
        df_4h = pl.DataFrame()

    # 2c. Load derivatives (funding + OI) via DataStore
    _log.info(f"[{symbol}] Loading funding + metrics (derivatives)...")
    funding_df, metrics_df = _load_derivatives(
        args.data_dir, args.raw_dir, symbol, start, end
    )

    # 3. Build feature matrix
    _log.info(f"[{symbol}] Building feature matrix...")
    df = build_feature_matrix(
        df_1h, df_4h, symbol,
        funding_df=funding_df, metrics_df=metrics_df,
    )
    if df.is_empty():
        _log.error(f"[{symbol}] Feature matrix empty — skipping")
        return pl.DataFrame()

    # 4. Create target — triple-barrier (AFML Ch.3, replaces the old
    #    fixed-horizon sign(return) target). future_return is the
    #    realized barrier-touch return (consistent with the label).
    _log.info(f"[{symbol}] Creating target variable (triple-barrier)...")
    from src.features.triple_barrier import (
        apply_triple_barrier,
        label_statistics,
    )

    df = apply_triple_barrier(
        df,
        pt_multiplier=_CFG.tb_pt_multiplier,
        sl_multiplier=_CFG.tb_sl_multiplier,
        max_holding_bars=_CFG.tb_max_holding_bars,  # 1H: 6 bars = 6 hours
    )
    st = label_statistics(df)
    print(f"\n[{symbol}] Triple-Barrier Label Statistics:")
    print(f"  Total bars:   {st['total']:>8,}")
    print(f"  Long (+1):    {st['long']:>8,} ({st['long_pct']:.1f}%)")
    print(f"  Short (-1):   {st['short']:>8,} ({st['short_pct']:.1f}%)")
    print(f"  Vertical (0): {st['vertical']:>8,} ({st['vertical_pct']:.1f}%) "
          f"← excluded")
    print(f"  Coverage:     {st['coverage']:>7.1f}% (actionable labels)")
    if st["coverage"] < 20.0:
        _log.warning(f"[{symbol}] Coverage < 20% — barriers too far")
    elif st["coverage"] > 70.0:
        _log.warning(f"[{symbol}] Coverage > 70% — barriers too close")

    # Drop vertical (0) labels; rename label → target for compatibility.
    df = df.filter(pl.col("label") != 0).rename({"label": "target"})

    # Tag the symbol. Consumed by LGBMTrainer → symbol_encoded; the raw
    # 'symbol' column is in _EXCLUDE_COLUMNS so it never becomes a feature.
    df = df.with_columns(pl.lit(symbol).alias("symbol"))

    return df


def _save_split(
    df: pl.DataFrame,
    symbol_label: str,
    output_root: Path,
    n_features: int,
) -> None:
    """Regime-split a feature+target frame and write trend/high_vol."""
    df_trend, df_high_vol = split_by_regime(df)

    output_dir = output_root / f"symbol={symbol_label}" / "interval=1h"
    output_dir.mkdir(parents=True, exist_ok=True)

    trend_path = output_dir / "dataset_trend.parquet"
    high_vol_path = output_dir / "dataset_high_vol.parquet"

    df_trend.write_parquet(trend_path, compression="zstd", compression_level=3)
    df_high_vol.write_parquet(high_vol_path, compression="zstd", compression_level=3)

    _log.info(f"[{symbol_label}] Saved: {trend_path} "
              f"({trend_path.stat().st_size / 1024:.1f} KB)")
    _log.info(f"[{symbol_label}] Saved: {high_vol_path} "
              f"({high_vol_path.stat().st_size / 1024:.1f} KB)")

    print(f"\n  ── {symbol_label} ──")
    print_dataset_summary(df_trend, df_high_vol, df, n_features)


def main() -> None:
    setup_logging()
    args = _parse_args()
    t0 = time.monotonic()

    symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — 1H Dataset Builder")
    print(f"{'='*60}")
    print(f"  Symbol(s): {', '.join(symbols)}")
    print(f"  Range    : {args.start} → {args.end}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Raw dir  : {args.raw_dir}")
    print(f"  Output   : {args.output_dir}")
    print(f"{'='*60}\n")

    from src.models.dataset_builder import _EXCLUDE_COLUMNS

    # Build each symbol independently (no cross-symbol leakage).
    per_symbol: list[pl.DataFrame] = []
    for symbol in symbols:
        print(f"\n{'─'*60}")
        print(f"  Processing: {symbol}")
        print(f"{'─'*60}")
        df = _build_symbol_df(symbol, args, start, end)
        if df.is_empty():
            continue
        per_symbol.append(df)

    if not per_symbol:
        print("ERROR: No data produced for any requested symbol.")
        sys.exit(1)

    # Feature-column count (schema identical across symbols).
    ref = per_symbol[0]
    feature_cols = [
        col for col in ref.columns
        if col not in _EXCLUDE_COLUMNS
        and (ref[col].dtype.is_float() or ref[col].dtype.is_integer())
    ]
    n_features = len(feature_cols)
    _log.info(f"Feature columns: {n_features}")

    # Per-symbol datasets (backward-compatible paths).
    for df in per_symbol:
        _save_split(df, df["symbol"][0], args.output_dir, n_features)

    # Combined MULTI dataset — concat the per-symbol *built* frames
    # (features + target already computed per-symbol → no boundary leak).
    if len(per_symbol) > 1:
        combined = pl.concat(per_symbol, how="diagonal")
        _log.info(
            f"Combined MULTI: {len(combined):,} rows across "
            f"{len(per_symbol)} symbols"
        )
        _save_split(combined, "MULTI", args.output_dir, n_features)

    elapsed = time.monotonic() - t0
    print(f"\n  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
