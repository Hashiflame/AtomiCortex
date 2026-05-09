#!/usr/bin/env python3
"""
scripts/build_15m_dataset.py

Builds feature + target dataset for 15m LightGBM training.

Key differences from 1H (build_1h_dataset.py):
- forward_bars=4 (4 × 15min = 1 hour ahead, vs 2H on 1H)
- atr_threshold_multiplier=0.35 (vs 0.4 on 1H)
- Two regime splits: trend AND orb (vs trend + high_vol)
- ORB features are critical (~20 columns from ORBDetector)
- HTF context: BOTH 1H and 4H (not just 4H like 1H strategy)
- Session trap filter: exclude first/last 2 bars of each session
- RegimeDetector15M with faster ADX/ATR (period=7, threshold=18)

Output:
  data/features/symbol=BTCUSDT/interval=15m/dataset_trend.parquet
  data/features/symbol=BTCUSDT/interval=15m/dataset_orb.parquet

Usage:
  python scripts/build_15m_dataset.py --symbol BTCUSDT
  python scripts/build_15m_dataset.py --symbol BTCUSDT --start 2023-01-01 --end 2025-12-31
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

from src.configs.strategy_15m import MLStrategyConfig15M
from src.features.feature_pipeline import FeaturePipeline
from src.features.microstructure import (
    add_cvd_features,
    add_price_features,
    add_volume_features,
)
from src.features.regime_detector import RegimeDetector, RegimeDetector1H, RegimeDetector15M
from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

# Default config
_CFG = MLStrategyConfig15M()

# Warmup rows to trim (NaN from rolling windows)
_WARMUP = 200

# Regime labels for each dataset split (from RegimeDetector15M)
_TREND_REGIMES = {"trend_up", "trend_down"}
_SKIP_REGIMES = {"range", "high_vol", "unknown"}


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
       (DataStore convention)
    2. ``data_dir/exchange=BINANCE_UM/symbol={symbol}/interval={interval}/``
       (MTF convention)
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
        df = pl.concat(dfs, how="diagonal").sort("open_time").unique(
            subset=["open_time"], maintain_order=True
        )
        _log.info(f"Loaded {len(df):,} bars from {base} ({symbol}/{interval})")
        return df

    _log.error(f"No klines data found for {symbol}/{interval} in {data_dir}")
    return pl.DataFrame()


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df_15m: pl.DataFrame,
    df_1h: pl.DataFrame,
    df_4h: pl.DataFrame,
    symbol: str,
) -> pl.DataFrame:
    """Build full feature matrix for 15m data.

    Steps:
    1. Ensure taker_buy_volume column exists
    2. Microstructure features (CVD, volume, price)
    3. Regime detection via RegimeDetector15M
    4. MTF features (session + ORB + 1H/4H HTF context)
    5. Drop warmup rows
    """
    # 1. Ensure taker_buy_volume
    if "taker_buy_volume" not in df_15m.columns:
        if "taker_buy_base_vol" in df_15m.columns:
            df_15m = df_15m.rename({"taker_buy_base_vol": "taker_buy_volume"})
        else:
            df_15m = df_15m.with_columns(
                (pl.col("volume") * 0.5).alias("taker_buy_volume")
            )

    # 2. Microstructure
    df_15m = add_cvd_features(df_15m)
    df_15m = add_volume_features(df_15m)
    df_15m = add_price_features(df_15m)

    # 3. Regime detection (15m-tuned parameters)
    detector = RegimeDetector15M()
    df_15m = detector.detect_all(df_15m, min_bars=detector.hurst_window)

    # 4. Prepare HTF data
    # 4a. 1H HTF — needs microstructure + regime detection
    df_htf_1h = None
    if not df_1h.is_empty():
        df_1h = _ensure_taker_buy_volume(df_1h)
        df_1h = add_cvd_features(df_1h)
        df_1h = add_volume_features(df_1h)
        df_1h = add_price_features(df_1h)
        det_1h = RegimeDetector1H()
        df_htf_1h = det_1h.detect_all(df_1h, min_bars=det_1h.hurst_window)

        # Session features for 1H (needed for htf_1h_vwap_position)
        pipeline_1h = FeaturePipeline(
            data_store=None,  # type: ignore[arg-type]
            symbol=symbol,
            interval="1h",
        )
        # Only add session features to 1H (not full MTF chain)
        from src.features.session_features import (
            SessionEncoder, SessionVWAP,
        )
        df_htf_1h = SessionEncoder().encode(df_htf_1h)
        df_htf_1h = SessionVWAP().calculate(df_htf_1h)

    # 4b. 4H HTF — needs regime detection
    df_htf_4h = None
    if not df_4h.is_empty():
        df_4h = _ensure_taker_buy_volume(df_4h)
        det_4h = RegimeDetector()
        df_htf_4h = det_4h.detect_all(df_4h)

    # 5. MTF features (session + ORB + 1H/4H HTF context)
    pipeline = FeaturePipeline(
        data_store=None,  # type: ignore[arg-type]
        symbol=symbol,
        interval="15m",
    )
    df_15m = pipeline.build_mtf(df_15m, df_htf_4h=df_htf_4h, df_htf_1h=df_htf_1h)

    # 6. Drop warmup rows
    if len(df_15m) > _WARMUP:
        df_15m = df_15m.slice(_WARMUP)
    _log.info(f"After warmup trim: {len(df_15m):,} rows, {len(df_15m.columns)} columns")

    return df_15m


def _ensure_taker_buy_volume(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure taker_buy_volume column exists."""
    if "taker_buy_volume" not in df.columns:
        if "taker_buy_base_vol" in df.columns:
            df = df.rename({"taker_buy_base_vol": "taker_buy_volume"})
        else:
            df = df.with_columns(
                (pl.col("volume") * 0.5).alias("taker_buy_volume")
            )
    return df


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def create_target_15m(
    df: pl.DataFrame,
    forward_bars: int = 4,
    atr_threshold_multiplier: float = 0.35,
) -> pl.DataFrame:
    """Create ternary target for 15m timeframe.

    Target:
      +1 if forward_return >  atr_pct * multiplier  (UP)
      -1 if forward_return < -atr_pct * multiplier  (DOWN)
       0 if |forward_return| <= threshold            (FLAT — excluded from training)

    Parameters
    ----------
    forward_bars : int
        Number of bars ahead for return calculation (4 = 1 hour on 15m).
    atr_threshold_multiplier : float
        Multiplied by atr_pct to get the dynamic threshold.
    """
    if "close" not in df.columns or "atr_pct" not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'atr_pct' columns")

    # Future return: (close[t+N] - close[t]) / close[t]
    future_close = df["close"].shift(-forward_bars)
    future_return = (future_close - df["close"]) / df["close"]

    # Dynamic ATR-based threshold
    atr_threshold = df["atr_pct"] * atr_threshold_multiplier

    # Ternary target
    target = (
        pl.when(future_return > atr_threshold)
        .then(pl.lit(1))
        .when(future_return < -atr_threshold)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
    )

    df = df.with_columns([
        future_return.alias("future_return"),
        target.alias("target"),
    ])

    # Drop last forward_bars rows (no target — future unknown)
    df = df.head(len(df) - forward_bars)

    n_total = len(df)
    n_up = int(df["target"].eq(1).sum())
    n_down = int(df["target"].eq(-1).sum())
    n_flat = int(df["target"].eq(0).sum())
    _log.info(
        f"Target created: {n_total:,} rows | "
        f"UP={n_up} ({100*n_up/n_total:.1f}%) | "
        f"DOWN={n_down} ({100*n_down/n_total:.1f}%) | "
        f"FLAT={n_flat} ({100*n_flat/n_total:.1f}%)"
    )

    return df


# ---------------------------------------------------------------------------
# Session trap filter
# ---------------------------------------------------------------------------

def filter_session_trap(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Remove session trap zone bars (too noisy for reliable signals).

    The ``is_session_trap_zone`` column is pre-computed by
    ``ORBDetector._add_session_meta()`` which marks the first/last 2 bars
    of each trading session.  The zone size is controlled there, not here.

    Applied AFTER target creation so we don't lose data for ATR/rolling
    calculations but BEFORE regime split so trap bars don't enter training.

    Returns (filtered_df, n_excluded).
    """
    if "is_session_trap_zone" not in df.columns:
        _log.warning("No is_session_trap_zone column — skipping trap filter")
        return df, 0

    # fill_null(False): if ORBDetector left nulls (e.g. bars without
    # session encoding), treat them as non-trap — don't silently drop.
    n_before = len(df)
    df_filtered = df.filter(
        ~pl.col("is_session_trap_zone").fill_null(False)
    )
    n_excluded = n_before - len(df_filtered)

    _log.info(
        f"Session trap filter: excluded {n_excluded:,} bars "
        f"({100*n_excluded/n_before:.1f}%) — {len(df_filtered):,} remaining"
    )
    return df_filtered, n_excluded


# ---------------------------------------------------------------------------
# Regime split
# ---------------------------------------------------------------------------

def split_by_regime(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split dataset into trend and ORB subsets.

    - trend:  regime in {trend_up, trend_down}, target != 0
    - orb:    orb_breakout_bull OR orb_breakout_bear == True, target != 0
    - Excluded: range, high_vol, unknown regimes (from trend dataset)
    - FLAT (target=0) rows are excluded from both datasets
    """
    if "regime" not in df.columns:
        raise ValueError("DataFrame must have 'regime' column")

    # Exclude FLAT target (not useful for training directional models)
    df_directional = df.filter(pl.col("target") != 0)
    _log.info(
        f"After FLAT exclusion: {len(df_directional):,} rows "
        f"(dropped {len(df) - len(df_directional):,} flat bars)"
    )

    # Trend dataset: trend_up/trend_down regimes
    df_trend = df_directional.filter(
        pl.col("regime").is_in(list(_TREND_REGIMES))
    )

    # ORB dataset: bars with ORB breakout signal (any regime)
    has_orb_bull = "orb_breakout_bull" in df_directional.columns
    has_orb_bear = "orb_breakout_bear" in df_directional.columns

    if has_orb_bull and has_orb_bear:
        df_orb = df_directional.filter(
            pl.col("orb_breakout_bull") | pl.col("orb_breakout_bear")
        )
    elif has_orb_bull:
        df_orb = df_directional.filter(pl.col("orb_breakout_bull"))
    elif has_orb_bear:
        df_orb = df_directional.filter(pl.col("orb_breakout_bear"))
    else:
        _log.warning("No ORB breakout columns found — ORB dataset will be empty")
        df_orb = df_directional.head(0)

    n_excluded = len(df_directional) - len(df_trend) - len(df_orb)
    # Some bars may be in both trend AND orb (that's fine — separate models)
    _log.info(
        f"Regime split: trend={len(df_trend):,}, "
        f"orb={len(df_orb):,}"
    )

    return df_trend, df_orb


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_dataset_summary(
    df_trend: pl.DataFrame,
    df_orb: pl.DataFrame,
    df_full: pl.DataFrame,
    n_features: int,
    n_trap_excluded: int,
) -> None:
    """Print formatted dataset summary."""
    print(f"\n{'═'*60}")
    print(f"  15m Dataset Summary")
    print(f"{'═'*60}")

    for name, ds in [("trend", df_trend), ("orb", df_orb)]:
        n = len(ds)
        if n == 0:
            print(f"\n  {name:10s}: 0 rows (empty)")
            continue
        n_up = int(ds["target"].eq(1).sum())
        n_down = int(ds["target"].eq(-1).sum())
        pct_up = 100 * n_up / n if n else 0
        pct_down = 100 * n_down / n if n else 0
        print(
            f"\n  {name:10s}: {n:>7,} rows | "
            f"+1: {pct_up:>5.1f}% | -1: {pct_down:>5.1f}%"
        )

    # Session trap stats
    print(f"\n  Session trap excluded: {n_trap_excluded:,} rows")

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
    p = argparse.ArgumentParser(description="Build 15m ML dataset")
    p.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    p.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    p.add_argument(
        "--data-dir",
        default="/mnt/hdd/AtomiCortex/data/features",
        type=Path,
        help="Root Parquet data directory",
    )
    p.add_argument(
        "--raw-dir",
        default="/mnt/hdd/AtomiCortex/data/raw",
        type=Path,
        help="Raw MTF data directory (interval=15m format)",
    )
    p.add_argument(
        "--output-dir",
        default="data/features",
        type=Path,
        help="Output directory for dataset parquets",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    t0 = time.monotonic()

    symbol = args.symbol.upper()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — 15m Dataset Builder")
    print(f"{'='*60}")
    print(f"  Symbol   : {symbol}")
    print(f"  Range    : {args.start} → {args.end}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Raw dir  : {args.raw_dir}")
    print(f"  Output   : {args.output_dir}")
    print(f"{'='*60}\n")

    # 1. Load 15m klines — try multiple locations
    _log.info("Loading 15m klines...")
    df_15m = _load_klines(args.raw_dir, symbol, "15m")
    if df_15m.is_empty():
        df_15m = _load_klines(args.data_dir, symbol, "15m")
    if df_15m.is_empty():
        print("ERROR: No 15m klines data found. Run download_mtf_data.py first.")
        sys.exit(1)

    # Filter by date range
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    df_15m = df_15m.filter(
        (pl.col("open_time") >= start_ms) & (pl.col("open_time") <= end_ms)
    )
    _log.info(f"After date filter: {len(df_15m):,} 15m bars")

    # 2. Load 1H klines for HTF context
    _log.info("Loading 1H klines for HTF context...")
    df_1h = _load_klines(args.data_dir, symbol, "1h")
    if df_1h.is_empty():
        df_1h = _load_klines(args.raw_dir, symbol, "1h")
    if df_1h.is_empty():
        _log.warning("No 1H data — 1H HTF context features will be absent")
        df_1h = pl.DataFrame()

    # 3. Load 4H klines for HTF context
    _log.info("Loading 4H klines for HTF context...")
    df_4h = _load_klines(args.data_dir, symbol, "4h")
    if df_4h.is_empty():
        df_4h = _load_klines(args.raw_dir, symbol, "4h")
    if df_4h.is_empty():
        _log.warning("No 4H data — 4H HTF context features will be absent")
        df_4h = pl.DataFrame()

    # 4. Build feature matrix
    _log.info("Building feature matrix...")
    df = build_feature_matrix(df_15m, df_1h, df_4h, symbol)

    if df.is_empty():
        print("ERROR: Feature matrix is empty after processing.")
        sys.exit(1)

    # 5. Create target (BEFORE session trap filter — preserve data for ATR)
    _log.info("Creating target variable...")
    df = create_target_15m(
        df,
        forward_bars=_CFG.forward_bars,
        atr_threshold_multiplier=_CFG.atr_threshold_multiplier,
    )

    # 6. Session trap filter (AFTER target — don't lose ATR data)
    _log.info("Applying session trap filter...")
    df, n_trap_excluded = filter_session_trap(df)

    # 7. Count feature columns (before split)
    from src.models.dataset_builder import DatasetBuilder, _EXCLUDE_COLUMNS
    feature_cols = [
        col for col in df.columns
        if col not in _EXCLUDE_COLUMNS
        and (df[col].dtype.is_float() or df[col].dtype.is_integer())
    ]
    n_features = len(feature_cols)
    _log.info(f"Feature columns: {n_features}")

    # 8. Split by regime
    _log.info("Splitting by regime...")
    df_trend, df_orb = split_by_regime(df)

    # 9. Save datasets
    output_dir = args.output_dir / f"symbol={symbol}" / "interval=15m"
    output_dir.mkdir(parents=True, exist_ok=True)

    trend_path = output_dir / "dataset_trend.parquet"
    orb_path = output_dir / "dataset_orb.parquet"

    df_trend.write_parquet(trend_path, compression="zstd", compression_level=3)
    df_orb.write_parquet(orb_path, compression="zstd", compression_level=3)

    _log.info(f"Saved: {trend_path} ({trend_path.stat().st_size / 1024:.1f} KB)")
    _log.info(f"Saved: {orb_path} ({orb_path.stat().st_size / 1024:.1f} KB)")

    # 10. Print summary
    print_dataset_summary(df_trend, df_orb, df, n_features, n_trap_excluded)

    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
