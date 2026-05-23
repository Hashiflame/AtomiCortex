#!/usr/bin/env python3
"""
scripts/build_meta_dataset.py

Block 4 / Step 2 — Build the meta-labeling dataset (López de Prado AFML §3.6).

For every 4H bar whose regime matches one of the v3 base models, score
the base model on that bar, then emit:

    meta features  — proba/confidence/direction of the firing base model
                     + 4H context (regime one-hot, vol, funding, hour)
    meta target    — 1 iff (future_return × direction_base − cost) > 0,
                     i.e. the base signal would have been profitable
                     net of round-trip cost.

The triple-barrier `future_return` already encodes the realized close
on the touched barrier (path-aware), so the cost-adjusted profitability
target is well-defined for every base signal.

Output
------
data/features/models/v3/meta_dataset.parquet

Usage
-----
    python3 scripts/build_meta_dataset.py
    python3 scripts/build_meta_dataset.py --cost-bps 6
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging
from src.models.dataset_builder import DatasetBuilder
from src.models.lgbm_trainer import SYMBOL_ENCODING

_log = get_logger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_FEATURES_DIR = Path("data/features/ml_features")
DEFAULT_MODELS_DIR = Path("data/features/models/v3")

# Per-regime base-model config — must match Block-2/Block-3 winners.
REGIME_BASE: dict[str, dict[str, Any]] = {
    "trend": {
        "model_path": "trend_model_v3_sel.pkl",
        "regime_values": ["trend_up", "trend_down"],
        "barrier": {"pt": 1.25, "sl": 1.0, "hold": 4},
    },
    "high_vol": {
        "model_path": "high_vol_model_v3.pkl",
        "regime_values": ["high_vol"],
        "barrier": {"pt": 1.25, "sl": 1.0, "hold": 6},
    },
}

# Meta features pulled from the 4H feature matrix at signal-time. These
# are NOT the base-model's feature set — they describe the *context* in
# which the base signal fires, which is what the meta-model conditions on.
META_CONTEXT_FEATURES: list[str] = [
    "atr_pct",
    "funding_rate",
    "funding_zscore_30d",
    "basis_approx",
    "returns_6",
    "volume_zscore",
]


def _score_base_model(
    df: pl.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    """Replicate LGBMTrainer._prepare_xy ordering and return P(UP)."""
    feature_cols: list[str] = bundle["feature_columns"]
    booster = bundle["booster"]

    df_cols = [c for c in feature_cols if c != "symbol_encoded" and c in df.columns]
    X = df.select(df_cols).to_numpy().astype(np.float64)
    if "symbol_encoded" in feature_cols and "symbol" in df.columns:
        sym = (
            df["symbol"]
            .replace(SYMBOL_ENCODING, default=-1)
            .cast(pl.Float64)
            .to_numpy()
            .reshape(-1, 1)
        )
        X = np.hstack([X, sym])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return booster.predict(X)


def _build_regime_slice(
    regime: str,
    symbols: list[str],
    features_dir: Path,
    models_dir: Path,
    cost_bps: float,
) -> pl.DataFrame:
    """Per-regime: triple-barrier → regime filter → score base model →
    build meta features + cost-adjusted meta target."""
    cfg = REGIME_BASE[regime]
    with open(models_dir / cfg["model_path"], "rb") as f:
        bundle = pickle.load(f)

    builder = DatasetBuilder(features_dir.parent, symbols)
    barrier = cfg["barrier"]
    out_parts: list[pl.DataFrame] = []

    for sym in symbols:
        df = builder.load_and_combine(features_dir, symbols=[sym])
        if df.is_empty():
            continue

        # Triple-barrier (target + future_return = realized close on touch)
        df = builder.create_target_triple_barrier(
            df,
            pt_multiplier=barrier["pt"],
            sl_multiplier=barrier["sl"],
            max_holding=barrier["hold"],
        )
        df = df.filter(pl.col("regime").is_in(cfg["regime_values"]))
        if df.is_empty():
            continue

        # Base-model probability — same X construction used in training
        proba_up = _score_base_model(df, bundle)
        direction = np.where(proba_up >= 0.5, 1, -1).astype(np.int32)
        confidence = np.maximum(proba_up, 1.0 - proba_up)

        # Meta target: net P&L positive after fixed round-trip cost.
        # `future_return` is realized close-to-touch return on the
        # triple-barrier exit, so (return * direction) is the per-trade
        # signed PnL pre-cost.
        future_ret = df["future_return"].to_numpy()
        cost = cost_bps / 10000.0
        net_pnl = future_ret * direction.astype(np.float64) - cost
        meta_target = (net_pnl > 0).astype(np.int32)

        # Hour-of-day cyclical encoding from open_time (ms epoch).
        # 4H bars land at hours 0/4/8/12/16/20 UTC; sin/cos preserves
        # cyclical distance and avoids spurious linear ordering.
        hour = (df["open_time"].to_numpy() // (60 * 60 * 1000)) % 24
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        # Regime one-hot (just trend_up/trend_down/high_vol — the
        # superset across both base models we plan to wrap).
        regime_str = df["regime"].to_numpy()

        # Assemble meta-row frame
        meta_df = df.select([
            "open_time", "datetime", "symbol", "regime",
            *[c for c in META_CONTEXT_FEATURES if c in df.columns],
            "future_return", "target",
        ]).with_columns([
            pl.Series("base_proba_up", proba_up),
            pl.Series("base_direction", direction),
            pl.Series("base_confidence", confidence),
            pl.Series("hour_sin", hour_sin),
            pl.Series("hour_cos", hour_cos),
            pl.Series("regime_trend_up",   (regime_str == "trend_up").astype(np.int32)),
            pl.Series("regime_trend_down", (regime_str == "trend_down").astype(np.int32)),
            pl.Series("regime_high_vol",   (regime_str == "high_vol").astype(np.int32)),
            pl.Series("net_pnl_after_cost", net_pnl),
            pl.Series("meta_target", meta_target),
            pl.lit(regime).alias("base_regime"),
        ])

        out_parts.append(meta_df)
        _log.info(
            f"[{regime}/{sym}] {len(meta_df)} rows | "
            f"meta_target +1 = {int(meta_target.sum())} "
            f"({100*meta_target.mean():.1f}%)"
        )

    if not out_parts:
        return pl.DataFrame()
    return pl.concat(out_parts, how="diagonal").sort("open_time")


def build_meta_dataset(
    symbols: list[str],
    features_dir: Path,
    models_dir: Path,
    cost_bps: float,
) -> pl.DataFrame:
    parts = [
        _build_regime_slice(r, symbols, features_dir, models_dir, cost_bps)
        for r in REGIME_BASE
    ]
    parts = [p for p in parts if not p.is_empty()]
    if not parts:
        raise RuntimeError("No meta rows produced — check base model paths / data")
    out = pl.concat(parts, how="diagonal").sort("open_time")
    return out


def _summary(df: pl.DataFrame) -> None:
    n = len(df)
    pos = int(df["meta_target"].sum())
    print(f"\n{'='*78}\n  META DATASET SUMMARY\n{'='*78}")
    print(f"  rows={n}  meta_target +1={pos} ({100*pos/n:.1f}%)  "
          f"−1={n-pos} ({100*(n-pos)/n:.1f}%)")
    print(f"  Per base_regime:")
    for r, sub in df.group_by("base_regime"):
        rname = r[0] if isinstance(r, tuple) else r
        p = int(sub["meta_target"].sum())
        print(f"    {rname:<10}: rows={len(sub):>5}  +1={p:>4} "
              f"({100*p/len(sub):.1f}%)   "
              f"avg base_conf={sub['base_confidence'].mean():.3f}")
    print(f"  Per symbol:")
    for s, sub in df.group_by("symbol"):
        sname = s[0] if isinstance(s, tuple) else s
        p = int(sub["meta_target"].sum())
        print(f"    {sname:<10}: rows={len(sub):>5}  +1={p:>4} ({100*p/len(sub):.1f}%)")
    print(f"  Net-PnL (after cost): mean={df['net_pnl_after_cost'].mean():+.4%}, "
          f"std={df['net_pnl_after_cost'].std():.4%}")
    print(f"{'='*78}\n")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--cost-bps", type=float, default=6.0,
                   help="Round-trip cost (bps) deducted from per-trade PnL "
                        "before computing meta_target. Default 6 bps.")
    p.add_argument("--output", type=Path,
                   default=DEFAULT_MODELS_DIR / "meta_dataset.parquet")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    df = build_meta_dataset(
        symbols=symbols, features_dir=args.features_dir,
        models_dir=args.models_dir, cost_bps=args.cost_bps,
    )
    _summary(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output, compression="zstd", compression_level=3)
    print(f"  ✓ Written: {args.output}  "
          f"({df.shape[0]} rows × {df.shape[1]} cols, "
          f"{args.output.stat().st_size/1024:.1f} KB)\n")


if __name__ == "__main__":
    main()
