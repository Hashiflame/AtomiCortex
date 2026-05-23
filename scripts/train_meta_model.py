#!/usr/bin/env python3
"""
scripts/train_meta_model.py

Block 4 / Step 3 — Train the take/skip meta-labeling booster.

Inputs
------
data/features/models/v3/meta_dataset.parquet   (from build_meta_dataset.py)

Walk-forward split: first 70% (chronological) → train, last 30% → test.
The base trend/high_vol models were OOS on the last 20% of bars, so the
meta-test slice mostly sits inside the base-OOS region — i.e. meta lift
measured on it is genuinely out-of-sample for *both* layers.

Outputs
-------
data/features/models/v3/meta_model_v3.pkl
data/features/models/v3/meta_eval.json

Reports precision / recall / WR-uplift at multiple meta-thresholds
(0.50, 0.55, 0.60, 0.65, 0.70). Compares meta-filtered WR vs the
unfiltered base WR on the same OOS slice — the core decision metric.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import precision_score, recall_score, roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

DEFAULT_DATASET = Path("data/features/models/v3/meta_dataset.parquet")
DEFAULT_MODELS_DIR = Path("data/features/models/v3")

META_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,
    "max_depth": 4,
    "feature_fraction": 0.8,
    "feature_fraction_seed": 42,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_seed": 42,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "verbose": -1,
    "random_state": 42,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING = 50

# Columns that are metadata or the label — NEVER fed as features.
NON_FEATURE_COLUMNS: set[str] = {
    "open_time", "datetime", "symbol", "regime",
    "future_return", "target", "net_pnl_after_cost",
    "meta_target", "base_regime",
}

EVAL_THRESHOLDS: list[float] = [0.50, 0.55, 0.60, 0.65, 0.70]


def _prepare_xy(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Return (X, y, feature_names, future_return) — future_return is
    passed through so we can measure WR uplift, not just classification
    precision."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df.select(feature_cols).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["meta_target"].to_numpy().astype(np.int32)
    future_return = df["future_return"].to_numpy().astype(np.float64)
    return X, y, feature_cols, future_return


def _walkforward_split(
    df: pl.DataFrame, train_frac: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = df.sort("open_time")
    n_tr = int(len(df) * train_frac)
    return df.head(n_tr), df.tail(len(df) - n_tr)


def _metrics_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    base_direction: np.ndarray,
    future_return: np.ndarray,
    threshold: float,
    cost_bps: float,
) -> dict[str, float]:
    """Compute take/skip metrics at a given meta-proba threshold."""
    take = proba >= threshold
    n_take = int(take.sum())
    n_total = len(proba)
    if n_take == 0:
        return {
            "threshold": threshold, "n_taken": 0, "signal_rate": 0.0,
            "precision": 0.0, "recall": 0.0,
            "wr_taken": 0.0, "pf_taken": 0.0,
            "net_pnl_mean_bps": 0.0,
        }
    # Take-only slices
    taken_pred = np.ones(n_take, dtype=np.int32)
    taken_true = y_true[take]
    # WR / PF measured on the actual signed P&L of taken trades
    cost = cost_bps / 10000.0
    taken_pnl = future_return[take] * base_direction[take].astype(np.float64) - cost
    wins = taken_pnl > 0
    wr = float(wins.mean())
    gains = taken_pnl[taken_pnl > 0].sum()
    losses = -taken_pnl[taken_pnl < 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
    return {
        "threshold": round(threshold, 4),
        "n_taken": n_take,
        "signal_rate": round(n_take / n_total, 4),
        "precision": round(precision_score(taken_true, taken_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, (proba >= threshold).astype(int),
                                     zero_division=0), 4),
        "wr_taken": round(100 * wr, 2),
        "pf_taken": round(min(pf, 999.0), 4),
        "net_pnl_mean_bps": round(10000.0 * float(taken_pnl.mean()), 2),
    }


def train(
    dataset: Path,
    models_dir: Path,
    train_frac: float = 0.70,
    cost_bps: float = 6.0,
) -> None:
    df = pl.read_parquet(dataset)
    _log.info(f"Loaded meta dataset: {df.shape}")

    train_df, test_df = _walkforward_split(df, train_frac)
    X_tr, y_tr, feature_cols, _ = _prepare_xy(train_df)
    X_te, y_te, _, future_ret_te = _prepare_xy(test_df)
    base_dir_te = test_df["base_direction"].to_numpy().astype(np.int32)

    base_rate_tr = float(y_tr.mean())
    base_rate_te = float(y_te.mean())
    _log.info(f"Train: {len(X_tr)} rows  | meta_target +1 = {base_rate_tr:.1%}")
    _log.info(f"Test : {len(X_te)} rows  | meta_target +1 = {base_rate_te:.1%}")
    _log.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Class imbalance — scale_pos_weight is unfriendly when positive
    # class dominates (75%), so flip and upweight the *minority* (skip)
    # class with sample weights instead. Keeps proba calibration sane.
    weights = np.where(y_tr == 0, base_rate_tr / (1 - base_rate_tr), 1.0)

    # Internal val split for early stopping (last 15% of train).
    val_split = int(len(X_tr) * 0.85)
    X_fit, y_fit, w_fit = X_tr[:val_split], y_tr[:val_split], weights[:val_split]
    X_val, y_val, w_val = X_tr[val_split:], y_tr[val_split:], weights[val_split:]

    train_data = lgb.Dataset(X_fit, label=y_fit, weight=w_fit,
                             feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val,
                           feature_name=feature_cols, reference=train_data)

    booster = lgb.train(
        META_LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    proba_te = booster.predict(X_te)
    auc = float(roc_auc_score(y_te, proba_te))

    # Baseline: take ALL base signals (no meta gate) on the test slice.
    # This is the bar the meta layer has to clear.
    cost = cost_bps / 10000.0
    base_pnl_te = future_ret_te * base_dir_te.astype(np.float64) - cost
    base_wins = base_pnl_te > 0
    base_wr_no_meta = float(100 * base_wins.mean())
    base_gains = base_pnl_te[base_pnl_te > 0].sum()
    base_losses = -base_pnl_te[base_pnl_te < 0].sum()
    base_pf = float(base_gains / base_losses) if base_losses > 0 else float("inf")

    thresholds = [
        _metrics_at_threshold(y_te, proba_te, base_dir_te, future_ret_te,
                              thr, cost_bps)
        for thr in EVAL_THRESHOLDS
    ]

    # Feature importance
    imp_gain = booster.feature_importance(importance_type="gain")
    fi = sorted(zip(feature_cols, imp_gain), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    print(f"\n{'='*82}\n  META MODEL — WALK-FORWARD ({train_frac:.0%} train / "
          f"{1-train_frac:.0%} test)\n{'='*82}")
    print(f"  Train rows = {len(X_tr)}   Test rows = {len(X_te)}")
    print(f"  Test base-rate (P(+1)) = {base_rate_te:.1%}")
    print(f"  ROC-AUC on test = {auc:.4f}")
    print(f"  Baseline (no meta gate): WR={base_wr_no_meta:.2f}%, "
          f"PF={base_pf:.3f}, signals={len(base_pnl_te)}")
    print()
    print(f"  {'thr':>5} {'n_take':>7} {'sig%':>6} {'prec':>6} {'recall':>7} "
          f"{'WR%':>6} {'PF':>6} {'mean_bps':>9}")
    for m in thresholds:
        print(f"  {m['threshold']:>5.2f} {m['n_taken']:>7} "
              f"{m['signal_rate']*100:>5.1f}% "
              f"{m['precision']*100:>5.1f}% {m['recall']*100:>6.1f}% "
              f"{m['wr_taken']:>5.2f}% {m['pf_taken']:>6.3f} "
              f"{m['net_pnl_mean_bps']:>+8.2f}")
    print()
    print(f"  Top-10 features by gain:")
    for name, g in fi[:10]:
        print(f"    {name:<25} gain={g:.1f}")
    print(f"{'='*82}\n")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "booster": booster,
        "feature_columns": feature_cols,
        "non_feature_columns": sorted(NON_FEATURE_COLUMNS),
        "train_frac": train_frac,
        "cost_bps": cost_bps,
        "lgbm_params": META_LGBM_PARAMS,
    }
    out_path = models_dir / "meta_model_v3.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    _log.info(f"Saved meta model: {out_path}")

    eval_path = models_dir / "meta_eval.json"
    eval_path.write_text(json.dumps({
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "test_base_rate": round(base_rate_te, 4),
        "roc_auc_test": round(auc, 4),
        "baseline_no_gate": {
            "wr": round(base_wr_no_meta, 2),
            "pf": round(base_pf, 4),
            "n_signals": int(len(base_pnl_te)),
        },
        "thresholds": thresholds,
        "feature_importance_top10": [
            {"feature": n, "gain": round(float(g), 2)} for n, g in fi[:10]
        ],
    }, indent=2))
    _log.info(f"Saved meta eval: {eval_path}")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--cost-bps", type=float, default=6.0)
    args = p.parse_args()
    train(args.dataset, args.models_dir, args.train_frac, args.cost_bps)


if __name__ == "__main__":
    main()
