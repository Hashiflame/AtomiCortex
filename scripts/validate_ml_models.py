#!/usr/bin/env python3
"""
scripts/validate_ml_models.py

ML model validation pipeline: Purged K-Fold CV + Walk-Forward + DSR/PBO/t-stat.

Usage
-----
    python scripts/validate_ml_models.py \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \
        --features-dir /mnt/hdd/AtomiCortex/data/features/ml_features \
        --models-dir /mnt/hdd/AtomiCortex/data/features/models \
        --regime trend

Phase 3 — Steps 3.5 + 3.6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging
from src.models.lgbm_trainer import LGBMTrainer, ModelConfig
from src.models.ml_validator import MLValidator, WalkForwardMLResult
from src.models.statistical_tests import run_all_tests

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML model validation pipeline")
    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,SOLUSDT",
    )
    p.add_argument(
        "--features-dir",
        required=True,
        type=Path,
        help="Directory with {SYMBOL}_4h_features.parquet files",
    )
    p.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="Directory with trained models",
    )
    p.add_argument(
        "--regime",
        required=True,
        help="Regime to validate: trend, range, or high_vol",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    p.add_argument(
        "--train-months",
        type=int,
        default=12,
        help="Walk-forward training window in months (default: 12)",
    )
    p.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Walk-forward test window in months (default: 3)",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    regime = args.regime.strip()

    print(f"\n{'═' * 60}")
    print(f"  AtomiCortex — ML Validation Report")
    print(f"  Regime: {regime}")
    print(f"{'═' * 60}")

    # ── Create trainer ────────────────────────────────────────────────
    config = ModelConfig(regime=regime, symbols=symbols)
    trainer = LGBMTrainer(
        config=config,
        features_dir=args.features_dir,
        models_dir=args.models_dir,
    )

    # ── Create validator ──────────────────────────────────────────────
    validator = MLValidator(
        n_splits=args.n_splits,
        embargo_pct=0.01,
        confidence_threshold=config.confidence_threshold,
    )

    # ── Load full data (for CV) ───────────────────────────────────────
    print(f"\n  Loading data for regime '{regime}'...")
    train_df, test_df = trainer.prepare_data()
    import polars as pl
    full_df = pl.concat([train_df, test_df], how="diagonal")
    print(f"  Total: {len(full_df)} rows")

    # ══════════════════════════════════════════════════════════════════
    # 1. Purged K-Fold CV
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(f"  Purged K-Fold CV ({args.n_splits} folds):")
    print(f"{'─' * 60}")

    cv_results = validator.purged_kfold_cv(trainer, full_df)

    for i, r in enumerate(cv_results):
        n_signals = int(r.signal_rate * len(full_df) / args.n_splits)
        print(
            f"  Fold {i + 1}: WR={r.win_rate:.1f}%, "
            f"PF={r.profit_factor:.2f}, "
            f"signals≈{n_signals}"
        )

    if cv_results:
        mean_wr = sum(r.win_rate for r in cv_results) / len(cv_results)
        mean_pf = sum(r.profit_factor for r in cv_results) / len(cv_results)
        print(f"  Mean:   WR={mean_wr:.1f}%, PF={mean_pf:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # 2. Walk-Forward ML
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(
        f"  Walk-Forward ({args.train_months}m train / "
        f"{args.test_months}m test):"
    )
    print(f"{'─' * 60}")

    wf_result = validator.walk_forward_ml(
        trainer=trainer,
        symbols=symbols,
        features_dir=args.features_dir,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.test_months,
    )

    for i, w in enumerate(wf_result.windows):
        profitable = "✅" if w.profit_factor > 1.0 else "❌"
        print(
            f"  Window {i + 1}: "
            f"{w.test_start.strftime('%Y-%m')} → "
            f"{w.test_end.strftime('%Y-%m')}: "
            f"WR={w.win_rate:.1f}%  {profitable}"
        )

    if wf_result.windows:
        n_prof = sum(1 for w in wf_result.windows if w.profit_factor > 1.0)
        n_total = len(wf_result.windows)
        pct = wf_result.profitable_windows_pct
        passes = "✅" if wf_result.passes_walk_forward_test else "❌"
        print(
            f"  Profitable windows: {n_prof}/{n_total} "
            f"({pct:.1f}%) {passes}"
        )
    else:
        print("  No walk-forward windows generated!")

    # ══════════════════════════════════════════════════════════════════
    # 3. Statistical Tests
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")

    if cv_results and wf_result.windows:
        stat_result = run_all_tests(
            cv_results=cv_results,
            wf_result=wf_result,
            n_experiments=10,
        )
        print(stat_result.summary())
    else:
        print("  ⚠️  Insufficient results for statistical tests")

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()
