#!/usr/bin/env python3
"""
scripts/train_15m_models.py

Trains LightGBM models for 15m timeframe.
Uses existing lgbm_trainer.py as backend.

Models trained:
  trend_model_15m.pkl — for trend_up / trend_down regimes
  orb_model_15m.pkl   — specialized for ORB breakout bars

Validation:
  Walk-forward: 10 months train / 3 months OOS
  Purged K-Fold with embargo = 16 bars (4 hours on 15m)

Usage:
  python scripts/train_15m_models.py --symbol BTCUSDT
  python scripts/train_15m_models.py --symbol BTCUSDT --tune  # with Optuna
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.strategy_15m import MLStrategyConfig15M
from src.logger import get_logger, setup_logging
from src.models.lgbm_trainer import LGBMTrainer, ModelConfig, EvaluationResult
from src.models.ml_validator import MLValidator, WalkForwardMLResult, WindowMLResult

_log = get_logger(__name__)

# 15m-specific config
_CFG = MLStrategyConfig15M()

# Walk-forward parameters for 15m
_WF_TRAIN_MONTHS = 10   # less than 1H (12) — 15m has 4× more bars per month
_WF_TEST_MONTHS = 3     # vs 4 for 1H
_WF_STEP_MONTHS = 2
_EMBARGO_BARS = 16       # 4 hours × 4 bars/hour on 15m

# Model types to train
_MODEL_TYPES = ["trend", "orb"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_regime_model(
    model_type: str,
    dataset_path: Path,
    models_dir: Path,
    symbol: str,
) -> tuple[EvaluationResult | None, Path | None]:
    """Train a single model from pre-built dataset.

    Returns (evaluation_result, model_path) or (None, None) on failure.
    """
    if not dataset_path.exists():
        _log.error(f"Dataset not found: {dataset_path}")
        return None, None

    df = pl.read_parquet(dataset_path, hive_partitioning=False)
    if df.is_empty():
        _log.error(f"Empty dataset: {dataset_path}")
        return None, None

    _log.info(f"Training {model_type} model: {len(df):,} rows from {dataset_path}")

    # Create ModelConfig for 15m
    config = ModelConfig(
        regime=model_type if model_type == "trend" else "all",
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=0.35,  # eval threshold (3-class baseline ~0.33)
    )

    # Build trainer
    trainer = LGBMTrainer(
        config=config,
        features_dir=dataset_path.parent,
        models_dir=models_dir,
    )

    # Split: walk-forward style (80% train / 20% test, temporal)
    # Apply embargo gap between train and test to prevent leakage
    n = len(df)
    train_n = int(n * 0.8)
    embargo = min(_EMBARGO_BARS, n - train_n)
    test_start = train_n + embargo
    train_df = df.head(train_n)
    test_df = df.slice(test_start, n - test_start)

    _log.info(
        f"  Split: train={train_n:,}, embargo={embargo}, "
        f"test={len(test_df):,}"
    )

    try:
        # Train
        model = trainer.train(train_df)

        # Evaluate
        result = trainer.evaluate(model, test_df)

        # Rename model file to 15m convention
        old_path = models_dir / f"{config.regime}_model.pkl"
        new_path = models_dir / f"{model_type}_model_15m.pkl"
        if old_path.exists():
            old_path.rename(new_path)
            _log.info(f"  Model saved: {new_path}")
        elif not new_path.exists():
            # If LGBMTrainer saved with a different name, find it
            _log.warning(f"  Expected {old_path} not found, check models_dir")

        return result, new_path

    except Exception as exc:
        _log.error(f"Training failed for {model_type}: {exc}")
        import traceback
        traceback.print_exc()
        return None, None


def run_walk_forward_validation(
    model_type: str,
    dataset_path: Path,
    models_dir: Path,
    symbol: str,
) -> WalkForwardMLResult | None:
    """Run walk-forward validation for a model type.

    Uses direct temporal splits on the pre-built dataset.
    """
    if not dataset_path.exists():
        return None

    df = pl.read_parquet(dataset_path, hive_partitioning=False)
    if df.is_empty() or len(df) < 500:
        _log.warning(f"Insufficient data for walk-forward: {len(df)} rows")
        return None

    from src.execution.walk_forward import WalkForwardValidator

    config = ModelConfig(
        regime=model_type if model_type == "trend" else "all",
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=0.35,
    )

    # Determine date range from open_time
    if "open_time" not in df.columns:
        _log.warning("No open_time column for WF splits")
        return None

    df = df.with_columns(
        (pl.col("open_time") * 1_000_000).cast(pl.Datetime("ns")).alias("_wf_dt")
    )

    data_start = df["_wf_dt"].min()
    data_end = df["_wf_dt"].max()

    if data_start is None or data_end is None:
        return None

    _log.info(
        f"  WF: data {data_start} → {data_end}, "
        f"train={_WF_TRAIN_MONTHS}m, test={_WF_TEST_MONTHS}m"
    )

    # Generate windows
    wf = WalkForwardValidator(
        train_months=_WF_TRAIN_MONTHS,
        test_months=_WF_TEST_MONTHS,
        step_months=_WF_STEP_MONTHS,
    )

    windows: list[WindowMLResult] = []

    # Embargo offset: skip first _EMBARGO_BARS × 15min of test window
    embargo_delta = timedelta(minutes=_EMBARGO_BARS * 15)

    for i, ((ts, te), (vs, ve)) in enumerate(wf.split(data_start, data_end)):
        vs_embargoed = vs + embargo_delta  # shift test start by embargo
        train_df = df.filter(
            (pl.col("_wf_dt") >= ts) & (pl.col("_wf_dt") < te)
        ).drop("_wf_dt")
        test_df = df.filter(
            (pl.col("_wf_dt") >= vs_embargoed) & (pl.col("_wf_dt") < ve)
        ).drop("_wf_dt")

        if len(train_df) < 100 or len(test_df) < 20:
            _log.debug(f"  Window {i+1}: skip (train={len(train_df)}, test={len(test_df)})")
            continue

        try:
            trainer = LGBMTrainer(
                config=config,
                features_dir=dataset_path.parent,
                models_dir=models_dir,
            )
            model = trainer.train(train_df)
            result = trainer.evaluate(model, test_df)

            n_signals = int(result.signal_rate * len(test_df))
            window = WindowMLResult(
                train_start=ts, train_end=te,
                test_start=vs, test_end=ve,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                signal_rate=result.signal_rate,
                n_signals=n_signals,
                n_test_bars=len(test_df),
            )
            windows.append(window)
            _log.info(
                f"  Window {i+1}: WR={result.win_rate:.1f}%, "
                f"PF={result.profit_factor:.2f}, signals={n_signals}"
            )
        except Exception as exc:
            _log.warning(f"  Window {i+1} failed: {exc}")
            continue

    wf_result = WalkForwardMLResult(regime=model_type, windows=windows)
    _log.info(
        f"  WF complete: {len(windows)} windows, "
        f"profitable={wf_result.profitable_windows_pct:.0f}%"
    )
    return wf_result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_training_report(
    results: dict[str, tuple[EvaluationResult | None, WalkForwardMLResult | None]],
) -> None:
    """Print formatted training report."""
    print(f"\n{'═'*80}")
    print(f"  AtomiCortex — 15m LightGBM Training Report")
    print(f"{'═'*80}")

    header = (
        f"  {'Model':<12} | {'Win Rate':>9} | {'PF':>7} | "
        f"{'Signal%':>8} | {'Accuracy':>9} | {'F1':>7} | {'Passes?':>8}"
    )
    print(f"\n{header}")
    print(f"  {'─'*12}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}")

    for model_type, (eval_result, wf_result) in results.items():
        if eval_result is None:
            print(f"  {model_type:<12} | {'FAILED':>45}")
            continue

        passes = "✅" if eval_result.passes_minimum_thresholds() else "❌"
        print(
            f"  {model_type:<12} | {eval_result.win_rate:>8.1f}% | "
            f"{eval_result.profit_factor:>7.2f} | "
            f"{eval_result.signal_rate * 100:>7.1f}% | "
            f"{eval_result.accuracy:>8.1f}% | "
            f"{eval_result.f1:>6.1f}% | "
            f"  {passes}"
        )

    # Walk-forward summary
    print(f"\n  {'─'*78}")
    print(f"  Walk-Forward Validation:")
    for model_type, (_, wf_result) in results.items():
        if wf_result is None:
            print(f"    {model_type}: skipped")
            continue
        n_windows = len(wf_result.windows)
        n_prof = sum(1 for w in wf_result.windows if w.profit_factor > 1.0)
        passes_wf = "✅" if wf_result.passes_walk_forward_test else "❌"
        print(
            f"    {model_type}: {n_prof}/{n_windows} profitable windows "
            f"({wf_result.profitable_windows_pct:.0f}%) "
            f"avg_WR={wf_result.avg_win_rate:.1f}% "
            f"avg_PF={wf_result.avg_profit_factor:.2f} {passes_wf}"
        )

    print(f"\n{'═'*80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 15m LightGBM models")
    p.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    p.add_argument(
        "--dataset-dir",
        default="data/features",
        type=Path,
        help="Directory containing dataset_trend.parquet etc.",
    )
    p.add_argument(
        "--models-dir",
        default="data/models/15m",
        type=Path,
        help="Directory to save trained models",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning (slow)",
    )
    p.add_argument(
        "--skip-wf",
        action="store_true",
        help="Skip walk-forward validation",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    t0 = time.monotonic()

    symbol = args.symbol.upper()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_base = Path(args.dataset_dir) / f"symbol={symbol}" / "interval=15m"

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — 15m Model Training")
    print(f"{'='*60}")
    print(f"  Symbol      : {symbol}")
    print(f"  Models      : {', '.join(_MODEL_TYPES)}")
    print(f"  Dataset dir : {dataset_base}")
    print(f"  Models dir  : {models_dir}")
    print(f"  Tune        : {args.tune}")
    print(f"  Walk-forward: train={_WF_TRAIN_MONTHS}m, test={_WF_TEST_MONTHS}m, "
          f"step={_WF_STEP_MONTHS}m, embargo={_EMBARGO_BARS} bars")
    print(f"{'='*60}\n")

    results: dict[str, tuple[EvaluationResult | None, WalkForwardMLResult | None]] = {}

    if args.tune:
        print("  ERROR: Optuna tuning for 15m not yet implemented.")
        print("  Run without --tune flag for standard training.")
        sys.exit(1)

    for model_type in _MODEL_TYPES:
        print(f"\n{'─'*60}")
        print(f"  Training: {model_type}")
        print(f"{'─'*60}")

        dataset_path = dataset_base / f"dataset_{model_type}.parquet"

        # Train
        eval_result, model_path = train_regime_model(
            model_type=model_type,
            dataset_path=dataset_path,
            models_dir=models_dir,
            symbol=symbol,
        )

        # Walk-forward validation
        wf_result = None
        if not args.skip_wf and eval_result is not None:
            _log.info(f"Running walk-forward validation for {model_type}...")
            wf_result = run_walk_forward_validation(
                model_type=model_type,
                dataset_path=dataset_path,
                models_dir=models_dir,
                symbol=symbol,
            )

        results[model_type] = (eval_result, wf_result)

    # Print report
    print_training_report(results)

    elapsed = time.monotonic() - t0
    print(f"  Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
