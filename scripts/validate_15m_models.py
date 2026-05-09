#!/usr/bin/env python3
"""
scripts/validate_15m_models.py

Validates 15m models against go/no-go criteria including
DSR, PBO, and t-stat (same statistical rigour as 1H/4H pipeline).

Go/No-go thresholds for 15m:
  OOS Sharpe Ratio  >= 0.85
  Win Rate          >= 51%
  Profit Factor     >= 1.20
  OOS trades        >= 1500 (trend) / 500 (orb)
  Walk-forward      >= 50% profitable windows
  Fee check         >= 5x round_trip_fees
  DSR               >= 0.95
  PBO               <= 0.30
  t-stat            >= 3.0

Usage:
  python scripts/validate_15m_models.py --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.strategy_15m import MLStrategyConfig15M
from src.logger import get_logger, setup_logging
from src.models.lgbm_trainer import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    LGBMTrainer,
    ModelConfig,
    EvaluationResult,
)
from src.models.ml_validator import MLValidator, WalkForwardMLResult, WindowMLResult
from src.models.statistical_tests import (
    StatTestResult,
    calculate_dsr,
    calculate_pbo,
    calculate_t_stat,
    run_all_tests,
)
from src.execution.walk_forward import PurgedKFoldCV, WalkForwardValidator

_log = get_logger(__name__)

_CFG = MLStrategyConfig15M()

# Go/no-go thresholds for 15m
_SHARPE_THRESHOLD = 0.85
_WIN_RATE_THRESHOLD = 51.0
_PROFIT_FACTOR_THRESHOLD = 1.20
_MIN_TRADES = {"trend": 1500, "orb": 500}  # orb has fewer bars
_WF_PROFITABLE_THRESHOLD = 50.0  # % of walk-forward windows
_FEE_MULTIPLIER_THRESHOLD = 5.0  # stricter than 1H (4.0)
_ROUND_TRIP_FEES_BPS = 7.0  # ~0.07% (maker + taker on Binance Futures)
_ROUND_TRIP_COST = _ROUND_TRIP_FEES_BPS / 10_000  # 0.0007
_EMBARGO_BARS = 16  # 4 hours × 4 bars/hour — embargo gap for WF
_DSR_THRESHOLD = 0.95
_PBO_THRESHOLD = 0.30
_TSTAT_THRESHOLD = 3.0

# Walk-forward parameters
_WF_TRAIN_MONTHS = 10
_WF_TEST_MONTHS = 3
_WF_STEP_MONTHS = 2

# N experiments estimate for DSR
# = n_wf_windows × n_model_types × n_tuning_attempts
_N_EXPERIMENTS = 10 * 2 * 1  # 20 (conservative for first run, no Optuna)


@dataclass
class ValidationResult:
    """Validation metrics for one model."""
    model_type: str
    oos_sharpe: float
    win_rate: float
    profit_factor: float
    n_trades: int
    wf_profitable_pct: float
    wf_windows_total: int
    wf_windows_profitable: int
    fee_multiplier: float
    avg_return_per_trade: float
    # Statistical tests
    dsr: float = 0.0
    pbo: float = 1.0
    t_stat: float = 0.0

    def passes(self) -> bool:
        """Check if ALL go/no-go criteria pass (including stat tests)."""
        min_trades = _MIN_TRADES.get(self.model_type, 1500)
        return (
            self.oos_sharpe >= _SHARPE_THRESHOLD
            and self.win_rate >= _WIN_RATE_THRESHOLD
            and self.profit_factor >= _PROFIT_FACTOR_THRESHOLD
            and self.n_trades >= min_trades
            and self.wf_profitable_pct >= _WF_PROFITABLE_THRESHOLD
            and self.fee_multiplier >= _FEE_MULTIPLIER_THRESHOLD
            and self.dsr >= _DSR_THRESHOLD
            and self.pbo <= _PBO_THRESHOLD
            and self.t_stat >= _TSTAT_THRESHOLD
        )


def _check(value: float, threshold: float, higher_better: bool = True) -> str:
    """Return pass/fail symbol."""
    if higher_better:
        return "✓" if value >= threshold else "✗"
    return "✓" if value <= threshold else "✗"


# ---------------------------------------------------------------------------
# Purged K-Fold CV (adapted for 15m pre-built datasets)
# ---------------------------------------------------------------------------

def _run_purged_kfold(
    df: pl.DataFrame,
    model_type: str,
    symbol: str,
    n_splits: int = 5,
    embargo_pct: float = 0.02,
) -> list[EvaluationResult]:
    """Run Purged K-Fold CV on a pre-built dataset.

    Returns list of EvaluationResult, one per fold.
    """
    config = ModelConfig(
        regime=model_type if model_type == "trend" else "all",
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=0.35,
    )

    cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)
    results: list[EvaluationResult] = []

    for fold_i, (train_df, test_df) in enumerate(cv.split(df)):
        if len(train_df) < 100 or len(test_df) < 20:
            _log.debug(f"  CV fold {fold_i+1}: skip (too small)")
            continue

        try:
            trainer = LGBMTrainer(
                config=config,
                features_dir=Path("."),
                models_dir=Path("data/models/15m"),
            )
            model = trainer.train(train_df)
            result = trainer.evaluate(model, test_df)
            results.append(result)
        except Exception as exc:
            _log.warning(f"  CV fold {fold_i+1} failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# Walk-Forward (adapted for 15m pre-built datasets)
# ---------------------------------------------------------------------------

def _run_walk_forward(
    df: pl.DataFrame,
    model_type: str,
    symbol: str,
) -> WalkForwardMLResult:
    """Run walk-forward validation on pre-built dataset.

    Returns WalkForwardMLResult with per-window metrics.
    """
    config = ModelConfig(
        regime=model_type if model_type == "trend" else "all",
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=0.35,
    )

    if "open_time" not in df.columns:
        return WalkForwardMLResult(regime=model_type)

    df = df.with_columns(
        (pl.col("open_time") * 1_000_000).cast(pl.Datetime("ns")).alias("_wf_dt")
    )
    data_start = df["_wf_dt"].min()
    data_end = df["_wf_dt"].max()

    if data_start is None or data_end is None:
        return WalkForwardMLResult(regime=model_type)

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
            continue

        try:
            trainer = LGBMTrainer(
                config=config,
                features_dir=Path("."),
                models_dir=Path("data/models/15m"),
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
        except Exception as exc:
            _log.warning(f"  WF window {i+1} failed: {exc}")

    return WalkForwardMLResult(regime=model_type, windows=windows)


# ---------------------------------------------------------------------------
# OOS metrics (static evaluation on final model)
# ---------------------------------------------------------------------------

def _compute_oos_metrics(
    model_path: Path,
    dataset_path: Path,
    symbol: str,
) -> dict:
    """Compute OOS metrics using the final trained model."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    booster = bundle["booster"]
    feature_columns = bundle.get("feature_columns", [])

    df = pl.read_parquet(dataset_path, hive_partitioning=False)
    n = len(df)

    # OOS split: last 20%
    oos_n = max(int(n * 0.2), 100)
    oos_df = df.tail(oos_n)

    # Prepare features
    feature_cols_in_df = [
        c for c in feature_columns
        if c in oos_df.columns and c != "symbol_encoded"
    ]
    X_oos = oos_df.select(feature_cols_in_df).to_numpy().astype(np.float64)

    if "symbol_encoded" in feature_columns:
        sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
        sym_val = sym_map.get(symbol, -1)
        sym_col = np.full((len(X_oos), 1), sym_val, dtype=np.float64)
        X_oos = np.hstack([X_oos, sym_col])

    X_oos = np.nan_to_num(X_oos, nan=0.0, posinf=0.0, neginf=0.0)

    proba = booster.predict(X_oos)
    y_pred_class = np.argmax(proba, axis=1)
    y_pred_labels = np.array([CLASS_TO_LABEL[int(c)] for c in y_pred_class])
    future_returns = oos_df["future_return"].to_numpy()
    max_proba = np.max(proba, axis=1)

    is_directional = y_pred_class != 1
    signal_mask = is_directional & (max_proba >= 0.35)

    signal_preds = y_pred_labels[signal_mask]
    signal_returns = future_returns[signal_mask]
    n_signals = int(signal_mask.sum())

    if n_signals > 0:
        correct = (signal_preds * signal_returns) > 0
        win_rate = float(correct.sum()) / n_signals * 100
        wins_abs = float(np.abs(signal_returns[correct]).sum())
        losses_abs = float(np.abs(signal_returns[~correct]).sum())
        profit_factor = wins_abs / losses_abs if losses_abs > 0 else 999.0

        # avg_return = mean SIGNED PnL per trade (pred × return - cost)
        signed_pnl = signal_preds * signal_returns - _ROUND_TRIP_COST
        avg_return = float(np.mean(signed_pnl)) * 100

        # Sharpe: daily P&L aggregation with costs, annualized by sqrt(252)
        from datetime import datetime as _dt, timezone
        trade_pnl_per_bar = np.zeros(len(oos_df))
        trade_pnl_per_bar[signal_mask] = (
            signal_preds * signal_returns - _ROUND_TRIP_COST
        )
        open_times = oos_df["open_time"].to_numpy()
        bar_dates = np.array([
            _dt.fromtimestamp(t / 1000, tz=timezone.utc).date()
            for t in open_times
        ])
        unique_dates = sorted(set(bar_dates))
        daily_pnl = np.array([
            float(np.sum(trade_pnl_per_bar[bar_dates == d]))
            for d in unique_dates
        ])
        if len(daily_pnl) > 1:
            mean_daily = float(np.mean(daily_pnl))
            std_daily = float(np.std(daily_pnl, ddof=1))
            sharpe = (
                mean_daily / std_daily * np.sqrt(252)
            ) if std_daily > 0 else 0.0
        else:
            sharpe = 0.0

        # fee_multiplier: how many times the avg gross return exceeds costs
        avg_gross_return = float(np.mean(signal_preds * signal_returns))
        fee_multiplier = avg_gross_return / _ROUND_TRIP_COST if _ROUND_TRIP_COST > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        avg_return = 0.0
        fee_multiplier = 0.0

    return {
        "oos_sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "n_trades": n_signals,
        "fee_multiplier": round(fee_multiplier, 1),
        "avg_return_per_trade": round(avg_return, 2),
    }


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

def validate_model(
    model_type: str,
    model_path: Path,
    dataset_path: Path,
    symbol: str,
) -> ValidationResult | None:
    """Validate a single model: OOS metrics + WF + CV + DSR/PBO/t-stat."""
    if not model_path.exists():
        _log.error(f"Model not found: {model_path}")
        return None
    if not dataset_path.exists():
        _log.error(f"Dataset not found: {dataset_path}")
        return None

    df = pl.read_parquet(dataset_path, hive_partitioning=False)
    if df.is_empty():
        _log.error(f"Empty dataset: {dataset_path}")
        return None

    # 1. OOS metrics from the final trained model.
    # NOTE: these use the last 20% of the dataset — the same slice that was
    # the test set during training.  This is a quick sanity check, NOT a
    # truly unseen evaluation.  The authoritative OOS evidence is the
    # walk-forward result computed in step 3 below.
    _log.info(f"  Computing OOS metrics...")
    oos = _compute_oos_metrics(model_path, dataset_path, symbol)

    # 2. Purged K-Fold CV (produces EvaluationResult list for DSR/PBO)
    _log.info(f"  Running Purged K-Fold CV (5 folds)...")
    cv_results = _run_purged_kfold(
        df, model_type, symbol, n_splits=5, embargo_pct=0.02,
    )
    _log.info(f"  CV complete: {len(cv_results)} folds")

    # 3. Walk-Forward validation
    _log.info(f"  Running Walk-Forward ({_WF_TRAIN_MONTHS}m/{_WF_TEST_MONTHS}m)...")
    wf_result = _run_walk_forward(df, model_type, symbol)
    n_windows = len(wf_result.windows)
    n_prof = sum(1 for w in wf_result.windows if w.profit_factor > 1.0)
    wf_pct = wf_result.profitable_windows_pct
    _log.info(f"  WF complete: {n_prof}/{n_windows} profitable ({wf_pct:.0f}%)")

    # 4. Statistical tests: DSR, PBO, t-stat
    _log.info(f"  Running statistical tests (DSR/PBO/t-stat)...")
    if cv_results and wf_result.windows:
        stat_result = run_all_tests(
            cv_results=cv_results,
            wf_result=wf_result,
            n_experiments=_N_EXPERIMENTS,
        )
        dsr = stat_result.dsr
        pbo = stat_result.pbo
        t_stat = stat_result.t_stat
    else:
        dsr = 0.0
        pbo = 1.0
        t_stat = 0.0
        _log.warning("  Insufficient data for statistical tests")

    return ValidationResult(
        model_type=model_type,
        oos_sharpe=oos["oos_sharpe"],
        win_rate=oos["win_rate"],
        profit_factor=oos["profit_factor"],
        n_trades=oos["n_trades"],
        wf_profitable_pct=round(wf_pct, 0),
        wf_windows_total=n_windows,
        wf_windows_profitable=n_prof,
        fee_multiplier=oos["fee_multiplier"],
        avg_return_per_trade=oos["avg_return_per_trade"],
        dsr=dsr,
        pbo=pbo,
        t_stat=t_stat,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_validation_report(results: dict[str, ValidationResult | None]) -> None:
    """Print go/no-go validation report with DSR/PBO/t-stat."""
    print(f"\n{'═'*60}")
    print(f"  15m Model Validation Report")
    print(f"{'═'*60}")

    for model_type, result in results.items():
        if result is None:
            print(f"\n  {model_type}_model_15m: NOT AVAILABLE")
            continue

        print(f"\n  {model_type}_model_15m:")
        print(
            f"    OOS Sharpe:     {result.oos_sharpe:>6.2f}  "
            f"{_check(result.oos_sharpe, _SHARPE_THRESHOLD)} (>= {_SHARPE_THRESHOLD})"
        )
        print(
            f"    Win Rate:       {result.win_rate:>5.1f}%  "
            f"{_check(result.win_rate, _WIN_RATE_THRESHOLD)} (>= {_WIN_RATE_THRESHOLD}%)"
        )
        print(
            f"    Profit Factor:  {result.profit_factor:>6.2f}  "
            f"{_check(result.profit_factor, _PROFIT_FACTOR_THRESHOLD)} "
            f"(>= {_PROFIT_FACTOR_THRESHOLD})"
        )
        min_trades = _MIN_TRADES.get(result.model_type, 1500)
        print(
            f"    OOS trades:     {result.n_trades:>6}  "
            f"{_check(result.n_trades, min_trades)} (>= {min_trades})"
        )
        print(
            f"    WF windows:     {result.wf_windows_profitable}/{result.wf_windows_total} "
            f"profitable  "
            f"{_check(result.wf_profitable_pct, _WF_PROFITABLE_THRESHOLD)} "
            f"(>= {_WF_PROFITABLE_THRESHOLD}%)"
        )
        print(
            f"    Fee check:      {result.fee_multiplier:>5.1f}x  "
            f"{_check(result.fee_multiplier, _FEE_MULTIPLIER_THRESHOLD)} "
            f"(>= {_FEE_MULTIPLIER_THRESHOLD}x)"
        )
        print(
            f"    DSR:            {result.dsr:>5.2f}  "
            f"{_check(result.dsr, _DSR_THRESHOLD)} (>= {_DSR_THRESHOLD})"
        )
        print(
            f"    PBO:            {result.pbo:>5.2f}  "
            f"{_check(result.pbo, _PBO_THRESHOLD, higher_better=False)} (<= {_PBO_THRESHOLD})"
        )
        print(
            f"    t-stat:         {result.t_stat:>5.1f}  "
            f"{_check(result.t_stat, _TSTAT_THRESHOLD)} (>= {_TSTAT_THRESHOLD})"
        )

        verdict = "✅ GO" if result.passes() else "❌ NO-GO"
        print(f"    VERDICT: {verdict}")

    print(f"\n{'═'*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate 15m ML models")
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
        help="Directory with trained models",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    t0 = time.monotonic()

    symbol = args.symbol.upper()
    dataset_base = Path(args.dataset_dir) / f"symbol={symbol}" / "interval=15m"
    models_dir = Path(args.models_dir)

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — 15m Model Validation")
    print(f"{'='*60}")
    print(f"  Symbol        : {symbol}")
    print(f"  Datasets      : {dataset_base}")
    print(f"  Models        : {models_dir}")
    print(f"  N experiments : {_N_EXPERIMENTS}")
    print(f"{'='*60}\n")

    results: dict[str, ValidationResult | None] = {}

    for model_type in ["trend", "orb"]:
        print(f"\n{'─'*60}")
        print(f"  Validating: {model_type}")
        print(f"{'─'*60}")

        model_path = models_dir / f"{model_type}_model_15m.pkl"
        dataset_path = dataset_base / f"dataset_{model_type}.parquet"

        result = validate_model(model_type, model_path, dataset_path, symbol)
        results[model_type] = result

    print_validation_report(results)

    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
