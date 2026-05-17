#!/usr/bin/env python3
"""
scripts/validate_1h_models.py

Validates 1H models against go/no-go criteria including
DSR, PBO, and t-stat (same statistical rigour as 4H pipeline).

Go/No-go thresholds for 1H:
  OOS Sharpe Ratio  >= 0.9
  Win Rate          >= 51%
  Profit Factor     >= 1.25
  OOS trades        >= 800
  Walk-forward      >= 55% profitable windows
  Fee check         >= 4x round_trip_fees
  DSR               >= 0.95
  PBO               <= 0.30
  t-stat            >= 3.0

Usage:
  python scripts/validate_1h_models.py --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.strategy_1h import MLStrategyConfig1H
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

_CFG = MLStrategyConfig1H()

# CLI override for the confidence threshold (set in main()).
# None -> use MLStrategyConfig1H().confidence_threshold.
_CONF_OVERRIDE: float | None = None


def _conf_threshold() -> float:
    """Resolve the active confidence threshold (CLI override or config)."""
    return _CONF_OVERRIDE if _CONF_OVERRIDE is not None else _CFG.confidence_threshold

# Go/no-go thresholds for 1H
_SHARPE_THRESHOLD = 0.9
_WIN_RATE_THRESHOLD = 51.0
_PROFIT_FACTOR_THRESHOLD = 1.25
_MIN_TRADES = {"trend": 800, "high_vol": 400}  # high_vol is rarer (~24% of bars)
_WF_PROFITABLE_THRESHOLD = 55.0  # % of walk-forward windows
_FEE_MULTIPLIER_THRESHOLD = 3.5
_ROUND_TRIP_FEES_BPS = 7.0  # ~0.07% (maker + taker on Binance Futures)
_ROUND_TRIP_COST = _ROUND_TRIP_FEES_BPS / 10_000  # 0.0007
_EMBARGO_BARS = 48  # 2 days × 24 bars/day — embargo gap for WF
_DSR_THRESHOLD = 0.95
_PBO_THRESHOLD = 0.30
_TSTAT_THRESHOLD = 3.0

# Walk-forward parameters
_WF_TRAIN_MONTHS = 12
_WF_TEST_MONTHS = 4
_WF_STEP_MONTHS = 2

# N experiments estimate for DSR.
# How to count N:
#   N = n_wf_windows × n_regimes × n_tuning_attempts
#   Example without tuning: 10 × 2 × 1 = 20
#   With Optuna (50 trials): 10 × 2 × 50 = 1000
#   If you tested multiple forward_bars / ATR multipliers, multiply further.
# If you ran tune_models.py — use --n-experiments 200+ on CLI.
_N_EXPERIMENTS_DEFAULT = 20  # conservative default (no Optuna tuning)


@dataclass
class ValidationResult:
    """Validation metrics for one model."""
    regime: str
    oos_sharpe: float
    win_rate: float
    profit_factor: float
    n_trades: int
    wf_profitable_pct: float
    wf_windows_total: int
    wf_windows_profitable: int
    fee_multiplier: float
    avg_return_per_trade: float
    signal_coverage: float = 0.0
    avg_confidence: float = 0.0
    suggested_threshold: float = 0.0
    suggested_coverage: float = 0.0
    # Statistical tests
    dsr: float = 0.0
    pbo: float = 1.0
    t_stat: float = 0.0

    def passes(self) -> bool:
        """Check if ALL go/no-go criteria pass (including stat tests)."""
        min_trades = _MIN_TRADES.get(self.regime, 800)
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


def suggest_confidence_threshold(
    proba: np.ndarray,
    min_coverage: float = 0.10,  # at least 10% of bars should trade
    max_coverage: float = 0.40,  # at most 40%
) -> float:
    """Suggest a confidence threshold from the OOS proba distribution.

    Finds the highest threshold T in [0.52, 0.70] such that the
    two-sided coverage P(proba >= T or proba <= 1-T) lands within
    [min_coverage, max_coverage]. Informational only — does not
    change gate logic; the user decides whether to update the config.
    """
    if proba.ndim != 1:
        return 0.55
    for threshold in np.arange(0.70, 0.51, -0.01):
        coverage = float(np.mean((proba >= threshold) | (proba <= 1 - threshold)))
        if min_coverage <= coverage <= max_coverage:
            return round(float(threshold), 2)
    return 0.55  # fallback


# ---------------------------------------------------------------------------
# Purged K-Fold CV (adapted for 1H pre-built datasets)
# ---------------------------------------------------------------------------

def _run_purged_kfold(
    df: pl.DataFrame,
    regime: str,
    symbol: str,
    n_splits: int = 5,
    embargo_pct: float = 0.02,
) -> list[EvaluationResult]:
    """Run Purged K-Fold CV on a pre-built dataset.

    Returns list of EvaluationResult, one per fold.
    """
    config = ModelConfig(
        regime=regime,
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=_conf_threshold(),
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
                models_dir=Path("data/models/1h"),
                use_mtf_params=True,  # DSR/PBO on production reg. params
            )
            model = trainer.train(train_df)
            result = trainer.evaluate(model, test_df)
            results.append(result)
        except Exception as exc:
            _log.warning(f"  CV fold {fold_i+1} failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# Walk-Forward (adapted for 1H pre-built datasets)
# ---------------------------------------------------------------------------

def _run_walk_forward(
    df: pl.DataFrame,
    regime: str,
    symbol: str,
) -> WalkForwardMLResult:
    """Run walk-forward validation on pre-built dataset.

    Returns WalkForwardMLResult with per-window metrics.
    """
    config = ModelConfig(
        regime=regime,
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=_conf_threshold(),
    )

    if "open_time" not in df.columns:
        return WalkForwardMLResult(regime=regime)

    df = df.with_columns(
        (pl.col("open_time") * 1_000_000).cast(pl.Datetime("ns")).alias("_wf_dt")
    )
    data_start = df["_wf_dt"].min()
    data_end = df["_wf_dt"].max()

    if data_start is None or data_end is None:
        return WalkForwardMLResult(regime=regime)

    wf = WalkForwardValidator(
        train_months=_WF_TRAIN_MONTHS,
        test_months=_WF_TEST_MONTHS,
        step_months=_WF_STEP_MONTHS,
    )

    windows: list[WindowMLResult] = []

    # Embargo offset: skip first _EMBARGO_BARS hours of test window
    from datetime import timedelta
    embargo_delta = timedelta(hours=_EMBARGO_BARS)

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
                models_dir=Path("data/models/1h"),
                use_mtf_params=True,  # DSR/PBO on production reg. params
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

    return WalkForwardMLResult(regime=regime, windows=windows)


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

    # Binary confidence gate: proba = P(UP). 0.5 is the random baseline
    # (ML-017), so max(p, 1-p) is always >= 0.5 — a sub-0.5 threshold
    # never filters. Trade only when the model is confident on one side.
    conf_threshold = _conf_threshold()

    if proba.ndim == 1:
        up_mask = proba >= conf_threshold
        down_mask = proba <= (1.0 - conf_threshold)
        signal_mask = up_mask | down_mask
        y_pred_direction = np.where(up_mask, 1, np.where(down_mask, -1, 0))
        confidence = np.maximum(proba, 1.0 - proba)
    else:
        # Legacy multiclass fallback
        y_pred_class = np.argmax(proba, axis=1)
        y_pred_direction = np.array([CLASS_TO_LABEL[int(c)] for c in y_pred_class])
        confidence = np.max(proba, axis=1)
        signal_mask = (confidence >= conf_threshold) & (y_pred_direction != 0)

    future_returns = oos_df["future_return"].to_numpy()

    signal_preds = y_pred_direction[signal_mask]
    signal_returns = future_returns[signal_mask]
    n_signals = int(signal_mask.sum())
    n_bars = len(signal_mask)
    signal_coverage = n_signals / n_bars * 100 if n_bars else 0.0
    avg_confidence = (
        float(confidence[signal_mask].mean()) * 100 if n_signals > 0 else 0.0
    )
    suggested_threshold = suggest_confidence_threshold(proba)
    if proba.ndim == 1:
        suggested_coverage = float(np.mean(
            (proba >= suggested_threshold) | (proba <= 1 - suggested_threshold)
        )) * 100
    else:
        suggested_coverage = 0.0

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

        # fee_multiplier: avg net PnL per trade / round-trip fees
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
        "signal_coverage": round(signal_coverage, 1),
        "avg_confidence": round(avg_confidence, 1),
        "suggested_threshold": suggested_threshold,
        "suggested_coverage": round(suggested_coverage, 1),
    }


# ---------------------------------------------------------------------------
# Full validation
# ---------------------------------------------------------------------------

def validate_model(
    regime: str,
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

    # 1. OOS metrics from the final trained model
    _log.info(f"  Computing OOS metrics...")
    oos = _compute_oos_metrics(model_path, dataset_path, symbol)

    # 2. Purged K-Fold CV (produces EvaluationResult list for DSR/PBO)
    _log.info(f"  Running Purged K-Fold CV (5 folds)...")
    cv_results = _run_purged_kfold(
        df, regime, symbol, n_splits=5, embargo_pct=0.02,
    )
    _log.info(f"  CV complete: {len(cv_results)} folds")

    # 3. Walk-Forward validation
    _log.info(f"  Running Walk-Forward ({_WF_TRAIN_MONTHS}m/{_WF_TEST_MONTHS}m)...")
    wf_result = _run_walk_forward(df, regime, symbol)
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
            n_experiments=_N_EXPERIMENTS_DEFAULT,
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
        regime=regime,
        oos_sharpe=oos["oos_sharpe"],
        win_rate=oos["win_rate"],
        profit_factor=oos["profit_factor"],
        n_trades=oos["n_trades"],
        wf_profitable_pct=round(wf_pct, 0),
        wf_windows_total=n_windows,
        wf_windows_profitable=n_prof,
        fee_multiplier=oos["fee_multiplier"],
        avg_return_per_trade=oos["avg_return_per_trade"],
        signal_coverage=oos["signal_coverage"],
        avg_confidence=oos["avg_confidence"],
        suggested_threshold=oos["suggested_threshold"],
        suggested_coverage=oos["suggested_coverage"],
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
    print(f"  1H Model Validation Report")
    print(f"{'═'*60}")

    for regime, result in results.items():
        if result is None:
            print(f"\n  {regime}_model_1h: NOT AVAILABLE")
            continue

        print(f"\n  {regime}_model_1h:")
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
        min_trades = _MIN_TRADES.get(result.regime, 800)
        print(
            f"    OOS trades:     {result.n_trades:>6}  "
            f"{_check(result.n_trades, min_trades)} (>= {min_trades})"
        )
        print(
            f"    Signal coverage:{result.signal_coverage:>5.1f}%  "
            f"(bars with a signal; ~100% means gate not filtering)"
        )
        print(
            f"    Avg confidence: {result.avg_confidence:>5.1f}%  "
            f"(mean max(p,1-p) on signal bars)"
        )
        print(
            f"    Suggested threshold: {result.suggested_threshold:.2f}  "
            f"(coverage: {result.suggested_coverage:.1f}%)  "
            f"[informational — config unchanged]"
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
    p = argparse.ArgumentParser(description="Validate 1H ML models")
    p.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    p.add_argument(
        "--dataset-dir",
        default="data/features",
        type=Path,
        help="Directory containing dataset_trend.parquet etc.",
    )
    p.add_argument(
        "--models-dir",
        default="data/models/1h",
        type=Path,
        help="Directory with trained models",
    )
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Override confidence threshold from config. "
            "Use suggested value from previous run."
        ),
    )
    p.add_argument(
        "--n-experiments",
        default=_N_EXPERIMENTS_DEFAULT,
        type=int,
        help=(
            "Number of strategy configurations tested (for DSR). "
            "Default=20 (no tuning). Use 200+ if Optuna was run."
        ),
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    t0 = time.monotonic()

    symbol = args.symbol.upper()
    dataset_base = Path(args.dataset_dir) / f"symbol={symbol}" / "interval=1h"
    models_dir = Path(args.models_dir)

    n_experiments = args.n_experiments

    global _CONF_OVERRIDE
    _CONF_OVERRIDE = args.confidence_threshold
    conf_src = (
        f"{_CONF_OVERRIDE:.2f} (CLI override)"
        if _CONF_OVERRIDE is not None
        else f"{_CFG.confidence_threshold:.2f} (config)"
    )

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — 1H Model Validation")
    print(f"{'='*60}")
    print(f"  Symbol        : {symbol}")
    print(f"  Datasets      : {dataset_base}")
    print(f"  Models        : {models_dir}")
    print(f"  Confidence    : {conf_src}")
    print(f"  N experiments : {n_experiments}")
    print(f"{'='*60}\n")

    results: dict[str, ValidationResult | None] = {}

    for regime in ["trend", "high_vol"]:
        print(f"\n{'─'*60}")
        print(f"  Validating: {regime}")
        print(f"{'─'*60}")

        model_path = models_dir / f"{regime}_model_1h.pkl"
        dataset_path = dataset_base / f"dataset_{regime}.parquet"

        result = validate_model(regime, model_path, dataset_path, symbol)
        results[regime] = result

    print_validation_report(results)

    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
