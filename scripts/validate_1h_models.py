#!/usr/bin/env python3
"""
scripts/validate_1h_models.py

Validates 1H models against go/no-go criteria.

Go/No-go thresholds for 1H:
  OOS Sharpe Ratio  >= 0.9
  Win Rate          >= 51%
  Profit Factor     >= 1.25
  OOS trades        >= 800
  Walk-forward      >= 55% profitable windows
  Fee check:        expected_return >= 4x round_trip_fees

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
)
from src.models.ml_validator import MLValidator

_log = get_logger(__name__)

_CFG = MLStrategyConfig1H()

# Go/no-go thresholds for 1H
_SHARPE_THRESHOLD = 0.9
_WIN_RATE_THRESHOLD = 51.0
_PROFIT_FACTOR_THRESHOLD = 1.25
_MIN_TRADES = 800
_WF_PROFITABLE_THRESHOLD = 55.0  # % of walk-forward windows
_FEE_MULTIPLIER_THRESHOLD = 4.0
_ROUND_TRIP_FEES_BPS = 7.0  # ~0.07% (maker + taker on Binance Futures)


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

    def passes(self) -> bool:
        """Check if all go/no-go criteria pass."""
        return (
            self.oos_sharpe >= _SHARPE_THRESHOLD
            and self.win_rate >= _WIN_RATE_THRESHOLD
            and self.profit_factor >= _PROFIT_FACTOR_THRESHOLD
            and self.n_trades >= _MIN_TRADES
            and self.wf_profitable_pct >= _WF_PROFITABLE_THRESHOLD
            and self.fee_multiplier >= _FEE_MULTIPLIER_THRESHOLD
        )


def _check(value: float, threshold: float, higher_better: bool = True) -> str:
    """Return pass/fail emoji."""
    if higher_better:
        return "✓" if value >= threshold else "✗"
    return "✓" if value <= threshold else "✗"


def validate_model(
    regime: str,
    model_path: Path,
    dataset_path: Path,
    symbol: str,
) -> ValidationResult | None:
    """Validate a single model against go/no-go criteria."""
    if not model_path.exists():
        _log.error(f"Model not found: {model_path}")
        return None
    if not dataset_path.exists():
        _log.error(f"Dataset not found: {dataset_path}")
        return None

    # Load model
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    booster = bundle["booster"]
    feature_columns = bundle.get("feature_columns", [])

    # Load dataset
    df = pl.read_parquet(dataset_path)
    if df.is_empty():
        _log.error(f"Empty dataset: {dataset_path}")
        return None

    n = len(df)

    # OOS split: last 20%
    oos_n = max(int(n * 0.2), 100)
    train_df = df.head(n - oos_n)
    oos_df = df.tail(oos_n)

    # Prepare features for OOS
    from src.models.dataset_builder import _EXCLUDE_COLUMNS
    feature_cols_in_df = [
        c for c in feature_columns
        if c in oos_df.columns and c != "symbol_encoded"
    ]

    X_oos = oos_df.select(feature_cols_in_df).to_numpy().astype(np.float64)

    # Add symbol_encoded if it was a training feature
    if "symbol_encoded" in feature_columns:
        sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
        sym_val = sym_map.get(symbol, -1)
        sym_col = np.full((len(X_oos), 1), sym_val, dtype=np.float64)
        X_oos = np.hstack([X_oos, sym_col])

    X_oos = np.nan_to_num(X_oos, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict
    proba = booster.predict(X_oos)
    y_pred_class = np.argmax(proba, axis=1)
    y_pred_labels = np.array([CLASS_TO_LABEL[int(c)] for c in y_pred_class])
    y_true = oos_df["target"].to_numpy()
    future_returns = oos_df["future_return"].to_numpy()
    max_proba = np.max(proba, axis=1)

    # Signal mask: directional + confident
    is_directional = y_pred_class != 1  # not FLAT
    signal_mask = is_directional & (max_proba >= 0.35)

    signal_preds = y_pred_labels[signal_mask]
    signal_returns = future_returns[signal_mask]
    n_signals = int(signal_mask.sum())

    # Win rate
    if n_signals > 0:
        correct = (signal_preds * signal_returns) > 0
        win_rate = float(correct.sum()) / n_signals * 100

        # Profit factor
        wins_abs = float(np.abs(signal_returns[correct]).sum())
        losses_abs = float(np.abs(signal_returns[~correct]).sum())
        profit_factor = wins_abs / losses_abs if losses_abs > 0 else 999.0

        # Average return per trade
        avg_return = float(np.mean(np.abs(signal_returns))) * 100  # in bps

        # Sharpe ratio approximation (annualized for 1H)
        trade_returns = signal_preds * signal_returns
        if len(trade_returns) > 1:
            mean_ret = float(np.mean(trade_returns))
            std_ret = float(np.std(trade_returns))
            # 8760 = hours per year; scale by sqrt(n_trades_per_year)
            hours_in_data = oos_n  # each row = 1H
            trades_per_year = n_signals / (hours_in_data / 8760) if hours_in_data > 0 else 0
            sharpe = (mean_ret / std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # Fee check: avg return vs round-trip fees
        fee_bps = _ROUND_TRIP_FEES_BPS / 10000  # 0.0007
        fee_multiplier = avg_return / 100 / fee_bps if fee_bps > 0 else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        sharpe = 0.0
        avg_return = 0.0
        fee_multiplier = 0.0

    # Walk-forward validation
    config = ModelConfig(
        regime=regime,
        symbols=[symbol],
        forward_bars=_CFG.forward_bars,
        threshold_atr_multiplier=_CFG.atr_threshold_multiplier,
        confidence_threshold=0.35,
    )

    trainer = LGBMTrainer(
        config=config,
        features_dir=dataset_path.parent,
        models_dir=model_path.parent,
    )

    # Create temp features file for validator
    temp_features = dataset_path.parent / f"{symbol}_1h_features.parquet"
    if not temp_features.exists():
        import shutil
        shutil.copy2(dataset_path, temp_features)

    validator = MLValidator(n_splits=5, embargo_pct=0.02)
    wf_result = None
    try:
        wf_result = validator.walk_forward_ml(
            trainer=trainer,
            symbols=[symbol],
            features_dir=dataset_path.parent,
            train_months=12,
            test_months=4,
            step_months=2,
        )
    except Exception as exc:
        _log.warning(f"Walk-forward failed: {exc}")
    finally:
        if temp_features.exists() and temp_features != dataset_path:
            temp_features.unlink(missing_ok=True)

    wf_pct = wf_result.profitable_windows_pct if wf_result else 0.0
    wf_total = len(wf_result.windows) if wf_result else 0
    wf_prof = sum(1 for w in wf_result.windows if w.profit_factor > 1.0) if wf_result else 0

    return ValidationResult(
        regime=regime,
        oos_sharpe=round(sharpe, 2),
        win_rate=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        n_trades=n_signals,
        wf_profitable_pct=round(wf_pct, 0),
        wf_windows_total=wf_total,
        wf_windows_profitable=wf_prof,
        fee_multiplier=round(fee_multiplier, 1),
        avg_return_per_trade=round(avg_return, 2),
    )


def print_validation_report(results: dict[str, ValidationResult | None]) -> None:
    """Print go/no-go validation report."""
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
        print(
            f"    OOS trades:     {result.n_trades:>6}  "
            f"{_check(result.n_trades, _MIN_TRADES)} (>= {_MIN_TRADES})"
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
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    t0 = time.monotonic()

    symbol = args.symbol.upper()
    dataset_base = Path(args.dataset_dir) / f"symbol={symbol}" / "interval=1h"
    models_dir = Path(args.models_dir)

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — 1H Model Validation")
    print(f"{'='*60}")
    print(f"  Symbol    : {symbol}")
    print(f"  Datasets  : {dataset_base}")
    print(f"  Models    : {models_dir}")
    print(f"{'='*60}\n")

    results: dict[str, ValidationResult | None] = {}

    for regime in ["trend", "high_vol"]:
        model_path = models_dir / f"{regime}_model_1h.pkl"
        dataset_path = dataset_base / f"dataset_{regime}.parquet"

        _log.info(f"Validating {regime} model...")
        result = validate_model(regime, model_path, dataset_path, symbol)
        results[regime] = result

    print_validation_report(results)

    elapsed = time.monotonic() - t0
    print(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
