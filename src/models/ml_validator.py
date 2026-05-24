"""
src/models/ml_validator.py

ML model validation via Purged K-Fold CV and Walk-Forward analysis.

Classes
-------
MLValidator       — Orchestrates CV and walk-forward on LGBMTrainer models.
WalkForwardMLResult — Walk-forward results with per-window metrics.
WindowMLResult    — Single walk-forward window metrics.

Phase 3 — Step 3.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.execution.walk_forward import PurgedKFoldCV, WalkForwardValidator, _add_months
from src.logger import get_logger
from src.models.lgbm_trainer import (
    EvaluationResult,
    LGBMTrainer,
    ModelConfig,
)

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WindowMLResult:
    """Metrics for a single walk-forward window."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    win_rate: float
    profit_factor: float
    signal_rate: float
    n_signals: int
    n_test_bars: int


@dataclass
class WalkForwardMLResult:
    """Walk-forward ML validation results across all windows."""

    regime: str
    windows: list[WindowMLResult] = field(default_factory=list)

    @property
    def profitable_windows_pct(self) -> float:
        """Percentage of windows where profit_factor > 1.0."""
        if not self.windows:
            return 0.0
        n_prof = sum(1 for w in self.windows if w.profit_factor > 1.0)
        return n_prof / len(self.windows) * 100

    @property
    def avg_win_rate(self) -> float:
        """Average win rate across all windows."""
        if not self.windows:
            return 0.0
        return sum(w.win_rate for w in self.windows) / len(self.windows)

    @property
    def avg_profit_factor(self) -> float:
        """Average profit factor across all windows."""
        if not self.windows:
            return 0.0
        pfs = [w.profit_factor for w in self.windows if w.profit_factor < 100]
        return sum(pfs) / len(pfs) if pfs else 0.0

    @property
    def passes_walk_forward_test(self) -> bool:
        """True when ≥ 60% of windows are profitable (go-live criterion)."""
        return self.profitable_windows_pct >= 60.0


# ---------------------------------------------------------------------------
# ML Validator
# ---------------------------------------------------------------------------

class MLValidator:
    """Validate ML models via Purged K-Fold CV and Walk-Forward analysis.

    Parameters
    ----------
    n_splits:
        Number of folds for K-Fold CV.
    embargo_pct:
        Fraction of data to embargo between train/test to prevent leakage.
    confidence_threshold:
        Minimum prediction confidence for a signal to fire.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Purged K-Fold CV
    # ------------------------------------------------------------------

    def purged_kfold_cv(
        self,
        trainer: LGBMTrainer,
        df: pl.DataFrame,
    ) -> list[EvaluationResult]:
        """Purged K-Fold CV for time-series data.

        Re-uses ``PurgedKFoldCV`` from ``src.execution.walk_forward``.

        For each fold:
        1. train_df = fold train (with embargo gap)
        2. test_df = fold test
        3. Train a new model on train_df
        4. Evaluate on test_df
        5. Store EvaluationResult

        Parameters
        ----------
        trainer:
            LGBMTrainer (provides train / evaluate methods and config).
        df:
            Full feature DataFrame (already has target created).

        Returns
        -------
        List of EvaluationResult — one per fold.
        """
        cv = PurgedKFoldCV(
            n_splits=self.n_splits,
            embargo_pct=self.embargo_pct,
        )

        # Pick the best available timestamp column. Prefer a real Datetime
        # ``datetime`` (cheaper comparison and human-readable); fall back to
        # the always-present ``open_time`` epoch-ms integer. Without this
        # the split would slice by row index and a multi-symbol concat
        # frame would produce temporally-incoherent folds.
        if (
            "datetime" in df.columns
            and df["datetime"].dtype == pl.Datetime
            and df["datetime"].null_count() < len(df)
        ):
            ts_col = "datetime"
        elif "open_time" in df.columns:
            ts_col = "open_time"
        else:
            raise ValueError(
                "purged_kfold_cv: no timestamp column ('datetime' or "
                "'open_time') in dataframe — cannot do a time-aware split"
            )
        # Per-symbol filtering when the concat dataset carries a symbol
        # column (the multi-symbol case that originally produced ML-018).
        sym_col = "symbol" if "symbol" in df.columns else None

        results: list[EvaluationResult] = []

        for fold_idx, (train_df, test_df) in enumerate(
            cv.split(df, timestamp_col=ts_col, symbol_col=sym_col)
        ):
            _log.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"train={len(train_df)}, test={len(test_df)}"
            )

            try:
                # Reset feature columns so trainer re-discovers them
                trainer._feature_columns = []
                model = trainer.train(train_df)
                result = trainer.evaluate(model, test_df)
                results.append(result)

                _log.info(
                    f"  Fold {fold_idx + 1}: WR={result.win_rate:.1f}%, "
                    f"PF={result.profit_factor:.2f}, "
                    f"sig={result.signal_rate * 100:.1f}%"
                )
            except Exception as exc:
                _log.warning(f"  Fold {fold_idx + 1} failed: {exc}")
                continue

        _log.info(
            f"Purged K-Fold CV complete: {len(results)}/{self.n_splits} folds"
        )
        return results

    # ------------------------------------------------------------------
    # Walk-Forward ML
    # ------------------------------------------------------------------

    def walk_forward_ml(
        self,
        trainer: LGBMTrainer,
        symbols: list[str],
        features_dir: Path,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3,
    ) -> WalkForwardMLResult:
        """Walk-forward validation for ML models.

        Sliding-window approach:
        1. Load all data for the regime.
        2. Split into rolling train/test windows by calendar months.
        3. For each window: train a fresh model → evaluate on test.

        Parameters
        ----------
        trainer:
            LGBMTrainer with config (regime, params, etc.).
        symbols:
            List of symbols to load data for.
        features_dir:
            Directory with feature parquet files.
        train_months:
            Training window length in months.
        test_months:
            Test window length in months.
        step_months:
            Step size in months between windows.

        Returns
        -------
        WalkForwardMLResult with per-window metrics.
        """
        features_dir = Path(features_dir)

        # Load full dataset (all symbols, regime-filtered)
        full_df = self._load_full_data(trainer, symbols, features_dir)

        if full_df.is_empty():
            _log.warning("No data loaded for walk-forward ML")
            return WalkForwardMLResult(regime=trainer.config.regime)

        # Determine date range from data
        # Prefer open_time (always present and reliable) over datetime
        # (which may be all-null in synthetic data)
        has_datetime = (
            "datetime" in full_df.columns
            and full_df["datetime"].dtype == pl.Datetime
            and full_df["datetime"].null_count() < len(full_df)
        )

        if has_datetime:
            ts_col = "datetime"
        elif "open_time" in full_df.columns:
            ts_col = "open_time"
        else:
            _log.error("No timestamp column found for walk-forward splits")
            return WalkForwardMLResult(regime=trainer.config.regime)

        # Convert open_time (ms epoch) to Datetime for filtering
        if ts_col == "open_time":
            full_df = full_df.with_columns(
                (pl.col("open_time") * 1_000_000)  # ms → ns
                .cast(pl.Datetime("ns"))
                .alias("_wf_datetime")
            )
            ts_col = "_wf_datetime"

        data_start = full_df[ts_col].min()
        data_end = full_df[ts_col].max()

        if data_start is None or data_end is None:
            return WalkForwardMLResult(regime=trainer.config.regime)

        # Convert to Python datetime if still numeric
        if isinstance(data_start, (int, float)):
            data_start = datetime.fromtimestamp(data_start / 1000)
            data_end = datetime.fromtimestamp(data_end / 1000)

        _log.info(
            f"Walk-forward ML: data {data_start} → {data_end}, "
            f"train={train_months}m, test={test_months}m, step={step_months}m"
        )

        # Generate windows using WalkForwardValidator's split
        wf_validator = WalkForwardValidator(
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )

        windows: list[WindowMLResult] = []

        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(
            wf_validator.split(data_start, data_end)
        ):
            _log.info(
                f"  Window {i + 1}: train {train_start.date()} → "
                f"{train_end.date()}, test {test_start.date()} → "
                f"{test_end.date()}"
            )

            # Filter data by date range
            train_df = full_df.filter(
                (pl.col(ts_col) >= train_start)
                & (pl.col(ts_col) < train_end)
            )
            test_df = full_df.filter(
                (pl.col(ts_col) >= test_start)
                & (pl.col(ts_col) < test_end)
            )

            # Drop temporary column if present
            if "_wf_datetime" in train_df.columns:
                train_df = train_df.drop("_wf_datetime")
            if "_wf_datetime" in test_df.columns:
                test_df = test_df.drop("_wf_datetime")

            if len(train_df) < 50 or len(test_df) < 10:
                _log.warning(
                    f"  Window {i + 1}: insufficient data "
                    f"(train={len(train_df)}, test={len(test_df)}) — skipping"
                )
                continue

            try:
                # Reset feature columns for fresh training
                trainer._feature_columns = []
                model = trainer.train(train_df)
                result = trainer.evaluate(model, test_df)

                # Count signals
                n_signals = int(result.signal_rate * len(test_df))

                window = WindowMLResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    win_rate=result.win_rate,
                    profit_factor=result.profit_factor,
                    signal_rate=result.signal_rate,
                    n_signals=n_signals,
                    n_test_bars=len(test_df),
                )
                windows.append(window)

                _log.info(
                    f"  Window {i + 1}: WR={result.win_rate:.1f}%, "
                    f"PF={result.profit_factor:.2f}, "
                    f"signals={n_signals}"
                )

            except Exception as exc:
                _log.warning(f"  Window {i + 1} failed: {exc}")
                continue

        result = WalkForwardMLResult(
            regime=trainer.config.regime,
            windows=windows,
        )

        _log.info(
            f"Walk-forward ML complete: {len(windows)} windows, "
            f"profitable={result.profitable_windows_pct:.1f}%, "
            f"passes={result.passes_walk_forward_test}"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_full_data(
        self,
        trainer: LGBMTrainer,
        symbols: list[str],
        features_dir: Path,
    ) -> pl.DataFrame:
        """Load and prepare the full dataset (all symbols, with target)."""
        parts: list[pl.DataFrame] = []

        for symbol in symbols:
            sym_df = trainer._builder.load_and_combine(
                features_dir, symbols=[symbol],
            )
            if sym_df.is_empty():
                _log.warning(f"No data for {symbol}")
                continue

            # Create target (must be done before regime filter).
            # Branch exactly as LGBMTrainer.train does so walk-forward
            # validates the SAME label the model was trained on. Previously
            # this always called the legacy 1-bar sign(return) target, so
            # a v3 triple-barrier model was being scored against a target
            # it had never seen — the "60% profitable windows" go-live
            # gate was checked on the wrong labels.
            if trainer.config.use_triple_barrier:
                sym_df = trainer._builder.create_target_triple_barrier(
                    sym_df,
                    pt_multiplier=trainer.config.barrier_pt_multiplier,
                    sl_multiplier=trainer.config.barrier_sl_multiplier,
                    max_holding=trainer.config.barrier_max_holding,
                )
            else:
                sym_df = trainer._builder.create_target(
                    sym_df,
                    forward_bars=trainer.config.forward_bars,
                    threshold_atr_multiplier=trainer.config.threshold_atr_multiplier,
                )

            # Apply regime filter
            if trainer.config.regime != "all":
                sym_df = trainer._filter_by_regime(sym_df, trainer.config.regime)
                if sym_df.is_empty():
                    _log.warning(
                        f"No data for {symbol} after regime filter "
                        f"'{trainer.config.regime}'"
                    )
                    continue

            parts.append(sym_df)

        if not parts:
            return pl.DataFrame()

        return pl.concat(parts, how="diagonal")
