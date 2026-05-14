#!/usr/bin/env python3
"""
scripts/tune_models.py

Optuna hyperparameter tuning for LightGBM regime models.

Usage
-----
    python scripts/tune_models.py \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \
        --features-dir /home/hashiflame/AtomiCortex/data/features/ml_features \
        --models-dir /home/hashiflame/AtomiCortex/data/features/models \
        --regimes trend,range,high_vol \
        --n-trials 100 \
        --n-jobs 4 \
        --timeout 3600

Phase 3 — Step 3.7.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import optuna
import polars as pl

from src.logger import get_logger, setup_logging
from src.models.lgbm_trainer import EvaluationResult, LGBMTrainer, ModelConfig

_log = get_logger(__name__)

# Fixed ATR threshold — NOT part of the Optuna search space.
# Tuning this would create data leakage: the optimizer would search
# for a target definition (UP/DOWN/FLAT) that flatters the model on
# the val set, rather than measuring genuine predictive ability.
# (OPT-002 fix)
FIXED_ATR_THRESHOLD: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def compute_optuna_score(
    win_rate: float,
    profit_factor: float,
    n_signals: int,
) -> float:
    """Normalised composite trading score for Optuna optimisation.

    Each component is normalised to approximately [0, 1] before combining
    with explicit weights, preventing any single metric from dominating.

    ``score = 0.4 × wr_norm + 0.2 × sig_norm + 0.4 × pf_norm``

    Guards
    ------
    - ``n_signals < 30 → 0.0``  (too few for statistical significance)
    - ``win_rate < 50   → 0.0``  (at or below random-baseline for
      directional binary prediction after FLAT is excluded)
    - ``profit_factor`` capped at 5.0 to prevent outlier distortion

    Parameters
    ----------
    win_rate : float
        Percent 0–100.
    profit_factor : float
        Ratio (>1 is profitable).
    n_signals : int
        Absolute count of directional signals on the val set.
    """
    # OPT-010: raised from 10 → 30 for statistical significance
    if n_signals < 30:
        return 0.0
    # OPT-009: raised from 48 → 50 (random baseline for binary direction)
    if win_rate < 50.0:
        return 0.0

    # OPT-001: normalise each component to ~[0, 1] with explicit weights
    pf = min(profit_factor, 5.0)                                # tighter cap
    wr_frac = win_rate / 100.0
    wr_norm = (wr_frac - 0.50) / (1.0 - 0.50)                  # [0, 1]
    sig_norm = min(math.log(1 + n_signals) / math.log(500), 1.0)  # saturate at 500
    pf_norm = min((pf - 1.0) / 4.0, 1.0)                       # [0, 1] for PF 1→5

    return wr_norm * 0.4 + sig_norm * 0.2 + pf_norm * 0.4


# ══════════════════════════════════════════════════════════════════════════════
# Objective factory
# ══════════════════════════════════════════════════════════════════════════════

def create_objective(
    base_config: ModelConfig,
    features_dir: Path,
    models_dir: Path,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    n_jobs: int = 1,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective closure.

    Each trial creates its own ``LGBMTrainer`` to avoid shared mutable
    state — critical for thread safety when ``n_jobs > 1`` (OPT-006 fix).

    The ``threshold_atr_multiplier`` is fixed at ``FIXED_ATR_THRESHOLD``
    and is NOT part of the search space (OPT-002 fix).
    """

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

        # ── Build per-trial LightGBM params ──────────────────────────
        lgbm_defaults: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": base_config.random_state,
            "verbose": -1,
            # OPT-012: cap threads per worker to avoid oversubscription
            "nthread": max(1, (os.cpu_count() or 1) // max(1, n_jobs)),
        }
        lgbm_defaults.update(params)

        # ── Create a fresh trainer for this trial (OPT-006) ──────────
        trial_config = ModelConfig(
            regime=base_config.regime,
            symbols=base_config.symbols,
            forward_bars=base_config.forward_bars,
            threshold_atr_multiplier=FIXED_ATR_THRESHOLD,
            test_size_pct=base_config.test_size_pct,
            confidence_threshold=base_config.confidence_threshold,
            random_state=base_config.random_state,
            lgbm_params=lgbm_defaults,
        )
        trial_trainer = LGBMTrainer(
            config=trial_config,
            features_dir=features_dir,
            models_dir=models_dir,
        )

        # ── Train & evaluate ─────────────────────────────────────────
        try:
            model = trial_trainer.train(train_df)
            result = trial_trainer.evaluate(model, val_df)
        except Exception as exc:
            _log.debug(f"Trial {trial.number} failed: {exc}")
            return 0.0

        n_signals = int(result.signal_rate * len(val_df))

        score = compute_optuna_score(
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            n_signals=n_signals,
        )

        # Attach metrics to trial for reporting
        trial.set_user_attr("win_rate", result.win_rate)
        trial.set_user_attr("profit_factor", result.profit_factor)
        trial.set_user_attr("signal_rate", result.signal_rate)
        trial.set_user_attr("n_signals", n_signals)

        return score

    return objective


# ══════════════════════════════════════════════════════════════════════════════
# OptunaResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptunaResult:
    """Outcome of an Optuna tuning run for one regime."""

    regime: str
    best_params: dict[str, Any]
    best_score: float
    best_win_rate: float
    best_profit_factor: float
    best_signal_rate: float
    n_trials: int
    study_name: str


# ══════════════════════════════════════════════════════════════════════════════
# OptunaTrainer
# ══════════════════════════════════════════════════════════════════════════════

class OptunaTrainer:
    """Orchestrate Optuna hyperparameter search across regimes.

    Parameters
    ----------
    n_trials:
        Maximum number of Optuna trials per regime.
    n_jobs:
        Number of parallel workers (1 = sequential).
    timeout:
        Maximum wall-clock seconds per study (None = unlimited).
    storage:
        SQLAlchemy URL for Optuna persistence.
        Use ``"sqlite:///optuna_studies.db"`` to survive restarts.
    """

    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: int | None = 3600,
        storage: str | None = None,
    ) -> None:
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.storage = storage

    # ------------------------------------------------------------------
    # Storage helper (OPT-015)
    # ------------------------------------------------------------------

    def _get_storage(self) -> str | optuna.storages.RDBStorage | None:
        """Return an appropriate Optuna storage backend.

        For ``n_jobs > 1`` with SQLite, wrap in ``RDBStorage`` with a
        connection pool to avoid ``database is locked`` errors.
        """
        if not self.storage:
            return None
        if self.n_jobs > 1:
            return optuna.storages.RDBStorage(
                url=self.storage,
                engine_kwargs={"pool_size": self.n_jobs + 2},
            )
        return self.storage

    # ------------------------------------------------------------------
    # Single regime
    # ------------------------------------------------------------------

    def tune_regime(
        self,
        regime: str,
        symbols: list[str],
        features_dir: Path,
        models_dir: Path,
    ) -> OptunaResult:
        """Run Optuna for a single regime and return the best result.

        Steps
        -----
        1. Build a ``LGBMTrainer`` for the regime.
        2. Load data → 80/20 walk-forward split.
        3. Create Optuna study (maximise composite score).
        4. Optimise ``n_trials`` trials.
        5. Package the best trial into ``OptunaResult``.
        """
        config = ModelConfig(
            regime=regime,
            symbols=symbols,
            threshold_atr_multiplier=FIXED_ATR_THRESHOLD,
        )
        trainer = LGBMTrainer(
            config=config,
            features_dir=features_dir,
            models_dir=models_dir,
        )

        train_df, val_df = trainer.prepare_data()

        study_name = f"atomicortex_{regime}"

        # Suppress Optuna per-trial chatter
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # OPT-005: NopPruner instead of MedianPruner (single-step
        # report is useless with MedianPruner; true pruning would
        # require multi-step reporting per boosting round).
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=self._get_storage(),
            load_if_exists=True,
            pruner=optuna.pruners.NopPruner(),
        )

        objective = create_objective(
            base_config=config,
            features_dir=features_dir,
            models_dir=models_dir,
            train_df=train_df,
            val_df=val_df,
            n_jobs=self.n_jobs,
        )

        # Progress callback
        def _progress_callback(
            study: optuna.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            n = trial.number + 1
            if n % 10 == 0 or n == 1:
                best = study.best_trial
                wr = best.user_attrs.get("win_rate", 0.0)
                pf = best.user_attrs.get("profit_factor", 0.0)
                sig = best.user_attrs.get("signal_rate", 0.0) * 100
                print(
                    f"  Trial {n}/{self.n_trials}: "
                    f"best_score={study.best_value:.3f}, "
                    f"WR={wr:.1f}%, PF={pf:.2f}, sig={sig:.1f}%"
                )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            callbacks=[_progress_callback],
            show_progress_bar=False,
        )

        # OPT-004: guard against all trials failing / being pruned
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError(
                f"All {len(study.trials)} trials failed or were pruned "
                f"for regime '{regime}'"
            )

        best = study.best_trial

        result = OptunaResult(
            regime=regime,
            best_params=best.params,
            best_score=study.best_value,
            best_win_rate=best.user_attrs.get("win_rate", 0.0),
            best_profit_factor=best.user_attrs.get("profit_factor", 0.0),
            best_signal_rate=best.user_attrs.get("signal_rate", 0.0),
            n_trials=len(study.trials),
            study_name=study_name,
        )

        _log.info(
            f"Optuna [{regime}]: best_score={result.best_score:.4f}, "
            f"WR={result.best_win_rate:.1f}%, "
            f"PF={result.best_profit_factor:.2f}, "
            f"trials={result.n_trials}"
        )
        return result

    # ------------------------------------------------------------------
    # All regimes
    # ------------------------------------------------------------------

    def tune_all_regimes(
        self,
        symbols: list[str],
        features_dir: Path,
        models_dir: Path,
        regimes: list[str] | None = None,
    ) -> dict[str, OptunaResult]:
        """Tune each regime sequentially and return a dict of results."""
        if regimes is None:
            regimes = ["trend", "range", "high_vol"]

        results: dict[str, OptunaResult] = {}
        for regime in regimes:
            print(f"\n{'═' * 60}")
            print(f"  Tuning regime: {regime}")
            print(f"{'═' * 60}")
            try:
                result = self.tune_regime(
                    regime=regime,
                    symbols=symbols,
                    features_dir=features_dir,
                    models_dir=models_dir,
                )
                results[regime] = result
            except Exception as exc:
                _log.error(f"Tuning failed for regime '{regime}': {exc}")
                continue

        return results

    # ------------------------------------------------------------------
    # Retrain with best params
    # ------------------------------------------------------------------

    def retrain_with_best_params(
        self,
        regime: str,
        best_params: dict[str, Any],
        symbols: list[str],
        features_dir: Path,
        models_dir: Path,
    ) -> EvaluationResult:
        """Retrain on FULL data (train+val) with the best parameters.

        The final production model is saved to ``models_dir/{regime}_model.pkl``.

        Parameters
        ----------
        regime:
            Market regime name.
        best_params:
            Best LightGBM parameters from Optuna.
            ``threshold_atr_multiplier`` is ignored if present — we always
            use ``FIXED_ATR_THRESHOLD``.
        symbols:
            Symbols to train on.
        features_dir:
            Path to feature parquet files.
        models_dir:
            Path to save the final model.

        Returns
        -------
        EvaluationResult on the held-out test portion (last 20%).
        Note: the production model is trained on 100% of data; the
        eval result is an *in-sample* quality check, not true OOS.
        """
        # OPT-014: use .get() instead of .pop() to avoid mutating caller's dict
        # Filter out non-LightGBM keys safely
        lgbm_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 42,
            "verbose": -1,
        }
        lgbm_params.update(
            {k: v for k, v in best_params.items()
             if k != "threshold_atr_multiplier"}
        )

        config = ModelConfig(
            regime=regime,
            symbols=symbols,
            threshold_atr_multiplier=FIXED_ATR_THRESHOLD,
            lgbm_params=lgbm_params,
        )

        trainer = LGBMTrainer(
            config=config,
            features_dir=features_dir,
            models_dir=models_dir,
        )

        train_df, test_df = trainer.prepare_data()

        # OPT-003 fix: train on full data (train + test combined).
        # prepare_data() does an 80/20 per-symbol temporal split; we
        # recombine so the production model sees ALL available data.
        # Evaluation on test_df is therefore in-sample and serves only
        # as a sanity-check on model quality — it is NOT true OOS.
        full_train = pl.concat([train_df, test_df], how="diagonal")
        model = trainer.train(full_train)
        result = trainer.evaluate(model, test_df)

        _log.info(
            f"Retrained [{regime}]: WR={result.win_rate}%, "
            f"PF={result.profit_factor}, sig={result.signal_rate*100:.1f}%"
        )

        # Log best params to MLflow
        try:
            mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
            mlflow.set_experiment("AtomiCortex_Optuna")
            with mlflow.start_run(run_name=f"optuna_best_{regime}"):
                mlflow.log_params({
                    f"best_{k}": v for k, v in best_params.items()
                    if k != "threshold_atr_multiplier"
                })
                mlflow.log_param("best_threshold_atr", FIXED_ATR_THRESHOLD)
                mlflow.log_metric("win_rate", result.win_rate)
                mlflow.log_metric("profit_factor", result.profit_factor)
                mlflow.log_metric("signal_rate", result.signal_rate)
        except Exception as exc:
            _log.warning(f"MLflow logging failed (non-fatal): {exc}")

        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for LightGBM regime models"
    )
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
        help="Directory to save trained models",
    )
    p.add_argument(
        "--regimes",
        default="trend,range,high_vol",
        help="Comma-separated regimes (default: trend,range,high_vol)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per regime (default: 100)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Max seconds per regime (default: 3600)",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna_studies.db",
    )
    return p.parse_args()


def _print_report(
    results: dict[str, OptunaResult],
    retrain_results: dict[str, EvaluationResult],
) -> None:
    """Print formatted Optuna + retrain summary."""
    print(f"\n{'═' * 70}")
    print(f"  AtomiCortex — Optuna Tuning Report")
    print(f"{'═' * 70}")

    for regime, optuna_res in results.items():
        print(f"\n{'─' * 70}")
        print(f"  Regime: {regime}")
        print(f"{'─' * 70}")
        print(f"  Best trial:       {optuna_res.n_trials} total")
        print(f"  Score:            {optuna_res.best_score:.4f}")
        print(f"  Win Rate:         {optuna_res.best_win_rate:.1f}%")
        print(f"  Profit Factor:    {optuna_res.best_profit_factor:.2f}")
        print(f"  Signal Rate:      {optuna_res.best_signal_rate * 100:.1f}%")
        print(f"  Best params:")
        for k, v in sorted(optuna_res.best_params.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

        if regime in retrain_results:
            r = retrain_results[regime]
            passes = "✅" if r.passes_minimum_thresholds() else "❌"
            print(f"\n  Retrained on full data:")
            print(
                f"    WR={r.win_rate:.1f}%, PF={r.profit_factor:.2f}, "
                f"sig={r.signal_rate*100:.1f}%  {passes}"
            )

    print(f"\n{'═' * 70}\n")


def main() -> None:
    setup_logging()
    args = _parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    regimes = [r.strip() for r in args.regimes.split(",")]

    print(f"\n{'═' * 60}")
    print(f"  AtomiCortex — Optuna Hyperparameter Tuning")
    print(f"{'═' * 60}")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Regimes:    {', '.join(regimes)}")
    print(f"  Trials:     {args.n_trials} per regime")
    print(f"  Workers:    {args.n_jobs}")
    print(f"  Timeout:    {args.timeout}s per regime")
    print(f"  Storage:    {args.storage or 'in-memory'}")
    print(f"  ATR thr:    {FIXED_ATR_THRESHOLD} (fixed, not tuned)")
    print(f"{'═' * 60}")

    tuner = OptunaTrainer(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        storage=args.storage,
    )

    # ── Tune ──────────────────────────────────────────────────────────
    results = tuner.tune_all_regimes(
        symbols=symbols,
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        regimes=regimes,
    )

    # ── Retrain with best params ──────────────────────────────────────
    retrain_results: dict[str, EvaluationResult] = {}
    for regime, optuna_res in results.items():
        print(f"\n  Retraining {regime} with best params...")
        try:
            eval_result = tuner.retrain_with_best_params(
                regime=regime,
                best_params=dict(optuna_res.best_params),  # copy
                symbols=symbols,
                features_dir=args.features_dir,
                models_dir=args.models_dir,
            )
            retrain_results[regime] = eval_result
        except Exception as exc:
            _log.error(f"Retrain failed for {regime}: {exc}")

    # ── Report ────────────────────────────────────────────────────────
    _print_report(results, retrain_results)


if __name__ == "__main__":
    main()
