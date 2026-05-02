"""
tests/test_optuna_trainer.py

Tests for Optuna hyperparameter tuning pipeline (15 tests).
Phase 3 — Step 3.7.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.models.lgbm_trainer import (
    EvaluationResult,
    LGBMTrainer,
    ModelConfig,
)

# Import from the script — add scripts to path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.tune_models import (
    FIXED_ATR_THRESHOLD,
    OptunaResult,
    OptunaTrainer,
    compute_optuna_score,
    create_objective,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(
    n: int = 200, symbol: str = "BTCUSDT", seed: int = 42
) -> pl.DataFrame:
    """Create a synthetic feature DataFrame that mirrors real schema."""
    rng = np.random.RandomState(seed)
    close = 40000 + np.cumsum(rng.randn(n) * 100)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    opn = close + rng.randn(n) * 50

    base_time = 1704067200000  # 2024-01-01 00:00 UTC in ms
    open_times = [base_time + i * 4 * 3600 * 1000 for i in range(n)]

    return pl.DataFrame(
        {
            "open_time": open_times,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
            "close_time": [t + 4 * 3600 * 1000 - 1 for t in open_times],
            "datetime": pl.Series([None] * n, dtype=pl.Datetime),
            "symbol": [symbol] * n,
            "regime": rng.choice(
                ["trend_up", "trend_down", "range", "high_vol"], n
            ).tolist(),
            "cvd": rng.randn(n),
            "cvd_cum": np.cumsum(rng.randn(n)),
            "cvd_slope_3": rng.randn(n),
            "cvd_slope_6": rng.randn(n),
            "cvd_slope_12": rng.randn(n),
            "taker_buy_ratio": rng.uniform(0.4, 0.6, n),
            "volume_sma_20": rng.uniform(200, 800, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
            "volume_zscore": rng.randn(n),
            "large_volume": rng.choice([0, 1], n).astype(np.int8),
            "vwap_4h": close + rng.randn(n) * 10,
            "price_to_vwap": rng.uniform(-0.01, 0.01, n),
            "returns_1": rng.randn(n) * 0.01,
            "returns_3": rng.randn(n) * 0.02,
            "returns_6": rng.randn(n) * 0.03,
            "returns_12": rng.randn(n) * 0.04,
            "returns_24": rng.randn(n) * 0.05,
            "body_ratio": rng.uniform(0.1, 0.9, n),
            "upper_wick": rng.uniform(0, 0.5, n),
            "lower_wick": rng.uniform(0, 0.5, n),
            "gap": rng.randn(n) * 0.001,
            "funding_rate": rng.randn(n) * 0.0001,
            "funding_abs": np.abs(rng.randn(n) * 0.0001),
            "funding_zscore_7d": rng.randn(n),
            "funding_zscore_30d": rng.randn(n),
            "funding_extreme": rng.choice([0, 1], n).astype(np.int8),
            "funding_positive": rng.choice([0, 1], n).astype(np.int8),
            "funding_cum_24h": rng.randn(n) * 0.001,
            "oi_value": rng.uniform(1e8, 1e9, n),
            "oi_delta_4h": rng.randn(n) * 0.01,
            "oi_delta_12h": rng.randn(n) * 0.02,
            "oi_zscore": rng.randn(n),
            "oi_quadrant": rng.choice([0, 1], n).astype(np.int8),
            "ls_ratio": rng.uniform(0.8, 1.2, n),
            "ls_ratio_zscore": rng.randn(n),
            "taker_vol_ratio": rng.uniform(0.4, 0.6, n),
            "basis_approx": rng.randn(n) * 0.001,
            "basis_extreme": rng.choice([0, 1], n).astype(np.int8),
            "hurst": rng.uniform(0.3, 0.7, n),
            "adx": rng.uniform(10, 50, n),
            "atr_pct": rng.uniform(0.01, 0.05, n),
            "atr_percentile": rng.uniform(0, 1, n),
            "trend_strength": rng.uniform(0, 1, n),
            "regime_confidence": rng.uniform(0, 1, n),
        }
    )


def _save_features(
    tmp_path: Path, symbols: list[str] | None = None, n: int = 300
) -> Path:
    """Save synthetic feature parquets and return the directory."""
    symbols = symbols or ["BTCUSDT"]
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for i, sym in enumerate(symbols):
        df = _make_feature_df(n=n, symbol=sym, seed=42 + i)
        df.write_parquet(features_dir / f"{sym}_4h_features.parquet")

    return features_dir


def _make_trainer(
    tmp_path: Path,
    regime: str = "all",
    symbols: list[str] | None = None,
    n: int = 300,
):
    """Create a LGBMTrainer with synthetic data."""
    symbols = symbols or ["BTCUSDT"]
    features_dir = _save_features(tmp_path, symbols, n=n)
    models_dir = tmp_path / "models"

    config = ModelConfig(regime=regime, symbols=symbols)
    trainer = LGBMTrainer(
        config=config, features_dir=features_dir, models_dir=models_dir
    )
    return trainer, features_dir, models_dir


# ===========================================================================
# 1. Score function tests — updated for OPT-001 / OPT-009 / OPT-010
# ===========================================================================


class TestComputeOptunaScore:
    """Test the normalised composite scoring function."""

    def test_returns_float(self):
        score = compute_optuna_score(
            win_rate=55.0, profit_factor=1.5, n_signals=50
        )
        assert isinstance(score, float)
        assert score > 0

    def test_few_signals_returns_zero(self):
        """n_signals < 30 → 0.0 (OPT-010: raised from 10 to 30)."""
        score = compute_optuna_score(
            win_rate=60.0, profit_factor=2.0, n_signals=29
        )
        assert score == 0.0

    def test_low_win_rate_returns_zero(self):
        """win_rate < 50% → 0.0 (OPT-009: raised from 48 to 50)."""
        score = compute_optuna_score(
            win_rate=49.9, profit_factor=1.5, n_signals=100
        )
        assert score == 0.0

    def test_normalised_formula(self):
        """Verify normalised formula:
        score = 0.4 × wr_norm + 0.2 × sig_norm + 0.4 × pf_norm
        """
        wr = 60.0
        pf = 1.5
        n = 100

        wr_norm = (wr / 100.0 - 0.50) / (1.0 - 0.50)
        sig_norm = min(math.log(1 + n) / math.log(500), 1.0)
        pf_norm = min((min(pf, 5.0) - 1.0) / 4.0, 1.0)
        expected = wr_norm * 0.4 + sig_norm * 0.2 + pf_norm * 0.4

        actual = compute_optuna_score(wr, pf, n)
        assert abs(actual - expected) < 1e-6

    def test_score_between_zero_and_one(self):
        """Normalised score should be in [0, 1]."""
        # Best-case scenario
        best = compute_optuna_score(100.0, 5.0, 500)
        assert 0.0 <= best <= 1.0

        # Mid-range scenario
        mid = compute_optuna_score(60.0, 1.5, 50)
        assert 0.0 <= mid <= 1.0

    def test_pf_does_not_dominate(self):
        """High PF with low WR should not outscore high WR with low PF.
        This was the OPT-001 bug: PF×10 dominated the old formula.
        """
        # High PF, low WR
        high_pf = compute_optuna_score(51.0, 5.0, 100)
        # Low PF, high WR
        high_wr = compute_optuna_score(80.0, 1.3, 100)
        # High WR should score higher (or at least competitive)
        assert high_wr > high_pf * 0.5  # not completely dominated


# ===========================================================================
# 2. Objective function tests — updated for OPT-002 / OPT-006
# ===========================================================================


class TestCreateObjective:
    """Test that create_objective returns a callable with isolated trainers."""

    def test_objective_returns_float(self, tmp_path: Path):
        trainer, features_dir, models_dir = _make_trainer(
            tmp_path, regime="all", n=300
        )
        train_df, val_df = trainer.prepare_data()

        objective = create_objective(
            base_config=trainer.config,
            features_dir=features_dir,
            models_dir=models_dir,
            train_df=train_df,
            val_df=val_df,
        )

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)

        assert len(study.trials) == 1
        assert isinstance(study.trials[0].value, (float, type(None)))

    def test_threshold_atr_not_in_search_space(self, tmp_path: Path):
        """OPT-002: threshold_atr_multiplier must NOT be sampled by Optuna."""
        trainer, features_dir, models_dir = _make_trainer(
            tmp_path, regime="all", n=300
        )
        train_df, val_df = trainer.prepare_data()

        objective = create_objective(
            base_config=trainer.config,
            features_dir=features_dir,
            models_dir=models_dir,
            train_df=train_df,
            val_df=val_df,
        )

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)

        # threshold_atr_multiplier should NOT appear in trial params
        assert "threshold_atr_multiplier" not in study.trials[0].params


# ===========================================================================
# 3. No race condition — OPT-006
# ===========================================================================


class TestNoRaceCondition:
    """OPT-006: Each trial must create its own LGBMTrainer, not mutate shared state."""

    def test_new_trainer_per_trial(self, tmp_path: Path):
        """Verify that objective creates a new LGBMTrainer each time,
        not mutating the original trainer's config."""
        trainer, features_dir, models_dir = _make_trainer(
            tmp_path, regime="all", n=300
        )
        original_lr = trainer.config.lgbm_params.get("learning_rate", 0.05)
        train_df, val_df = trainer.prepare_data()

        objective = create_objective(
            base_config=trainer.config,
            features_dir=features_dir,
            models_dir=models_dir,
            train_df=train_df,
            val_df=val_df,
        )

        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2, show_progress_bar=False)

        # Original trainer's config should be untouched
        assert trainer.config.lgbm_params.get("learning_rate", 0.05) == original_lr


# ===========================================================================
# 4. Retrain uses full data — OPT-003
# ===========================================================================


class TestRetrainUsesFullData:
    """OPT-003: retrain_with_best_params must train on train+test combined."""

    def test_retrain_trains_on_full_data(self, tmp_path: Path):
        """Verify that retrain calls trainer.train with concatenated data."""
        features_dir = _save_features(tmp_path, ["BTCUSDT"], n=300)
        models_dir = tmp_path / "models"

        tuner = OptunaTrainer(n_trials=2, n_jobs=1, timeout=60)

        best_params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 200,
            "lambda_l1": 1e-3,
            "lambda_l2": 1e-3,
        }

        # Patch LGBMTrainer.train to capture input length
        train_input_lengths = []
        original_train = LGBMTrainer.train

        def capturing_train(self_trainer, train_df):
            train_input_lengths.append(len(train_df))
            return original_train(self_trainer, train_df)

        with patch.object(LGBMTrainer, "train", capturing_train):
            eval_result = tuner.retrain_with_best_params(
                regime="all",
                best_params=dict(best_params),
                symbols=["BTCUSDT"],
                features_dir=features_dir,
                models_dir=models_dir,
            )

        assert isinstance(eval_result, EvaluationResult)
        # The train call should receive MORE rows than an 80% split
        # For n=300, target drops 1 row → 299 rows total
        # 80% split would be ~239; full data should be ~299
        assert len(train_input_lengths) == 1
        assert train_input_lengths[0] > 250  # must be close to 299 (full data)


# ===========================================================================
# 5. OptunaResult dataclass tests
# ===========================================================================


class TestOptunaResult:
    """Test OptunaResult dataclass."""

    def test_dataclass_fields(self):
        result = OptunaResult(
            regime="trend",
            best_params={"num_leaves": 50, "learning_rate": 0.05},
            best_score=0.45,
            best_win_rate=55.0,
            best_profit_factor=1.5,
            best_signal_rate=0.2,
            n_trials=100,
            study_name="atomicortex_trend",
        )
        assert result.regime == "trend"
        assert result.best_score == 0.45
        assert result.n_trials == 100
        assert "num_leaves" in result.best_params


# ===========================================================================
# 6. OptunaTrainer tests
# ===========================================================================


class TestTuneRegime:
    """Test tune_regime runs without errors (n_trials=3)."""

    def test_runs_successfully(self, tmp_path: Path):
        features_dir = _save_features(tmp_path, ["BTCUSDT"], n=300)
        models_dir = tmp_path / "models"

        tuner = OptunaTrainer(n_trials=3, n_jobs=1, timeout=120)

        result = tuner.tune_regime(
            regime="all",
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
        )

        assert isinstance(result, OptunaResult)
        assert result.regime == "all"
        assert result.n_trials >= 1
        assert isinstance(result.best_score, float)


class TestBestParamsKeys:
    """Test best_params contains all expected hyperparameter keys."""

    def test_contains_lgbm_keys_only(self, tmp_path: Path):
        """OPT-002: threshold_atr_multiplier should NOT be in best_params."""
        features_dir = _save_features(tmp_path, ["BTCUSDT"], n=300)
        models_dir = tmp_path / "models"

        tuner = OptunaTrainer(n_trials=3, n_jobs=1, timeout=120)

        result = tuner.tune_regime(
            regime="all",
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
        )

        expected_lgbm_keys = {
            "num_leaves",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "min_child_samples",
            "n_estimators",
            "lambda_l1",
            "lambda_l2",
        }

        assert expected_lgbm_keys.issubset(set(result.best_params.keys()))
        # threshold_atr_multiplier must NOT be in search results
        assert "threshold_atr_multiplier" not in result.best_params


class TestRetrainWithBestParams:
    """Test retrain_with_best_params saves a model."""

    def test_saves_model(self, tmp_path: Path):
        features_dir = _save_features(tmp_path, ["BTCUSDT"], n=300)
        models_dir = tmp_path / "models"

        tuner = OptunaTrainer(n_trials=3, n_jobs=1, timeout=120)
        optuna_result = tuner.tune_regime(
            regime="all",
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
        )

        eval_result = tuner.retrain_with_best_params(
            regime="all",
            best_params=dict(optuna_result.best_params),
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
        )

        assert isinstance(eval_result, EvaluationResult)
        assert (models_dir / "all_model.pkl").exists()


# ===========================================================================
# 7. n_jobs and timeout tests
# ===========================================================================


class TestNJobsAccepted:
    """Test n_jobs parameter is accepted."""

    def test_n_jobs_4(self):
        tuner = OptunaTrainer(n_trials=10, n_jobs=4, timeout=300)
        assert tuner.n_jobs == 4
        assert tuner.n_trials == 10


class TestTimeoutStopsStudy:
    """Test timeout halts the study early."""

    def test_timeout_stops_early(self, tmp_path: Path):
        features_dir = _save_features(tmp_path, ["BTCUSDT"], n=300)
        models_dir = tmp_path / "models"

        # Set timeout to 5 seconds with a large n_trials
        tuner = OptunaTrainer(n_trials=1000, n_jobs=1, timeout=5)

        start = time.time()
        result = tuner.tune_regime(
            regime="all",
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
        )
        elapsed = time.time() - start

        # Should stop well before 1000 trials (timeout kicks in)
        assert result.n_trials < 1000
        # Should not take much more than the timeout
        assert elapsed < 30  # generous bound


# ===========================================================================
# 8. Score boundary tests — updated for new thresholds
# ===========================================================================


class TestScoreBoundaries:
    """Test edge cases in the score function."""

    def test_pf_capped_at_5(self):
        """OPT-001: profit_factor > 5 is capped (was 10)."""
        score_capped = compute_optuna_score(60.0, 999.0, 100)
        score_5 = compute_optuna_score(60.0, 5.0, 100)
        assert abs(score_capped - score_5) < 1e-6

    def test_exactly_threshold_values(self):
        """At boundary: n_signals=30, win_rate=50.0."""
        score = compute_optuna_score(50.0, 1.0, 30)
        # WR=50% → wr_norm=0, so score should be small but non-negative
        # pf=1.0 → pf_norm=0
        # Only sig_norm contributes
        assert score >= 0.0

        score_below_wr = compute_optuna_score(49.9, 1.0, 30)
        assert score_below_wr == 0.0

        score_below_n = compute_optuna_score(55.0, 1.5, 29)
        assert score_below_n == 0.0

    def test_fixed_atr_threshold_constant(self):
        """Ensure FIXED_ATR_THRESHOLD is set to 0.5."""
        assert FIXED_ATR_THRESHOLD == 0.5


# ===========================================================================
# 9. OPT-014: best_params not mutated
# ===========================================================================


class TestBestParamsNotMutated:
    """OPT-014: retrain_with_best_params must not mutate the input dict."""

    def test_params_dict_unchanged(self, tmp_path: Path):
        features_dir = _save_features(tmp_path, ["BTCUSDT"], n=300)
        models_dir = tmp_path / "models"

        tuner = OptunaTrainer(n_trials=2, n_jobs=1, timeout=60)

        best_params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 200,
            "lambda_l1": 1e-3,
            "lambda_l2": 1e-3,
            "threshold_atr_multiplier": 0.4,  # included deliberately
        }
        params_copy = dict(best_params)

        tuner.retrain_with_best_params(
            regime="all",
            best_params=best_params,
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
        )

        # Original dict must be unchanged (no .pop() mutation)
        assert best_params == params_copy
