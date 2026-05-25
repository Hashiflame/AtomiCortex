"""Tests for Step H26 — unbiased feature importance via MDA.

The codebase already uses ``importance_type='gain'`` everywhere (gain
MDI < split MDI in cardinality bias), but gain still over-weights
continuous features. The fix adds a model-agnostic permutation MDA
helper that ranks features purely by their out-of-sample effect on
accuracy — the only fully unbiased option per AFML Ch.8.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from src.models.lgbm_trainer import LGBMTrainer


# ---------------------------------------------------------------------------
# H26 part 1: canonical IMPORTANCE_TYPE constant
# ---------------------------------------------------------------------------


class TestImportanceTypeConstant:
    def test_constant_is_gain(self):
        assert LGBMTrainer.IMPORTANCE_TYPE == "gain"

    def test_constant_is_exposed_on_class(self):
        """Must be addressable as a class attr (used by the trainer
        and may be overridden in tests / subclasses)."""
        assert hasattr(LGBMTrainer, "IMPORTANCE_TYPE")
        assert isinstance(LGBMTrainer.IMPORTANCE_TYPE, str)


# ---------------------------------------------------------------------------
# H26 part 2: permutation_importance helper
# ---------------------------------------------------------------------------


def _train_booster(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    train = lgb.Dataset(X, label=y)
    return lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "verbose": -1,
        },
        train,
        num_boost_round=40,
    )


class TestPermutationImportanceShape:
    def test_returns_one_score_per_feature(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        booster = _train_booster(X, y)
        out = LGBMTrainer.permutation_importance(
            booster, X, y, ["a", "b", "c", "d"], n_repeats=3,
        )
        assert set(out.keys()) == {"a", "b", "c", "d"}
        for v in out.values():
            assert isinstance(v, float)

    def test_reproducible_with_same_seed(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(150, 3))
        y = (X[:, 0] > 0).astype(int)
        booster = _train_booster(X, y)
        a = LGBMTrainer.permutation_importance(
            booster, X, y, ["x", "y", "z"], n_repeats=3, random_state=42,
        )
        b = LGBMTrainer.permutation_importance(
            booster, X, y, ["x", "y", "z"], n_repeats=3, random_state=42,
        )
        assert a == b


class TestPermutationImportanceCorrectness:
    def test_informative_feature_outranks_noise(self):
        """Standard sanity: the feature that drives ``y`` must produce
        the largest accuracy drop when permuted."""
        rng = np.random.default_rng(2)
        n = 600
        # Feature 0 deterministically drives the label; features 1-3
        # are independent noise of varying cardinality.
        signal = rng.integers(0, 2, size=n)
        noise_cont = rng.normal(size=n)
        noise_int = rng.integers(0, 50, size=n)
        noise_const = np.zeros(n)
        X = np.column_stack([signal, noise_cont, noise_int, noise_const])
        y = signal.copy()
        booster = _train_booster(X.astype(float), y)
        out = LGBMTrainer.permutation_importance(
            booster, X.astype(float), y,
            ["signal", "noise_cont", "noise_int", "noise_const"],
            n_repeats=5,
        )
        # The signal feature must have the largest permutation drop.
        ordered = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
        assert ordered[0][0] == "signal", out

    def test_binary_signal_not_dominated_by_high_cardinality_noise(self):
        """The core H26 anti-bias check: a binary feature that fully
        determines y must out-rank a noise feature with many unique
        values, even though MDI gain would over-weight the latter."""
        rng = np.random.default_rng(3)
        n = 800
        binary_signal = rng.integers(0, 2, size=n)
        # High-cardinality noise — irrelevant to y but rich in unique
        # values, which biases gain MDI upward.
        hi_card_noise = rng.normal(0, 1, size=n) + rng.integers(
            -1_000_000, 1_000_000, size=n,
        )
        X = np.column_stack([binary_signal, hi_card_noise]).astype(float)
        y = binary_signal.copy()
        booster = _train_booster(X, y)
        out = LGBMTrainer.permutation_importance(
            booster, X, y, ["binary_signal", "hi_card_noise"], n_repeats=5,
        )
        assert out["binary_signal"] > out["hi_card_noise"], out


# ---------------------------------------------------------------------------
# Edge cases / fail-soft
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_repeats_returns_empty(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        booster = _train_booster(X, y)
        out = LGBMTrainer.permutation_importance(
            booster, X, y, ["a", "b"], n_repeats=0,
        )
        assert out == {}

    def test_empty_X_returns_empty(self):
        # Pre-trained booster on real data, but ask for importance on
        # an empty matrix.
        rng = np.random.default_rng(5)
        Xtrain = rng.normal(size=(50, 2))
        ytrain = (Xtrain[:, 0] > 0).astype(int)
        booster = _train_booster(Xtrain, ytrain)
        out = LGBMTrainer.permutation_importance(
            booster, np.zeros((0, 2)), np.zeros(0, dtype=int),
            ["a", "b"], n_repeats=3,
        )
        assert out == {}

    def test_shape_mismatch_returns_empty(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        booster = _train_booster(X, y)
        # feature_names has 3 entries vs X.shape[1] == 2.
        out = LGBMTrainer.permutation_importance(
            booster, X, y, ["a", "b", "c"], n_repeats=3,
        )
        assert out == {}

    def test_broken_booster_returns_empty(self):
        class _Boom:
            def predict(self, *a, **kw):
                raise RuntimeError("synthetic")
        out = LGBMTrainer.permutation_importance(
            _Boom(), np.ones((10, 2)), np.zeros(10, dtype=int),
            ["a", "b"], n_repeats=2,
        )
        assert out == {}
