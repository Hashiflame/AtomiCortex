"""Regression tests for Phase 3 Step 3.3 — NaN preservation through to LightGBM.

Pre-fix every code path between the feature pipeline and the booster
called ``np.nan_to_num(X, nan=0, posinf=0, neginf=0)``. That collapsed
two semantically distinct cases into the same value:

* "data unavailable" (e.g. a rolling feature still warming up) — should
  be ``NaN`` and routed by LightGBM to the missing-value branch of every
  split; and
* "value is actually zero" (e.g. funding_rate measured at 0 %) —
  a legitimate real-valued zero.

After the fix:

* ``LGBMTrainer._encode_features`` only converts ±inf to NaN (LightGBM
  cannot consume ±inf) and leaves NaN intact.
* ``MLTradingStrategy``'s feature-vector builders use ``_safe_float``
  which returns NaN for missing / None / non-numeric / ±inf inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# 1. _safe_float helper — the per-cell NaN-preserving converter
# ---------------------------------------------------------------------------

class TestSafeFloat:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.execution.strategies.ml_strategy import _safe_float
        self.f = _safe_float

    def test_none_returns_nan(self) -> None:
        assert np.isnan(self.f(None))

    def test_missing_dict_key_returns_nan(self) -> None:
        d: dict = {}
        assert np.isnan(self.f(d.get("missing")))

    def test_positive_inf_returns_nan(self) -> None:
        assert np.isnan(self.f(float("inf")))

    def test_negative_inf_returns_nan(self) -> None:
        assert np.isnan(self.f(float("-inf")))

    def test_string_returns_nan(self) -> None:
        assert np.isnan(self.f("not a number"))

    def test_real_zero_passes_through(self) -> None:
        """The whole point: real 0.0 must survive — it is NOT missing."""
        assert self.f(0.0) == 0.0
        assert not np.isnan(self.f(0.0))

    def test_finite_floats_pass_through(self) -> None:
        for v in [-1.5, 0.0, 42.0, 1e-9, 1e9]:
            assert self.f(v) == pytest.approx(v)

    def test_integer_converted_to_float(self) -> None:
        assert self.f(7) == 7.0


# ---------------------------------------------------------------------------
# 2. LGBMTrainer._encode_features — NaN preserved, ±inf → NaN
# ---------------------------------------------------------------------------

def _trainer(tmp_path) -> "LGBMTrainer":
    from src.models.lgbm_trainer import LGBMTrainer, ModelConfig
    cfg = ModelConfig(regime="all", symbols=["BTCUSDT"])
    return LGBMTrainer(cfg, features_dir=tmp_path, models_dir=tmp_path)


def _toy_df_with_nans_and_inf() -> pl.DataFrame:
    """Small DF with one feature carrying NaN, +inf, -inf, and 0.0."""
    return pl.DataFrame({
        "open_time": list(range(8)),
        "atr_pct":   [0.01] * 8,
        "regime":    ["all"] * 8,
        # `feat_a` deliberately mixes the four cases we care about:
        #   index 0 → NaN  (missing)
        #   index 1 → +inf (must become NaN downstream)
        #   index 2 → -inf (must become NaN downstream)
        #   index 3 → 0.0  (real zero, must survive)
        #   indices 4..7 → ordinary floats
        "feat_a": [
            float("nan"), float("inf"), float("-inf"),
            0.0, 1.0, 2.0, 3.0, 4.0,
        ],
        "feat_b": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "target": [1, -1, 1, -1, 1, -1, 1, -1],
    })


class TestEncoderPreservesNaN:
    def test_nan_cells_pass_through(self, tmp_path) -> None:
        trainer = _trainer(tmp_path)
        # Discover features (so feat_a / feat_b enter the matrix).
        trainer._feature_columns = ["feat_a", "feat_b"]
        X, _y, cols = trainer._prepare_xy(_toy_df_with_nans_and_inf())
        # cols may include extras; locate feat_a deterministically.
        a_idx = cols.index("feat_a")
        # Index 0 was NaN → must STILL be NaN (not 0.0).
        assert np.isnan(X[0, a_idx])
        # Index 3 was real 0.0 → must remain 0.0 (NOT NaN, NOT replaced).
        assert X[3, a_idx] == 0.0
        assert not np.isnan(X[3, a_idx])

    def test_inf_converted_to_nan(self, tmp_path) -> None:
        trainer = _trainer(tmp_path)
        trainer._feature_columns = ["feat_a", "feat_b"]
        X, _y, cols = trainer._prepare_xy(_toy_df_with_nans_and_inf())
        a_idx = cols.index("feat_a")
        # +inf @ row 1 and -inf @ row 2 must become NaN (LightGBM cannot
        # consume ±inf; mapping them to NaN routes them to the missing
        # branch instead of to a literal large/small number).
        assert np.isnan(X[1, a_idx])
        assert np.isnan(X[2, a_idx])
        # And NO cell anywhere should be ±inf.
        assert not np.any(np.isinf(X))


# ---------------------------------------------------------------------------
# 3. End-to-end: LightGBM actually sees NaN (proves train/serve consistency)
# ---------------------------------------------------------------------------

class TestLightGBMSeesNaN:
    def test_booster_fit_receives_nan(self, tmp_path) -> None:
        """The encoder hands NaN-containing arrays to LightGBM. We assert
        the captured matrix has NaN in the original cells — proof the
        ``nan_to_num`` zeroing is gone."""
        import lightgbm as lgb
        trainer = _trainer(tmp_path)
        trainer._feature_columns = ["feat_a", "feat_b"]
        df = _toy_df_with_nans_and_inf()
        X, _y, cols = trainer._prepare_xy(df)
        a_idx = cols.index("feat_a")
        # Round-trip through LightGBM's Dataset to confirm it accepts NaN.
        ds = lgb.Dataset(X, label=np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        ds.construct()  # raises if X is invalid for LightGBM
        # And the data we handed in still has NaN where we put it.
        assert np.isnan(X[0, a_idx])

    def test_predictions_differ_from_zero_imputed_model(self, tmp_path) -> None:
        """A model trained with NaN-preserved features behaves
        differently from one trained with NaN→0 imputation. If they
        matched, the fix would be a no-op. The setup deliberately
        encodes the warmup vs zero distinction the bug erased."""
        import lightgbm as lgb

        rng = np.random.default_rng(7)
        n = 400
        feat = rng.normal(0, 1, n)
        # Engineer the leakage: when the feature is "missing", the label
        # is biased toward 1; when the feature is actually 0, biased
        # toward 0. A NaN-aware model can split on missing-vs-not and
        # learn the distinction; a 0-imputed model cannot.
        missing_mask = rng.random(n) < 0.3
        zero_mask = (~missing_mask) & (rng.random(n) < 0.2)
        feat[zero_mask] = 0.0
        label_probs = np.where(missing_mask, 0.9, np.where(zero_mask, 0.1, 0.5))
        y = (rng.random(n) < label_probs).astype(int)

        X_nan = feat.copy().reshape(-1, 1)
        X_nan[missing_mask, 0] = np.nan
        X_zero = feat.copy().reshape(-1, 1)
        X_zero[missing_mask, 0] = 0.0  # the buggy imputation

        params = dict(
            objective="binary", num_leaves=4,
            learning_rate=0.1, verbose=-1,
            min_data_in_leaf=5,
        )
        booster_nan = lgb.train(
            params, lgb.Dataset(X_nan, label=y), num_boost_round=30,
        )
        booster_zero = lgb.train(
            params, lgb.Dataset(X_zero, label=y), num_boost_round=30,
        )

        # Predict on a probe row that has a NaN feature vs a 0.0 feature.
        probe = np.array([[np.nan], [0.0]])
        # The 0-imputed model collapses both probes to the same prediction;
        # the NaN-aware one routes them differently → predictions differ
        # measurably between the two models.
        p_nan = booster_nan.predict(probe)
        p_zero = booster_zero.predict(probe)
        # Bug demo: under the zero-imputed model, the two probe rows are
        # indistinguishable. Under the NaN-aware model they are not.
        assert p_zero[0] == pytest.approx(p_zero[1])
        assert abs(p_nan[0] - p_nan[1]) > 1e-4


# ---------------------------------------------------------------------------
# 4. ml_strategy.py vector builders: NaN preserved through to predict()
# ---------------------------------------------------------------------------

class TestStrategyVectorPreservesNaN:
    """The four-bug surface in ml_strategy lived in the row→vector
    conversion: missing key → 0.0, None → 0.0, then nan_to_num at the
    end. These tests exercise the post-fix code by feeding crafted dicts
    through ``_safe_float`` exactly as the strategy does."""

    def test_vector_built_from_dict_keeps_nan_for_missing_keys(self) -> None:
        from src.execution.strategies.ml_strategy import _safe_float
        # Mirror line 1158/1318 idiom: ``[_safe_float(d.get(f)) for f in feature_names]``
        d = {"a": 1.0, "b": 0.0}
        feature_names = ["a", "b", "c"]  # "c" is missing
        vec = np.array([_safe_float(d.get(f)) for f in feature_names])
        assert vec[0] == 1.0
        assert vec[1] == 0.0           # real zero survives
        assert np.isnan(vec[2])        # missing → NaN (NOT 0.0)

    def test_vector_keeps_nan_for_none_values(self) -> None:
        from src.execution.strategies.ml_strategy import _safe_float
        d = {"a": None, "b": 2.5}
        vec = np.array([_safe_float(d.get(f)) for f in ["a", "b"]])
        assert np.isnan(vec[0])
        assert vec[1] == 2.5

    def test_vector_collapses_inf_to_nan_not_to_zero(self) -> None:
        from src.execution.strategies.ml_strategy import _safe_float
        d = {"a": float("inf"), "b": float("-inf"), "c": 7.0}
        vec = np.array([_safe_float(d.get(f)) for f in ["a", "b", "c"]])
        assert np.isnan(vec[0])
        assert np.isnan(vec[1])
        assert vec[2] == 7.0
        assert not np.any(np.isinf(vec))


# ---------------------------------------------------------------------------
# 5. No nan_to_num call anywhere in the touched files
# ---------------------------------------------------------------------------

class TestNoNanToNumCallsLeft:
    """Belt-and-braces: read the source files and assert the
    ``nan_to_num`` call is gone from the touched modules. A regression
    that re-introduces zero-imputation will trip this immediately."""

    def test_lgbm_trainer_has_no_nan_to_num(self) -> None:
        src = open("src/models/lgbm_trainer.py", encoding="utf-8").read()
        assert "nan_to_num" not in src

    def test_ml_strategy_has_no_nan_to_num(self) -> None:
        src = open(
            "src/execution/strategies/ml_strategy.py", encoding="utf-8",
        ).read()
        assert "nan_to_num" not in src

    def test_ml_strategy_15m_has_no_nan_to_num(self) -> None:
        src = open(
            "src/execution/strategies/ml_strategy_15m.py", encoding="utf-8",
        ).read()
        assert "nan_to_num" not in src

    def test_meta_strategy_has_no_nan_to_num(self) -> None:
        src = open(
            "src/execution/strategies/meta_strategy.py", encoding="utf-8",
        ).read()
        assert "nan_to_num" not in src


# ---------------------------------------------------------------------------
# 6. 15m strategy: NaN survives through _vector()
# ---------------------------------------------------------------------------

class TestStrategy15mPreservesNaN:
    """Phase 5.6 — mirror the 4H fix in the 15m strategy.

    ``_vector`` is a staticmethod that takes a 1-row polars DataFrame
    and a feature-name list. We feed it crafted rows containing the
    same boundary values that broke 4H pre-fix (None, missing column,
    ±inf) and assert each one becomes NaN at the boundary into the
    LightGBM predictor.
    """

    def test_missing_column_becomes_nan(self) -> None:
        from src.execution.strategies.ml_strategy_15m import (
            MLTradingStrategy15M,
        )
        row = pl.DataFrame({"a": [1.0], "b": [0.0]})
        vec = MLTradingStrategy15M._vector(row, ["a", "b", "c"])
        assert vec[0] == 1.0
        assert vec[1] == 0.0          # real zero survives
        assert np.isnan(vec[2])       # absent column → NaN, NOT 0.0

    def test_none_value_becomes_nan(self) -> None:
        from src.execution.strategies.ml_strategy_15m import (
            MLTradingStrategy15M,
        )
        row = pl.DataFrame({"a": [None], "b": [2.5]}, strict=False)
        vec = MLTradingStrategy15M._vector(row, ["a", "b"])
        assert np.isnan(vec[0])
        assert vec[1] == 2.5

    def test_inf_collapses_to_nan(self) -> None:
        from src.execution.strategies.ml_strategy_15m import (
            MLTradingStrategy15M,
        )
        row = pl.DataFrame({
            "a": [float("inf")],
            "b": [float("-inf")],
            "c": [7.0],
        })
        vec = MLTradingStrategy15M._vector(row, ["a", "b", "c"])
        assert np.isnan(vec[0])
        assert np.isnan(vec[1])
        assert vec[2] == 7.0
        assert not np.any(np.isinf(vec))

    def test_real_nan_passthrough(self) -> None:
        from src.execution.strategies.ml_strategy_15m import (
            MLTradingStrategy15M,
        )
        row = pl.DataFrame({"a": [float("nan")], "b": [3.0]})
        vec = MLTradingStrategy15M._vector(row, ["a", "b"])
        assert np.isnan(vec[0])
        assert vec[1] == 3.0


# ---------------------------------------------------------------------------
# 7. Meta gate: build_feature_vector preserves NaN
# ---------------------------------------------------------------------------

class _FakeBooster:
    """Picklable stand-in for the trained LightGBM booster."""

    def predict(self, X):
        return np.zeros((X.shape[0],), dtype=np.float64)


class TestMetaGatePreservesNaN:
    """Phase 5.6 — same fix on MetaSignalGate.build_feature_vector.

    Constructs a gate with a hand-built fake bundle so we don't depend
    on the real meta_model_v3.pkl bytes, only on the vector-assembly
    code path that previously called ``nan_to_num``.
    """

    @staticmethod
    def _gate(feature_columns, tmp_path):
        import pickle
        from src.execution.strategies.meta_strategy import MetaSignalGate
        bundle = {
            "booster": _FakeBooster(),
            "feature_columns": feature_columns,
        }
        path = tmp_path / "fake_meta.pkl"
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        return MetaSignalGate(path, threshold=0.6, min_size=0.0)

    def test_missing_key_becomes_nan(self, tmp_path):
        gate = self._gate(["a", "b", "c"], tmp_path)
        vec = gate.build_feature_vector({"a": 1.0, "b": 0.0})
        assert vec.shape == (1, 3)
        assert vec[0, 0] == 1.0
        assert vec[0, 1] == 0.0
        assert np.isnan(vec[0, 2])

    def test_inf_and_none_collapse_to_nan(self, tmp_path):
        gate = self._gate(["a", "b", "c", "d"], tmp_path)
        vec = gate.build_feature_vector({
            "a": None,
            "b": float("inf"),
            "c": float("-inf"),
            "d": 4.2,
        })
        assert np.isnan(vec[0, 0])
        assert np.isnan(vec[0, 1])
        assert np.isnan(vec[0, 2])
        assert vec[0, 3] == 4.2
        assert not np.any(np.isinf(vec))

    def test_real_zero_is_not_treated_as_missing(self, tmp_path):
        """The whole point: a genuine 0.0 (e.g. funding_rate at 0%)
        must reach the booster as 0.0, not be silently treated as NaN."""
        gate = self._gate(["funding_rate"], tmp_path)
        vec = gate.build_feature_vector({"funding_rate": 0.0})
        assert vec[0, 0] == 0.0
        assert not np.isnan(vec[0, 0])
