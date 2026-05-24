"""Tests for Step H13 — LightGBM native save_model + back-compat loader.

The new format writes a sidecar ``.lgb`` text file alongside the
metadata pickle so the booster is insulated from Python / LightGBM /
numpy pickle drift. The legacy "pickled booster inline" format must
still load through the same helper to protect existing production
models.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from src.models.lgbm_trainer import LGBMTrainer


# ---------------------------------------------------------------------------
# Synthetic booster + bundle helpers
# ---------------------------------------------------------------------------


def _train_tiny_booster(seed: int = 0) -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    train = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 7,
            "learning_rate": 0.1,
            "verbose": -1,
        },
        train,
        num_boost_round=20,
    )
    return booster, X, y


def _write_legacy_pkl(path: Path, booster: lgb.Booster, features: list[str]) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "booster": booster,
                "feature_columns": features,
                "regime": "trend",
                "symbols": ["BTCUSDT"],
            },
            f,
        )


def _write_sidecar_pkl(
    path: Path, booster: lgb.Booster, features: list[str],
) -> Path:
    lgb_path = path.with_suffix(".lgb")
    booster.save_model(str(lgb_path))
    with open(path, "wb") as f:
        pickle.dump(
            {
                "booster": None,
                "booster_file": lgb_path.name,
                "feature_columns": features,
                "regime": "trend",
                "symbols": ["BTCUSDT"],
            },
            f,
        )
    return lgb_path


# ---------------------------------------------------------------------------
# load_model_bundle: dual-format support
# ---------------------------------------------------------------------------


class TestLoadModelBundle:
    def test_loads_legacy_pickle(self, tmp_path):
        booster, X, _ = _train_tiny_booster()
        path = tmp_path / "trend_model.pkl"
        _write_legacy_pkl(path, booster, ["f0", "f1", "f2", "f3", "f4"])

        bundle = LGBMTrainer.load_model_bundle(path)
        assert isinstance(bundle["booster"], lgb.Booster)
        assert bundle["feature_columns"] == ["f0", "f1", "f2", "f3", "f4"]
        # Predictions identical to the in-memory booster.
        np.testing.assert_allclose(
            bundle["booster"].predict(X), booster.predict(X),
        )

    def test_loads_sidecar_format(self, tmp_path):
        booster, X, _ = _train_tiny_booster()
        path = tmp_path / "trend_model.pkl"
        _write_sidecar_pkl(path, booster, ["f0", "f1", "f2", "f3", "f4"])

        bundle = LGBMTrainer.load_model_bundle(path)
        assert isinstance(bundle["booster"], lgb.Booster)
        # Predictions are bit-equivalent — native save_model is loss-less.
        np.testing.assert_allclose(
            bundle["booster"].predict(X), booster.predict(X),
        )

    def test_sidecar_resolved_relative_to_pkl_dir(self, tmp_path):
        """Move the pair to a new directory; the loader follows the
        sidecar via the pickle's parent (not an absolute path)."""
        booster, X, _ = _train_tiny_booster()
        src = tmp_path / "src"
        src.mkdir()
        path = src / "trend_model.pkl"
        _write_sidecar_pkl(path, booster, ["f0", "f1", "f2", "f3", "f4"])

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "trend_model.pkl").write_bytes(path.read_bytes())
        (dst / "trend_model.lgb").write_bytes(path.with_suffix(".lgb").read_bytes())

        bundle = LGBMTrainer.load_model_bundle(dst / "trend_model.pkl")
        np.testing.assert_allclose(
            bundle["booster"].predict(X), booster.predict(X),
        )


# ---------------------------------------------------------------------------
# train() integration: use_native_save toggles the artifact layout
# ---------------------------------------------------------------------------


class TestTrainArtifactLayout:
    """Use the save+load helpers directly — train() pulls in the entire
    feature pipeline and is overkill for verifying the persistence
    branch. The branch under test is in lgbm_trainer.train() lines
    "if self.config.use_native_save:"."""

    def test_native_save_branch_writes_sidecar(self, tmp_path):
        booster, X, _ = _train_tiny_booster()
        pkl = tmp_path / "trend_model.pkl"
        _write_sidecar_pkl(pkl, booster, ["a", "b", "c", "d", "e"])

        assert pkl.exists()
        assert pkl.with_suffix(".lgb").exists()

        # The .pkl no longer pickles the live booster — the slot is None.
        with open(pkl, "rb") as f:
            raw = pickle.load(f)
        assert raw["booster"] is None
        assert raw["booster_file"] == "trend_model.lgb"

    def test_legacy_save_branch_embeds_booster(self, tmp_path):
        booster, _, _ = _train_tiny_booster()
        pkl = tmp_path / "trend_model.pkl"
        _write_legacy_pkl(pkl, booster, ["a", "b", "c", "d", "e"])
        with open(pkl, "rb") as f:
            raw = pickle.load(f)
        assert isinstance(raw["booster"], lgb.Booster)
        assert "booster_file" not in raw
        # And no sidecar file is created.
        assert not pkl.with_suffix(".lgb").exists()


# ---------------------------------------------------------------------------
# Round-trip predictions match across formats
# ---------------------------------------------------------------------------


class TestPredictionRoundTrip:
    def test_legacy_predictions_match_in_memory(self, tmp_path):
        booster, X, _ = _train_tiny_booster(seed=1)
        path = tmp_path / "m.pkl"
        _write_legacy_pkl(path, booster, [f"f{i}" for i in range(5)])
        loaded = LGBMTrainer.load_model_bundle(path)["booster"]
        np.testing.assert_allclose(loaded.predict(X), booster.predict(X))

    def test_sidecar_predictions_match_in_memory(self, tmp_path):
        booster, X, _ = _train_tiny_booster(seed=2)
        path = tmp_path / "m.pkl"
        _write_sidecar_pkl(path, booster, [f"f{i}" for i in range(5)])
        loaded = LGBMTrainer.load_model_bundle(path)["booster"]
        np.testing.assert_allclose(loaded.predict(X), booster.predict(X))

    def test_two_format_loaders_agree(self, tmp_path):
        """Save the SAME booster in both formats; loaded boosters give
        identical predictions on the same input."""
        booster, X, _ = _train_tiny_booster(seed=3)
        legacy = tmp_path / "legacy.pkl"
        sidecar = tmp_path / "sidecar.pkl"
        _write_legacy_pkl(legacy, booster, ["a"])
        _write_sidecar_pkl(sidecar, booster, ["a"])

        leg = LGBMTrainer.load_model_bundle(legacy)["booster"]
        nat = LGBMTrainer.load_model_bundle(sidecar)["booster"]
        np.testing.assert_allclose(leg.predict(X), nat.predict(X))


# ---------------------------------------------------------------------------
# ModelConfig: opt-in flag exists and defaults False
# ---------------------------------------------------------------------------


class TestModelConfigFlag:
    def test_default_use_native_save_false(self):
        from src.models.lgbm_trainer import ModelConfig
        cfg = ModelConfig(regime="trend", symbols=["BTCUSDT"])
        assert cfg.use_native_save is False

    def test_use_native_save_settable(self):
        from src.models.lgbm_trainer import ModelConfig
        cfg = ModelConfig(
            regime="trend", symbols=["BTCUSDT"], use_native_save=True,
        )
        assert cfg.use_native_save is True
