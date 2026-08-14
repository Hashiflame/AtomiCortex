"""Tests for PR-G — model bundle manifest.

The bundle used to carry four keys (booster / feature_columns / regime /
symbols), which made a model's provenance unrecoverable: barriers, the
eval metrics it was accepted on, the data window it saw and the feature
count were all lost the moment training finished.

PR-G moves the disk write out of ``train()`` into an explicit
``save_bundle(booster, result, train_df, test_df)`` — which is the first
point where the evaluation actually exists — and stamps a ``manifest``
key into the same pickle bundle.

Scope guards encoded here:
* ``train()`` must not touch the disk at all.
* ``passes`` is informational — a failing model is still written (the
  refuse-to-write gate is PR-H).
* ``load_model_bundle`` on a pre-PR-G bundle WARNs and keeps working —
  fail-closed is PR-I. VMs hold manifest-less bundles today.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from loguru import logger as _loguru_logger

from src.models.lgbm_trainer import (
    MTF_LGBM_PARAMS,
    EvaluationResult,
    LGBMTrainer,
    ModelConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers — synthetic feature frames (self-contained; mirrors the real
# schema closely enough for get_feature_columns / create_target).
# ---------------------------------------------------------------------------


def _make_feature_df(n: int = 300, symbol: str = "BTCUSDT", seed: int = 42) -> pl.DataFrame:
    rng = np.random.RandomState(seed)
    close = 40000 + np.cumsum(rng.randn(n) * 100)

    base_time = 1704067200000  # 2024-01-01 00:00 UTC in ms
    open_times = [base_time + i * 4 * 3600 * 1000 for i in range(n)]

    return pl.DataFrame({
        "open_time": open_times,
        "open": close + rng.randn(n) * 50,
        "high": close + rng.uniform(50, 200, n),
        "low": close - rng.uniform(50, 200, n),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
        "close_time": [t + 4 * 3600 * 1000 - 1 for t in open_times],
        "symbol": [symbol] * n,
        "regime": rng.choice(["trend_up", "trend_down", "range", "high_vol"], n).tolist(),
        # Numeric feature columns
        "cvd": rng.randn(n),
        "cvd_slope_3": rng.randn(n),
        "taker_buy_ratio": rng.uniform(0.4, 0.6, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n),
        "volume_zscore": rng.randn(n),
        "returns_1": rng.randn(n) * 0.01,
        "returns_3": rng.randn(n) * 0.02,
        "returns_6": rng.randn(n) * 0.03,
        "body_ratio": rng.uniform(0.1, 0.9, n),
        "funding_rate": rng.randn(n) * 0.0001,
        "funding_zscore_7d": rng.randn(n),
        "oi_delta_4h": rng.randn(n) * 0.01,
        "oi_zscore": rng.randn(n),
        "ls_ratio": rng.uniform(0.8, 1.2, n),
        "hurst": rng.uniform(0.3, 0.7, n),
        "adx": rng.uniform(10, 50, n),
        "atr_pct": rng.uniform(0.01, 0.05, n),
        "atr_percentile": rng.uniform(0, 1, n),
        "trend_strength": rng.uniform(0, 1, n),
        "regime_confidence": rng.uniform(0, 1, n),
    })


def _save_features(base: Path, symbols: list[str], n: int = 300) -> Path:
    features_dir = base / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols):
        _make_feature_df(n=n, symbol=sym, seed=42 + i).write_parquet(
            features_dir / f"{sym}_4h_features.parquet"
        )
    return features_dir


def _make_trainer(
    base: Path,
    *,
    symbols: list[str] | None = None,
    n: int = 300,
    models_subdir: str = "models",
    **config_kwargs,
) -> tuple[LGBMTrainer, Path]:
    """Build a trainer over freshly written synthetic features."""
    symbols = symbols or ["BTCUSDT"]
    features_dir = _save_features(base, symbols, n=n)
    models_dir = base / models_subdir
    trainer_kwargs = {
        k: config_kwargs.pop(k)
        for k in ("use_mtf_params", "min_child_samples")
        if k in config_kwargs
    }
    config = ModelConfig(
        regime=config_kwargs.pop("regime", "all"),
        symbols=symbols,
        **config_kwargs,
    )
    trainer = LGBMTrainer(
        config=config,
        features_dir=features_dir,
        models_dir=models_dir,
        **trainer_kwargs,
    )
    return trainer, models_dir


def _run(trainer: LGBMTrainer) -> SimpleNamespace:
    """prepare → train → evaluate, returning everything save_bundle needs."""
    train_df, test_df = trainer.prepare_data()
    booster = trainer.train(train_df)
    result = trainer.evaluate(booster, test_df)
    return SimpleNamespace(
        trainer=trainer,
        booster=booster,
        result=result,
        train_df=train_df,
        test_df=test_df,
    )


def _write_legacy_pkl(path: Path, booster: lgb.Booster) -> None:
    """Pre-PR-G bundle shape: four keys, no manifest."""
    with open(path, "wb") as f:
        pickle.dump(
            {
                "booster": booster,
                "feature_columns": ["f0", "f1"],
                "regime": "trend",
                "symbols": ["BTCUSDT"],
            },
            f,
        )


@pytest.fixture(scope="module")
def default_run(tmp_path_factory) -> SimpleNamespace:
    """One trained 4H model on default config, shared by most tests."""
    base = tmp_path_factory.mktemp("manifest_default")
    trainer, models_dir = _make_trainer(base)
    run = _run(trainer)
    run.models_dir = models_dir
    return run


@pytest.fixture(scope="module")
def default_manifest(default_run: SimpleNamespace) -> dict:
    path = default_run.trainer.save_bundle(
        default_run.booster,
        default_run.result,
        default_run.train_df,
        default_run.test_df,
        filename="manifest_probe.pkl",
        # PR-H: synthetic fixtures never clear the go-live thresholds,
        # and this file tests the manifest, not the gate.
        allow_failing=True,
    )
    with open(path, "rb") as f:
        return pickle.load(f)["manifest"]


@pytest.fixture
def loguru_warnings():
    """Capture loguru WARNING records into a list."""
    sink: list[str] = []
    sink_id = _loguru_logger.add(
        lambda msg: sink.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield sink
    finally:
        _loguru_logger.remove(sink_id)


_EXPECTED_MANIFEST_KEYS = {
    "schema_version",
    "created_at",
    "regime",
    "interval",
    "symbols",
    "symbols_in_train",
    "n_features",
    "feature_columns_hash",
    "barriers",
    "target_kind",
    "forward_bars",
    "confidence_threshold",
    "eval",
    "passes",
    "written_despite_failing",
    "data_range",
    "n_train_rows",
    "n_test_rows",
    "embargo_rows",
    "train_rows_after_embargo",
    "class_balance",
    "lgbm_params",
    "use_mtf_params",
    "use_uniqueness_weights",
    "feature_whitelist_size",
    "best_iteration",
    "num_trees",
    "git_commit",
    "lightgbm_version",
    "python_version",
}


# ===========================================================================
# 1. train() no longer writes to disk
# ===========================================================================


class TestTrainWritesNothing:
    """train() returns a Booster and touches no file."""

    def test_train_writes_no_pkl(self, tmp_path: Path):
        trainer, models_dir = _make_trainer(tmp_path)
        train_df, _ = trainer.prepare_data()
        booster = trainer.train(train_df)

        assert isinstance(booster, lgb.Booster)
        assert list(models_dir.glob("*.pkl")) == []

    def test_train_writes_no_lgb_sidecar(self, tmp_path: Path):
        trainer, models_dir = _make_trainer(tmp_path, use_native_save=True)
        train_df, _ = trainer.prepare_data()
        trainer.train(train_df)

        assert list(models_dir.glob("*.pkl")) == []
        assert list(models_dir.glob("*.lgb")) == []


# ===========================================================================
# 2. save_bundle writes
# ===========================================================================


class TestSaveBundleWrites:
    """save_bundle is now the only writer."""

    def test_save_bundle_creates_file_and_returns_path(self, tmp_path: Path):
        trainer, models_dir = _make_trainer(tmp_path)
        run = _run(trainer)

        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )

        assert path == models_dir / "all_model.pkl"
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_bundle_filename_override(self, tmp_path: Path):
        """1H/15m scripts name the artifact directly instead of renaming
        a file that train() happened to leave behind."""
        trainer, models_dir = _make_trainer(tmp_path, regime="trend")
        run = _run(trainer)

        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            filename="trend_model_1h.pkl",
            allow_failing=True,
        )

        assert path == models_dir / "trend_model_1h.pkl"
        assert path.exists()
        assert not (models_dir / "trend_model.pkl").exists()


# ===========================================================================
# 3. Control assert — the OLD bundle shape carries no manifest
# ===========================================================================


class TestLegacyBundleHasNoManifest:
    """Proves 'manifest is non-empty' is a real assertion, not a tautology."""

    def test_legacy_bundle_has_no_manifest(self, tmp_path: Path):
        trainer, _ = _make_trainer(tmp_path)
        run = _run(trainer)

        # ---- pre-PR-G shape: manifest absent ----
        legacy_path = tmp_path / "legacy.pkl"
        _write_legacy_pkl(legacy_path, run.booster)
        with open(legacy_path, "rb") as f:
            legacy_raw = pickle.load(f)
        assert "manifest" not in legacy_raw
        assert set(legacy_raw) == {
            "booster", "feature_columns", "regime", "symbols",
        }

        # ---- PR-G shape: manifest present and non-empty ----
        new_path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(new_path, "rb") as f:
            new_raw = pickle.load(f)
        assert "manifest" in new_raw
        assert isinstance(new_raw["manifest"], dict)
        assert new_raw["manifest"]
        # The four legacy keys survive untouched.
        assert set(legacy_raw).issubset(set(new_raw))


# ===========================================================================
# 4. Manifest contents
# ===========================================================================


class TestManifestContents:
    """Field-by-field contract of the manifest."""

    def test_manifest_key_set(self, default_manifest: dict):
        assert set(default_manifest) == _EXPECTED_MANIFEST_KEYS

    def test_manifest_schema_version_and_created_at(self, default_manifest: dict):
        # Imported here (not at module scope) so a missing constant fails
        # this one test instead of aborting collection of the whole file.
        from src.models.lgbm_trainer import MANIFEST_SCHEMA_VERSION

        assert default_manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert isinstance(default_manifest["schema_version"], int)
        assert default_manifest["schema_version"] == 1

        created = datetime.fromisoformat(default_manifest["created_at"])
        assert created.tzinfo is not None
        assert created.utcoffset() == timezone.utc.utcoffset(None)

    def test_manifest_barriers_match_config(self, tmp_path: Path):
        trainer, _ = _make_trainer(
            tmp_path,
            n=600,
            use_triple_barrier=True,
            barrier_pt_multiplier=1.5,
            barrier_sl_multiplier=0.75,
            barrier_max_holding=8,
        )
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        assert manifest["barriers"] == {
            "pt": 1.5,
            "sl": 0.75,
            "max_holding": 8,
            "use_triple_barrier": True,
        }

    def test_manifest_confidence_threshold_is_config_value(self, tmp_path: Path):
        """The threshold eval actually ran at — not the production 0.65."""
        trainer, _ = _make_trainer(tmp_path, confidence_threshold=0.35)
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        assert manifest["confidence_threshold"] == 0.35

    def test_manifest_eval_matches_result(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        r = default_run.result
        assert default_manifest["eval"] == {
            "accuracy": r.accuracy,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "signal_rate": r.signal_rate,
            "avg_confidence": r.avg_confidence,
        }
        assert default_manifest["passes"] == r.passes_minimum_thresholds()
        assert isinstance(default_manifest["passes"], bool)

    def test_manifest_n_features_matches_feature_columns(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        n = default_manifest["n_features"]
        assert n == len(default_run.trainer._feature_columns)
        assert n > 0

    def test_manifest_interval_default_and_override(self, tmp_path: Path):
        assert ModelConfig(regime="all", symbols=["BTCUSDT"]).interval == "4h"

        trainer, _ = _make_trainer(tmp_path, interval="1h")
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        assert manifest["interval"] == "1h"


# ===========================================================================
# 5. data_range comes from open_time
# ===========================================================================


class TestDataRange:
    """open_time is the time key DataStore.get_klines sorts and filters on."""

    def test_data_range_from_open_time_ms(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        dr = default_manifest["data_range"]
        train_ot = default_run.train_df["open_time"]
        test_ot = default_run.test_df["open_time"]

        assert dr["train_start_ms"] == int(train_ot.min())
        assert dr["train_end_ms"] == int(train_ot.max())
        assert dr["test_start_ms"] == int(test_ot.min())
        assert dr["test_end_ms"] == int(test_ot.max())
        # Walk-forward: test starts after train ends.
        assert dr["train_end_ms"] < dr["test_start_ms"]

    def test_data_range_iso_matches_ms(self, default_manifest: dict):
        dr = default_manifest["data_range"]
        for field in ("train_start", "train_end", "test_start", "test_end"):
            ms = dr[f"{field}_ms"]
            iso = dr[f"{field}_iso"]
            expected = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            assert datetime.fromisoformat(iso) == expected
            assert datetime.fromisoformat(iso).tzinfo is not None

    def test_data_range_none_without_open_time(
        self, tmp_path: Path, default_run: SimpleNamespace, loguru_warnings,
    ):
        """1H/15m datasets are fed in directly — never crash on a frame
        that lacks the time key; write None and warn."""
        trainer, models_dir = _make_trainer(tmp_path)
        run = _run(trainer)

        path = trainer.save_bundle(
            run.booster,
            run.result,
            run.train_df.drop("open_time"),
            run.test_df.drop("open_time"),
            filename="no_time.pkl",
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        assert path.exists()
        assert all(v is None for v in manifest["data_range"].values())
        assert len(manifest["data_range"]) == 8
        assert any("open_time" in m for m in loguru_warnings)


# ===========================================================================
# 6. Row counts, params, flags
# ===========================================================================


class TestRowsParamsFlags:

    def test_n_rows_match_frames(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        assert default_manifest["n_train_rows"] == len(default_run.train_df)
        assert default_manifest["n_test_rows"] == len(default_run.test_df)

    def test_lgbm_params_reflect_mtf_substitution(self, tmp_path: Path):
        """The manifest records what lgb.train actually got, after the
        MTF profile swap and the min_child_samples override."""
        trainer, _ = _make_trainer(
            tmp_path, use_mtf_params=True, min_child_samples=20,
        )
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        params = manifest["lgbm_params"]
        assert manifest["use_mtf_params"] is True
        assert params["min_child_samples"] == 20
        assert params["num_leaves"] == MTF_LGBM_PARAMS["num_leaves"]
        assert params["learning_rate"] == MTF_LGBM_PARAMS["learning_rate"]
        # Not lgb.train kwargs — filtered out before the call.
        assert "n_estimators" not in params
        assert "early_stopping_rounds" not in params

    def test_flags_and_whitelist_size(self, tmp_path: Path):
        whitelist = ["adx", "hurst", "atr_pct", "cvd", "returns_1"]
        trainer, _ = _make_trainer(
            tmp_path,
            n=600,
            use_triple_barrier=True,
            use_uniqueness_weights=True,
            feature_whitelist=whitelist,
        )
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        assert manifest["use_uniqueness_weights"] is True
        assert manifest["use_mtf_params"] is False
        assert manifest["feature_whitelist_size"] == len(whitelist)

    def test_whitelist_size_none_when_unset(self, default_manifest: dict):
        assert default_manifest["feature_whitelist_size"] is None

    def test_passes_is_informational(self, tmp_path: Path):
        """PR-G writes the bundle even when the model fails go-live."""
        trainer, models_dir = _make_trainer(tmp_path)
        run = _run(trainer)

        failing = EvaluationResult(
            regime="all",
            accuracy=48.0, precision=48.0, recall=48.0, f1=48.0,
            win_rate=40.0, profit_factor=0.8, signal_rate=0.01,
            avg_confidence=0.51, per_symbol={},
        )
        assert failing.passes_minimum_thresholds() is False

        path = trainer.save_bundle(
            run.booster, failing, run.train_df, run.test_df,
            filename="failing.pkl",
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]

        assert path.exists()
        assert manifest["passes"] is False


# ===========================================================================
# 7. load_model_bundle on pre-PR-G bundles: warn, never raise
# ===========================================================================


class TestLoadBundleWithoutManifest:

    def test_load_bundle_without_manifest_warns(
        self, tmp_path: Path, default_run: SimpleNamespace, loguru_warnings,
    ):
        path = tmp_path / "legacy_warn.pkl"
        _write_legacy_pkl(path, default_run.booster)

        LGBMTrainer.load_model_bundle(path)

        matched = [m for m in loguru_warnings if "provenance unknown" in m]
        assert len(matched) == 1
        assert "corrupt" not in matched[0].lower()

    def test_load_bundle_without_manifest_still_usable(
        self, tmp_path: Path, default_run: SimpleNamespace,
    ):
        path = tmp_path / "legacy_usable.pkl"
        _write_legacy_pkl(path, default_run.booster)

        bundle = LGBMTrainer.load_model_bundle(path)

        assert isinstance(bundle["booster"], lgb.Booster)
        assert bundle["feature_columns"] == ["f0", "f1"]
        assert "manifest" not in bundle
        X, _, _ = default_run.trainer._prepare_xy(default_run.test_df)
        np.testing.assert_allclose(
            bundle["booster"].predict(X), default_run.booster.predict(X),
        )

    def test_load_bundle_warning_deduplicated_per_path(
        self, tmp_path: Path, default_run: SimpleNamespace, loguru_warnings,
    ):
        """A strategy that reloads a bundle must not spam the log."""
        path = tmp_path / "legacy_dedup.pkl"
        _write_legacy_pkl(path, default_run.booster)

        LGBMTrainer.load_model_bundle(path)
        LGBMTrainer.load_model_bundle(path)
        LGBMTrainer.load_model_bundle(path)

        matched = [m for m in loguru_warnings if "provenance unknown" in m]
        assert len(matched) == 1

    def test_load_bundle_with_manifest_does_not_warn(
        self, tmp_path: Path, loguru_warnings,
    ):
        trainer, _ = _make_trainer(tmp_path)
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )

        bundle = LGBMTrainer.load_model_bundle(path)

        assert isinstance(bundle["manifest"], dict)
        assert [m for m in loguru_warnings if "provenance unknown" in m] == []


# ===========================================================================
# 8. Provenance fields (embargo, booster stats, hashes, versions)
# ===========================================================================


class TestManifestProvenance:

    def test_manifest_embargo_fields(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        """n_train_rows is the frame; train_rows_after_embargo is what
        LightGBM actually fitted on."""
        n_train = len(default_run.train_df)
        val_split = int(n_train * 0.90)

        assert default_manifest["embargo_rows"] == 1  # forward_bars=1
        assert default_manifest["train_rows_after_embargo"] == val_split - 1
        assert default_manifest["train_rows_after_embargo"] < n_train

    def test_manifest_booster_iteration_fields(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        assert default_manifest["num_trees"] == default_run.booster.num_trees()
        assert default_manifest["num_trees"] >= 1
        assert isinstance(default_manifest["best_iteration"], int)
        assert default_manifest["best_iteration"] >= 0

    def test_manifest_feature_columns_hash(
        self, default_manifest: dict, default_run: SimpleNamespace,
    ):
        expected = hashlib.sha256(
            json.dumps(sorted(default_run.trainer._feature_columns)).encode()
        ).hexdigest()
        assert default_manifest["feature_columns_hash"] == expected
        assert len(default_manifest["feature_columns_hash"]) == 64

    def test_manifest_target_kind_and_forward_bars(self, tmp_path: Path):
        # Legacy sign(return) target
        trainer, _ = _make_trainer(tmp_path / "legacy", forward_bars=3)
        run = _run(trainer)
        path = trainer.save_bundle(
            run.booster, run.result, run.train_df, run.test_df,
            allow_failing=True,
        )
        with open(path, "rb") as f:
            legacy_manifest = pickle.load(f)["manifest"]

        assert legacy_manifest["target_kind"] == "sign_return"
        assert legacy_manifest["forward_bars"] == 3

        # Triple-barrier target
        tb_trainer, _ = _make_trainer(
            tmp_path / "tb", n=600, use_triple_barrier=True,
        )
        tb_run = _run(tb_trainer)
        tb_path = tb_trainer.save_bundle(
            tb_run.booster, tb_run.result, tb_run.train_df, tb_run.test_df,
            allow_failing=True,
        )
        with open(tb_path, "rb") as f:
            tb_manifest = pickle.load(f)["manifest"]

        assert tb_manifest["target_kind"] == "triple_barrier"

    def test_manifest_class_balance_and_symbols_in_train(
        self, default_manifest: dict, default_run: SimpleNamespace, tmp_path: Path,
    ):
        train_df = default_run.train_df
        expected = float((train_df["target"] == 1).sum()) / len(train_df)
        assert default_manifest["class_balance"] == pytest.approx(expected)
        assert 0.0 <= default_manifest["class_balance"] <= 1.0

        # symbols_in_train reflects the data, not config.symbols — a
        # symbol whose parquet is missing is skipped silently upstream.
        assert default_manifest["symbols_in_train"] == sorted(
            train_df["symbol"].unique().to_list()
        )

        # No target column → class_balance is None, no crash.
        path = default_run.trainer.save_bundle(
            default_run.booster,
            default_run.result,
            train_df.drop("target"),
            default_run.test_df,
            filename="no_target.pkl",
            allow_failing=True,
        )
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]
        assert manifest["class_balance"] is None

    def test_manifest_provenance_versions(self, default_manifest: dict):
        assert default_manifest["lightgbm_version"] == lgb.__version__
        assert default_manifest["python_version"] == sys.version.split()[0]

        commit = default_manifest["git_commit"]
        assert commit is None or (isinstance(commit, str) and len(commit) >= 7)
        if commit is not None:
            actual = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            assert commit == actual


# ===========================================================================
# 9. Stale-state guard — params must not leak between runs
# ===========================================================================


class TestEffectiveParamsState:

    def test_save_bundle_without_train_has_empty_params(
        self, tmp_path: Path, default_run: SimpleNamespace, loguru_warnings,
    ):
        """Two halves of one stale-state guard.

        A trainer that never ran train() has no feature list, and PR-H
        refuses that bundle outright — a broken artifact is not covered
        by the escape hatch. So the manifest half of the guard (no stale
        params bleeding in from a previous run) has to be exercised on a
        trainer that DID train, with its captured state wiped by hand.
        """
        from src.models.lgbm_trainer import ModelRejectedError

        # --- half 1: no feature columns → rejected, nothing written ---
        fresh, models_dir = _make_trainer(tmp_path)
        assert fresh._feature_columns == []

        with pytest.raises(ModelRejectedError) as excinfo:
            fresh.save_bundle(
                default_run.booster,
                default_run.result,
                default_run.train_df,
                default_run.test_df,
                allow_failing=True,   # does not waive a broken artifact
            )
        assert excinfo.value.reason == "no_feature_columns"
        assert list(models_dir.glob("*")) == []

        # --- half 2: trained trainer, captured training state cleared ---
        trainer = default_run.trainer
        keep = (
            dict(trainer._effective_lgbm_params),
            trainer._embargo_rows,
            trainer._train_rows_after_embargo,
        )
        trainer._effective_lgbm_params = {}
        trainer._embargo_rows = None
        trainer._train_rows_after_embargo = None
        try:
            path = trainer.save_bundle(
                default_run.booster,
                default_run.result,
                default_run.train_df,
                default_run.test_df,
                filename="stale_state.pkl",
                allow_failing=True,
            )
        finally:
            # Module-scoped fixture — leave it as we found it.
            (
                trainer._effective_lgbm_params,
                trainer._embargo_rows,
                trainer._train_rows_after_embargo,
            ) = keep

        with open(path, "rb") as f:
            raw = pickle.load(f)
        manifest = raw["manifest"]

        assert manifest["lgbm_params"] == {}
        assert manifest["embargo_rows"] is None
        assert manifest["train_rows_after_embargo"] is None
        assert any("save_bundle" in m for m in loguru_warnings)
        # The feature list is real here — only the training state was wiped.
        assert raw["feature_columns"] == trainer._feature_columns
        assert manifest["n_features"] == len(trainer._feature_columns)


# ===========================================================================
# 10. Caller lint — the single automated trap for "forgot save_bundle"
# ===========================================================================


class TestProductionCallersSave:
    """train() no longer writes, so every production training path must
    call save_bundle explicitly. Forgetting one loses the model silently:
    the script still exits 0 and still prints metrics."""

    @pytest.mark.parametrize("rel_path", [
        "scripts/retrain_v3.py",
        "scripts/train_1h_models.py",
        "scripts/train_15m_models.py",
        "scripts/tune_models.py",
        "src/models/training_pipeline.py",
    ])
    def test_prod_caller_invokes_save_bundle(self, rel_path: str):
        source = (_REPO_ROOT / rel_path).read_text()
        assert "save_bundle" in source, (
            f"{rel_path} trains a model but never calls save_bundle — "
            "the artifact would never reach disk"
        )
