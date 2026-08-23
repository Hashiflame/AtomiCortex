"""Tests for PR-H — save_bundle refuses to write a failing model.

PR-G made every bundle carry its own verdict (``manifest["passes"]``) but
still wrote the file regardless. That is how a model with
``signal_rate ≈ 0`` — one that fails the trainer's own go-live gate —
ended up overwriting the production artifact.

PR-H turns the verdict into a gate:

* ``passes_minimum_thresholds()`` False  → ``ModelRejectedError``,
  nothing is written. ``allow_failing=True`` is the deliberate escape
  hatch (grid cells need it so DSR still counts every trial).
* empty ``feature_columns``              → ``ModelRejectedError`` always.
  A bundle nobody can build a feature vector for is a broken artifact,
  not a weak model, so the escape hatch does not apply.

Both checks run before anything touches the filesystem, so a rejected
save cannot truncate or replace the bundle already on disk.
"""
from __future__ import annotations

import ast
import pickle
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from loguru import logger as _loguru_logger

from src.models.lgbm_trainer import EvaluationResult, LGBMTrainer, ModelConfig
from src.models.training_pipeline import TrainingPipeline

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _gate_error() -> type[BaseException]:
    """The rejection class, imported lazily.

    Module-level import would turn a missing class into a collection
    error and hide every other test in this file behind it.
    """
    from src.models.lgbm_trainer import ModelRejectedError

    return ModelRejectedError


# ---------------------------------------------------------------------------
# Synthetic data (self-contained; mirrors the real feature schema closely
# enough for get_feature_columns / create_target)
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


def _save_features(base: Path, symbols: list[str] | None = None, n: int = 300) -> Path:
    symbols = symbols or ["BTCUSDT"]
    features_dir = base / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols):
        _make_feature_df(n=n, symbol=sym, seed=42 + i).write_parquet(
            features_dir / f"{sym}_4h_features.parquet"
        )
    return features_dir


def _make_trainer(base: Path, **config_kwargs) -> tuple[LGBMTrainer, Path]:
    features_dir = _save_features(base)
    models_dir = base / "models"
    config = ModelConfig(
        regime=config_kwargs.pop("regime", "all"),
        symbols=["BTCUSDT"],
        **config_kwargs,
    )
    trainer = LGBMTrainer(
        config=config, features_dir=features_dir, models_dir=models_dir,
    )
    return trainer, models_dir


def _trained(base: Path) -> SimpleNamespace:
    """A trainer that has run train() — feature_columns populated."""
    trainer, models_dir = _make_trainer(base)
    train_df, test_df = trainer.prepare_data()
    booster = trainer.train(train_df)
    return SimpleNamespace(
        trainer=trainer,
        models_dir=models_dir,
        booster=booster,
        train_df=train_df,
        test_df=test_df,
    )


def _tiny_booster() -> lgb.Booster:
    """Standalone booster — for trainers that never ran train()."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return lgb.train(
        {"objective": "binary", "num_leaves": 7, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=10,
    )


# ---------------------------------------------------------------------------
# Hand-built verdicts — the gate reads result, never the booster
# ---------------------------------------------------------------------------


def _passing_result(regime: str = "all") -> EvaluationResult:
    """WR 60 / PF 1.5 / sig 0.5 — clears 52.0 / 1.3 / 0.10."""
    return EvaluationResult(
        regime=regime,
        accuracy=58.0, precision=58.0, recall=58.0, f1=58.0,
        win_rate=60.0, profit_factor=1.5, signal_rate=0.5,
        avg_confidence=0.62, per_symbol={},
    )


def _failing_result(regime: str = "all") -> EvaluationResult:
    """WR 40 / PF 0.8 / sig 0.01 — fails all three thresholds."""
    return EvaluationResult(
        regime=regime,
        accuracy=48.0, precision=48.0, recall=48.0, f1=48.0,
        win_rate=40.0, profit_factor=0.8, signal_rate=0.01,
        avg_confidence=0.51, per_symbol={},
    )


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


# ===========================================================================
# 1. Rejection writes nothing at all
# ===========================================================================


class TestRejectionWritesNothing:

    @pytest.mark.parametrize("bad", ["failing", "none"])
    def test_reject_raises_and_writes_nothing(self, tmp_path: Path, bad: str):
        """Control: no file before the call, still no file after it.

        The ``none`` case covers a caller that never ran evaluate() —
        that must be a clean rejection, not an AttributeError from
        poking at None.
        """
        run = _trained(tmp_path)
        target = run.models_dir / "all_model.pkl"

        # ---- before: nothing on disk ----
        assert not target.exists()
        assert list(run.models_dir.glob("*")) == []

        result = _failing_result() if bad == "failing" else None
        with pytest.raises(_gate_error()):
            run.trainer.save_bundle(
                run.booster, result, run.train_df, run.test_df,
            )

        # ---- after: still nothing, and no orphan .lgb either ----
        assert not target.exists()
        assert list(run.models_dir.glob("*")) == []

    def test_existing_bundle_not_overwritten_on_reject(self, tmp_path: Path):
        """The original defect: a failing model replacing a good one."""
        run = _trained(tmp_path)
        good = run.trainer.save_bundle(
            run.booster, _passing_result(), run.train_df, run.test_df,
        )
        before = good.read_bytes()

        with pytest.raises(_gate_error()):
            run.trainer.save_bundle(
                run.booster, _failing_result(), run.train_df, run.test_df,
            )

        assert good.exists()
        assert good.read_bytes() == before


# ===========================================================================
# 2. Escape hatch
# ===========================================================================


class TestAllowFailing:

    def test_allow_failing_writes_failing_model(
        self, tmp_path: Path, loguru_warnings,
    ):
        run = _trained(tmp_path)

        path = run.trainer.save_bundle(
            run.booster, _failing_result(), run.train_df, run.test_df,
            allow_failing=True,
        )

        assert path.exists()
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]
        assert manifest["passes"] is False
        assert manifest["written_despite_failing"] is True
        # A deliberate override must be loud in the log.
        assert any("allow_failing" in m for m in loguru_warnings)

    def test_passing_model_writes_by_default(self, tmp_path: Path):
        """Positive control — the gate is not a blanket block."""
        run = _trained(tmp_path)

        path = run.trainer.save_bundle(
            run.booster, _passing_result(), run.train_df, run.test_df,
        )

        assert path.exists()
        with open(path, "rb") as f:
            manifest = pickle.load(f)["manifest"]
        assert manifest["passes"] is True
        assert manifest["written_despite_failing"] is False


# ===========================================================================
# 3. Broken artifact — no escape hatch
# ===========================================================================


class TestEmptyFeatureColumns:

    @pytest.mark.parametrize("allow_failing", [False, True])
    def test_empty_feature_columns_always_rejected(
        self, tmp_path: Path, allow_failing: bool,
    ):
        """Empty feature_columns is a broken artifact, not a weak model:
        allow_failing does not buy a way past it. Result is a PASSING
        one, so the only possible rejection reason is the feature list."""
        trainer, models_dir = _make_trainer(tmp_path)
        assert trainer._feature_columns == []

        with pytest.raises(_gate_error()) as excinfo:
            trainer.save_bundle(
                _tiny_booster(),
                _passing_result(),
                pl.DataFrame({"target": [1, -1]}),
                pl.DataFrame({"target": [1, -1]}),
                allow_failing=allow_failing,
            )

        assert excinfo.value.reason == "no_feature_columns"
        assert list(models_dir.glob("*")) == []


# ===========================================================================
# 4. The exception itself
# ===========================================================================


class TestModelRejectedError:

    def test_model_rejected_error_type_and_fields(self, tmp_path: Path):
        run = _trained(tmp_path)
        failing = _failing_result()

        with pytest.raises(_gate_error()) as excinfo:
            run.trainer.save_bundle(
                run.booster, failing, run.train_df, run.test_df,
            )

        exc = excinfo.value
        assert isinstance(exc, Exception)
        assert exc.reason == "thresholds"
        assert exc.path == run.models_dir / "all_model.pkl"
        assert exc.regime == "all"
        assert exc.result is failing

        msg = str(exc)
        assert "all_model.pkl" in msg
        # The operator must see what the model actually scored.
        assert str(failing.win_rate) in msg
        assert str(failing.profit_factor) in msg
        assert str(failing.signal_rate) in msg

    def test_model_rejected_distinct_from_io_errors(self, tmp_path: Path):
        """A bad model must not be mistaken for an unwritable disk."""
        err = _gate_error()
        assert not issubclass(err, OSError)

        run = _trained(tmp_path)
        caught = None
        try:
            run.trainer.save_bundle(
                run.booster, _failing_result(), run.train_df, run.test_df,
            )
        except OSError:  # pragma: no cover — must not be taken
            pytest.fail("ModelRejectedError was caught as an OSError")
        except err as exc:
            caught = exc

        assert caught is not None


# ===========================================================================
# 5. Production callers — which ones carry the escape hatch
# ===========================================================================


def _save_bundle_calls(rel_path: str) -> list[bool]:
    """For each save_bundle call in the module: does it pass
    allow_failing? Parsed via ast so reformatting cannot fool it."""
    tree = ast.parse((_REPO_ROOT / rel_path).read_text())
    out: list[bool] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "save_bundle":
            out.append(any(kw.arg == "allow_failing" for kw in node.keywords))
    return out


class TestProductionCallerGateState:
    """Six production paths must run gated; only the research grid cell
    carries the escape hatch."""

    @pytest.mark.parametrize("rel_path,with_hatch,without_hatch", [
        ("src/models/training_pipeline.py", 0, 1),
        # run_best_only + winner refit gated; grid cell exempt
        ("scripts/retrain_v3.py", 1, 2),
        ("scripts/train_1h_models.py", 0, 1),
        ("scripts/train_15m_models.py", 0, 1),
        ("scripts/tune_models.py", 0, 1),
    ])
    def test_prod_caller_gate_state(
        self, rel_path: str, with_hatch: int, without_hatch: int,
    ):
        calls = _save_bundle_calls(rel_path)

        assert len(calls) == with_hatch + without_hatch, (
            f"{rel_path}: expected {with_hatch + without_hatch} save_bundle "
            f"calls, found {len(calls)}"
        )
        assert sum(calls) == with_hatch, (
            f"{rel_path}: expected {with_hatch} call(s) with allow_failing, "
            f"found {sum(calls)}"
        )
        assert calls.count(False) == without_hatch, (
            f"{rel_path}: expected {without_hatch} gated call(s), "
            f"found {calls.count(False)}"
        )


# ===========================================================================
# 6. Pipeline behaviour on rejection
# ===========================================================================


class TestPipelineOnRejection:

    def test_pipeline_skips_rejected_regime(
        self, tmp_path: Path, loguru_warnings,
    ):
        """A rejected regime is skipped, not crashed on — and the log
        must not call it a training failure."""
        features_dir = _save_features(tmp_path)
        models_dir = tmp_path / "models"

        results = TrainingPipeline().run(
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            models_dir=models_dir,
            regimes=["all"],
        )

        assert results == {}
        assert list(models_dir.glob("*.pkl")) == []
        rejected = [m for m in loguru_warnings if "rejected" in m.lower()]
        assert len(rejected) == 1
        assert "Training failed" not in rejected[0]


# ===========================================================================
# 7. PR-Э1.4 — the two production names are reserved for triple_barrier
# ===========================================================================
#
# A2-036: ``train_models.py`` and ``tune_models.py`` wrote ``sign_return``
# models straight to ``trend_model.pkl`` / ``high_vol_model.pkl`` -- the two
# files the 4H loader reads.  The reservation is expressed here rather than
# in the scripts because it is the artifact that is illegal, not the caller:
# whoever computes that name, with that target, is refused.
#
# Keyed on the FINAL filename, never on ``config.model_suffix``: the 1H and
# 15m scripts legitimately leave the suffix empty and take their name from
# ``filename=``, so a config-keyed predicate would refuse two production
# paths that never approach a reserved name (test_filename_override_... below).


def _prod_reserved_names() -> list[str]:
    """The names the 4H loader opens, built through the one primitive."""
    from src.models.model_paths import PROD_STEMS_4H, SUFFIX_PROD, bundle_filename

    return [bundle_filename(stem, SUFFIX_PROD) for stem in PROD_STEMS_4H]


def _trained_config(
    base: Path, *, n: int = 300, **config_kwargs,
) -> SimpleNamespace:
    """A trainer that has run train(), over an arbitrary ModelConfig.

    ``_trained`` above is fixed to the default config; the reservation
    turns on ``use_triple_barrier`` and on the regime, so this section
    needs both to be free.
    """
    features_dir = _save_features(base, n=n)
    models_dir = base / "models"
    config = ModelConfig(symbols=["BTCUSDT"], **config_kwargs)
    trainer = LGBMTrainer(
        config=config, features_dir=features_dir, models_dir=models_dir,
    )
    train_df, test_df = trainer.prepare_data()
    booster = trainer.train(train_df)
    return SimpleNamespace(
        trainer=trainer,
        models_dir=models_dir,
        booster=booster,
        train_df=train_df,
        test_df=test_df,
    )


class TestProdNameReservation:

    @pytest.mark.parametrize("stem", ["trend", "high_vol"])
    def test_prod_name_on_sign_return_rejected(self, tmp_path: Path, stem: str):
        """The defect itself: a 1-bar sign(return) model under the name the
        live 4H strategy loads."""
        run = _trained_config(tmp_path, regime=stem)

        with pytest.raises(_gate_error()) as excinfo:
            run.trainer.save_bundle(
                run.booster, _passing_result(stem),
                run.train_df, run.test_df,
            )

        exc = excinfo.value
        assert exc.reason == "prod_name_wrong_target"
        assert exc.path == run.models_dir / f"{stem}_model.pkl"
        assert list(run.models_dir.glob("*")) == []

    def test_allow_failing_does_not_waive_prod_reservation(self, tmp_path: Path):
        """Like the empty-feature-columns gate, this is a statement about
        the artifact, not about its metrics -- the research escape hatch
        has nothing to say about it."""
        run = _trained_config(tmp_path, regime="trend")

        with pytest.raises(_gate_error()) as excinfo:
            run.trainer.save_bundle(
                run.booster, _passing_result("trend"),
                run.train_df, run.test_df,
                allow_failing=True,
            )

        assert excinfo.value.reason == "prod_name_wrong_target"
        assert list(run.models_dir.glob("*")) == []

    def test_prod_name_on_triple_barrier_written(self, tmp_path: Path):
        """Positive control: the reservation reserves, it does not forbid.

        Keyed on the name, so the name is what this test supplies -- a
        triple-barrier model is entitled to it.
        """
        run = _trained_config(tmp_path, n=600, regime="all", use_triple_barrier=True)

        path = run.trainer.save_bundle(
            run.booster, _passing_result(), run.train_df, run.test_df,
            filename=_prod_reserved_names()[0],
        )

        assert path.name == _prod_reserved_names()[0]
        assert path.exists()

    def test_suffixed_name_on_sign_return_written(self, tmp_path: Path):
        """What the scripts do after Д-1: same model, a name nothing loads."""
        run = _trained_config(tmp_path, regime="trend", model_suffix="_v4")

        path = run.trainer.save_bundle(
            run.booster, _passing_result("trend"), run.train_df, run.test_df,
        )

        assert path.name == "trend_model_v4.pkl"
        assert path.exists()
        assert not (run.models_dir / "trend_model.pkl").exists()

    def test_filename_override_is_not_reserved(self, tmp_path: Path):
        """Regression against the config-keyed predicate that was rejected.

        ``train_1h_models.py`` and ``train_15m_models.py`` build their
        ModelConfig with an empty ``model_suffix`` and a ``sign_return``
        target, then name the artifact through ``filename=``.  A gate
        reading ``config.model_suffix`` would refuse both of them for a
        name neither one ever produces.
        """
        from src.models.model_paths import SUFFIX_1H, bundle_filename

        run = _trained_config(tmp_path, regime="trend")

        path = run.trainer.save_bundle(
            run.booster, _passing_result("trend"), run.train_df, run.test_df,
            filename=bundle_filename("trend", SUFFIX_1H),
        )

        assert path.name == "trend_model_1h.pkl"
        assert path.exists()
