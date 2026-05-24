"""Walk-forward must validate the SAME target the model is trained on.

Pre-fix, ``MLValidator._load_full_data`` always called the legacy
1-bar sign(return) ``create_target`` — so a v3 model trained with the
triple-barrier label was being scored on a label it had never seen.
The go-live gate ("≥60 % profitable walk-forward windows") was therefore
checking the wrong model.

The fix replicates exactly the same branch ``LGBMTrainer.train`` uses,
so both code paths produce identical targets for the same config.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from src.models.lgbm_trainer import LGBMTrainer, ModelConfig
from src.models.ml_validator import MLValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trainer(
    use_triple_barrier: bool, tmp_path: Path, **overrides,
) -> LGBMTrainer:
    cfg = ModelConfig(
        regime="all",
        symbols=["BTCUSDT"],
        use_triple_barrier=use_triple_barrier,
        forward_bars=overrides.pop("forward_bars", 1),
        threshold_atr_multiplier=overrides.pop("threshold_atr_multiplier", 0.5),
        barrier_pt_multiplier=overrides.pop("barrier_pt_multiplier", 1.0),
        barrier_sl_multiplier=overrides.pop("barrier_sl_multiplier", 1.0),
        barrier_max_holding=overrides.pop("barrier_max_holding", 6),
        **overrides,
    )
    return LGBMTrainer(cfg, features_dir=tmp_path, models_dir=tmp_path)


def _stub_builder(trainer: LGBMTrainer) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Replace trainer._builder with mocks for load/legacy/triple-barrier targets."""
    df = pl.DataFrame({
        "open_time": [1, 2, 3, 4, 5],
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        "atr_pct": [0.01] * 5,
        "regime": ["all"] * 5,
    })
    load_mock = MagicMock(return_value=df)
    legacy_mock = MagicMock(return_value=df.with_columns(pl.lit(1).alias("target")))
    triple_mock = MagicMock(return_value=df.with_columns(pl.lit(1).alias("target")))
    trainer._builder = MagicMock()
    trainer._builder.load_and_combine = load_mock
    trainer._builder.create_target = legacy_mock
    trainer._builder.create_target_triple_barrier = triple_mock
    # _filter_by_regime is a passthrough for regime="all", but stub it
    # anyway so a refactor can't accidentally bring it back.
    trainer._filter_by_regime = MagicMock(side_effect=lambda d, r: d)
    return load_mock, legacy_mock, triple_mock


@pytest.fixture
def validator() -> MLValidator:
    return MLValidator(n_splits=3)


# ---------------------------------------------------------------------------
# Branching: which target builder is called
# ---------------------------------------------------------------------------

class TestTargetBranchSelection:
    def test_triple_barrier_true_calls_triple_barrier_only(
        self, validator: MLValidator, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(use_triple_barrier=True, tmp_path=tmp_path)
        _, legacy_mock, triple_mock = _stub_builder(trainer)
        validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)
        triple_mock.assert_called_once()
        legacy_mock.assert_not_called()

    def test_triple_barrier_false_calls_legacy_only(
        self, validator: MLValidator, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(use_triple_barrier=False, tmp_path=tmp_path)
        _, legacy_mock, triple_mock = _stub_builder(trainer)
        validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)
        legacy_mock.assert_called_once()
        triple_mock.assert_not_called()

    def test_default_config_uses_legacy(
        self, validator: MLValidator, tmp_path: Path
    ) -> None:
        """ModelConfig.use_triple_barrier defaults to False — production
        legacy models must keep getting the legacy target."""
        trainer = LGBMTrainer(
            ModelConfig(regime="all", symbols=["BTCUSDT"]),
            features_dir=tmp_path, models_dir=tmp_path,
        )
        _, legacy_mock, triple_mock = _stub_builder(trainer)
        validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)
        legacy_mock.assert_called_once()
        triple_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Parameter pass-through
# ---------------------------------------------------------------------------

class TestParameterPassthrough:
    def test_triple_barrier_params_forwarded(
        self, validator: MLValidator, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(
            use_triple_barrier=True,
            tmp_path=tmp_path,
            barrier_pt_multiplier=2.5,
            barrier_sl_multiplier=1.7,
            barrier_max_holding=9,
        )
        _, _, triple_mock = _stub_builder(trainer)
        validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)
        kwargs = triple_mock.call_args.kwargs
        assert kwargs["pt_multiplier"] == 2.5
        assert kwargs["sl_multiplier"] == 1.7
        assert kwargs["max_holding"] == 9

    def test_legacy_params_forwarded(
        self, validator: MLValidator, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(
            use_triple_barrier=False,
            tmp_path=tmp_path,
            forward_bars=4,
            threshold_atr_multiplier=0.75,
        )
        _, legacy_mock, _ = _stub_builder(trainer)
        validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)
        kwargs = legacy_mock.call_args.kwargs
        assert kwargs["forward_bars"] == 4
        assert kwargs["threshold_atr_multiplier"] == 0.75


# ---------------------------------------------------------------------------
# Consistency: validator and trainer agree on which builder to call
# ---------------------------------------------------------------------------

class TestTrainerValidatorParity:
    """The whole point of the fix — train() and _load_full_data() must
    pick the same target builder with the same kwargs for any config."""

    @pytest.mark.parametrize("use_triple_barrier", [True, False])
    def test_branch_matches_trainer_train(
        self, use_triple_barrier: bool, validator: MLValidator, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(
            use_triple_barrier=use_triple_barrier,
            tmp_path=tmp_path,
            barrier_pt_multiplier=1.3,
            barrier_sl_multiplier=0.8,
            barrier_max_holding=7,
            forward_bars=3,
            threshold_atr_multiplier=0.4,
        )
        _, legacy_mock, triple_mock = _stub_builder(trainer)
        validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)

        if use_triple_barrier:
            triple_mock.assert_called_once()
            legacy_mock.assert_not_called()
            kw = triple_mock.call_args.kwargs
            assert kw["pt_multiplier"] == trainer.config.barrier_pt_multiplier
            assert kw["sl_multiplier"] == trainer.config.barrier_sl_multiplier
            assert kw["max_holding"] == trainer.config.barrier_max_holding
        else:
            legacy_mock.assert_called_once()
            triple_mock.assert_not_called()
            kw = legacy_mock.call_args.kwargs
            assert kw["forward_bars"] == trainer.config.forward_bars
            assert (
                kw["threshold_atr_multiplier"]
                == trainer.config.threshold_atr_multiplier
            )

    def test_target_column_present_after_load(
        self, validator: MLValidator, tmp_path: Path
    ) -> None:
        """Whichever branch fires, the resulting dataframe must carry
        the ``target`` column — both builders produce it under contract."""
        trainer = _make_trainer(use_triple_barrier=True, tmp_path=tmp_path)
        _stub_builder(trainer)
        df = validator._load_full_data(trainer, ["BTCUSDT"], tmp_path)
        assert "target" in df.columns
