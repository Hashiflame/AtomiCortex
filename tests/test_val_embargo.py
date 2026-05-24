"""Phase 3 Step 3.4 — eval_set embargo regression tests.

Pre-fix the train/val split inside ``LGBMTrainer.train`` had zero gap.
Triple-barrier labels of the last ``max_holding`` rows of train looked
forward by exactly that many bars — i.e. straight into the val window
— so LightGBM's early-stopping callback saw a leaked, falsely-optimistic
val loss and halted too early. AFML Ch.7 demands an embargo equal to
(at least) the label horizon between train end and val start.

The fix drops one label-horizon's worth of rows from the tail of
``X_train_fit`` so no train label's forward window reaches into
``X_val``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pytest

from src.models.lgbm_trainer import LGBMTrainer, ModelConfig


def _make_trainer(
    tmp_path: Path,
    use_triple_barrier: bool = False,
    forward_bars: int = 1,
    barrier_max_holding: int = 6,
) -> LGBMTrainer:
    cfg = ModelConfig(
        regime="all",
        symbols=["BTCUSDT"],
        use_triple_barrier=use_triple_barrier,
        forward_bars=forward_bars,
        barrier_max_holding=barrier_max_holding,
    )
    return LGBMTrainer(cfg, features_dir=tmp_path, models_dir=tmp_path)


def _capture_lgb_train_args() -> dict:
    """Patch ``lgb.train`` to record the train/val Datasets passed in.

    Returns a dict that gets populated with ``train_data`` and
    ``val_data`` once the patched train() runs. Returns a no-op
    Booster-like object so the rest of ``train()`` can proceed.
    """
    capture: dict = {}

    class _DummyBooster:
        # Just enough surface for the post-fit code path.
        best_iteration = 1
        best_score = {"valid_0": {"binary_logloss": 0.5}}

        def feature_importance(self, importance_type="gain"):
            return np.array([])

        def feature_name(self):
            return []

        def save_model(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return np.zeros(1)

    def fake_train(params, train_set, num_boost_round, valid_sets=None,
                   valid_names=None, callbacks=None, **kwargs):
        capture["train_data"] = train_set
        capture["val_data"] = valid_sets[-1] if valid_sets else None
        capture["params"] = params
        return _DummyBooster()

    return capture, fake_train


def _toy_X_y(n: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Deterministic feature matrix + labels that survive the booster path."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(n, 4)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int32)
    return X, y, [f"f{i}" for i in range(4)]


def _run_train_with_embargo(
    trainer: LGBMTrainer, n: int,
) -> dict:
    """Drive train() through the embargo branch by stubbing _prepare_xy."""
    capture, fake_train = _capture_lgb_train_args()
    X, y, names = _toy_X_y(n)

    # Stub _prepare_xy so we control n_rows exactly; bypass uniqueness
    # weights to keep the trace clean.
    with patch.object(
        LGBMTrainer, "_prepare_xy", return_value=(X, y, names),
    ), patch.object(
        LGBMTrainer, "_log_to_mlflow", return_value=None,
    ), patch("src.models.lgbm_trainer.lgb.train", side_effect=fake_train), \
         patch("src.models.lgbm_trainer.pickle.dump", return_value=None):
        import polars as pl
        # Empty placeholder df — _prepare_xy is stubbed so contents
        # don't matter.
        trainer.train(pl.DataFrame({"target": [1] * n}))
    return capture


# ---------------------------------------------------------------------------
# Embargo arithmetic — proves the train tail is trimmed by exactly the
# label horizon, and the val set still starts at the legacy val_split.
# ---------------------------------------------------------------------------

class TestEmbargoArithmetic:
    def test_triple_barrier_embargo_equals_max_holding(self, tmp_path):
        n = 1000
        trainer = _make_trainer(
            tmp_path, use_triple_barrier=True, barrier_max_holding=6,
        )
        cap = _run_train_with_embargo(trainer, n)
        # val_frac = 0.90 for 4H profile
        val_split = int(n * 0.90)            # 900
        expected_fit_end = val_split - 6     # 894
        assert cap["train_data"].data.shape[0] == expected_fit_end
        assert cap["val_data"].data.shape[0] == n - val_split  # 100

    def test_legacy_target_embargo_equals_forward_bars(self, tmp_path):
        n = 1000
        trainer = _make_trainer(
            tmp_path, use_triple_barrier=False, forward_bars=3,
        )
        cap = _run_train_with_embargo(trainer, n)
        val_split = int(n * 0.90)
        assert cap["train_data"].data.shape[0] == val_split - 3
        assert cap["val_data"].data.shape[0] == n - val_split

    def test_default_legacy_embargo_is_one_row(self, tmp_path):
        n = 500
        trainer = _make_trainer(tmp_path)  # default: forward_bars=1
        cap = _run_train_with_embargo(trainer, n)
        val_split = int(n * 0.90)
        assert cap["train_data"].data.shape[0] == val_split - 1


# ---------------------------------------------------------------------------
# Train tail labels no longer reach into val
# ---------------------------------------------------------------------------

class TestNoLeakageBetweenFitAndVal:
    def test_max_train_label_horizon_does_not_reach_val(self, tmp_path):
        """Per-row: the embargo means the *last* train_fit row's label
        horizon (row_index + max_holding) is strictly less than
        val_start (val_split). No label peeks across."""
        n = 1000
        max_h = 6
        trainer = _make_trainer(
            tmp_path, use_triple_barrier=True, barrier_max_holding=max_h,
        )
        cap = _run_train_with_embargo(trainer, n)
        fit_end = cap["train_data"].data.shape[0]            # 894
        last_train_row_idx = fit_end - 1                  # 893
        label_reaches_to = last_train_row_idx + max_h     # 899
        val_start = n - cap["val_data"].data.shape[0]        # 900
        assert label_reaches_to < val_start


# ---------------------------------------------------------------------------
# Guard against shrinking train_fit too much on tiny datasets
# ---------------------------------------------------------------------------

class TestEmbargoGuard:
    def test_tiny_dataset_caps_embargo_at_half_val_split(self, tmp_path):
        """val_split * 2 = 10 → val_split // 2 = 5 → embargo capped at 5
        even though max_holding=6. Ensures we never collapse train_fit
        to zero on synthetic fixtures."""
        n = 12  # val_split = int(12 * 0.90) = 10
        trainer = _make_trainer(
            tmp_path, use_triple_barrier=True, barrier_max_holding=6,
        )
        cap = _run_train_with_embargo(trainer, n)
        val_split = int(n * 0.90)  # 10
        # cap = val_split // 2 = 5
        assert cap["train_data"].data.shape[0] == val_split - 5
        # Train fit must still have rows.
        assert cap["train_data"].data.shape[0] > 0


# ---------------------------------------------------------------------------
# Uniqueness weights stay length-aligned with the embargoed train_fit
# ---------------------------------------------------------------------------

class TestUniquenessAlignment:
    def test_uniq_fit_length_matches_train_fit(self, tmp_path):
        """When ``use_uniqueness_weights=True`` the uniq_fit slice has to
        be length-aligned with the (smaller) embargoed train_fit, not
        the pre-embargo val_split."""
        n = 1000
        cfg = ModelConfig(
            regime="all",
            symbols=["BTCUSDT"],
            use_triple_barrier=True,
            use_uniqueness_weights=True,
            barrier_max_holding=6,
        )
        trainer = LGBMTrainer(cfg, features_dir=tmp_path, models_dir=tmp_path)

        capture, fake_train = _capture_lgb_train_args()
        X, y, names = _toy_X_y(n)

        # Pre-built uniqueness weights of length n (one per row).
        uniq = np.linspace(0.5, 1.5, n)

        import polars as pl
        with patch.object(
            LGBMTrainer, "_prepare_xy", return_value=(X, y, names),
        ), patch.object(
            LGBMTrainer, "_log_to_mlflow", return_value=None,
        ), patch.object(
            trainer._builder,
            "compute_uniqueness_weights_by_symbol",
            return_value=uniq,
        ), patch(
            "src.models.lgbm_trainer.lgb.train", side_effect=fake_train,
        ), patch(
            "src.models.lgbm_trainer.pickle.dump", return_value=None,
        ):
            trainer.train(pl.DataFrame({"target": [1] * n}))

        val_split = int(n * 0.90)         # 900
        fit_end = val_split - 6           # 894
        # The training Dataset's per-row weight must have fit_end entries.
        weights = capture["train_data"].weight
        assert weights.shape[0] == fit_end


# ---------------------------------------------------------------------------
# Backward compat — public train() signature unchanged
# ---------------------------------------------------------------------------

class TestApiSurface:
    def test_train_signature_unchanged(self):
        import inspect
        sig = inspect.signature(LGBMTrainer.train)
        # self + train_df only — no new required arg.
        assert list(sig.parameters.keys()) == ["self", "train_df"]
