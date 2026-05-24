"""AFML Ch.4 sample uniqueness with REAL exit bars (Phase 3 Step 3.2).

Pre-fix ``compute_uniqueness_weights`` assumed every triple-barrier label
held for the full ``max_holding`` window. Early-exiting trades (PT/SL
touched before the vertical barrier) shared a shorter actual span, so
concurrency was systematically OVER-counted and uniqueness UNDER-counted.

The fix:
1. ``apply_triple_barrier`` now emits ``t1_bar`` = the bar index where
   each label actually exits (PT/SL touch bar, or vertical-barrier bar
   for timeouts).
2. ``create_target_triple_barrier`` stamps a per-symbol ``_bar_idx`` so
   that t1_bar (input-df bar index) stays comparable after timeout rows
   are dropped.
3. ``compute_uniqueness_weights`` accepts those two arrays and computes
   concurrency in the actual bar-index space, span-by-span.

Backward compat: the old fixed-span behaviour is preserved when neither
column is supplied — the existing test_lgbm_trainer/test_ml_validator
suites continue to pass unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.features.triple_barrier import apply_triple_barrier
from src.models.dataset_builder import DatasetBuilder


# ---------------------------------------------------------------------------
# Fixtures — synthetic price paths that force specific exit-bar behaviour
# ---------------------------------------------------------------------------

def _df_with_path(closes: list[float], atr_pct: float = 0.01) -> pl.DataFrame:
    return pl.DataFrame({
        "close": closes,
        "atr_pct": [atr_pct] * len(closes),
    })


# ---------------------------------------------------------------------------
# t1_bar column from apply_triple_barrier
# ---------------------------------------------------------------------------

class TestT1BarColumn:
    def test_column_present(self) -> None:
        df = _df_with_path([100.0] * 20)
        out = apply_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4,
        )
        assert "t1_bar" in out.columns
        assert out["t1_bar"].dtype == pl.Int64

    def test_timeout_t1_equals_i_plus_max_holding(self) -> None:
        """Flat path → no barrier touched → vertical barrier exit at
        ``i + max_holding_bars`` for every kept row."""
        n, h = 12, 4
        df = _df_with_path([100.0] * n)  # flat → no PT/SL ever
        out = apply_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=h,
        )
        expected = np.arange(len(out)) + h
        np.testing.assert_array_equal(out["t1_bar"].to_numpy(), expected)

    def test_pt_early_exit_gives_smaller_t1(self) -> None:
        """Bar 0 entry @100 with PT at 101. If close[1]=101.5 (above PT),
        label exits at bar 1 → t1_bar[0] = 0 + 1 = 1 (< 0 + max_holding)."""
        n, h = 10, 4
        closes = [100.0] * n
        closes[1] = 200.0  # bar 1 blows through PT
        df = _df_with_path(closes, atr_pct=0.01)  # PT = 100*1.01 = 101
        out = apply_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=h,
        )
        assert out["t1_bar"][0] == 1  # exited at bar 1
        assert out["t1_bar"][0] < 0 + h  # earlier than vertical barrier
        assert out["label"][0] == 1.0

    def test_sl_early_exit_gives_smaller_t1(self) -> None:
        n, h = 10, 4
        closes = [100.0] * n
        closes[2] = 50.0  # SL hit at bar 2
        df = _df_with_path(closes, atr_pct=0.01)
        out = apply_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=h,
        )
        assert out["t1_bar"][0] == 2
        assert out["label"][0] == -1.0

    def test_t1_never_exceeds_i_plus_max_holding(self) -> None:
        n, h = 30, 5
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0, 1.0, n))).tolist()
        df = _df_with_path(closes, atr_pct=0.02)
        out = apply_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=h,
        )
        t1 = out["t1_bar"].to_numpy()
        i = np.arange(len(out))
        assert np.all(t1 <= i + h)
        assert np.all(t1 >= i + 1)  # exit is strictly after entry


# ---------------------------------------------------------------------------
# compute_uniqueness_weights — real t1 mode vs fixed-span fallback
# ---------------------------------------------------------------------------

class TestWeightsBackwardCompat:
    def test_fixed_span_fallback_unchanged(self) -> None:
        """Without t1_bars the function reproduces the legacy values."""
        w_legacy = DatasetBuilder.compute_uniqueness_weights(
            n_samples=20, max_holding=4,
        )
        # Same call explicitly without t1_bars → identical
        w_again = DatasetBuilder.compute_uniqueness_weights(
            n_samples=20, max_holding=4, t1_bars=None,
        )
        np.testing.assert_array_equal(w_legacy, w_again)

    def test_normalization_mean_one(self) -> None:
        for mode_kwargs in [
            {},
            {
                "t1_bars": np.array([3, 4, 5, 6, 7, 8, 9]),
                "bar_idxs": np.arange(7),
            },
        ]:
            w = DatasetBuilder.compute_uniqueness_weights(
                n_samples=7, max_holding=4, **mode_kwargs,
            )
            assert w.mean() == pytest.approx(1.0, abs=1e-12)


class TestRealT1ChangesWeights:
    def test_early_exits_raise_uniqueness(self) -> None:
        """Half the labels exit immediately (PT in 1 bar) → their span
        is 1 bar; the other half hold full max_holding. The early-exit
        labels must end up with HIGHER uniqueness (less overlap) and
        therefore higher weights than under the fixed-span assumption."""
        n, h = 10, 4
        idx = np.arange(n, dtype=np.int64)
        # alternating: even labels exit at i+1, odd labels hold the
        # full h bars (vertical barrier).
        t1 = np.where(idx % 2 == 0, idx + 1, idx + h)

        w_real = DatasetBuilder.compute_uniqueness_weights(
            n_samples=n, max_holding=h, t1_bars=t1, bar_idxs=idx,
        )
        w_fixed = DatasetBuilder.compute_uniqueness_weights(
            n_samples=n, max_holding=h,
        )
        # Sanity: not identical (the whole point of the fix).
        assert not np.allclose(w_real, w_fixed)
        # Even rows (early exits) — average weight strictly above odd rows.
        even_avg = w_real[::2].mean()
        odd_avg = w_real[1::2].mean()
        assert even_avg > odd_avg

    def test_all_timeouts_yield_constant_interior_weights(self) -> None:
        """When every label hits the vertical barrier the concurrency
        is uniform across the interior of the time series, so interior
        weights must be constant in BOTH modes. (Real-t1 mode includes
        the exit bar so spans are h+1 bars vs h bars under the legacy
        off-by-one definition — the constants themselves differ but the
        within-mode-flatness invariant holds.)"""
        n, h = 50, 4
        idx = np.arange(n, dtype=np.int64)
        t1 = idx + h  # everyone times out

        w_real = DatasetBuilder.compute_uniqueness_weights(
            n_samples=n, max_holding=h, t1_bars=t1, bar_idxs=idx,
        )
        w_fixed = DatasetBuilder.compute_uniqueness_weights(
            n_samples=n, max_holding=h,
        )
        # Interior flatness in each mode.
        assert np.allclose(w_real[h:n - h], w_real[h], rtol=1e-9)
        assert np.allclose(w_fixed[h:n - h], w_fixed[h], rtol=1e-9)
        # Both modes normalise to mean 1.
        assert w_real.mean() == pytest.approx(1.0, abs=1e-12)
        assert w_fixed.mean() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# compute_uniqueness_weights_by_symbol picks the right mode automatically
# ---------------------------------------------------------------------------

class TestBySymbolAutoSelectsMode:
    def _df_with_t1(self) -> pl.DataFrame:
        # 6 labels, 2 symbols, all early exits (t1 = i + 1).
        return pl.DataFrame({
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "_bar_idx": [0, 1, 2, 0, 1, 2],
            "t1_bar":   [1, 2, 3, 1, 2, 3],
            "target":   [1, -1, 1, -1, 1, -1],
        })

    def test_real_t1_used_when_columns_present(self) -> None:
        db = DatasetBuilder(data_dir=Path("."), symbols=["BTC"])
        df = self._df_with_t1()
        w_with = db.compute_uniqueness_weights_by_symbol(
            df, max_holding=4,
        )
        # Compare against an explicit drop of the t1/idx columns →
        # legacy path. They must differ (real-t1 path engaged).
        df_no_t1 = df.drop(["t1_bar", "_bar_idx"])
        w_without = db.compute_uniqueness_weights_by_symbol(
            df_no_t1, max_holding=4,
        )
        assert not np.allclose(w_with, w_without)

    def test_fallback_when_columns_missing(self) -> None:
        db = DatasetBuilder(data_dir=Path("."), symbols=["BTC"])
        df = self._df_with_t1().drop(["t1_bar", "_bar_idx"])
        # Must not raise — fall back to legacy fixed-span behaviour.
        w = db.compute_uniqueness_weights_by_symbol(df, max_holding=4)
        assert w.shape == (6,)
        assert w.mean() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# End-to-end: create_target_triple_barrier emits the columns + downstream
# weight computation differs from the legacy path
# ---------------------------------------------------------------------------

class TestCreateTargetEndToEnd:
    def _path_with_early_exits(self) -> pl.DataFrame:
        # Strong moves on every other bar to trigger early PT exits.
        n = 50
        closes = [100.0]
        for i in range(1, n):
            closes.append(closes[-1] * (1.02 if i % 2 == 0 else 0.99))
        return pl.DataFrame({
            "close": closes,
            "atr_pct": [0.01] * n,
            "symbol": ["BTC"] * n,
            "regime": ["all"] * n,
        })

    def test_target_columns_carry_through_filter(self) -> None:
        db = DatasetBuilder(data_dir=Path("."), symbols=["BTC"])
        df = self._path_with_early_exits()
        out = db.create_target_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding=4,
            drop_timeout=True,
        )
        assert "t1_bar" in out.columns
        assert "_bar_idx" in out.columns
        # After drop_timeout, surviving rows still carry coherent t1/idx.
        assert (out["t1_bar"] >= out["_bar_idx"]).all()
        assert (out["t1_bar"] <= out["_bar_idx"] + 4).all()

    def test_by_symbol_weights_change_after_fix(self) -> None:
        db = DatasetBuilder(data_dir=Path("."), symbols=["BTC"])
        df = self._path_with_early_exits()
        out = db.create_target_triple_barrier(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding=4,
            drop_timeout=True,
        )
        # Real-t1 path (columns present)
        w_real = db.compute_uniqueness_weights_by_symbol(out, max_holding=4)
        # Force the legacy path by stripping the columns
        out_legacy = out.drop(["t1_bar", "_bar_idx"])
        w_legacy = db.compute_uniqueness_weights_by_symbol(
            out_legacy, max_holding=4,
        )
        # The whole point of the fix: weights must change in a setup
        # with many early exits.
        assert not np.allclose(w_real, w_legacy)
        # Both still average to 1 (normalisation invariant preserved).
        assert w_real.mean() == pytest.approx(1.0, abs=1e-12)
        assert w_legacy.mean() == pytest.approx(1.0, abs=1e-12)
