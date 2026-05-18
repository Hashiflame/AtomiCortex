"""Tests for triple-barrier labeling (AFML Ch.3).

Covers barrier-hit logic, the vertical barrier, no-lookahead,
volatility scaling, the min_atr guard, label-value domain,
asymmetric barriers, label_statistics, and the 1H/15m presets.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.triple_barrier import apply_triple_barrier, label_statistics


def _df(close, atr=0.01):
    """Build a frame; atr may be scalar or per-bar array."""
    n = len(close)
    atr_arr = np.full(n, atr) if np.isscalar(atr) else np.asarray(atr)
    return pl.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * 3_600_000,
        "close": np.asarray(close, dtype=np.float64),
        "atr_pct": atr_arr.astype(np.float64),
    })


# --- barrier-hit basics ---------------------------------------------------

def test_upper_barrier_hit_returns_plus1():
    # atr=0.01, pt=1.5 → upper = 100×1.015 = 101.5; bar1 jumps to 102.
    df = _df([100.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
             atr=0.01)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3)
    assert out["label"][0] == 1


def test_lower_barrier_hit_returns_minus1():
    # lower = 100×(1-1.0×0.01) = 99; bar1 drops to 98.
    df = _df([100.0, 98.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
             atr=0.01)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3)
    assert out["label"][0] == -1


def test_vertical_barrier_returns_zero():
    # Flat path, no barrier ever touched within the window.
    df = _df([100.0] * 10, atr=0.05)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3)
    assert (out["label"] == 0).all()


def test_upper_before_lower_returns_plus1():
    # Up first (bar1), then a deep drop later (bar2). First touch wins.
    df = _df([100.0, 102.0, 90.0, 100.0, 100.0, 100.0, 100.0],
             atr=0.01)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3)
    assert out["label"][0] == 1


def test_lower_before_upper_returns_minus1():
    df = _df([100.0, 98.0, 110.0, 100.0, 100.0, 100.0, 100.0],
             atr=0.01)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3)
    assert out["label"][0] == -1


# --- structural guarantees ------------------------------------------------

def test_last_rows_are_dropped():
    n = 12
    df = _df(100.0 + np.zeros(n), atr=0.05)
    out = apply_triple_barrier(df, max_holding_bars=4)
    assert len(out) == n - 4
    assert out["label"].null_count() == 0


def test_label_values_only_minus1_0_plus1():
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, 200))
    out = apply_triple_barrier(_df(close, atr=0.01), max_holding_bars=6)
    assert set(out["label"].unique().to_list()).issubset({-1.0, 0.0, 1.0})


def test_no_lookahead_in_label():
    # label[t] must not depend on bars beyond t+max_holding_bars.
    rng = np.random.default_rng(2)
    close = list(100.0 + np.cumsum(rng.normal(0, 0.8, 80)))
    H = 5
    base = apply_triple_barrier(_df(close, 0.01), max_holding_bars=H)
    # Mutate a far-future bar only; rows whose window ends before it
    # must be unchanged.
    close2 = list(close)
    close2[60] = close2[60] * 5.0
    mut = apply_triple_barrier(_df(close2, 0.01), max_holding_bars=H)
    safe = 60 - H - 1  # last row whose [t+1..t+H] window excludes idx 60
    assert np.array_equal(
        base["label"].to_numpy()[:safe],
        mut["label"].to_numpy()[:safe],
    )


def test_atr_scaling_correct():
    # On an upper hit the realized future_return == pt × atr_pct.
    df = _df([100.0, 105.0, 100.0, 100.0, 100.0, 100.0], atr=0.02)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3)
    assert out["label"][0] == 1
    assert out["future_return"][0] == pytest.approx(1.5 * 0.02, rel=1e-9)


def test_min_atr_protection():
    # atr_pct = 0 everywhere → must use the min_atr floor, no div-zero,
    # finite barriers/returns.
    df = _df([100.0, 100.5, 100.0, 100.0, 100.0, 100.0], atr=0.0)
    out = apply_triple_barrier(df, pt_multiplier=1.5, sl_multiplier=1.0,
                               max_holding_bars=3, min_atr=0.001)
    assert np.isfinite(out["future_return"].to_numpy()).all()
    assert set(out["label"].unique().to_list()).issubset({-1.0, 0.0, 1.0})


def test_vertical_future_return_is_real_path_return():
    # No barrier touched → future_return = (close[t+H] - close[t]) / close[t].
    close = [100.0, 100.2, 100.4, 100.6, 100.0, 100.0, 100.0]
    df = _df(close, atr=0.5)  # huge ATR → barriers unreachable
    out = apply_triple_barrier(df, max_holding_bars=3)
    assert out["label"][0] == 0
    assert out["future_return"][0] == pytest.approx(
        (close[3] - close[0]) / close[0], rel=1e-9
    )


# --- coverage / distribution ---------------------------------------------

def test_coverage_between_20_and_80():
    # Barrier width on the order of a typical multi-bar move:
    # upper≈1.5 / lower≈1.0 price units vs ~6-bar move σ≈0.45·√6≈1.1.
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.45, 3000))
    out = apply_triple_barrier(_df(close, atr=0.01), pt_multiplier=1.5,
                               sl_multiplier=1.0, max_holding_bars=6)
    cov = label_statistics(out)["coverage"]
    assert 20.0 <= cov <= 80.0


def test_asymmetric_barriers_more_vertical_than_symmetric():
    # Wider PT (2.0) vs SL (1.0): harder to hit the upper → at least as
    # many vertical labels as the symmetric (1.0/1.0) case.
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 1500))
    asym = label_statistics(apply_triple_barrier(
        _df(close, 0.01), pt_multiplier=2.0, sl_multiplier=1.0,
        max_holding_bars=6))["vertical"]
    sym = label_statistics(apply_triple_barrier(
        _df(close, 0.01), pt_multiplier=1.0, sl_multiplier=1.0,
        max_holding_bars=6))["vertical"]
    assert asym >= sym


def test_label_statistics_correct():
    df = _df([100.0] * 10, atr=0.05)
    out = apply_triple_barrier(df, max_holding_bars=3)
    st = label_statistics(out)
    assert st["total"] == 7
    assert st["vertical"] == 7
    assert st["long"] == 0 and st["short"] == 0
    assert st["coverage"] == pytest.approx(0.0)
    assert st["vertical_pct"] == pytest.approx(100.0)


def test_label_statistics_sums_to_total():
    rng = np.random.default_rng(9)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, 500))
    out = apply_triple_barrier(_df(close, 0.01), max_holding_bars=6)
    st = label_statistics(out)
    assert st["long"] + st["short"] + st["vertical"] == st["total"]
    assert st["long_pct"] + st["short_pct"] + st["vertical_pct"] == \
        pytest.approx(100.0)


# --- timeframe presets ----------------------------------------------------

def test_works_with_1h_params():
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, 800))
    out = apply_triple_barrier(_df(close, 0.01), pt_multiplier=1.5,
                               sl_multiplier=1.0, max_holding_bars=6)
    assert len(out) == 800 - 6
    assert out["label"].null_count() == 0


def test_works_with_15m_params():
    rng = np.random.default_rng(12)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, 800))
    out = apply_triple_barrier(_df(close, 0.01), pt_multiplier=2.0,
                               sl_multiplier=1.0, max_holding_bars=8)
    assert len(out) == 800 - 8
    assert set(out["label"].unique().to_list()).issubset({-1.0, 0.0, 1.0})
    # future_return consistent with label sign on actionable trades.
    lab = out["label"].to_numpy()
    fr = out["future_return"].to_numpy()
    assert np.all(fr[lab == 1] > 0)
    assert np.all(fr[lab == -1] < 0)
