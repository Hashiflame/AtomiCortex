"""Tests for the post-lookahead alpha-v2 feature expansion.

NOTE: the spec asked for tests/test_new_features.py, but that file
already holds an unrelated suite (scalar live-enrichment: compute_vpin,
compute_liquidation_proximity, ...). To avoid destroying those tests
the alpha-v2 suite lives here instead.

Covers funding momentum, OI-derived, CVD-derived (derivatives.py),
EMA slopes / volume-session / fractal / candle (microstructure.py),
and session momentum (session_features.py). All new features are
MTF-only (1H/15m) and must be lookahead-free.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.derivatives import (
    add_cvd_derived_features,
    add_funding_momentum_features,
    add_oi_derived_features,
)
from src.features.microstructure import (
    add_candle_structure_features,
    add_ema_slope_features,
    add_fractal_features,
    add_volume_session_features,
)
from src.features.session_features import SessionMomentum

_HOUR_MS = 3_600_000


def _ohlcv(n: int = 120, seed: int = 7) -> pl.DataFrame:
    """Deterministic hourly OHLCV with the upstream columns the
    alpha-v2 functions depend on (funding_rate, oi_value, cvd,
    volume_sma_20, atr_pct, returns_1)."""
    rng = np.random.default_rng(seed)
    close = np.maximum(100.0 + np.cumsum(rng.normal(0, 1.0, n)), 1.0)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.uniform(10, 100, n)
    df = pl.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * _HOUR_MS,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        "taker_buy_volume": volume * rng.uniform(0.3, 0.7, n),
        "funding_rate": rng.normal(0.0001, 0.00005, n),
        "oi_value": 1_000_000 + np.cumsum(rng.normal(0, 5_000, n)),
        "cvd": np.cumsum(rng.normal(0, 50, n)),
        "volume_sma_20": rng.uniform(20, 80, n),
        "atr_pct": rng.uniform(0.005, 0.02, n),
    })
    return df.with_columns(
        ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))
        .fill_null(0.0).alias("returns_1")
    )


# --- Funding momentum -----------------------------------------------------

def test_funding_rate_change_1_is_diff():
    df = add_funding_momentum_features(_ohlcv())
    fr = df["funding_rate"].to_numpy()
    chg = df["funding_rate_change_1"].to_numpy()
    assert np.allclose(chg[1:], fr[1:] - fr[:-1], atol=1e-12)
    assert chg[0] == 0.0


def test_funding_rate_zscore_bounded():
    z = add_funding_momentum_features(_ohlcv())["funding_rate_zscore_rolling"].to_numpy()
    assert np.isfinite(z).all()
    assert np.abs(z).max() < 10.0


def test_funding_rate_change_no_lookahead():
    df = _ohlcv()
    full = add_funding_momentum_features(df)["funding_rate_change_1"].to_numpy()
    head = add_funding_momentum_features(df.head(60))["funding_rate_change_1"].to_numpy()
    assert np.allclose(full[:60], head, atol=1e-12)


def test_funding_rate_acceleration_is_second_diff():
    df = add_funding_momentum_features(_ohlcv())
    c1 = df["funding_rate_change_1"].to_numpy()
    acc = df["funding_rate_acceleration"].to_numpy()
    assert np.allclose(acc[2:], c1[2:] - c1[1:-1], atol=1e-12)


# --- OI derived -----------------------------------------------------------

def test_oi_delta_1h_normalized():
    df = add_oi_derived_features(_ohlcv())
    oi = df["oi_value"].to_numpy()
    d = df["oi_delta_1h"].to_numpy()
    assert np.allclose(d[1:], (oi[1:] - oi[:-1]) / oi[:-1], atol=1e-9)


def test_oi_price_div_vec_values():
    df = add_oi_derived_features(_ohlcv())
    assert set(df["oi_price_div_vec"].unique().to_list()).issubset({-1, 0, 1})


def test_oi_features_missing_base_is_noop_safe():
    out = add_oi_derived_features(_ohlcv().drop("oi_value"))
    assert out["oi_delta_1h"].sum() == 0.0
    assert set(out["oi_price_div_vec"].unique().to_list()) == {0}


# --- CVD derived ----------------------------------------------------------

def test_cvd_slope_3bar_correct():
    df = add_cvd_derived_features(_ohlcv())
    cvd = df["cvd"].to_numpy()
    vsma = df["volume_sma_20"].to_numpy()
    slope = df["cvd_slope_3bar"].to_numpy()
    assert np.allclose(slope[3:], (cvd[3:] - cvd[:-3]) / vsma[3:], atol=1e-6)


def test_cvd_divergence_binary():
    df = add_cvd_derived_features(_ohlcv())
    assert set(df["cvd_divergence"].unique().to_list()).issubset({-1, 0, 1})


# --- EMA slopes -----------------------------------------------------------

def test_ema9_slope_normalized_uses_atr():
    df = add_ema_slope_features(_ohlcv())
    ema9 = df["ema9"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr_pct"].to_numpy()
    slope = df["ema9_slope_normalized"].to_numpy()
    atr_abs = np.maximum(atr * close, 1e-10)
    assert np.allclose(slope[3:], (ema9[3:] - ema9[:-3]) / atr_abs[3:], atol=1e-6)


def test_ema9_cross_ema21_sign():
    df = add_ema_slope_features(_ohlcv())
    cross = df["ema9_cross_ema21"].to_numpy()
    diff = df["ema9"].to_numpy() - df["ema21"].to_numpy()
    assert np.all(np.sign(cross) == np.sign(diff))


def test_ema_slopes_no_lookahead():
    df = _ohlcv()
    full = add_ema_slope_features(df)["ema9_cross_ema21"].to_numpy()
    head = add_ema_slope_features(df.head(50))["ema9_cross_ema21"].to_numpy()
    assert np.allclose(full[:50], head, atol=1e-6)


# --- Volume vs session ----------------------------------------------------

def test_volume_vs_session_avg_positive():
    v = add_volume_session_features(_ohlcv())["volume_vs_session_avg"].to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all()


def test_volume_vs_session_avg_no_lookahead():
    df = _ohlcv(n=120)
    full = add_volume_session_features(df)["volume_vs_session_avg"].to_numpy()
    head = add_volume_session_features(df.head(72))["volume_vs_session_avg"].to_numpy()
    assert np.allclose(full[:72], head, atol=1e-9, equal_nan=True)


# --- Fractal efficiency ---------------------------------------------------

def test_efficiency_ratio_between_0_and_1():
    df = add_fractal_features(_ohlcv())
    for col in ("efficiency_ratio_10", "efficiency_ratio_20"):
        x = df[col].to_numpy()
        assert np.isfinite(x).all()
        assert (x >= 0.0).all() and (x <= 1.0).all()


def test_efficiency_ratio_trend_close_to_1():
    n = 60
    df = pl.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * _HOUR_MS,
        "close": np.arange(1.0, n + 1.0),
    })
    er = add_fractal_features(df)["efficiency_ratio_10"].to_numpy()
    assert er[-1] == pytest.approx(1.0, abs=1e-9)


def test_efficiency_ratio_choppy_close_to_0():
    n = 60
    df = pl.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * _HOUR_MS,
        "close": 100.0 + np.array([0.0, 1.0] * (n // 2)),
    })
    er = add_fractal_features(df)["efficiency_ratio_20"].to_numpy()
    assert er[-1] < 0.2


# --- Candle structure -----------------------------------------------------

def test_candle_body_pct_between_0_and_1():
    x = add_candle_structure_features(_ohlcv())["candle_body_pct"].to_numpy()
    assert (x >= 0.0).all() and (x <= 1.0 + 1e-9).all()


def test_upper_wick_pct_non_negative():
    df = add_candle_structure_features(_ohlcv())
    assert (df["upper_wick_pct"].to_numpy() >= 0.0).all()
    assert (df["lower_wick_pct"].to_numpy() >= 0.0).all()


def test_candle_direction_sign():
    df = add_candle_structure_features(_ohlcv())
    d = df["candle_direction"].to_numpy()
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    assert set(np.unique(d)).issubset({-1, 0, 1})
    assert np.all(d[c > o] == 1)
    assert np.all(d[c < o] == -1)


# --- Session momentum -----------------------------------------------------

def test_session_open_return_forward_filled():
    df = SessionMomentum().calculate(_ohlcv(n=96))
    chk = df.with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("_t")
    ).with_columns([
        pl.col("_t").dt.date().cast(pl.Utf8).alias("d"),
        pl.when(pl.col("_t").dt.hour() < 8).then(0)
        .when(pl.col("_t").dt.hour() < 13).then(1)
        .otherwise(2).alias("ph"),
    ])
    nuniq = chk.group_by("d", "ph").agg(
        pl.col("session_open_return").n_unique().alias("u")
    )
    assert (nuniq["u"] == 1).all()


def test_session_return_cumulative_no_lookahead():
    df = _ohlcv(n=96)
    full = SessionMomentum().calculate(df)["session_return_cumulative"].to_numpy()
    head = SessionMomentum().calculate(df.head(48))["session_return_cumulative"].to_numpy()
    assert np.allclose(full[:48], head, atol=1e-9)


def test_session_momentum_3bar_correct():
    df = SessionMomentum().calculate(_ohlcv())
    close = df["close"].to_numpy()
    ret = np.zeros_like(close)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
    mom = df["session_momentum_3bar"].to_numpy()
    assert np.allclose(mom[3:], ret[3:] + ret[2:-1] + ret[1:-2], atol=1e-9)


def test_session_momentum_columns_present():
    df = SessionMomentum().calculate(_ohlcv())
    for c in (
        "session_open_return", "session_momentum_3bar",
        "session_return_cumulative", "vwap_slope_3bar",
        "vwap_slope_6bar", "price_above_vwap",
    ):
        assert c in df.columns
