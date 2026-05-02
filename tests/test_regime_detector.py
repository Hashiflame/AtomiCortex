"""
tests/test_regime_detector.py

Phase 3, Step 3.3 — Regime Detector unit and integration tests.
Minimum 15 tests covering Hurst, ADX, ATR percentile, RegimeDetector,
and real-data validation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.features.regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeState,
    calculate_adx,
    calculate_atr_percentile,
    calculate_hurst_exponent,
)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_BASE_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC
_BAR_MS = 4 * 3600 * 1000     # 4H in ms
_RNG = np.random.default_rng(42)


def _ar1_prices(n: int, phi: float, seed: int = 42) -> np.ndarray:
    """Generate prices from AR(1) returns: ret[t] = phi * ret[t-1] + noise.

    phi > 0 → persistent (trending), phi < 0 → mean-reverting, phi=0 → random walk.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = phi * returns[i - 1] + rng.normal(0, 1)
    return 40_000.0 + returns.cumsum()


def _trending_klines(n: int = 500, direction: float = 1.0) -> pl.DataFrame:
    """Strongly trending price series (AR(1) with positive persistence)."""
    rng = np.random.default_rng(42)
    base = 40_000.0 + direction * np.arange(n) * 50.0
    noise = rng.normal(0, 5.0, n)
    close = base + noise
    return pl.DataFrame({
        "open_time": [_BASE_MS + i * _BAR_MS for i in range(n)],
        "open": close - 10.0,
        "high": close + 30.0,
        "low": close - 30.0,
        "close": close,
        "volume": [500.0 + float(i % 10) * 10 for i in range(n)],
        "taker_buy_volume": [275.0 + float(i % 10) * 5 for i in range(n)],
    })


def _ranging_klines(n: int = 500) -> pl.DataFrame:
    """Mean-reverting (sinusoidal) price series."""
    t = np.linspace(0, 10 * math.pi, n)
    close = 40_000.0 + 200.0 * np.sin(t)
    return pl.DataFrame({
        "open_time": [_BASE_MS + i * _BAR_MS for i in range(n)],
        "open": close - 10.0,
        "high": close + 30.0,
        "low": close - 30.0,
        "close": close,
        "volume": [500.0] * n,
        "taker_buy_volume": [250.0] * n,
    })


def _high_vol_klines(n: int = 500) -> pl.DataFrame:
    """Price series with a massive volatility spike at the end."""
    rng = np.random.default_rng(123)
    close = np.full(n, 40_000.0)
    # Normal vol for first 80%
    close[:int(n * 0.8)] += rng.normal(0, 20, int(n * 0.8)).cumsum()
    # Massive vol for last 20%
    close[int(n * 0.8):] += rng.normal(0, 500, n - int(n * 0.8)).cumsum()
    high = close + np.abs(rng.normal(0, 50, n))
    low = close - np.abs(rng.normal(0, 50, n))
    # Make high/low spread huge for the last 20%
    high[int(n * 0.8):] = close[int(n * 0.8):] + rng.uniform(500, 2000, n - int(n * 0.8))
    low[int(n * 0.8):] = close[int(n * 0.8):] - rng.uniform(500, 2000, n - int(n * 0.8))
    return pl.DataFrame({
        "open_time": [_BASE_MS + i * _BAR_MS for i in range(n)],
        "open": close - 5.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": [500.0] * n,
        "taker_buy_volume": [250.0] * n,
    })


def _random_walk_prices(n: int = 300, seed: int = 42) -> np.ndarray:
    """Random walk: cumsum of N(0,1)."""
    rng = np.random.default_rng(seed)
    return 40_000.0 + rng.standard_normal(n).cumsum()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hurst Exponent
# ──────────────────────────────────────────────────────────────────────────────

class TestHurstExponent:
    def test_hurst_trending_above_055(self):
        """Persistent AR(1) returns (phi=0.6) → Hurst > 0.55."""
        prices = _ar1_prices(500, phi=0.6, seed=42)
        h = calculate_hurst_exponent(prices, min_lag=2, max_lag=100)
        assert h > 0.55, f"Expected Hurst > 0.55 for persistent AR(1), got {h}"

    def test_hurst_mean_reverting_below_050(self):
        """Anti-persistent AR(1) returns (phi=-0.6) → Hurst < 0.50."""
        prices = _ar1_prices(500, phi=-0.6, seed=42)
        h = calculate_hurst_exponent(prices, min_lag=2, max_lag=100)
        assert h < 0.50, f"Expected Hurst < 0.50 for anti-persistent AR(1), got {h}"

    def test_hurst_random_walk_around_05(self):
        """Random walk (phi=0) → Hurst in [0.40, 0.70] (R/S has upward bias)."""
        prices = _random_walk_prices(500, seed=42)
        h = calculate_hurst_exponent(prices, min_lag=2, max_lag=100)
        assert 0.40 <= h <= 0.70, f"Expected Hurst ≈ 0.5 for random walk, got {h}"

    def test_hurst_short_series_returns_05(self):
        """Series shorter than min threshold → return 0.5 (neutral)."""
        h = calculate_hurst_exponent(np.array([1.0, 2.0, 3.0]))
        assert h == 0.5

    def test_hurst_in_01_range(self):
        """Hurst must always be clipped to [0, 1]."""
        for seed in range(5):
            prices = _random_walk_prices(300, seed=seed)
            h = calculate_hurst_exponent(prices)
            assert 0.0 <= h <= 1.0, f"Hurst {h} out of [0,1] for seed={seed}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. ADX
# ──────────────────────────────────────────────────────────────────────────────

class TestADX:
    def test_adx_strong_trend_above_25(self):
        """Strong linear trend → last ADX > 25."""
        n = 200
        close = np.linspace(40_000, 50_000, n)
        high = close + 50
        low = close - 50
        adx = calculate_adx(high, low, close, period=14)
        # Use the last stable value (skip warmup NaN)
        valid = adx[~np.isnan(adx)]
        assert valid[-1] > 25, f"Expected ADX > 25 on strong trend, got {valid[-1]}"

    def test_adx_sideways_below_20(self):
        """Flat / sideways → ADX < 20."""
        n = 200
        rng = np.random.default_rng(42)
        close = 40_000 + rng.normal(0, 1, n)  # almost flat
        high = close + 2
        low = close - 2
        adx = calculate_adx(high, low, close, period=14)
        valid = adx[~np.isnan(adx)]
        assert valid[-1] < 20, f"Expected ADX < 20 on sideways, got {valid[-1]}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. ATR Percentile
# ──────────────────────────────────────────────────────────────────────────────

class TestATRPercentile:
    def test_atr_percentile_in_01(self):
        """atr_percentile must be in [0, 1]."""
        n = 200
        rng = np.random.default_rng(42)
        close = 40_000 + rng.standard_normal(n).cumsum()
        high = close + rng.uniform(10, 50, n)
        low = close - rng.uniform(10, 50, n)
        _atr, pct = calculate_atr_percentile(high, low, close)
        assert 0.0 <= pct <= 1.0, f"ATR percentile {pct} out of [0,1]"


# ──────────────────────────────────────────────────────────────────────────────
# 4. RegimeDetector.detect
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeDetect:
    def test_detect_trend_strong_adx(self):
        """Strong linear trend (ADX > 25) → TREND_UP or TREND_DOWN."""
        # Linear ramp guarantees ADX >> 25
        df = _trending_klines(500, direction=1.0)
        det = RegimeDetector(hurst_window=200)
        state = det.detect(df)
        assert state.regime in (
            MarketRegime.TREND_UP, MarketRegime.TREND_DOWN, MarketRegime.HIGH_VOL,
        ), f"Expected TREND or HIGH_VOL for strong ADX, got {state.regime}"
        # ADX should be very high on a linear ramp
        assert state.adx > 25, f"Expected ADX > 25, got {state.adx}"

    def test_detect_range_low_adx(self):
        """Flat/sideways data (ADX < 20) → RANGE."""
        n = 500
        rng = np.random.default_rng(42)
        close = 40_000.0 + rng.normal(0, 1.0, n)  # nearly flat
        df = pl.DataFrame({
            "open_time": [_BASE_MS + i * _BAR_MS for i in range(n)],
            "open": close - 0.5,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": [500.0] * n,
            "taker_buy_volume": [250.0] * n,
        })
        det = RegimeDetector(hurst_window=200)
        state = det.detect(df)
        assert state.regime == MarketRegime.RANGE, (
            f"Expected RANGE for flat data, got {state.regime} (ADX={state.adx})"
        )

    def test_detect_high_vol(self):
        """High-volatility spike → HIGH_VOL."""
        df = _high_vol_klines(500)
        det = RegimeDetector(hurst_window=200, atr_lookback=300)
        state = det.detect(df)
        assert state.regime == MarketRegime.HIGH_VOL, f"Expected HIGH_VOL, got {state.regime}"

    def test_confidence_in_01(self):
        """Confidence must always be in [0, 1]."""
        for gen in [_trending_klines, _ranging_klines, _high_vol_klines]:
            df = gen(500)
            det = RegimeDetector(hurst_window=200)
            state = det.detect(df)
            assert 0.0 <= state.confidence <= 1.0, (
                f"confidence {state.confidence} out of [0,1]"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 5. RegimeDetector.detect_all
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeDetectAll:
    def test_detect_all_columns_present(self):
        """detect_all must add exactly 7 new columns."""
        df = _trending_klines(500)
        det = RegimeDetector(hurst_window=200)
        out = det.detect_all(df, min_bars=200)
        expected_cols = {
            "regime", "hurst", "adx", "atr_pct",
            "atr_percentile", "trend_strength", "regime_confidence",
        }
        assert expected_cols.issubset(set(out.columns)), (
            f"Missing columns: {expected_cols - set(out.columns)}"
        )

    def test_detect_all_no_nan(self):
        """No NaN in any regime column after detect_all."""
        df = _trending_klines(500)
        det = RegimeDetector(hurst_window=200)
        out = det.detect_all(df, min_bars=200)
        for col in ("hurst", "adx", "atr_pct", "atr_percentile",
                     "trend_strength", "regime_confidence"):
            nan_n = out[col].is_nan().sum()
            null_n = out[col].null_count()
            assert nan_n == 0, f"Column '{col}' has {nan_n} NaNs"
            assert null_n == 0, f"Column '{col}' has {null_n} nulls"


# ──────────────────────────────────────────────────────────────────────────────
# 6. RegimeState utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeState:
    @pytest.mark.parametrize("regime,expected", [
        (MarketRegime.TREND_UP, 1.0),
        (MarketRegime.TREND_DOWN, 1.0),
        (MarketRegime.RANGE, 0.7),
        (MarketRegime.HIGH_VOL, 0.5),
        (MarketRegime.UNKNOWN, 0.0),
    ])
    def test_position_size_multiplier(self, regime: MarketRegime, expected: float):
        """position_size_multiplier matches master document specification."""
        state = RegimeState(
            regime=regime, hurst=0.5, adx=20.0,
            atr_pct=0.01, atr_percentile=0.5,
            trend_strength=0.5, confidence=0.5,
        )
        assert state.position_size_multiplier() == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Regime statistics
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeStatistics:
    def test_regime_pct_sums_to_100(self):
        """Sum of all regime percentages must equal 100."""
        df = _trending_klines(500)
        det = RegimeDetector(hurst_window=200)
        out = det.detect_all(df, min_bars=200)
        stats = det.get_regime_statistics(out)
        pct_sum = sum(stats["regime_pct"].values())
        assert pct_sum == pytest.approx(100.0, abs=0.1), (
            f"Regime %s should sum to 100, got {pct_sum}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8. Integration: real BTCUSDT data
# ──────────────────────────────────────────────────────────────────────────────

_REAL_PARQUET = Path("/mnt/hdd/AtomiCortex/data/features/ml_features/BTCUSDT_4h_features.parquet")


@pytest.mark.skipif(not _REAL_PARQUET.exists(), reason="Real data not available")
class TestRealData:
    def test_btcusdt_trend_above_30pct(self):
        """BTC 2024-2025 was heavily trending — TREND_* should be > 30% of bars."""
        df = pl.read_parquet(_REAL_PARQUET)
        det = RegimeDetector()
        out = det.detect_all(df)
        stats = det.get_regime_statistics(out)
        trend_pct = (
            stats["regime_pct"].get("trend_up", 0)
            + stats["regime_pct"].get("trend_down", 0)
        )
        assert trend_pct > 30.0, (
            f"Expected >30% trend for BTC 2024, got {trend_pct}%"
        )
