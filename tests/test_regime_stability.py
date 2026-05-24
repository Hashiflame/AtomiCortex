"""Tests for Step H3 — regime trend-direction stability.

Old behavior: TREND_UP/TREND_DOWN flipped on sign(close[t]-close[t-1])
in choppy markets. New behavior: direction = sign of EMA(close, span=
adx_period) slope over max(2, adx_period//3) bars — stable, no lookahead.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeDetector1H,
    RegimeDetector15M,
)


def _df_from_close(close: np.ndarray) -> pl.DataFrame:
    n = len(close)
    # Tight bars (low ATR) so HIGH_VOL doesn't take over.
    high = close + np.abs(close) * 0.0005
    low = close - np.abs(close) * 0.0005
    return pl.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * 4 * 3_600_000,
        "open":  close,
        "high":  high,
        "low":   low,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def _count_trend_flips(regimes: list[str]) -> int:
    flips = 0
    prev = None
    for r in regimes:
        if r in ("trend_up", "trend_down"):
            if prev is not None and r != prev:
                flips += 1
            prev = r
    return flips


def _count_naive_sign_flips(close: np.ndarray) -> int:
    """How many times sign(close[t]-close[t-1]) flipped — the old behavior."""
    diffs = np.diff(close)
    signs = np.sign(diffs)
    flips = 0
    last = 0
    for s in signs:
        if s != 0 and s != last:
            if last != 0:
                flips += 1
            last = s
    return flips


# ---------------------------------------------------------------------------
# Direct unit test on _classify / _ema
# ---------------------------------------------------------------------------


class TestEmaAndSlopeWindow:
    def test_ema_basic(self):
        d = RegimeDetector(adx_period=14)
        ema = d._ema(np.array([1.0, 2.0, 3.0, 4.0]), span=3)
        assert ema[0] == pytest.approx(1.0)
        assert ema[3] > ema[0]
        assert len(ema) == 4

    def test_ema_short_circuits_for_span_1(self):
        d = RegimeDetector()
        x = np.array([5.0, 1.0, 7.0])
        np.testing.assert_array_equal(d._ema(x, span=1), x)

    @pytest.mark.parametrize("adx_period,expected_w", [
        (14, 4),  # 4H default
        (10, 3),  # 1H/15m
        (6, 2),
        (3, 2),   # floor
    ])
    def test_slope_window_scales(self, adx_period, expected_w):
        d = RegimeDetector(adx_period=adx_period)
        assert d._trend_slope_window() == expected_w


# ---------------------------------------------------------------------------
# Stability — flicker tests on synthetic series
# ---------------------------------------------------------------------------


class TestRegimeStabilityChoppy:
    def test_choppy_uptrend_does_not_flicker_every_bar(self):
        """Strong drift + alternating-bar noise.

        Old code would flip TREND_UP/DOWN on every bar of noise. New code
        should stay mostly TREND_UP (drift dominates the EMA slope).
        """
        n = 400
        rng = np.random.default_rng(42)
        # Strong drift + symmetric per-bar noise that crosses zero often.
        drift = np.linspace(100.0, 130.0, n)
        noise = rng.normal(0, 0.4, n)
        # Alternating saw on top so sign(close[t]-close[t-1]) flips a LOT.
        saw = 0.3 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        close = drift + noise + saw

        det = RegimeDetector(hurst_window=100, adx_trend_threshold=10.0)
        out = det.detect_all(_df_from_close(close), min_bars=50)
        regimes = out["regime"].to_list()

        naive_flips = _count_naive_sign_flips(close)
        trend_flips = _count_trend_flips(regimes)

        # Naive sign flips ≈ n/2 (~200). New regime should be massively
        # more stable — at most a small fraction of that.
        assert naive_flips > 100, f"sanity: noise should flip a lot ({naive_flips})"
        assert trend_flips < naive_flips // 10, (
            f"regime flickered too much: trend_flips={trend_flips}, "
            f"naive_flips={naive_flips}"
        )

    def test_pure_uptrend_settles_to_trend_up(self):
        n = 200
        close = np.linspace(100.0, 200.0, n)
        # Disable HIGH_VOL gate so the constant-ATR fixture doesn't flip
        # the regime — we're testing direction, not vol.
        det = RegimeDetector(
            hurst_window=80, adx_trend_threshold=10.0, atr_vol_threshold=1.01,
        )
        out = det.detect_all(_df_from_close(close), min_bars=50)
        tail = out["regime"].to_list()[-30:]
        assert tail.count("trend_up") == 30, f"tail={set(tail)}"

    def test_pure_downtrend_settles_to_trend_down(self):
        n = 200
        close = np.linspace(200.0, 100.0, n)
        det = RegimeDetector(
            hurst_window=80, adx_trend_threshold=10.0, atr_vol_threshold=1.01,
        )
        out = det.detect_all(_df_from_close(close), min_bars=50)
        tail = out["regime"].to_list()[-30:]
        assert tail.count("trend_down") == 30, f"tail={set(tail)}"

    def test_reversal_switches_direction_within_bounded_window(self):
        """Up 100 bars, then down 100 bars. New regime must reach
        TREND_DOWN within a bounded transition window (not instantly,
        but well before the new leg ends)."""
        n_each = 100
        up = np.linspace(100.0, 200.0, n_each)
        down = np.linspace(200.0, 100.0, n_each)
        close = np.concatenate([up, down])

        det = RegimeDetector(hurst_window=60, adx_trend_threshold=10.0)
        out = det.detect_all(_df_from_close(close), min_bars=40)
        regimes = out["regime"].to_list()

        # By the end of the down-leg we must be TREND_DOWN.
        assert regimes[-1] == "trend_down"
        # The first TREND_DOWN bar appears within `adx_period + slope_window
        # + cushion` bars after the turning point (idx=n_each). Cushion
        # accounts for ADX warmup on the new leg.
        w = det._trend_slope_window()
        budget = det.adx_period + w + 20
        switched_at = next(
            (i for i in range(n_each, len(regimes)) if regimes[i] == "trend_down"),
            None,
        )
        assert switched_at is not None, "never switched to TREND_DOWN"
        assert switched_at - n_each <= budget, (
            f"switch took {switched_at - n_each} bars > budget {budget}"
        )


class TestHighVolAndRangeUntouched:
    def test_high_vol_takes_precedence_over_trend_direction(self):
        """A series with a recent volatility spike should be HIGH_VOL
        regardless of the new EMA-slope rule."""
        n = 300
        rng = np.random.default_rng(0)
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        # Spike volatility in the last 10 bars: huge high-low ranges.
        df = _df_from_close(close)
        high = df["high"].to_numpy().copy()
        low = df["low"].to_numpy().copy()
        # Massive bars near the end → ATR percentile → 1.0.
        high[-10:] = close[-10:] + 50.0
        low[-10:] = close[-10:] - 50.0
        df = df.with_columns([
            pl.Series("high", high),
            pl.Series("low", low),
        ])

        det = RegimeDetector(
            hurst_window=80, atr_lookback=100, atr_vol_threshold=0.80,
        )
        out = det.detect_all(df, min_bars=50)
        assert out["regime"].to_list()[-1] == "high_vol"

    def test_low_adx_stays_range(self):
        """Flat market → ADX low → RANGE regardless of EMA slope."""
        n = 300
        rng = np.random.default_rng(1)
        # Non-cumulative noise around a constant level — keeps ADX low.
        close = 100.0 + rng.normal(0, 0.05, n)
        det = RegimeDetector(hurst_window=80)  # default adx_trend_threshold=20
        out = det.detect_all(_df_from_close(close), min_bars=50)
        tail = out["regime"].to_list()[-75:]
        # No directional regime should appear — only RANGE (or HIGH_VOL on
        # fixture-specific spikes; we only assert "no trend").
        trend_count = tail.count("trend_up") + tail.count("trend_down")
        assert trend_count == 0, f"unexpected trend bars in flat fixture: {tail}"


# ---------------------------------------------------------------------------
# Per-TF detectors inherit the fix (no extra wiring required)
# ---------------------------------------------------------------------------


class TestSubclassesInheritFix:
    @pytest.mark.parametrize("cls", [RegimeDetector1H, RegimeDetector15M])
    def test_subclass_does_not_flicker(self, cls):
        n = 400
        rng = np.random.default_rng(7)
        drift = np.linspace(100.0, 120.0, n)
        noise = rng.normal(0, 0.3, n)
        saw = 0.25 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        close = drift + noise + saw

        det = cls(adx_trend_threshold=10.0)
        out = det.detect_all(_df_from_close(close), min_bars=det.hurst_window)
        regimes = out["regime"].to_list()
        naive = _count_naive_sign_flips(close)
        flips = _count_trend_flips(regimes)
        # 15m uses adx_period=7 → slope window = 2 (faster by design), so
        # it tolerates more flips than the 4H/1H detectors. Still must be
        # ≥5× more stable than the naive single-bar sign.
        assert flips < naive // 5, (
            f"{cls.__name__} flickered: flips={flips}, naive={naive}"
        )


# ---------------------------------------------------------------------------
# Backward-compat smoke
# ---------------------------------------------------------------------------


class TestBackwardCompatColumns:
    def test_detect_returns_regime_state_unchanged(self):
        close = np.linspace(100.0, 150.0, 100)
        det = RegimeDetector(hurst_window=50)
        state = det.detect(_df_from_close(close), idx=-1)
        # All the legacy fields are present and well-typed.
        assert isinstance(state.regime, MarketRegime)
        for attr in ("hurst", "adx", "atr_pct", "atr_percentile",
                     "trend_strength", "confidence"):
            assert isinstance(getattr(state, attr), float)

    def test_detect_all_columns_unchanged(self):
        close = np.linspace(100.0, 150.0, 100)
        det = RegimeDetector(hurst_window=50)
        out = det.detect_all(_df_from_close(close), min_bars=50)
        for col in ("regime", "hurst", "adx", "atr_pct", "atr_percentile",
                    "trend_strength", "regime_confidence"):
            assert col in out.columns
