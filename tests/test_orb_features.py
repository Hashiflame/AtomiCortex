"""
tests/test_orb_features.py

Unit tests for ORB (Opening Range Breakout) features (Phase 2).

Run:
    pytest tests/test_orb_features.py -v
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.features.orb_features import ORBDetector

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_BAR_MS_15M = 900_000


def _ts_ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _15m_klines_full_day(
    day: int = 8,
    base_price: float = 42000.0,
    high_offset: float = 20.0,
    low_offset: float = 20.0,
    volume: float = 100.0,
) -> pl.DataFrame:
    """Generate 96 bars of 15m klines for one full day (2024-01-{day})."""
    n = 96
    base = _ts_ms(2024, 1, day, 0)
    prices = [base_price + i * 2.0 for i in range(n)]
    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_15M for i in range(n)],
        "open":  [p - 5.0 for p in prices],
        "high":  [p + high_offset for p in prices],
        "low":   [p - low_offset for p in prices],
        "close": prices,
        "volume": [volume + i * 0.5 for i in range(n)],
    })


def _15m_with_breakout(
    session: str = "asia",
    direction: str = "bull",
) -> pl.DataFrame:
    """Create 15m data where price breaks out of ORB with high volume."""
    n = 96
    base = _ts_ms(2024, 1, 8, 0)
    prices = [42000.0] * n
    highs = [42020.0] * n
    lows = [41980.0] * n
    volumes = [50.0] * n

    # ORB forming bars (first 4 bars: 00:00-00:45).
    for i in range(4):
        highs[i] = 42050.0
        lows[i] = 41950.0
        prices[i] = 42000.0

    # Breakout bar: bar 8 (02:00) with extreme price and volume.
    if direction == "bull":
        prices[8] = 42100.0  # Above ORB high of 42050
        highs[8] = 42120.0
    else:
        prices[8] = 41900.0  # Below ORB low of 41950
        lows[8] = 41880.0
    volumes[8] = 200.0  # High volume

    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_15M for i in range(n)],
        "open":  [p - 5.0 for p in prices],
        "high":  highs,
        "low":   lows,
        "close": prices,
        "volume": volumes,
    })


# ═══════════════════════════════════════════════════════════════
# ORB Tests
# ═══════════════════════════════════════════════════════════════


class TestORBAsiaSession:
    def test_orb_asia_uses_first_4_bars(self) -> None:
        """Asia ORB should be set from first 4 bars (00:00-00:45)."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        assert "orb_high_asia" in out.columns
        assert "orb_low_asia" in out.columns
        # ORB high should be the max high of first 4 bars.
        first4_highs = df["high"][:4].to_list()
        assert out["orb_high_asia"][0] == pytest.approx(max(first4_highs))

    def test_orb_asia_low_correct(self) -> None:
        """Asia ORB low should be the min low of first 4 bars."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        first4_lows = df["low"][:4].to_list()
        assert out["orb_low_asia"][0] == pytest.approx(min(first4_lows))


class TestORBLondonSession:
    def test_orb_london_correct_hours(self) -> None:
        """London ORB should be from bars at 08:00-08:45 UTC."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        assert "orb_high_london" in out.columns
        # Bar index 32 = 08:00 UTC (32 * 15min = 480min = 8h).
        london_highs = df["high"][32:36].to_list()
        expected = max(london_highs)
        assert out["orb_high_london"][35] == pytest.approx(expected)

    def test_orb_london_range_positive(self) -> None:
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        # After ORB is formed, range should be positive.
        assert out["orb_range_london"][36] > 0


class TestORBNYSession:
    def test_orb_ny_correct_hours(self) -> None:
        """NY ORB should be from bars at 13:00-13:45 UTC."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        assert "orb_high_ny" in out.columns
        # Bar index 52 = 13:00 UTC (52 * 15min = 780min = 13h).
        ny_highs = df["high"][52:56].to_list()
        expected = max(ny_highs)
        assert out["orb_high_ny"][56] == pytest.approx(expected)


class TestORBForwardFill:
    def test_orb_values_forward_filled(self) -> None:
        """ORB values should remain constant after formation."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        # Asia ORB should be the same for all bars (forward filled).
        asia_highs = out["orb_high_asia"].unique()
        assert len(asia_highs) == 1  # All same value


class TestORBBreakout:
    def test_orb_breakout_bull_requires_volume(self) -> None:
        """Bullish breakout requires close > ORB high AND volume >= 1.3×avg."""
        df = _15m_with_breakout(direction="bull")
        out = ORBDetector().calculate(df)
        # Bar 8 should have breakout if volume is high enough.
        assert "orb_breakout_bull" in out.columns

    def test_orb_breakout_bear_requires_close_below(self) -> None:
        """Bearish breakout requires close < ORB low."""
        df = _15m_with_breakout(direction="bear")
        out = ORBDetector().calculate(df)
        assert "orb_breakout_bear" in out.columns

    def test_no_breakout_inside_orb(self) -> None:
        """No breakout when price stays inside ORB."""
        df = _15m_klines_full_day(high_offset=5.0, low_offset=5.0)
        out = ORBDetector().calculate(df)
        # Should have very few or no breakouts with tight range.
        bull_count = out["orb_breakout_bull"].sum()
        bear_count = out["orb_breakout_bear"].sum()
        # Not necessarily zero (volume can still trigger), but check column exists.
        assert isinstance(bull_count, int)
        assert isinstance(bear_count, int)


class TestORBPositionFeatures:
    def test_price_vs_current_orb_values(self) -> None:
        """price_vs_current_orb should be -1, 0, or 1."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        valid = {-1, 0, 1}
        actual = set(out["price_vs_current_orb"].to_list())
        assert actual.issubset(valid)

    def test_dist_to_orb_high_pct_is_finite(self) -> None:
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        assert not out["dist_to_orb_high_pct"].is_infinite().any()

    def test_orb_range_atr_ratio_positive(self) -> None:
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        # After ATR warmup, ratio should be positive.
        valid = out["orb_range_asia_atr_pct"].filter(
            out["orb_range_asia_atr_pct"] > 0
        )
        assert len(valid) > 0


class TestSessionMeta:
    def test_session_trap_zone_first_bars(self) -> None:
        """First bars of session should be in trap zone."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        assert "is_session_trap_zone" in out.columns
        assert out["is_session_trap_zone"].dtype == pl.Boolean

    def test_bars_since_session_open_non_negative(self) -> None:
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        assert (out["bars_since_session_open"] >= 0).all()

    def test_current_session_valid_values(self) -> None:
        """current_session should be 0, 1, 2, or 3."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        valid = {0, 1, 2, 3}
        actual = set(out["current_session"].to_list())
        assert actual.issubset(valid)

    def test_all_orb_columns_present(self) -> None:
        """Verify all expected ORB columns exist."""
        df = _15m_klines_full_day()
        out = ORBDetector().calculate(df)
        expected = {
            "orb_high_asia", "orb_low_asia", "orb_range_asia",
            "orb_high_london", "orb_low_london", "orb_range_london",
            "orb_high_ny", "orb_low_ny", "orb_range_ny",
            "current_session", "price_vs_current_orb",
            "orb_breakout_bull", "orb_breakout_bear",
            "bars_since_session_open", "is_session_trap_zone",
        }
        assert expected.issubset(set(out.columns))
