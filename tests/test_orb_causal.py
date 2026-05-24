"""Causality (no-lookahead) tests for ``ORBDetector``.

Pre-fix, ``_compute_session_orb`` used ``max/min().over("_date")`` which
broadcast the full-day ORB back to *every* row in the day — so bar 00:00
already "knew" the highs of bars 00:30 and 00:45. That was the chief
cause of the 76.6 % WR mirage on the 15m model.

Post-fix:
* In-window bars  → ``cum_max / cum_min`` (expanding so far) — bar T only
  sees data with ``open_time <= T``.
* Post-window bars → frozen at the final ORB (every ORB bar has closed
  by then, so reading the full max is legitimate).
* Pre-window bars  → forward-filled from yesterday's ORB.

These tests pin those invariants symbol-by-bar so a regression cannot
hide behind aggregate statistics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from src.features.orb_features import ORBDetector


_BAR_MS_15M = 15 * 60 * 1000
_UTC = timezone.utc


def _ts_ms(year: int, month: int, day: int, hour: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, tzinfo=_UTC).timestamp() * 1000
    )


def _asia_session_df(
    asia_highs: list[float],
    asia_lows: list[float],
    *,
    n: int = 96,
    other_high: float = 41_900.0,
    other_low: float = 41_800.0,
) -> pl.DataFrame:
    """One full UTC day of 15m bars; Asia ORB highs/lows controlled."""
    assert len(asia_highs) >= 4 and len(asia_lows) >= 4
    base = _ts_ms(2024, 1, 8, 0)
    highs = [other_high] * n
    lows = [other_low] * n
    closes = [(other_high + other_low) / 2] * n
    for i in range(4):
        highs[i] = asia_highs[i]
        lows[i] = asia_lows[i]
        closes[i] = (asia_highs[i] + asia_lows[i]) / 2
    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_15M for i in range(n)],
        "open":  [c - 5.0 for c in closes],
        "high":  highs,
        "low":   lows,
        "close": closes,
        "volume": [50.0] * n,
    })


# ---------------------------------------------------------------------------
# In-window expanding semantics — the canonical lookahead test
# ---------------------------------------------------------------------------

class TestExpandingWithinORBWindow:
    def test_bar_0_does_not_see_later_bars(self) -> None:
        """Bar 00:00 must reflect ONLY bar 0 — it cannot know bars 1/2/3."""
        # Later bars are deliberately higher so the buggy broadcast would
        # surface at bar 0 (==max(50, 60, 70, 80) = 80) whereas the causal
        # value is 50 (bar 0's own high).
        df = _asia_session_df(
            asia_highs=[50.0, 60.0, 70.0, 80.0],
            asia_lows=[45.0, 40.0, 35.0, 30.0],
        )
        out = ORBDetector().calculate(df)
        assert out["orb_high_asia"][0] == pytest.approx(50.0)
        assert out["orb_low_asia"][0] == pytest.approx(45.0)

    def test_bar_1_sees_only_bars_0_and_1(self) -> None:
        df = _asia_session_df(
            asia_highs=[50.0, 60.0, 70.0, 80.0],
            asia_lows=[45.0, 40.0, 35.0, 30.0],
        )
        out = ORBDetector().calculate(df)
        assert out["orb_high_asia"][1] == pytest.approx(60.0)  # max(50, 60)
        assert out["orb_low_asia"][1] == pytest.approx(40.0)   # min(45, 40)

    def test_bar_2_sees_only_bars_0_1_2(self) -> None:
        df = _asia_session_df(
            asia_highs=[50.0, 60.0, 70.0, 80.0],
            asia_lows=[45.0, 40.0, 35.0, 30.0],
        )
        out = ORBDetector().calculate(df)
        assert out["orb_high_asia"][2] == pytest.approx(70.0)
        assert out["orb_low_asia"][2] == pytest.approx(35.0)

    def test_bar_3_completes_orb(self) -> None:
        df = _asia_session_df(
            asia_highs=[50.0, 60.0, 70.0, 80.0],
            asia_lows=[45.0, 40.0, 35.0, 30.0],
        )
        out = ORBDetector().calculate(df)
        assert out["orb_high_asia"][3] == pytest.approx(80.0)
        assert out["orb_low_asia"][3] == pytest.approx(30.0)

    def test_expanding_max_is_monotonic_within_window(self) -> None:
        """orb_high cannot decrease while inside the ORB window."""
        df = _asia_session_df(
            asia_highs=[50.0, 60.0, 70.0, 80.0],
            asia_lows=[45.0, 40.0, 35.0, 30.0],
        )
        out = ORBDetector().calculate(df)
        window_high = out["orb_high_asia"][:4].to_list()
        assert window_high == sorted(window_high)
        window_low = out["orb_low_asia"][:4].to_list()
        # min sequence is monotonically non-increasing
        assert window_low == sorted(window_low, reverse=True)


# ---------------------------------------------------------------------------
# Post-window: frozen at the final ORB
# ---------------------------------------------------------------------------

class TestPostWindowFrozen:
    def test_post_window_equals_full_orb(self) -> None:
        asia_highs = [50.0, 60.0, 70.0, 80.0]
        asia_lows = [45.0, 40.0, 35.0, 30.0]
        df = _asia_session_df(asia_highs=asia_highs, asia_lows=asia_lows)
        out = ORBDetector().calculate(df)
        # After ORB window closes (bar 4 onward, within same day) value
        # equals max/min of all 4 ORB bars.
        assert out["orb_high_asia"][4] == pytest.approx(max(asia_highs))
        assert out["orb_low_asia"][4] == pytest.approx(min(asia_lows))

    def test_post_window_value_does_not_drift(self) -> None:
        """Later non-ORB bars cannot revise ORB."""
        df = _asia_session_df(
            asia_highs=[50.0, 60.0, 70.0, 80.0],
            asia_lows=[45.0, 40.0, 35.0, 30.0],
            # Other bars feature highs > 80 — these MUST NOT bleed into ORB.
            other_high=200.0, other_low=10.0,
        )
        out = ORBDetector().calculate(df)
        # All post-window bars same as the final ORB.
        for i in range(4, 32):  # bars 4..31 still within "asia day window"
            assert out["orb_high_asia"][i] == pytest.approx(80.0)
            assert out["orb_low_asia"][i] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Breakout signal cannot fire while ORB is still forming
# ---------------------------------------------------------------------------

class TestBreakoutCausality:
    def test_no_premature_breakout_on_forming_bars(self) -> None:
        """Set up bar 0's high so the buggy code would say 'breakout'
        (close > buggy-orb-high). With the fix, orb_high at bar 0 is
        only bar 0's own high, so close cannot exceed it on the same bar."""
        # Force a "breakout-shaped" trap: bar 1 close above bar 0 high,
        # but bar 2 high is much higher. Pre-fix orb_high at bar 1 was
        # max(0,1,2,3) — buggy could either suppress or fire breakout
        # by reading the future. Post-fix orb_high at bar 1 is max(h0,h1)
        # and we ensure the close strictly cannot exceed that without
        # the bar's own range covering it.
        n = 96
        base = _ts_ms(2024, 1, 8, 0)
        highs = [42_000.0] * n
        lows = [41_900.0] * n
        closes = [41_950.0] * n
        # Force later ORB bars to dominate, ensuring buggy code would
        # see large later highs at bar 1.
        highs[0], highs[1], highs[2], highs[3] = 42_050, 42_060, 42_500, 42_600
        lows[0], lows[1], lows[2], lows[3] = 41_950, 41_940, 41_500, 41_400
        closes[0], closes[1] = 42_000, 42_010
        # Average volume baseline; bar 1 must NOT have breakout regardless.
        vols = [50.0] * n
        vols[1] = 1_000.0  # would-be confirmation
        df = pl.DataFrame({
            "open_time": [base + i * _BAR_MS_15M for i in range(n)],
            "open":  [c - 5.0 for c in closes],
            "high":  highs,
            "low":   lows,
            "close": closes,
            "volume": vols,
        })
        out = ORBDetector().calculate(df)
        # At bar 1: orb_high_asia = max(highs[0], highs[1]) = 42_060.
        # close[1] = 42_010 < 42_060 → no bull breakout.
        assert out["orb_breakout_bull"][1] is False or not out["orb_breakout_bull"][1]


# ---------------------------------------------------------------------------
# Multi-day isolation — yesterday's ORB cannot bleed into today's expanding
# ---------------------------------------------------------------------------

class TestMultiDayIsolation:
    def test_today_expanding_does_not_include_yesterday(self) -> None:
        """Across two days, today's bar 0 sees only today's bar 0 high
        (not yesterday's). Forward-fill carries yesterday's frozen ORB
        through pre-window bars of today, but the expanding cum_max
        restarts inside today's window."""
        n_per_day = 96
        days_df = []
        for day, top in [(8, 100.0), (9, 200.0)]:
            base = _ts_ms(2024, 1, day, 0)
            highs = [50.0] * n_per_day
            lows = [40.0] * n_per_day
            highs[0:4] = [top, top + 10, top + 20, top + 30]
            lows[0:4] = [top - 50, top - 60, top - 70, top - 80]
            closes = [(h + l) / 2 for h, l in zip(highs, lows)]
            days_df.append(pl.DataFrame({
                "open_time": [base + i * _BAR_MS_15M for i in range(n_per_day)],
                "open":  [c - 5.0 for c in closes],
                "high":  highs,
                "low":   lows,
                "close": closes,
                "volume": [50.0] * n_per_day,
            }))
        df = pl.concat(days_df)
        out = ORBDetector().calculate(df)
        # Day 2 bar 0 is at index n_per_day. Its orb_high is bar's own high
        # (200), NOT max with day 1's frozen ORB (which was 130).
        idx_day2_bar0 = n_per_day
        assert out["orb_high_asia"][idx_day2_bar0] == pytest.approx(200.0)
        # Day 2 bar 3 carries day 2's full ORB max.
        assert out["orb_high_asia"][idx_day2_bar0 + 3] == pytest.approx(230.0)
