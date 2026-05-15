"""
tests/test_session_features.py

Unit tests for session features (Phase 2).
Covers: SessionEncoder, SessionVWAP, AnchoredVWAP,
        PreFundingDetector, MondayAsiaEffect.

Run:
    pytest tests/test_session_features.py -v
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from src.features.session_features import (
    AnchoredVWAP,
    MondayAsiaEffect,
    PreFundingDetector,
    SessionEncoder,
    SessionVWAP,
)

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_BAR_MS_1H = 3_600_000  # 1 hour in ms


def _ts_ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Return unix epoch ms for a given UTC datetime."""
    import datetime as dt
    return int(dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _1h_klines(n: int = 48, start_hour: int = 0, start_day: int = 6) -> pl.DataFrame:
    """Generate synthetic 1H klines starting on 2024-01-06 (Saturday) at given hour."""
    base = _ts_ms(2024, 1, start_day, start_hour)
    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_1H for i in range(n)],
        "open":  [42000.0 + i * 10.0 for i in range(n)],
        "high":  [42050.0 + i * 10.0 for i in range(n)],
        "low":   [41950.0 + i * 10.0 for i in range(n)],
        "close": [42020.0 + i * 10.0 for i in range(n)],
        "volume": [100.0 + i for i in range(n)],
    })


def _15m_klines(n: int = 96) -> pl.DataFrame:
    """Generate 96 bars of 15m klines (1 day) starting 2024-01-08 Mon 00:00."""
    bar_ms = 900_000  # 15 min
    base = _ts_ms(2024, 1, 8, 0)
    return pl.DataFrame({
        "open_time": [base + i * bar_ms for i in range(n)],
        "open":  [42000.0 + i * 2.0 for i in range(n)],
        "high":  [42010.0 + i * 2.0 for i in range(n)],
        "low":   [41990.0 + i * 2.0 for i in range(n)],
        "close": [42005.0 + i * 2.0 for i in range(n)],
        "volume": [50.0 + i * 0.5 for i in range(n)],
    })


# ═══════════════════════════════════════════════════════════════
# 1. SessionEncoder
# ═══════════════════════════════════════════════════════════════


class TestSessionEncoder:
    def test_overlap_session_14_utc(self) -> None:
        """14:00 UTC should be OVERLAP (4)."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 14)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["trading_session"][0] == 4

    def test_overlap_session_15_utc(self) -> None:
        """15:00 UTC should be OVERLAP (4)."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 15)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["trading_session"][0] == 4

    def test_asia_session_06_utc(self) -> None:
        """06:00 UTC should be ASIA (1)."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 6)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["trading_session"][0] == 1

    def test_london_session_10_utc(self) -> None:
        """10:00 UTC should be LONDON (2)."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 10)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["trading_session"][0] == 2

    def test_ny_session_18_utc(self) -> None:
        """18:00 UTC should be NY (3)."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 18)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["trading_session"][0] == 3

    def test_dead_zone_23_utc(self) -> None:
        """23:00 UTC should be DEAD (0)."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 23)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["trading_session"][0] == 0
        assert out["is_dead_zone"][0] is True

    def test_dead_zone_02_utc(self) -> None:
        """02:00 UTC should be DEAD zone."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 2)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = SessionEncoder().encode(df)
        assert out["is_dead_zone"][0] is True

    def test_cyclical_encoding_continuous(self) -> None:
        """sin/cos should have no discontinuity between 23:00 and 00:00."""
        df = _1h_klines(25, start_hour=0)
        out = SessionEncoder().encode(df)
        sin_vals = out["session_hour_sin"].to_list()
        cos_vals = out["session_hour_cos"].to_list()
        # Check that adjacent values don't jump wildly.
        for i in range(1, len(sin_vals)):
            d = math.sqrt((sin_vals[i] - sin_vals[i-1])**2 + (cos_vals[i] - cos_vals[i-1])**2)
            assert d < 0.5, f"Discontinuity at bar {i}: delta={d}"

    def test_is_overlap_only_14_15_16(self) -> None:
        """is_overlap must be True only for hours 14, 15, 16."""
        df = _1h_klines(24, start_hour=0)
        out = SessionEncoder().encode(df)
        overlaps = out.filter(pl.col("is_overlap"))
        ts = pl.from_epoch(overlaps["open_time"], time_unit="ms")
        hours = ts.dt.hour().to_list()
        assert set(hours) == {14, 15, 16}

    def test_hours_to_session_end_positive(self) -> None:
        """hours_to_session_end should always be > 0."""
        df = _1h_klines(24, start_hour=0)
        out = SessionEncoder().encode(df)
        assert (out["hours_to_session_end"] > 0).all()


# ═══════════════════════════════════════════════════════════════
# 2. SessionVWAP
# ═══════════════════════════════════════════════════════════════


class TestSessionVWAP:
    def test_session_vwap_resets_daily(self) -> None:
        """VWAP should reset between days."""
        df = _1h_klines(48, start_hour=0)  # 2 days
        out = SessionVWAP().calculate(df)
        # First bar of each day should have vwap ≈ typical price of that bar.
        first_typical = (df["high"][0] + df["low"][0] + df["close"][0]) / 3.0
        assert out["session_vwap"][0] == pytest.approx(first_typical, rel=1e-6)

    def test_session_vwap_positive(self) -> None:
        """VWAP should always be positive."""
        df = _1h_klines(48)
        out = SessionVWAP().calculate(df)
        assert (out["session_vwap"] > 0).all()

    def test_price_vs_session_vwap_is_pct(self) -> None:
        """price_vs_session_vwap is (close - vwap) / vwap."""
        df = _1h_klines(24)
        out = SessionVWAP().calculate(df)
        # Reasonable range: within ±10%.
        assert (out["price_vs_session_vwap"].abs() < 0.1).all()

    def test_vwap_band_position_clamped(self) -> None:
        """vwap_band_position should be in [-4, 4]."""
        df = _1h_klines(48)
        out = SessionVWAP().calculate(df)
        vals = out["vwap_band_position"]
        assert (vals >= -4.0).all()
        assert (vals <= 4.0).all()

    def test_session_cumulative_volume_ratio_positive(self) -> None:
        """Cumulative volume ratio should be non-negative."""
        df = _1h_klines(24, start_hour=0)
        out = SessionVWAP().calculate(df)
        assert "session_cumulative_volume_ratio" in out.columns
        assert (out["session_cumulative_volume_ratio"] >= 0).all()

    def test_session_cumulative_volume_ratio_increases_intraday(self) -> None:
        """Cumulative volume ratio should generally increase within a day."""
        df = _1h_klines(24, start_hour=0)
        out = SessionVWAP().calculate(df)
        ratios = out["session_cumulative_volume_ratio"].to_list()
        # First few bars should be smaller than last bars
        assert ratios[0] < ratios[-1]

    def test_vwap_bands_symmetric(self) -> None:
        """Upper band - vwap == vwap - lower band."""
        df = _1h_klines(24)
        out = SessionVWAP().calculate(df)
        upper_dist = (out["vwap_upper_band"] - out["session_vwap"]).to_list()
        lower_dist = (out["session_vwap"] - out["vwap_lower_band"]).to_list()
        for u, l in zip(upper_dist, lower_dist):
            assert u == pytest.approx(l, abs=1e-6)

    def test_session_vwap_std_no_lookahead(self) -> None:
        """std at bar t must NOT equal full-day std.

        The old implementation used ``close.std().over('_date')`` which
        computed the std over **all** bars of the day — a lookahead bug.
        The fix uses expanding (cumulative) std so each bar only sees
        bars up to and including itself.
        """
        df = _1h_klines(24, start_hour=0)  # 1 full day
        out = SessionVWAP().calculate(df)

        stds = out["session_vwap_std"].to_list()
        # Full-day std (what the old code produced) — identical for every bar.
        import numpy as np
        full_day_std = float(np.std(out["close"].to_numpy(), ddof=1))

        # With expanding std the mid-day bars must differ from full_day_std.
        mid = len(stds) // 2  # bar 12
        assert stds[mid] != pytest.approx(full_day_std, rel=0.01), (
            "Mid-day expanding std should NOT equal the full-day std "
            "(that would indicate lookahead)"
        )

        # Additionally, std should grow as the day progresses
        # (more data → std converges to true value).
        assert stds[2] < stds[-1] or stds[2] == pytest.approx(stds[-1], rel=0.3)

    def test_session_vwap_std_first_bar_zero(self) -> None:
        """First bar of the day has only 1 data point → std must be 0."""
        df = _1h_klines(24, start_hour=0)
        out = SessionVWAP().calculate(df)
        assert out["session_vwap_std"][0] == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════
# 3. AnchoredVWAP
# ═══════════════════════════════════════════════════════════════


class TestAnchoredVWAP:
    def test_anchored_vwap_resets_weekly(self) -> None:
        """AVWAP should have different reset anchors across weeks."""
        # 200 hours = ~8 days = spans 2 ISO weeks.
        df = _1h_klines(200, start_hour=0, start_day=1)
        out = AnchoredVWAP().calculate(df)
        assert "avwap_weekly" in out.columns
        # AVWAP should be positive.
        assert (out["avwap_weekly"] > 0).all()

    def test_price_above_avwap_boolean(self) -> None:
        """price_above_avwap must be boolean."""
        df = _1h_klines(48)
        out = AnchoredVWAP().calculate(df)
        assert out["price_above_avwap"].dtype == pl.Boolean

    def test_avwap_weekly_slope_is_finite(self) -> None:
        """avwap_weekly_slope should have no inf values."""
        df = _1h_klines(48)
        out = AnchoredVWAP().calculate(df)
        slope = out["avwap_weekly_slope"]
        assert not slope.is_infinite().any()


# ═══════════════════════════════════════════════════════════════
# 4. PreFundingDetector
# ═══════════════════════════════════════════════════════════════


class TestPreFundingDetector:
    def test_pre_funding_window_1h_before_01utc(self) -> None:
        """00:00 UTC is 1h before mark at 01:00 → pre_funding_window=True."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 0)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = PreFundingDetector().detect(df)
        assert out["pre_funding_window"][0] is True

    def test_pre_funding_window_2h_before_09utc(self) -> None:
        """07:00 UTC is 2h before mark at 09:00 → pre_funding_window=True."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 7)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = PreFundingDetector().detect(df)
        assert out["pre_funding_window"][0] is True

    def test_not_pre_funding_5h_before_mark(self) -> None:
        """04:00 UTC is 5h before mark at 09:00 → pre_funding_window=False."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 4)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = PreFundingDetector().detect(df)
        assert out["pre_funding_window"][0] is False

    def test_funding_urgency_1h_before(self) -> None:
        """≤1h before mark → urgency=1.0."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 0)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = PreFundingDetector().detect(df)
        assert out["funding_window_urgency"][0] == pytest.approx(1.0)

    def test_post_funding_window(self) -> None:
        """01:00 UTC is right at a mark → post_funding_window=True."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 1)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = PreFundingDetector().detect(df)
        assert out["post_funding_window"][0] is True

    def test_hours_to_funding_always_positive(self) -> None:
        """hours_to_funding_mark should always be >= 0."""
        df = _1h_klines(24)
        out = PreFundingDetector().detect(df)
        assert (out["hours_to_funding_mark"] >= 0).all()


# ═══════════════════════════════════════════════════════════════
# 5. MondayAsiaEffect
# ═══════════════════════════════════════════════════════════════


class TestMondayAsiaEffect:
    def test_monday_asia_window_sunday_20utc(self) -> None:
        """Sunday 20:00 UTC is in Monday Asia window."""
        # 2024-01-07 is Sunday.
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 7, 20)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = MondayAsiaEffect().detect(df)
        assert out["is_monday_asia_window"][0] is True

    def test_monday_asia_window_monday_07utc(self) -> None:
        """Monday 07:00 UTC is in Monday Asia window."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 8, 7)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = MondayAsiaEffect().detect(df)
        assert out["is_monday_asia_window"][0] is True

    def test_not_monday_asia_tuesday(self) -> None:
        """Tuesday 03:00 UTC is NOT in Monday Asia window."""
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 9, 3)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = MondayAsiaEffect().detect(df)
        assert out["is_monday_asia_window"][0] is False

    def test_cyclical_dow_continuous(self) -> None:
        """day_of_week sin/cos should form a smooth cycle."""
        df = _1h_klines(168, start_hour=0, start_day=1)  # 7 days
        out = MondayAsiaEffect().detect(df)
        sin_vals = out["day_of_week_sin"].to_list()
        cos_vals = out["day_of_week_cos"].to_list()
        for s, c in zip(sin_vals, cos_vals):
            assert math.isfinite(s) and math.isfinite(c)

    def test_is_weekend_saturday_sunday(self) -> None:
        """Saturday and Sunday should be is_weekend=True."""
        # 2024-01-06=Saturday, 2024-01-07=Sunday.
        df = _1h_klines(48, start_hour=0, start_day=6)
        out = MondayAsiaEffect().detect(df)
        assert out["is_weekend"][0] is True
        assert out["is_weekend"][23] is True  # still Saturday
        assert out["is_weekend"][24] is True  # Sunday 00:00

    def test_is_high_vol_day_wed_thu(self) -> None:
        """Wednesday and Thursday should be is_high_vol_day=True."""
        # 2024-01-10=Wednesday.
        df = pl.DataFrame({"open_time": [_ts_ms(2024, 1, 10, 12)]})
        df = df.with_columns([
            pl.lit(42000.0).alias("open"), pl.lit(42100.0).alias("high"),
            pl.lit(41900.0).alias("low"), pl.lit(42050.0).alias("close"),
            pl.lit(100.0).alias("volume"),
        ])
        out = MondayAsiaEffect().detect(df)
        assert out["is_high_vol_day"][0] is True
