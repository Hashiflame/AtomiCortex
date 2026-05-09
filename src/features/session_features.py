"""
src/features/session_features.py

Session-specific features for 1H and 15m trading strategies.

These features capture intraday market structure that doesn't
meaningfully exist on 4H timeframe.

Academic basis:
- ScienceDirect 2024: trading activity peaks 14:00-17:00 UTC
  across 1940 pairs on 38 exchanges
- Concretum Group 2026: Monday Asia Open Effect, BTC 2018-2025
- Pre-funding patterns: position squeezing 1-2H before marks

All methods take Polars DataFrame and return Polars DataFrame.
No side effects, pure transformations.
"""

from __future__ import annotations

import math

import polars as pl

from src.features.utils import safe_divide
from src.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi

# Session boundaries (UTC hours) — priority: OVERLAP > NY > LONDON > ASIA > DEAD
# Maps hour (0-23) → session int (0=DEAD, 1=ASIA, 2=LONDON, 3=NY, 4=OVERLAP)
_SESSION_MAP: list[int] = [
    0, 0, 0, 0, 0,           # 00-04: DEAD
    1, 1, 1,                  # 05-07: ASIA
    2, 2, 2, 2, 2,           # 08-12: LONDON
    3,                        # 13:    NY
    4, 4, 4,                  # 14-16: OVERLAP
    3, 3, 3, 3, 3,           # 17-21: NY
    0, 0,                     # 22-23: DEAD
]

# Hours remaining until current session ends, indexed by hour (0-23).
_HOURS_TO_END: list[float] = [
    5, 4, 3, 2, 1,           # DEAD → ends at 05
    3, 2, 1,                  # ASIA → ends at 08
    5, 4, 3, 2, 1,           # LONDON → ends at 13
    9,                        # NY → ends at 22
    3, 2, 1,                  # OVERLAP → ends at 17
    5, 4, 3, 2, 1,           # NY → ends at 22
    7, 6,                     # DEAD → ends at 05 next day
]

# Funding marks at 01:00, 09:00, 17:00 UTC (every 8 hours)
_FUNDING_MARKS = [1, 9, 17]
_PRE_FUNDING_WINDOW_H = 2


# ---------------------------------------------------------------------------
# Helper: derive _ts (Datetime) from open_time (Int64 unix ms)
# ---------------------------------------------------------------------------

def _ensure_ts(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``_ts`` column (Datetime) derived from ``open_time``."""
    if "_ts" not in df.columns:
        df = df.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("_ts")
        )
    return df


# ═══════════════════════════════════════════════════════════════
# 1. SessionEncoder
# ═══════════════════════════════════════════════════════════════


class SessionEncoder:
    """Encode the trading session and cyclical time features.

    Added columns:
        trading_session, session_hour_sin, session_hour_cos,
        is_overlap, is_dead_zone, hours_to_session_end
    """

    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add session-encoding columns to *df*."""
        df = _ensure_ts(df)
        hour = pl.col("_ts").dt.hour()

        # Build mapping Series for session and hours_to_end.
        session_map = pl.Series("_s", _SESSION_MAP, dtype=pl.Int32)
        hours_map = pl.Series("_h", _HOURS_TO_END, dtype=pl.Float64)

        df = df.with_columns([
            hour.replace(
                old=list(range(24)),
                new=_SESSION_MAP,
                default=0,
            ).cast(pl.Int32).alias("trading_session"),

            (hour.cast(pl.Float64) / 24.0 * _TWO_PI).sin().alias("session_hour_sin"),
            (hour.cast(pl.Float64) / 24.0 * _TWO_PI).cos().alias("session_hour_cos"),

            ((hour >= 14) & (hour <= 16)).alias("is_overlap"),

            ((hour >= 22) | (hour <= 4)).alias("is_dead_zone"),

            hour.replace(
                old=list(range(24)),
                new=_HOURS_TO_END,
                default=0.0,
            ).cast(pl.Float64).alias("hours_to_session_end"),
        ])

        df = df.drop("_ts") if "_ts" in df.columns else df
        _log.debug("SessionEncoder.encode: done")
        return df


# ═══════════════════════════════════════════════════════════════
# 2. SessionVWAP
# ═══════════════════════════════════════════════════════════════


class SessionVWAP:
    """Daily VWAP (resets at 00:00 UTC) with Bollinger-style bands.

    Added columns:
        session_vwap, price_vs_session_vwap, session_vwap_std,
        vwap_upper_band, vwap_lower_band, vwap_band_position,
        session_volume_pct
    """

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add session VWAP columns to *df*."""
        df = _ensure_ts(df)
        df = df.sort("open_time")

        # Date partition key.
        df = df.with_columns(
            pl.col("_ts").dt.date().cast(pl.Utf8).alias("_date")
        )

        typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0

        df = df.with_columns([
            safe_divide(
                (typical * pl.col("volume")).cum_sum().over("_date"),
                pl.col("volume").cum_sum().over("_date"),
            ).alias("session_vwap"),

            # Daily volume share (%) for current bar.
            safe_divide(
                pl.col("volume"),
                pl.col("volume").sum().over("_date"),
            ).alias("session_volume_pct"),
        ])

        # Price vs VWAP (%).
        df = df.with_columns(
            safe_divide(
                pl.col("close") - pl.col("session_vwap"),
                pl.col("session_vwap"),
            ).alias("price_vs_session_vwap")
        )

        # Intraday std of close prices.
        df = df.with_columns(
            pl.col("close").std().over("_date").fill_null(0.0).alias("session_vwap_std")
        )

        # Bands: vwap ± 1×std.
        df = df.with_columns([
            (pl.col("session_vwap") + pl.col("session_vwap_std")).alias("vwap_upper_band"),
            (pl.col("session_vwap") - pl.col("session_vwap_std")).alias("vwap_lower_band"),
        ])

        # Band position: (close - vwap) / std, clamped.
        df = df.with_columns(
            pl.when(pl.col("session_vwap_std") > 0)
            .then(
                ((pl.col("close") - pl.col("session_vwap")) / pl.col("session_vwap_std"))
                .clip(-4.0, 4.0)
            )
            .otherwise(0.0)
            .alias("vwap_band_position")
        )

        df = df.drop([c for c in ["_date", "_ts"] if c in df.columns])
        _log.debug("SessionVWAP.calculate: done")
        return df


# ═══════════════════════════════════════════════════════════════
# 3. AnchoredVWAP
# ═══════════════════════════════════════════════════════════════


class AnchoredVWAP:
    """Weekly anchored VWAP (resets Monday 00:00 UTC).

    Added columns:
        avwap_weekly, price_vs_avwap_weekly, avwap_weekly_slope,
        price_above_avwap
    """

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add weekly anchored VWAP columns to *df*."""
        df = _ensure_ts(df)
        df = df.sort("open_time")

        # ISO week key: "YYYY-WW"
        df = df.with_columns(
            (
                pl.col("_ts").dt.iso_year().cast(pl.Utf8)
                + "-"
                + pl.col("_ts").dt.week().cast(pl.Utf8).str.pad_start(2, "0")
            ).alias("_week")
        )

        typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0

        df = df.with_columns(
            safe_divide(
                (typical * pl.col("volume")).cum_sum().over("_week"),
                pl.col("volume").cum_sum().over("_week"),
            ).alias("avwap_weekly")
        )

        # Price vs weekly AVWAP (%).
        df = df.with_columns(
            safe_divide(
                pl.col("close") - pl.col("avwap_weekly"),
                pl.col("avwap_weekly"),
            ).alias("price_vs_avwap_weekly")
        )

        # Slope: change over last 5 bars, normalised by price.
        df = df.with_columns(
            safe_divide(
                pl.col("avwap_weekly") - pl.col("avwap_weekly").shift(5),
                pl.col("avwap_weekly").shift(5),
            ).alias("avwap_weekly_slope")
        )

        # Boolean flag.
        df = df.with_columns(
            (pl.col("close") > pl.col("avwap_weekly")).alias("price_above_avwap")
        )

        df = df.drop([c for c in ["_week", "_ts"] if c in df.columns])
        _log.debug("AnchoredVWAP.calculate: done")
        return df


# ═══════════════════════════════════════════════════════════════
# 4. PreFundingDetector
# ═══════════════════════════════════════════════════════════════


class PreFundingDetector:
    """Detect proximity to Binance funding rate marks (01, 09, 17 UTC).

    Added columns:
        hours_to_funding_mark, pre_funding_window, post_funding_window,
        funding_window_urgency
    """

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add pre-funding columns to *df*."""
        df = _ensure_ts(df)
        hour = pl.col("_ts").dt.hour().cast(pl.Float64)
        minute = pl.col("_ts").dt.minute().cast(pl.Float64)
        fractional_hour = hour + minute / 60.0

        # Hours to next funding mark (circular, 8-hour cycle).
        # Marks at 1, 9, 17.  Compute distance to each, take minimum.
        distances = []
        for mark in _FUNDING_MARKS:
            # Forward distance (always positive, wraps at 24).
            dist = (pl.lit(float(mark)) - fractional_hour + 24.0) % 24.0
            # But funding is every 8h, so cap at 8.
            dist = pl.when(dist > 8.0).then(pl.lit(8.0)).otherwise(dist)
            distances.append(dist)

        # Minimum distance to any mark.
        hours_to_mark = pl.min_horizontal(*distances)

        # Hours since last mark (for post-funding window).
        hours_since: list[pl.Expr] = []
        for mark in _FUNDING_MARKS:
            since = (fractional_hour - pl.lit(float(mark)) + 24.0) % 24.0
            since = pl.when(since > 8.0).then(pl.lit(8.0)).otherwise(since)
            hours_since.append(since)
        min_hours_since = pl.min_horizontal(*hours_since)

        df = df.with_columns([
            hours_to_mark.alias("hours_to_funding_mark"),
            (hours_to_mark <= _PRE_FUNDING_WINDOW_H).alias("pre_funding_window"),
            (min_hours_since <= 1.0).alias("post_funding_window"),
            # Urgency: 1.0 at ≤1H, 0.5 at ≤2H, 0.0 otherwise.
            pl.when(hours_to_mark <= 1.0).then(1.0)
              .when(hours_to_mark <= 2.0).then(0.5)
              .otherwise(0.0)
              .alias("funding_window_urgency"),
        ])

        df = df.drop([c for c in ["_ts"] if c in df.columns])
        _log.debug("PreFundingDetector.detect: done")
        return df


# ═══════════════════════════════════════════════════════════════
# 5. MondayAsiaEffect
# ═══════════════════════════════════════════════════════════════


class MondayAsiaEffect:
    """Day-of-week features and Monday Asia Open window detection.

    Added columns:
        day_of_week_sin, day_of_week_cos, is_weekend,
        is_monday_asia_window, is_high_vol_day
    """

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add day-of-week columns to *df*."""
        df = _ensure_ts(df)

        # Polars weekday: Monday=1 .. Sunday=7
        dow = pl.col("_ts").dt.weekday().cast(pl.Float64)  # 1-7
        hour = pl.col("_ts").dt.hour()

        df = df.with_columns([
            # Cyclical: use 0-indexed (Mon=0 .. Sun=6) for sin/cos.
            ((dow - 1.0) / 7.0 * _TWO_PI).sin().alias("day_of_week_sin"),
            ((dow - 1.0) / 7.0 * _TWO_PI).cos().alias("day_of_week_cos"),

            # Weekend: Saturday (6) or Sunday (7).
            (dow >= 6).alias("is_weekend"),

            # Monday Asia window: Sunday 20:00 UTC → Monday 08:00 UTC.
            (
                ((dow == 7) & (hour >= 20))   # Sunday 20:00+
                | ((dow == 1) & (hour < 8))   # Monday 00:00-07:59
            ).alias("is_monday_asia_window"),

            # High-vol days: Wednesday (3) and Thursday (4).
            ((dow == 3) | (dow == 4)).alias("is_high_vol_day"),
        ])

        df = df.drop([c for c in ["_ts"] if c in df.columns])
        _log.debug("MondayAsiaEffect.detect: done")
        return df
