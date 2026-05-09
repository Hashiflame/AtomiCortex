"""
src/features/orb_features.py

Opening Range Breakout (ORB) features for 15m strategy.

ORB = high/low of first 4 bars (1 hour) after session open.
Breakout beyond ORB with volume confirmation = high probability move.

Sessions and their opens (UTC):
  Asia:   00:00 UTC
  London: 08:00 UTC
  NY:     13:00 UTC

Only meaningful on 15m timeframe (ORB period = 4 bars × 15min = 1 hour).
All methods take Polars DataFrame and return Polars DataFrame.
"""

from __future__ import annotations

import polars as pl

from src.features.utils import safe_divide
from src.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Session open hours (UTC) and the ORB formation period (4 bars = 1 hour).
_SESSION_OPENS: dict[str, int] = {
    "asia": 0,
    "london": 8,
    "ny": 13,
}
_ORB_BARS = 4  # 4 × 15m = 1 hour
_VOLUME_CONFIRM_MULT = 1.3  # volume must be >= 1.3× average for breakout
_AVG_VOLUME_WINDOW = 20     # bars for average volume


# ═══════════════════════════════════════════════════════════════
# ORBDetector
# ═══════════════════════════════════════════════════════════════


class ORBDetector:
    """Opening Range Breakout detection for 15m bars.

    Added columns (per session: asia, london, ny):
        orb_high_{session}, orb_low_{session}, orb_range_{session},
        orb_range_{session}_atr_pct

    Position columns:
        current_session, price_vs_current_orb,
        dist_to_orb_high_pct, dist_to_orb_low_pct

    Breakout columns:
        orb_breakout_bull, orb_breakout_bear,
        bars_since_session_open, is_session_trap_zone
    """

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add ORB feature columns to *df*."""
        df = df.sort("open_time")

        # Derive hour from open_time.
        ts = pl.from_epoch(pl.col("open_time"), time_unit="ms")
        df = df.with_columns([
            ts.dt.hour().alias("_hour"),
            ts.dt.date().cast(pl.Utf8).alias("_date"),
        ])

        # ATR14 for normalisation (simple: high - low rolling mean).
        df = df.with_columns(
            (pl.col("high") - pl.col("low"))
            .rolling_mean(window_size=14)
            .fill_null(0.0)
            .alias("_atr14")
        )

        # Average volume for breakout confirmation.
        df = df.with_columns(
            pl.col("volume")
            .rolling_mean(window_size=_AVG_VOLUME_WINDOW)
            .fill_null(0.0)
            .alias("_avg_vol")
        )

        # Compute ORB for each session.
        for session_name, open_hour in _SESSION_OPENS.items():
            df = self._compute_session_orb(df, session_name, open_hour)

        # Current session and position.
        df = self._add_position_features(df)

        # Breakout signals.
        df = self._add_breakout_features(df)

        # Session trap zone and bars since open.
        df = self._add_session_meta(df)

        # Cleanup temporary columns.
        drop_cols = [c for c in df.columns if c.startswith("_")]
        if drop_cols:
            df = df.drop(drop_cols)

        _log.debug("ORBDetector.calculate: done")
        return df

    # ----------------------------------------------------------------
    # Internal: compute ORB for one session
    # ----------------------------------------------------------------

    def _compute_session_orb(
        self, df: pl.DataFrame, session: str, open_hour: int,
    ) -> pl.DataFrame:
        """Add orb_high/low/range for one session, forward-filled daily."""
        end_hour = open_hour + 1  # ORB = first hour

        # Mark ORB-forming bars.
        is_orb_bar = (
            (pl.col("_hour") >= open_hour)
            & (pl.col("_hour") < end_hour)
        )

        # Compute ORB high/low within each day's ORB window.
        df = df.with_columns([
            pl.when(is_orb_bar)
            .then(pl.col("high"))
            .otherwise(None)
            .max()
            .over("_date")
            .alias(f"orb_high_{session}"),

            pl.when(is_orb_bar)
            .then(pl.col("low"))
            .otherwise(None)
            .min()
            .over("_date")
            .alias(f"orb_low_{session}"),
        ])

        # Fill null (days with no ORB data) with forward-fill.
        df = df.with_columns([
            pl.col(f"orb_high_{session}").forward_fill().fill_null(0.0),
            pl.col(f"orb_low_{session}").forward_fill().fill_null(0.0),
        ])

        # ORB range and ATR ratio.
        df = df.with_columns(
            (pl.col(f"orb_high_{session}") - pl.col(f"orb_low_{session}"))
            .alias(f"orb_range_{session}")
        )
        df = df.with_columns(
            safe_divide(
                pl.col(f"orb_range_{session}"),
                pl.col("_atr14"),
            ).alias(f"orb_range_{session}_atr_pct")
        )

        return df

    # ----------------------------------------------------------------
    # Internal: position features
    # ----------------------------------------------------------------

    def _add_position_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add current_session, price_vs_current_orb, distance columns."""
        hour = pl.col("_hour")

        # Current session (simplified): 0-4=DEAD(0), 5-7=ASIA(1),
        # 8-12=LONDON(2), 13-21=NY(3), 22-23=DEAD(0)
        df = df.with_columns(
            pl.when(hour < 5).then(0)
              .when(hour < 8).then(1)
              .when(hour < 13).then(2)
              .when(hour < 22).then(3)
              .otherwise(0)
              .cast(pl.Int32)
              .alias("current_session")
        )

        # Select the ORB for the current session.
        # Use the most relevant active ORB (Asia for Asia, London for London, NY for NY).
        current_orb_high = (
            pl.when(pl.col("current_session") == 1).then(pl.col("orb_high_asia"))
              .when(pl.col("current_session") == 2).then(pl.col("orb_high_london"))
              .when(pl.col("current_session") == 3).then(pl.col("orb_high_ny"))
              .otherwise(pl.col("orb_high_asia"))
        )
        current_orb_low = (
            pl.when(pl.col("current_session") == 1).then(pl.col("orb_low_asia"))
              .when(pl.col("current_session") == 2).then(pl.col("orb_low_london"))
              .when(pl.col("current_session") == 3).then(pl.col("orb_low_ny"))
              .otherwise(pl.col("orb_low_asia"))
        )

        close = pl.col("close")
        df = df.with_columns([
            # +1 = above ORB, -1 = below ORB, 0 = inside
            pl.when(close > current_orb_high).then(1)
              .when(close < current_orb_low).then(-1)
              .otherwise(0)
              .cast(pl.Int32)
              .alias("price_vs_current_orb"),

            safe_divide(current_orb_high - close, close).alias("dist_to_orb_high_pct"),
            safe_divide(close - current_orb_low, close).alias("dist_to_orb_low_pct"),
        ])

        return df

    # ----------------------------------------------------------------
    # Internal: breakout signals
    # ----------------------------------------------------------------

    def _add_breakout_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add orb_breakout_bull, orb_breakout_bear flags."""
        # Use the current session's ORB for breakout detection.
        current_orb_high = (
            pl.when(pl.col("current_session") == 1).then(pl.col("orb_high_asia"))
              .when(pl.col("current_session") == 2).then(pl.col("orb_high_london"))
              .when(pl.col("current_session") == 3).then(pl.col("orb_high_ny"))
              .otherwise(pl.col("orb_high_asia"))
        )
        current_orb_low = (
            pl.when(pl.col("current_session") == 1).then(pl.col("orb_low_asia"))
              .when(pl.col("current_session") == 2).then(pl.col("orb_low_london"))
              .when(pl.col("current_session") == 3).then(pl.col("orb_low_ny"))
              .otherwise(pl.col("orb_low_asia"))
        )

        vol_confirmed = pl.col("volume") >= pl.col("_avg_vol") * _VOLUME_CONFIRM_MULT

        df = df.with_columns([
            ((pl.col("close") > current_orb_high) & vol_confirmed)
                .alias("orb_breakout_bull"),
            ((pl.col("close") < current_orb_low) & vol_confirmed)
                .alias("orb_breakout_bear"),
        ])

        return df

    # ----------------------------------------------------------------
    # Internal: session metadata
    # ----------------------------------------------------------------

    def _add_session_meta(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add bars_since_session_open and is_session_trap_zone."""
        hour = pl.col("_hour")

        # Bars since session open (assuming 15m bars → 4 bars/hour).
        # Session opens: Asia=0, London=8, NY=13.
        session_start_hour = (
            pl.when(pl.col("current_session") == 1).then(0)
              .when(pl.col("current_session") == 2).then(8)
              .when(pl.col("current_session") == 3).then(13)
              .otherwise(0)
        )

        # Approximate: (hour - start) * 4. Actual bar count within the hour
        # would need minute data, so this is a good proxy.
        df = df.with_columns(
            ((hour - session_start_hour) * 4)
            .cast(pl.Int32)
            .clip(0, 200)
            .alias("bars_since_session_open")
        )

        # Session lengths in bars: Asia=32 (8h), London=20 (5h), NY=36 (9h).
        session_len_bars = (
            pl.when(pl.col("current_session") == 1).then(32)
              .when(pl.col("current_session") == 2).then(20)
              .when(pl.col("current_session") == 3).then(36)
              .otherwise(20)
        )

        # Trap zone: first 2 or last 2 bars of session.
        bars_from_start = pl.col("bars_since_session_open")
        bars_to_end = session_len_bars - bars_from_start

        df = df.with_columns(
            ((bars_from_start <= 2) | (bars_to_end <= 2))
            .alias("is_session_trap_zone")
        )

        return df
