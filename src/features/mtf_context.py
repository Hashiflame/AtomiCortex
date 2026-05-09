"""
src/features/mtf_context.py

Multi-timeframe context provider.

Extracts features from higher timeframe (HTF) to use as context
in lower timeframe (LTF) strategies.

Usage:
  1H strategy: uses 4H as HTF filter
  15m strategy: uses 1H + 4H as HTF filters

CRITICAL: All joins use ASOF JOIN (as-of join) to prevent lookahead bias.
A 4H bar closing at 12:00 is only available AFTER 12:00.
We never use future data — only the latest CLOSED bar.

Polars ASOF JOIN: df_ltf.join_asof(df_htf, on='timestamp', strategy='backward')
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.features.utils import safe_divide
from src.logger import get_logger

_log = get_logger(__name__)

# EMA smoothing factor helpers.
_EMA_SPAN_20 = 20


def _ema_column(col: str, span: int) -> pl.Expr:
    """Exponential moving average with given span (as Polars expression)."""
    return pl.col(col).ewm_mean(span=span, adjust=False, ignore_nulls=True)


def _trend_direction(close: pl.Expr, lookback: int = 3) -> pl.Expr:
    """Trend direction from recent return: +1 up, -1 down, 0 neutral."""
    change = close - close.shift(lookback)
    return (
        pl.when(change > 0).then(1)
          .when(change < 0).then(-1)
          .otherwise(0)
          .cast(pl.Int32)
    )


# ═══════════════════════════════════════════════════════════════
# MTFContextBuilder
# ═══════════════════════════════════════════════════════════════


class MTFContextBuilder:
    """Build multi-timeframe context features via backward ASOF joins.

    All joins use ``strategy='backward'`` to prevent lookahead bias:
    a 4H bar at timestamp T is only visible to LTF bars with timestamp >= T.
    """

    # ----------------------------------------------------------------
    # Public: 1H ← 4H context
    # ----------------------------------------------------------------

    def build_for_1h(
        self,
        df_1h: pl.DataFrame,
        df_4h: pl.DataFrame,
    ) -> pl.DataFrame:
        """Enrich *df_1h* with 4H HTF context.

        Added columns:
            htf_4h_regime, htf_4h_adx, htf_4h_trend_dir, htf_4h_hurst,
            htf_4h_atr_pct, price_vs_4h_ema20, mtf_1h_4h_aligned,
            mtf_alignment_score, htf_4h_last_n_bars_dir
        """
        # Prepare 4H features for joining.
        htf = self._prepare_htf(df_4h, prefix="4h")

        if htf.is_empty():
            _log.warning("build_for_1h: empty 4H data — filling defaults")
            return self._fill_1h_defaults(df_1h)

        # Backward ASOF join: each 1H bar gets the latest closed 4H bar.
        df = (
            df_1h.sort("open_time")
            .join_asof(
                htf.sort("_htf_time"),
                left_on="open_time",
                right_on="_htf_time",
                strategy="backward",
            )
        )

        # Price vs 4H EMA20 (%).
        df = df.with_columns(
            safe_divide(
                pl.col("close") - pl.col("_htf_ema20"),
                pl.col("_htf_ema20"),
            ).alias("price_vs_4h_ema20")
        )

        # 1H trend direction (from recent close changes).
        df = df.with_columns(
            _trend_direction(pl.col("close"), lookback=3).alias("_1h_trend")
        )

        # MTF alignment: both 1H and 4H trending same direction.
        df = df.with_columns([
            (
                (pl.col("_1h_trend") == pl.col("htf_4h_trend_dir"))
                & (pl.col("_1h_trend") != 0)
            ).alias("mtf_1h_4h_aligned"),
        ])

        # Alignment score: count how many TFs are trending.
        trend_1h_active = pl.col("_1h_trend").abs()
        trend_4h_active = pl.col("htf_4h_trend_dir").abs()
        df = df.with_columns(
            (trend_1h_active + trend_4h_active).cast(pl.Int32).alias("mtf_alignment_score")
        )

        # Cleanup.
        df = df.drop(
            [c for c in df.columns if c.startswith("_")],
        )

        _log.debug("MTFContextBuilder.build_for_1h: done")
        return df

    # ----------------------------------------------------------------
    # Public: 15m ← 1H + 4H context
    # ----------------------------------------------------------------

    def build_for_15m(
        self,
        df_15m: pl.DataFrame,
        df_1h: pl.DataFrame,
        df_4h: pl.DataFrame,
    ) -> pl.DataFrame:
        """Enrich *df_15m* with 1H and 4H HTF context.

        Added columns:
            htf_1h_regime, htf_1h_trend_dir, htf_1h_adx, htf_1h_vwap_position,
            htf_4h_regime, htf_4h_trend_dir,
            mtf_3tf_alignment, htf_conflict, htf_both_strong_trend
        """
        htf_1h = self._prepare_htf(df_1h, prefix="1h")
        htf_4h = self._prepare_htf(df_4h, prefix="4h")

        df = df_15m.sort("open_time")

        # Join 1H context.
        if not htf_1h.is_empty():
            # Add session_vwap from 1H if available.
            vwap_cols = ["_htf_time"]
            if "session_vwap" in df_1h.columns:
                htf_vwap = df_1h.select([
                    pl.col("open_time").alias("_vwap_time"),
                    pl.col("session_vwap").alias("_htf_1h_vwap"),
                ]).sort("_vwap_time")

                df = df.join_asof(
                    htf_vwap,
                    left_on="open_time",
                    right_on="_vwap_time",
                    strategy="backward",
                )
                df = df.with_columns(
                    safe_divide(
                        pl.col("close") - pl.col("_htf_1h_vwap"),
                        pl.col("_htf_1h_vwap"),
                    ).alias("htf_1h_vwap_position")
                )
            else:
                df = df.with_columns(pl.lit(0.0).alias("htf_1h_vwap_position"))

            df = df.join_asof(
                htf_1h.sort("_htf_time"),
                left_on="open_time",
                right_on="_htf_time",
                strategy="backward",
            )
            # Rename 1H columns.
            renames = {
                "htf_4h_regime": "htf_1h_regime",
                "htf_4h_adx": "htf_1h_adx",
                "htf_4h_trend_dir": "htf_1h_trend_dir",
            }
            for old, new in renames.items():
                if old in df.columns:
                    df = df.rename({old: new})
        else:
            df = df.with_columns([
                pl.lit("unknown").alias("htf_1h_regime"),
                pl.lit(0).cast(pl.Int32).alias("htf_1h_trend_dir"),
                pl.lit(0.0).alias("htf_1h_adx"),
                pl.lit(0.0).alias("htf_1h_vwap_position"),
            ])

        # Join 4H context.
        if not htf_4h.is_empty():
            df = df.join_asof(
                htf_4h.sort("_htf_time"),
                left_on="open_time",
                right_on="_htf_time",
                strategy="backward",
            )
        else:
            df = df.with_columns([
                pl.lit("unknown").alias("htf_4h_regime"),
                pl.lit(0).cast(pl.Int32).alias("htf_4h_trend_dir"),
            ])

        # 15m trend direction.
        df = df.with_columns(
            _trend_direction(pl.col("close"), lookback=4).alias("_15m_trend")
        )

        # 3-TF alignment score.
        t15 = pl.col("_15m_trend").abs()
        t1h = pl.col("htf_1h_trend_dir").abs() if "htf_1h_trend_dir" in df.columns else pl.lit(0)
        t4h = pl.col("htf_4h_trend_dir").abs() if "htf_4h_trend_dir" in df.columns else pl.lit(0)

        df = df.with_columns(
            (t15 + t1h + t4h).clip(0, 3).cast(pl.Int32).alias("mtf_3tf_alignment")
        )

        # Conflict: 1H and 4H disagree.
        if "htf_1h_trend_dir" in df.columns and "htf_4h_trend_dir" in df.columns:
            df = df.with_columns(
                (
                    (pl.col("htf_1h_trend_dir") * pl.col("htf_4h_trend_dir") < 0)
                    & (pl.col("htf_1h_trend_dir") != 0)
                    & (pl.col("htf_4h_trend_dir") != 0)
                ).alias("htf_conflict")
            )
        else:
            df = df.with_columns(pl.lit(False).alias("htf_conflict"))

        # Both strong trend: 1H ADX > 25 AND 4H ADX > 25.
        adx_1h = pl.col("htf_1h_adx") if "htf_1h_adx" in df.columns else pl.lit(0.0)
        adx_4h = pl.col("htf_4h_adx") if "htf_4h_adx" in df.columns else pl.lit(0.0)
        df = df.with_columns(
            ((adx_1h > 25.0) & (adx_4h > 25.0)).alias("htf_both_strong_trend")
        )

        # Cleanup.
        df = df.drop(
            [c for c in df.columns if c.startswith("_")],
        )

        _log.debug("MTFContextBuilder.build_for_15m: done")
        return df

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _prepare_htf(self, df_htf: pl.DataFrame, prefix: str) -> pl.DataFrame:
        """Extract key HTF columns for joining."""
        if df_htf.is_empty() or "open_time" not in df_htf.columns:
            return pl.DataFrame()

        cols = [pl.col("open_time").alias("_htf_time")]

        # Regime.
        if "regime" in df_htf.columns:
            cols.append(pl.col("regime").alias(f"htf_{prefix}_regime"))
        else:
            cols.append(pl.lit("unknown").alias(f"htf_{prefix}_regime"))

        # ADX.
        if "adx" in df_htf.columns:
            cols.append(pl.col("adx").alias(f"htf_{prefix}_adx"))
        else:
            cols.append(pl.lit(0.0).alias(f"htf_{prefix}_adx"))

        # Trend direction from close.
        if "close" in df_htf.columns:
            # Pre-compute trend direction and EMA20 on the HTF df.
            df_htf = df_htf.sort("open_time").with_columns([
                _trend_direction(pl.col("close"), lookback=3).alias("_trend_dir"),
                _ema_column("close", _EMA_SPAN_20).alias("_ema20"),
            ])
            cols.extend([
                pl.col("_trend_dir").alias(f"htf_{prefix}_trend_dir"),
                pl.col("_ema20").alias("_htf_ema20"),
            ])
        else:
            cols.extend([
                pl.lit(0).cast(pl.Int32).alias(f"htf_{prefix}_trend_dir"),
                pl.lit(0.0).alias("_htf_ema20"),
            ])

        # Hurst and ATR%.
        if "hurst" in df_htf.columns:
            cols.append(pl.col("hurst").alias(f"htf_{prefix}_hurst"))
        if "atr_pct" in df_htf.columns:
            cols.append(pl.col("atr_pct").alias(f"htf_{prefix}_atr_pct"))

        # Last N bars direction.
        if "close" in df_htf.columns:
            last_n = _trend_direction(pl.col("close"), lookback=3)
            cols.append(last_n.alias(f"htf_{prefix}_last_n_bars_dir"))

        return df_htf.select(cols).sort("_htf_time")

    def _fill_1h_defaults(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill all 1H←4H context columns with defaults."""
        return df.with_columns([
            pl.lit("unknown").alias("htf_4h_regime"),
            pl.lit(0.0).alias("htf_4h_adx"),
            pl.lit(0).cast(pl.Int32).alias("htf_4h_trend_dir"),
            pl.lit(0.5).alias("htf_4h_hurst"),
            pl.lit(0.0).alias("htf_4h_atr_pct"),
            pl.lit(0.0).alias("price_vs_4h_ema20"),
            pl.lit(False).alias("mtf_1h_4h_aligned"),
            pl.lit(0).cast(pl.Int32).alias("mtf_alignment_score"),
            pl.lit(0).cast(pl.Int32).alias("htf_4h_last_n_bars_dir"),
        ])
