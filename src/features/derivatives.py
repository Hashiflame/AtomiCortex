"""
src/features/derivatives.py

Derivatives-specific features: funding rate, open interest, and basis.
All functions take a pl.DataFrame and return it with new columns appended.
"""

from __future__ import annotations

import polars as pl

from src.features.utils import rolling_zscore, safe_divide
from src.logger import get_logger

_log = get_logger(__name__)

# Funding rate is published every 8 hours → 3 events per day (24 / 8 = 3)
_FUNDING_PERIODS_PER_DAY = 3

# --- helpers -----------------------------------------------------------------

def _zero_funding(df: pl.DataFrame) -> pl.DataFrame:
    """Add zero-filled funding columns when no funding data is available."""
    return df.with_columns([
        pl.lit(0.0).alias("funding_rate"),
        pl.lit(0.0).alias("funding_abs"),
        pl.lit(0.0).alias("funding_zscore_7d"),
        pl.lit(0.0).alias("funding_zscore_30d"),
        pl.lit(0).cast(pl.Int8).alias("funding_extreme"),
        pl.lit(0).cast(pl.Int8).alias("funding_positive"),
        pl.lit(0.0).alias("funding_cum_24h"),
    ])


def _zero_oi(df: pl.DataFrame) -> pl.DataFrame:
    """Add zero-filled OI columns when no metrics data is available."""
    return df.with_columns([
        pl.lit(0.0).alias("oi_value"),
        pl.lit(0.0).alias("oi_delta_4h"),
        pl.lit(0.0).alias("oi_delta_12h"),
        pl.lit(0.0).alias("oi_zscore"),
        pl.lit(0).cast(pl.Int8).alias("oi_quadrant"),
        pl.lit(0.0).alias("ls_ratio"),
        pl.lit(0.0).alias("ls_ratio_zscore"),
        pl.lit(0.0).alias("taker_vol_ratio"),
    ])


# --- public functions --------------------------------------------------------

def add_funding_features(
    df: pl.DataFrame,
    funding_df: pl.DataFrame,
) -> pl.DataFrame:
    """Funding rate features via asof join on open_time → fundingTime.

    Required df columns: open_time
    funding_df columns: fundingTime, fundingRate (or legacy: calc_time, funding_rate)

    Added columns: funding_rate, funding_abs, funding_zscore_7d,
                   funding_zscore_30d, funding_extreme, funding_positive,
                   funding_cum_24h
    """
    if funding_df.is_empty():
        _log.warning("add_funding_features: empty funding_df — using zeros")
        return _zero_funding(df)

    # Detect timestamp and rate column names (handle both data conventions)
    time_col = next(
        (c for c in ("fundingTime", "calc_time") if c in funding_df.columns),
        funding_df.columns[0],
    )
    rate_col = next(
        (c for c in ("fundingRate", "funding_rate") if c in funding_df.columns),
        None,
    )
    if rate_col is None:
        _log.warning("add_funding_features: no rate column found — using zeros")
        return _zero_funding(df)

    f = (
        funding_df
        .select([
            pl.col(time_col).cast(pl.Int64).alias("_f_time"),
            pl.col(rate_col).cast(pl.Float64).alias("funding_rate"),
        ])
        .sort("_f_time")
    )

    df = (
        df.sort("open_time")
        .join_asof(f, left_on="open_time", right_on="_f_time", strategy="backward")
    )
    df = df.with_columns(
        pl.col("funding_rate").fill_null(0.0).fill_nan(0.0)
    )

    df = df.with_columns([
        pl.col("funding_rate").abs().alias("funding_abs"),
        rolling_zscore(pl.col("funding_rate"), 42).alias("funding_zscore_7d"),   # 42 bars × 4H = 7 days
        rolling_zscore(pl.col("funding_rate"), 180).alias("funding_zscore_30d"), # 180 bars × 4H = 30 days
    ])
    df = df.with_columns([
        (pl.col("funding_zscore_7d").abs() > 2.0).cast(pl.Int8).alias("funding_extreme"),
        (pl.col("funding_rate") > 0).cast(pl.Int8).alias("funding_positive"),
        pl.col("funding_rate")
            .rolling_sum(window_size=6)   # 6 bars × 4H = 24 hours
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("funding_cum_24h"),
    ])
    _log.debug("add_funding_features: done")
    return df


def add_oi_features(
    df: pl.DataFrame,
    metrics_df: pl.DataFrame,
) -> pl.DataFrame:
    """Open interest and long/short ratio features via asof join.

    Required df columns: open_time, close
    metrics_df columns: create_time, sum_open_interest_value,
                        count_long_short_ratio, sum_taker_long_short_vol_ratio

    Added columns: oi_value, oi_delta_4h, oi_delta_12h, oi_zscore, oi_quadrant,
                   ls_ratio, ls_ratio_zscore, taker_vol_ratio
    """
    if metrics_df.is_empty():
        _log.warning("add_oi_features: empty metrics_df — using zeros")
        return _zero_oi(df)

    select_cols = [pl.col("create_time").cast(pl.Int64).alias("_m_time")]
    for src, dst in [
        ("sum_open_interest_value", "oi_value"),
        ("count_long_short_ratio", "ls_ratio"),
        ("sum_taker_long_short_vol_ratio", "taker_vol_ratio"),
    ]:
        if src in metrics_df.columns:
            select_cols.append(pl.col(src).cast(pl.Float64).alias(dst))
        else:
            select_cols.append(pl.lit(0.0).alias(dst))

    m = metrics_df.select(select_cols).sort("_m_time")

    df = (
        df.sort("open_time")
        .join_asof(m, left_on="open_time", right_on="_m_time", strategy="backward")
    )
    for col in ("oi_value", "ls_ratio", "taker_vol_ratio"):
        df = df.with_columns(pl.col(col).fill_null(0.0).fill_nan(0.0))

    # OI deltas: (current - past) / current
    df = df.with_columns([
        safe_divide(
            pl.col("oi_value") - pl.col("oi_value").shift(1),
            pl.col("oi_value").shift(1),
        ).alias("oi_delta_4h"),   # pct change over 1 bar (4H)
        safe_divide(
            pl.col("oi_value") - pl.col("oi_value").shift(3),
            pl.col("oi_value").shift(3),
        ).alias("oi_delta_12h"),  # pct change over 3 bars (12H)
        rolling_zscore(pl.col("oi_value"), 180).alias("oi_zscore"),
        rolling_zscore(pl.col("ls_ratio"), 180).alias("ls_ratio_zscore"),
    ])

    # oi_quadrant: sign(price_change) × sign(oi_delta_4h)
    price_up = pl.col("close") >= pl.col("close").shift(1)
    oi_up = pl.col("oi_delta_4h") >= 0
    df = df.with_columns(
        pl.when(price_up & oi_up).then(1)
        .when(price_up & ~oi_up).then(2)
        .when(~price_up & oi_up).then(-1)
        .otherwise(-2)
        .cast(pl.Int8)
        .alias("oi_quadrant")
    )
    _log.debug("add_oi_features: done")
    return df


def add_basis_features(df: pl.DataFrame) -> pl.DataFrame:
    """Basis approximation from accumulated funding rate.

    Requires funding_cum_24h column (from add_funding_features).

    Added columns: basis_approx, basis_extreme
    """
    # Daily basis = sum of 3 daily funding payments (already accumulated)
    df = df.with_columns(
        pl.col("funding_cum_24h").alias("basis_approx")
    )
    df = df.with_columns(
        (pl.col("basis_approx").abs() > 0.001).cast(pl.Int8).alias("basis_extreme")
    )
    _log.debug("add_basis_features: done")
    return df
