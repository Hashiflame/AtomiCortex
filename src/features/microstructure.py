"""
src/features/microstructure.py

Microstructure features: CVD, volume, and price-pattern features.
All functions take a pl.DataFrame and return it with new columns appended.
"""

from __future__ import annotations

import math

import polars as pl

from src.features.utils import rolling_zscore, safe_divide
from src.logger import get_logger

_log = get_logger(__name__)


def add_cvd_features(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulative Volume Delta and derived slope / ratio features.

    Required columns: taker_buy_volume, volume
    Added columns: cvd, cvd_cum, cvd_slope_3, cvd_slope_6, cvd_slope_12,
                   taker_buy_ratio
    """
    df = df.with_columns(
        (2.0 * pl.col("taker_buy_volume") - pl.col("volume")).alias("cvd")
    )
    df = df.with_columns([
        pl.col("cvd").cum_sum().alias("cvd_cum"),
        (pl.col("cvd") - pl.col("cvd").shift(3)).fill_null(0.0).fill_nan(0.0).alias("cvd_slope_3"),
        (pl.col("cvd") - pl.col("cvd").shift(6)).fill_null(0.0).fill_nan(0.0).alias("cvd_slope_6"),
        (pl.col("cvd") - pl.col("cvd").shift(12)).fill_null(0.0).fill_nan(0.0).alias("cvd_slope_12"),
        safe_divide(pl.col("taker_buy_volume"), pl.col("volume")).alias("taker_buy_ratio"),
    ])
    _log.debug("add_cvd_features: done")
    return df


def add_volume_features(df: pl.DataFrame) -> pl.DataFrame:
    """Relative volume, z-score, VWAP, and price-to-VWAP features.

    Required columns: volume, high, low, close
    Added columns: volume_sma_20, volume_ratio, volume_zscore, large_volume,
                   vwap_4h, price_to_vwap
    """
    df = df.with_columns(
        pl.col("volume").rolling_mean(window_size=20).fill_null(0.0).alias("volume_sma_20")
    )
    df = df.with_columns([
        safe_divide(pl.col("volume"), pl.col("volume_sma_20")).alias("volume_ratio"),
        rolling_zscore(pl.col("volume"), 50).alias("volume_zscore"),
    ])
    df = df.with_columns(
        (pl.col("volume_ratio") > 2.0).cast(pl.Int8).alias("large_volume")
    )
    # VWAP over last 6 bars (1 trading day on 4H)
    df = df.with_columns(
        safe_divide(
            (pl.col("close") * pl.col("volume")).rolling_sum(window_size=6),
            pl.col("volume").rolling_sum(window_size=6),
        ).alias("vwap_4h")
    )
    df = df.with_columns(
        (safe_divide(pl.col("close"), pl.col("vwap_4h"), fill=1.0) - 1.0)
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("price_to_vwap")
    )
    _log.debug("add_volume_features: done")
    return df


def add_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """Log returns, candle-shape ratios, and gap features.

    Required columns: open, high, low, close
    Added columns: returns_1, returns_3, returns_6, returns_12, returns_24,
                   body_ratio, upper_wick, lower_wick, gap
    """
    log_close = pl.col("close").log(base=math.e)

    hl = pl.col("high") - pl.col("low")
    hi_body = pl.max_horizontal(pl.col("open"), pl.col("close"))
    lo_body = pl.min_horizontal(pl.col("open"), pl.col("close"))
    body = (pl.col("close") - pl.col("open")).abs()

    df = df.with_columns([
        (log_close - log_close.shift(1)).fill_null(0.0).fill_nan(0.0).alias("returns_1"),
        (log_close - log_close.shift(3)).fill_null(0.0).fill_nan(0.0).alias("returns_3"),
        (log_close - log_close.shift(6)).fill_null(0.0).fill_nan(0.0).alias("returns_6"),
        (log_close - log_close.shift(12)).fill_null(0.0).fill_nan(0.0).alias("returns_12"),
        (log_close - log_close.shift(24)).fill_null(0.0).fill_nan(0.0).alias("returns_24"),
        safe_divide(body, hl).alias("body_ratio"),
        safe_divide(pl.col("high") - hi_body, hl).alias("upper_wick"),
        safe_divide(lo_body - pl.col("low"), hl).alias("lower_wick"),
        (pl.col("open") - pl.col("close").shift(1)).fill_null(0.0).fill_nan(0.0).alias("gap"),
    ])
    _log.debug("add_price_features: done")
    return df
