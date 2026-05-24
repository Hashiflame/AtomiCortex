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
    """Volume Delta and derived rolling / slope / ratio features.

    Required columns: taker_buy_volume, volume
    Added columns: cvd, cvd_rolling_24, cvd_rolling_96, cvd_slope_3,
                   cvd_slope_6, cvd_slope_12, taker_buy_ratio

    cvd_rolling_N replaces the previous non-stationary cvd_cum (full-history
    cumulative sum). Rolling windows are bounded and epoch-independent, so
    LightGBM splits learned on them transfer to live where the buffer starts
    fresh on each bot restart.
    """
    df = df.with_columns(
        (2.0 * pl.col("taker_buy_volume") - pl.col("volume")).alias("cvd")
    )
    df = df.with_columns([
        pl.col("cvd").rolling_sum(window_size=24, min_periods=1)
        .fill_null(0.0).fill_nan(0.0).alias("cvd_rolling_24"),
        pl.col("cvd").rolling_sum(window_size=96, min_periods=1)
        .fill_null(0.0).fill_nan(0.0).alias("cvd_rolling_96"),
        ((pl.col("cvd") - pl.col("cvd").shift(3)) / 3.0).fill_null(0.0).fill_nan(0.0).alias("cvd_slope_3"),
        ((pl.col("cvd") - pl.col("cvd").shift(6)) / 6.0).fill_null(0.0).fill_nan(0.0).alias("cvd_slope_6"),
        ((pl.col("cvd") - pl.col("cvd").shift(12)) / 12.0).fill_null(0.0).fill_nan(0.0).alias("cvd_slope_12"),
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
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    df = df.with_columns(
        safe_divide(
            (typical * pl.col("volume")).rolling_sum(window_size=6),
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
        safe_divide(
            pl.col("open") - pl.col("close").shift(1),
            pl.col("close").shift(1),
        ).alias("gap"),
    ])
    _log.debug("add_price_features: done")
    return df


# =============================================================================
# VPIN — Volume-Synchronized Probability of Informed Trading
# Reference: Easley, López de Prado, O'Hara (2012).
# =============================================================================

import math as _math
from statistics import pstdev as _pstdev


def _phi(z: float) -> float:
    """Standard-normal CDF via math.erf."""
    return 0.5 * (1.0 + _math.erf(z / _math.sqrt(2.0)))


def compute_vpin(
    trades: list[dict],
    bucket_size: int = 50,
    num_buckets: int = 50,
) -> float:
    """Volume-synchronized probability of informed trading.

    Parameters
    ----------
    trades:
        List of aggTrade-like dicts.  Each must carry a price (``p`` or
        ``price``) and a quantity (``q`` or ``qty``); a side flag is not
        required — bulk volume classification (BVC) splits buy/sell
        volume from the return distribution alone.
    bucket_size:
        Volume units per bucket (in quantity, not notional).
    num_buckets:
        Rolling window of buckets to average over.

    Returns
    -------
    float in [0, 1].  Returns ``0.5`` (neutral) when there is too little
    data to form even one bucket — matches the project-wide fallback
    convention for missing data.
    """
    if not trades or bucket_size <= 0 or num_buckets <= 0:
        return 0.5

    # Extract (price, qty) pairs robustly.
    series: list[tuple[float, float]] = []
    for t in trades:
        try:
            p = float(t.get("p", t.get("price", 0.0)))
            q = float(t.get("q", t.get("qty", 0.0)))
        except (TypeError, ValueError):
            continue
        if p > 0 and q > 0 and not _math.isnan(p) and not _math.isnan(q):
            series.append((p, q))

    if len(series) < 2:
        return 0.5

    # Per-trade log returns for BVC.
    returns: list[float] = [0.0]
    for i in range(1, len(series)):
        prev_p = series[i - 1][0]
        cur_p = series[i][0]
        if prev_p > 0 and cur_p > 0:
            returns.append(_math.log(cur_p / prev_p))
        else:
            returns.append(0.0)

    sigma = _pstdev(returns) if len(returns) > 1 else 0.0
    if sigma <= 0.0 or _math.isnan(sigma):
        return 0.5

    # Walk trades, filling fixed-size volume buckets.
    buckets: list[float] = []  # |V_buy - V_sell| per bucket
    cur_buy = 0.0
    cur_sell = 0.0
    cur_vol = 0.0

    for (_p, q), r in zip(series, returns):
        z = r / sigma
        buy_frac = _phi(z)
        v_buy = q * buy_frac
        v_sell = q - v_buy

        remaining = q
        while remaining > 0 and len(buckets) < num_buckets:
            capacity = bucket_size - cur_vol
            take = min(capacity, remaining)
            share = take / q if q > 0 else 0.0
            cur_buy += v_buy * share
            cur_sell += v_sell * share
            cur_vol += take
            remaining -= take

            if cur_vol >= bucket_size - 1e-12:
                buckets.append(abs(cur_buy - cur_sell))
                cur_buy = 0.0
                cur_sell = 0.0
                cur_vol = 0.0

        if len(buckets) >= num_buckets:
            break

    if not buckets:
        return 0.5

    vpin = sum(buckets) / (len(buckets) * bucket_size)
    if _math.isnan(vpin) or _math.isinf(vpin):
        return 0.5
    return float(max(0.0, min(1.0, vpin)))


# =============================================================================
# MTF derived features (1H / 15m only) — appended for the post-lookahead
# feature expansion. Wired into FeaturePipeline.build_mtf() for 1h/15m;
# never run on the 4H pipeline. Only ewm/rolling/shift/diff ops.
# =============================================================================


def _atr_abs(df: pl.DataFrame) -> pl.Expr:
    """Absolute ATR proxy. RegimeDetector emits atr_pct (= ATR/price),
    there is no atr_14 column → reconstruct atr_abs = atr_pct * close.
    Falls back to 1% of close when atr_pct is unavailable.
    """
    if "atr_pct" in df.columns:
        return (pl.col("atr_pct") * pl.col("close")).clip(lower_bound=1e-10)
    return (pl.col("close") * 0.01).clip(lower_bound=1e-10)


def add_ema_slope_features(df: pl.DataFrame) -> pl.DataFrame:
    """EMA-difference ratios normalized by ATR (scale-free for LightGBM).

    Required: close (+ atr_pct for normalization). Added columns:
    ema9, ema21, ema9_slope_normalized, ema21_slope_normalized,
    ema9_cross_ema21, ema9_cross_ema21_change.
    """
    if "ema9" not in df.columns:
        df = df.with_columns(
            pl.col("close").ewm_mean(span=9, ignore_nulls=True).alias("ema9")
        )
    if "ema21" not in df.columns:
        df = df.with_columns(
            pl.col("close").ewm_mean(span=21, ignore_nulls=True).alias("ema21")
        )

    atr_abs = _atr_abs(df)
    df = df.with_columns([
        safe_divide(pl.col("ema9") - pl.col("ema9").shift(3), atr_abs)
        .alias("ema9_slope_normalized"),
        safe_divide(pl.col("ema21") - pl.col("ema21").shift(3), atr_abs)
        .alias("ema21_slope_normalized"),
        safe_divide(pl.col("ema9") - pl.col("ema21"), atr_abs)
        .alias("ema9_cross_ema21"),
    ])
    df = df.with_columns(
        (pl.col("ema9_cross_ema21") - pl.col("ema9_cross_ema21").shift(1))
        .fill_null(0.0).fill_nan(0.0)
        .alias("ema9_cross_ema21_change")
    )
    _log.debug("add_ema_slope_features: done")
    return df


def add_volume_session_features(df: pl.DataFrame) -> pl.DataFrame:
    """Volume relative to the same-hour historical average (no lookahead).

    Required: open_time, volume. Added columns: volume_vs_session_avg,
    volume_momentum_3bar.

    ``volume_vs_session_avg`` divides current volume by the trailing
    20-observation mean of volume at the *same hour of day*, computed
    from prior bars only (shift(1) before the rolling mean), so it never
    uses the current or future bars.
    """
    df = df.sort("open_time")
    df = df.with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.hour().alias("_hour_vsa")
    )
    hist_same_hour = (
        pl.col("volume").shift(1)
        .rolling_mean(window_size=20, min_periods=1)
        .over("_hour_vsa")
    )
    df = df.with_columns(
        safe_divide(pl.col("volume"), hist_same_hour, fill=1.0)
        .alias("volume_vs_session_avg")
    )
    prior_3_mean = pl.mean_horizontal(
        pl.col("volume").shift(1),
        pl.col("volume").shift(2),
        pl.col("volume").shift(3),
    )
    df = df.with_columns(
        safe_divide(pl.col("volume"), prior_3_mean, fill=1.0)
        .alias("volume_momentum_3bar")
    )
    df = df.drop("_hour_vsa")
    _log.debug("add_volume_session_features: done")
    return df


def _efficiency_ratio(window: int) -> pl.Expr:
    """Kaufman fractal efficiency over *window* bars, clamped to [0, 1]."""
    net_move = (pl.col("close") - pl.col("close").shift(window)).abs()
    path = pl.col("close").diff().abs().rolling_sum(window_size=window)
    return safe_divide(net_move, path).clip(0.0, 1.0)


def add_fractal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Fractal efficiency ratio: trend (≈1) vs choppy (≈0).

    Required: close. Added columns: efficiency_ratio_10,
    efficiency_ratio_20.
    """
    df = df.with_columns([
        _efficiency_ratio(10).fill_null(0.0).fill_nan(0.0).alias("efficiency_ratio_10"),
        _efficiency_ratio(20).fill_null(0.0).fill_nan(0.0).alias("efficiency_ratio_20"),
    ])
    _log.debug("add_fractal_features: done")
    return df


def add_candle_structure_features(df: pl.DataFrame) -> pl.DataFrame:
    """Volatility-normalized candle anatomy.

    Required: open, high, low, close. Added columns: candle_range,
    candle_body, candle_body_pct, upper_wick_pct, lower_wick_pct,
    candle_direction.
    """
    rng = pl.col("high") - pl.col("low")
    body = (pl.col("close") - pl.col("open")).abs()
    hi_body = pl.max_horizontal(pl.col("open"), pl.col("close"))
    lo_body = pl.min_horizontal(pl.col("open"), pl.col("close"))

    df = df.with_columns([
        rng.alias("candle_range"),
        body.alias("candle_body"),
    ])
    df = df.with_columns([
        safe_divide(pl.col("candle_body"), pl.col("candle_range"))
        .alias("candle_body_pct"),
        safe_divide(pl.col("high") - hi_body, pl.col("candle_range"))
        .alias("upper_wick_pct"),
        safe_divide(lo_body - pl.col("low"), pl.col("candle_range"))
        .alias("lower_wick_pct"),
        pl.when(pl.col("close") > pl.col("open")).then(1)
        .when(pl.col("close") < pl.col("open")).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("candle_direction"),
    ])
    _log.debug("add_candle_structure_features: done")
    return df
