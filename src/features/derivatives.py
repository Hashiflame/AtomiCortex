"""
src/features/derivatives.py

Derivatives-specific features: funding rate, open interest, and basis.
All functions take a pl.DataFrame and return it with new columns appended.

Also exposes scalar compute_* helpers (liquidation proximity, basis
annualized, OI velocity, sentiment) that operate on plain Python inputs —
intended for live signal-time enrichment, not vectorized pipeline use.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

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


# =============================================================================
# Scalar compute_* helpers (live signal-time enrichment).
#
# These functions work on plain Python inputs and return dicts/scalars.
# They are NOT polars-vectorized and are intended to be called once per
# signal — either at signal-emit time or via a thin per-bar wrapper.
#
# All functions implement a fail-soft contract: empty / malformed inputs
# return a neutral-valued dict (no NaN, no exceptions raised).
# =============================================================================


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _zscore(value: float, series: list[float]) -> float:
    if not series:
        return 0.0
    m = mean(series)
    s = pstdev(series)
    if s == 0.0:
        return 0.0
    z = (value - m) / s
    if math.isnan(z) or math.isinf(z):
        return 0.0
    return z


def compute_liquidation_proximity(
    current_price: float,
    liquidations: list[dict],
    lookback_hours: int = 24,
    atr: float | None = None,
) -> dict:
    """Liquidation cluster proximity / imbalance / cascade risk.

    Parameters
    ----------
    current_price:
        Spot or perp price at evaluation time.
    liquidations:
        List of recent force-orders from Binance ``/fapi/v1/allForceOrders``.
        Each item is expected to contain:

          - ``price``        : execution price (float-like)
          - ``origQty`` or ``executedQty`` : quantity
          - ``side``         : ``"BUY"`` (short liquidation) or ``"SELL"``
                               (long liquidation)
          - ``time``         : ms epoch (optional — used for the 1h window)

    lookback_hours:
        Window (hours) used for the cascade-risk numerator vs. baseline.
    atr:
        ATR (absolute price units) used to size cluster bins.  Falls back
        to ``current_price * 0.005`` if not provided.

    Returns
    -------
    dict with keys ``liq_cluster_long_pct``, ``liq_cluster_short_pct``,
    ``liq_imbalance``, ``liq_cascade_risk``, ``liq_volume_1h``.
    """
    neutral = {
        "liq_cluster_long_pct": 0.0,
        "liq_cluster_short_pct": 0.0,
        "liq_imbalance": 0.0,
        "liq_cascade_risk": 0.0,
        "liq_volume_1h": 0.0,
    }

    cp = _safe_float(current_price)
    if cp <= 0 or not liquidations:
        return neutral

    bin_size = _safe_float(atr, cp * 0.005) * 0.1
    if bin_size <= 0:
        bin_size = cp * 0.0005

    # Aggregate notional volume per (side, price-bin).
    long_clusters: dict[int, float] = defaultdict(float)
    short_clusters: dict[int, float] = defaultdict(float)
    long_vol = 0.0
    short_vol = 0.0
    vol_1h = 0.0

    now_ms = 0
    for liq in liquidations:
        t = _safe_float(liq.get("time"), 0.0)
        if t > now_ms:
            now_ms = int(t)
    one_hour_ago = now_ms - 3_600_000 if now_ms > 0 else 0
    lookback_ms = now_ms - lookback_hours * 3_600_000 if now_ms > 0 else 0

    for liq in liquidations:
        price = _safe_float(liq.get("price"))
        if price <= 0:
            continue
        qty = _safe_float(liq.get("origQty") or liq.get("executedQty") or liq.get("qty"))
        if qty <= 0:
            continue
        notional = price * qty
        side = str(liq.get("side", "")).upper()
        t = _safe_float(liq.get("time"), 0.0)

        if lookback_ms and t < lookback_ms:
            continue
        if one_hour_ago and t >= one_hour_ago:
            vol_1h += notional

        # Binance force-order convention: side=SELL → a LONG was liquidated;
        # side=BUY → a SHORT was liquidated.
        bucket = int(price // bin_size)
        if side == "SELL":
            long_clusters[bucket] += notional
            long_vol += notional
        elif side == "BUY":
            short_clusters[bucket] += notional
            short_vol += notional

    # Top-3 by notional, restricted to the relevant side of current price.
    def _nearest_above(clusters: dict[int, float]) -> float:
        top = sorted(clusters.items(), key=lambda kv: kv[1], reverse=True)[:3]
        dists = []
        for bucket, _vol in top:
            px = (bucket + 0.5) * bin_size
            if px > cp:
                dists.append((px - cp) / cp * 100.0)
        return min(dists) if dists else 0.0

    def _nearest_below(clusters: dict[int, float]) -> float:
        top = sorted(clusters.items(), key=lambda kv: kv[1], reverse=True)[:3]
        dists = []
        for bucket, _vol in top:
            px = (bucket + 0.5) * bin_size
            if px < cp:
                dists.append((cp - px) / cp * 100.0)
        return min(dists) if dists else 0.0

    # LONG liquidations sit BELOW current price (longs get stopped on dips)
    # — but the cluster the price is magnetically drawn TO that produces
    # further long liquidations is the one below; the spec asks for the
    # nearest cluster of LONG liquidations *above* the price (drawing it
    # up).  We honor the spec literally: report nearest long-liq cluster
    # above, nearest short-liq cluster below.
    liq_cluster_long_pct = min(_nearest_above(long_clusters), 10.0)
    liq_cluster_short_pct = min(_nearest_below(short_clusters), 10.0)

    total = long_vol + short_vol
    liq_imbalance = (long_vol - short_vol) / total if total > 0 else 0.0

    # Cascade risk: share of the lookback window's volume that landed in
    # the most-recent hour, clamped to [0, 1].
    if total > 0:
        cascade = vol_1h / total
        liq_cascade_risk = max(0.0, min(1.0, cascade))
    else:
        liq_cascade_risk = 0.0

    return {
        "liq_cluster_long_pct": float(liq_cluster_long_pct),
        "liq_cluster_short_pct": float(liq_cluster_short_pct),
        "liq_imbalance": float(max(-1.0, min(1.0, liq_imbalance))),
        "liq_cascade_risk": float(liq_cascade_risk),
        "liq_volume_1h": float(vol_1h),
    }


def compute_basis_annualized(
    perp_price: float,
    spot_price: float,
    funding_rate: float,
    funding_interval_hours: int = 8,
    basis_history_bps: list[float] | None = None,
) -> dict:
    """Spot–perp basis (bps) plus annualized funding yield and z-score.

    ``basis_history_bps`` is an optional list of recent basis-bps readings
    (e.g. last 30 days, one per funding period) used for the z-score.
    """
    perp = _safe_float(perp_price)
    spot = _safe_float(spot_price)
    fr = _safe_float(funding_rate)
    interval = max(1, int(funding_interval_hours or 8))

    if spot <= 0 or perp <= 0:
        return {
            "basis_bps": 0.0,
            "basis_annualized": 0.0,
            "basis_zscore_30d": 0.0,
            "basis_extreme": 0,
        }

    basis_bps = (perp - spot) / spot * 10_000.0
    # funding paid every `interval` hours → (24/interval) payments/day × 365.
    basis_annualized = fr * (24.0 / interval) * 365.0 * 100.0  # in % p.a.

    zscore = _zscore(basis_bps, basis_history_bps or [])

    if basis_annualized > 30.0:
        extreme = -1   # overheated longs → bearish
    elif basis_annualized < -10.0:
        extreme = 1    # overheated shorts → bullish
    else:
        extreme = 0

    return {
        "basis_bps": float(basis_bps),
        "basis_annualized": float(basis_annualized),
        "basis_zscore_30d": float(zscore),
        "basis_extreme": int(extreme),
    }


def compute_oi_velocity(
    oi_series: list[float],
    price_series: list[float],
) -> dict:
    """OI velocity (Δ%), acceleration, price-divergence, exhaustion flag."""
    neutral = {
        "oi_velocity": 0.0,
        "oi_acceleration": 0.0,
        "oi_price_divergence": 0.0,
        "oi_exhaustion": 0,
        "oi_velocity_zscore": 0.0,
    }
    if not oi_series or len(oi_series) < 3:
        return neutral

    oi = [_safe_float(x) for x in oi_series]
    px = [_safe_float(x) for x in (price_series or [])]

    # Per-bar velocity (% change).
    vel = []
    for i in range(1, len(oi)):
        prev = oi[i - 1]
        vel.append((oi[i] - prev) / prev if prev > 0 else 0.0)

    oi_velocity = vel[-1]
    oi_acceleration = vel[-1] - vel[-2] if len(vel) >= 2 else 0.0

    # Price-OI divergence: correlation sign × magnitude over the window.
    divergence = 0.0
    if len(px) >= len(oi) and len(oi) >= 3:
        oi_ret = vel
        px_ret = []
        # Align: vel covers oi[1..N-1], so use px[1..N-1] vs px[0..N-2].
        for i in range(len(oi) - len(vel), len(oi)):
            p_prev = px[i - 1] if i - 1 >= 0 else 0.0
            px_ret.append((px[i] - p_prev) / p_prev if p_prev > 0 else 0.0)
        n = min(len(oi_ret), len(px_ret))
        if n >= 2:
            mx, my = mean(px_ret[-n:]), mean(oi_ret[-n:])
            num = sum((px_ret[-n + i] - mx) * (oi_ret[-n + i] - my) for i in range(n))
            denx = math.sqrt(sum((px_ret[-n + i] - mx) ** 2 for i in range(n)))
            deny = math.sqrt(sum((oi_ret[-n + i] - my) ** 2 for i in range(n)))
            if denx > 0 and deny > 0:
                divergence = num / (denx * deny)
                divergence = max(-1.0, min(1.0, divergence))

    # Exhaustion: long up-trend (positive vel) followed by current negative vel.
    exhaustion = 0
    if len(vel) >= 4:
        prior = vel[:-1]
        up_count = sum(1 for v in prior if v > 0)
        if up_count >= max(2, int(0.6 * len(prior))) and oi_velocity < 0:
            exhaustion = 1

    # Velocity z-score across the available velocity history.
    z_window = vel[:-1] if len(vel) > 1 else []
    velocity_z = _zscore(oi_velocity, z_window)

    return {
        "oi_velocity": float(oi_velocity),
        "oi_acceleration": float(oi_acceleration),
        "oi_price_divergence": float(divergence),
        "oi_exhaustion": int(exhaustion),
        "oi_velocity_zscore": float(velocity_z),
    }


def compute_sentiment_features(
    fear_greed_index: int,
    long_short_ratio: float,
    top_trader_ls_ratio: float,
    fear_greed_history: list[int] | None = None,
) -> dict:
    """Composite sentiment from F&G + retail vs. top-trader L/S ratios."""
    # Clamp F&G to [0, 100] and normalize.
    fg_raw = _safe_float(fear_greed_index, 50.0)
    fg = max(0.0, min(100.0, fg_raw))
    fg_norm = fg / 100.0

    if fg < 20.0:
        fg_extreme = 1     # extreme fear → contrarian bullish
    elif fg > 80.0:
        fg_extreme = -1    # extreme greed → contrarian bearish
    else:
        fg_extreme = 0

    retail = _safe_float(long_short_ratio, 1.0)
    top = _safe_float(top_trader_ls_ratio, 1.0)

    # Log-ratio divergence between top traders and retail, clamped to [-1, +1].
    if retail > 0 and top > 0:
        raw = math.log(top / retail)
        ls_divergence = max(-1.0, min(1.0, raw / 1.5))
    else:
        ls_divergence = 0.0

    # Composite sentiment:
    #   - F&G centered at 0 (greed positive),
    #   - top-trader divergence pushes signal toward smart-money direction,
    #   - extreme F&G flips a portion of the score (contrarian).
    base = (fg_norm - 0.5) * 2.0          # -1..+1
    sentiment_score = 0.6 * base + 0.4 * ls_divergence + 0.2 * fg_extreme
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    return {
        "fear_greed_norm": float(fg_norm),
        "fear_greed_extreme": int(fg_extreme),
        "ls_divergence": float(ls_divergence),
        "sentiment_score": float(sentiment_score),
    }


# =============================================================================
# MTF derived features (1H / 15m only) — appended for the post-lookahead
# feature expansion. These build on columns produced by the *existing*
# add_funding_features / add_oi_features / add_cvd_features functions and
# are wired into FeaturePipeline.build_mtf() for 1h/15m. They do not run
# on the 4H pipeline. Only shift/rolling/diff ops — no .over("_date").
# =============================================================================


def add_funding_momentum_features(
    df: pl.DataFrame,
    zscore_window: int = 72,  # 72 bars × 1H = 3 days
) -> pl.DataFrame:
    """Funding-rate momentum: change is more predictive than level.

    Requires ``funding_rate`` (from add_funding_features; zero-filled if
    no funding data). Added columns: funding_rate_change_1,
    funding_rate_change_3, funding_rate_zscore_rolling,
    funding_rate_acceleration.
    """
    if "funding_rate" not in df.columns:
        return df.with_columns([
            pl.lit(0.0).alias("funding_rate_change_1"),
            pl.lit(0.0).alias("funding_rate_change_3"),
            pl.lit(0.0).alias("funding_rate_zscore_rolling"),
            pl.lit(0.0).alias("funding_rate_acceleration"),
        ])

    fr = pl.col("funding_rate")
    df = df.with_columns([
        (fr - fr.shift(1)).fill_null(0.0).fill_nan(0.0).alias("funding_rate_change_1"),
        (fr - fr.shift(3)).fill_null(0.0).fill_nan(0.0).alias("funding_rate_change_3"),
    ])
    roll_mean = fr.rolling_mean(window_size=zscore_window, min_periods=10)
    roll_std = fr.rolling_std(window_size=zscore_window, min_periods=10)
    df = df.with_columns(
        safe_divide(fr - roll_mean, roll_std).alias("funding_rate_zscore_rolling")
    )
    df = df.with_columns(
        (pl.col("funding_rate_change_1") - pl.col("funding_rate_change_1").shift(1))
        .fill_null(0.0).fill_nan(0.0)
        .alias("funding_rate_acceleration")
    )
    _log.debug("add_funding_momentum_features: done")
    return df


def add_oi_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Open-interest momentum + price/OI divergence.

    Requires ``oi_value`` (from add_oi_features; zero-filled if no metrics)
    and ``returns_1`` (from add_price_features). Vectorized columns use a
    ``_vec`` suffix to avoid colliding with the scalar live-enrichment
    names (oi_acceleration / oi_price_divergence).

    Added columns: oi_delta_1h, oi_accel_vec, oi_price_div_vec.
    """
    if "oi_value" not in df.columns:
        return df.with_columns([
            pl.lit(0.0).alias("oi_delta_1h"),
            pl.lit(0.0).alias("oi_accel_vec"),
            pl.lit(0).cast(pl.Int8).alias("oi_price_div_vec"),
        ])

    df = df.with_columns(
        safe_divide(
            pl.col("oi_value") - pl.col("oi_value").shift(1),
            pl.col("oi_value").shift(1),
        ).alias("oi_delta_1h")
    )
    df = df.with_columns(
        (pl.col("oi_delta_1h") - pl.col("oi_delta_1h").shift(1))
        .fill_null(0.0).fill_nan(0.0)
        .alias("oi_accel_vec")
    )

    # Divergence on returns_1 (log return) vs OI direction:
    #   +1  price up   & OI down  (weak/unsupported move)
    #   -1  price down  & OI down  (panic / potential reversal)
    #    0  price & OI aligned
    price_ret = pl.col("returns_1") if "returns_1" in df.columns else pl.lit(0.0)
    price_up = price_ret > 0
    oi_down = pl.col("oi_delta_1h") < 0
    df = df.with_columns(
        pl.when(price_up & oi_down).then(1)
        .when(~price_up & oi_down).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("oi_price_div_vec")
    )
    _log.debug("add_oi_derived_features: done")
    return df


def add_cvd_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """CVD slope (volume-normalized) + hidden-flow divergence.

    Requires ``cvd`` (from add_cvd_features), ``volume_sma_20``
    (from add_volume_features), ``returns_1`` (from add_price_features).

    Added columns: cvd_slope_3bar, cvd_slope_6bar, cvd_divergence.
    """
    if "cvd" not in df.columns:
        return df.with_columns([
            pl.lit(0.0).alias("cvd_slope_3bar"),
            pl.lit(0.0).alias("cvd_slope_6bar"),
            pl.lit(0).cast(pl.Int8).alias("cvd_divergence"),
        ])

    vol_norm = (
        pl.col("volume_sma_20")
        if "volume_sma_20" in df.columns
        else pl.col("volume")
    )
    df = df.with_columns([
        safe_divide(pl.col("cvd") - pl.col("cvd").shift(3), vol_norm)
        .alias("cvd_slope_3bar"),
        safe_divide(pl.col("cvd") - pl.col("cvd").shift(6), vol_norm)
        .alias("cvd_slope_6bar"),
    ])

    # Hidden-flow divergence:
    #   +1  price up   & cvd_slope_3bar < 0  (hidden sellers — bearish)
    #   -1  price down & cvd_slope_3bar > 0  (hidden buyers — bullish)
    #    0  otherwise
    price_ret = pl.col("returns_1") if "returns_1" in df.columns else pl.lit(0.0)
    price_up = price_ret > 0
    df = df.with_columns(
        pl.when(price_up & (pl.col("cvd_slope_3bar") < 0)).then(1)
        .when(~price_up & (pl.col("cvd_slope_3bar") > 0)).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("cvd_divergence")
    )
    _log.debug("add_cvd_derived_features: done")
    return df
