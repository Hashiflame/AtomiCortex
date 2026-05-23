"""
src/features/agg_15m.py

15m → 4H aggregated micro-structure features (Block-3 Step 3).

For each 4H bar with open_time T, we summarize the 16 child 15m bars
whose open_time falls in [T, T + 4h). All four features therefore only
read data inside the 4H bar itself — same causal contract as the
existing 4H OHLCV-derived features (which are also known at close T+4h).

Emitted columns
---------------
agg_15m_cvd_sum     Σ (2·taker_buy_volume − volume) over the 16 child bars.
                    Same sign convention as the existing `cvd` column.
agg_15m_cvd_slope   OLS slope of cumulative CVD vs bar index over the
                    window, normalized by max(|cum_cvd|, 1). Captures
                    *direction* of intra-bar order flow build-up.
agg_15m_realized_vol  √Σ rᵢ²  with rᵢ = ln(close[i] / close[i-1]).
                    A finer-grain realized-volatility proxy than 4H ATR.
agg_15m_orb_ratio   Donchian-style intra-bar breakout fraction: the
                    fraction of 15m bars whose `high` exceeds the prior
                    4-bar window maximum, OR whose `low` falls below the
                    prior 4-bar window minimum. Proxies for opening-range
                    breakouts without needing a session calendar.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.logger import get_logger

_log = get_logger(__name__)

AGG_15M_FEATURE_NAMES: list[str] = [
    "agg_15m_cvd_sum",
    "agg_15m_cvd_slope",
    "agg_15m_realized_vol",
    "agg_15m_orb_ratio",
]


def _bucket_stats(group: pl.DataFrame) -> dict[str, float]:
    """Compute the 4 features from a single 4H-aligned 15m bucket.

    Pure NumPy on small (≤16) arrays — vectorized polars apply over the
    grouped 15m frame in the caller.
    """
    n = len(group)
    if n == 0:
        return {f: 0.0 for f in AGG_15M_FEATURE_NAMES}

    close = group["close"].to_numpy().astype(np.float64)
    high = group["high"].to_numpy().astype(np.float64)
    low = group["low"].to_numpy().astype(np.float64)
    vol = group["volume"].to_numpy().astype(np.float64)
    tbv = group["taker_buy_volume"].to_numpy().astype(np.float64)

    # 1. CVD sum (signed bar-by-bar order flow imbalance)
    cvd_bars = 2.0 * tbv - vol
    cvd_sum = float(cvd_bars.sum())

    # 2. CVD slope (OLS over cumulative CVD; normalized for cross-asset
    #    comparability so SOL vs BTC stay on the same scale)
    cum = np.cumsum(cvd_bars)
    if n >= 2:
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        cov = float(((x - x_mean) * (cum - cum.mean())).sum())
        var = float(((x - x_mean) ** 2).sum())
        slope = cov / var if var > 0 else 0.0
        denom = max(abs(cum).max(), 1.0)
        cvd_slope = slope / denom
    else:
        cvd_slope = 0.0

    # 3. Realized volatility (sum of squared log returns inside the bar)
    if n >= 2 and (close > 0).all():
        log_ret = np.diff(np.log(close))
        realized_vol = float(np.sqrt(np.sum(log_ret ** 2)))
    else:
        realized_vol = 0.0

    # 4. ORB-style breakout ratio — count bars whose high/low pierces a
    #    rolling 4-bar prior window. Requires at least 5 bars; below that
    #    we fall back to 0 (no breakouts measurable).
    if n >= 5:
        breakouts = 0
        for i in range(4, n):
            win_hi = high[i - 4 : i].max()
            win_lo = low[i - 4 : i].min()
            if high[i] > win_hi or low[i] < win_lo:
                breakouts += 1
        orb_ratio = breakouts / (n - 4)
    else:
        orb_ratio = 0.0

    return {
        "agg_15m_cvd_sum": cvd_sum,
        "agg_15m_cvd_slope": cvd_slope,
        "agg_15m_realized_vol": realized_vol,
        "agg_15m_orb_ratio": orb_ratio,
    }


def add_15m_aggregated_features(
    df_4h: pl.DataFrame,
    df_15m: pl.DataFrame,
    *,
    open_time_col: str = "open_time",
) -> pl.DataFrame:
    """Join 4-column 15m aggregates onto a 4H feature DataFrame.

    The mapping is by floor(15m.open_time, 4h). Missing 4H buckets (e.g.
    data gaps on the 15m side) are filled with 0.0 — same convention as
    the warmup-tail and gap-tolerance used elsewhere in the pipeline.

    Both inputs must carry an `open_time` column expressed in the same
    integer millisecond epoch units (matches DataStore.get_klines output).
    """
    if df_4h.is_empty():
        return df_4h
    if df_15m.is_empty():
        _log.warning("add_15m_aggregated_features: empty 15m frame, "
                     "writing zeros for all agg_15m_* columns")
        zeros = {f: pl.Series(f, [0.0] * len(df_4h)) for f in AGG_15M_FEATURE_NAMES}
        return df_4h.with_columns(list(zeros.values()))

    # Bucket each 15m bar to its parent 4H open_time (ms = 4*60*60*1000).
    bucket_ms = 4 * 60 * 60 * 1000
    df_15m = df_15m.with_columns(
        ((pl.col(open_time_col) // bucket_ms) * bucket_ms).alias("_bucket")
    )

    # Group → per-bucket stats. Partition-by gives one mini-DataFrame per
    # bucket; ≤16 rows each, so the python loop cost is bounded.
    rows: list[dict[str, float | int]] = []
    for bucket_df in df_15m.partition_by("_bucket", maintain_order=True):
        bucket = int(bucket_df["_bucket"][0])
        stats = _bucket_stats(bucket_df)
        rows.append({"_bucket": bucket, **stats})

    agg_df = pl.DataFrame(rows)

    # Join onto 4H frame; fill missing buckets with 0.0.
    out = (
        df_4h.join(agg_df, left_on=open_time_col, right_on="_bucket", how="left")
        .with_columns([
            pl.col(c).fill_null(0.0).cast(pl.Float64).alias(c)
            for c in AGG_15M_FEATURE_NAMES
        ])
    )

    n_missing = int(
        out[AGG_15M_FEATURE_NAMES[0]].eq(0.0).sum() - agg_df.height + len(out)
    )
    _log.info(
        f"15m aggregates joined: {len(agg_df)} buckets → {len(out)} 4H rows "
        f"(missing buckets filled with 0: ≈{max(n_missing,0)})"
    )
    return out
