"""
Triple-Barrier Labeling (López de Prado, AFML Chapter 3).

Replaces fixed-horizon sign(return) with economically meaningful labels
that respect the actual path of prices.

Labels:
  +1: upper barrier hit first (profitable long / loss on short)
  -1: lower barrier hit first (profitable short / loss on long)
   0: vertical barrier hit (no clear direction, excluded from training)

Barriers are volatility-scaled:
  upper = entry × (1 + pt_multiplier × atr_pct)
  lower = entry × (1 - sl_multiplier × atr_pct)
  vertical = entry_bar + max_holding_bars

This module also emits ``future_return`` — the *realized* close-to-close
return at the bar the trade actually exits — so the validator / DSR /
PBO P&L reflects real price paths (the touched barrier is the trade
outcome, not an arbitrary fixed horizon):

  label=+1 → future_return = (close[t+k] - close[t]) / close[t]
             where t+k is the bar the upper barrier is first breached
  label=-1 → future_return = (close[t+k] - close[t]) / close[t]
             where t+k is the bar the lower barrier is first breached
  label= 0 → future_return = (close[t+max_holding] - close[t]) / close[t]
             (real return at the vertical barrier)

NB: ``future_return`` is the *real close at the exit bar*, NOT the
barrier constant ±pt/sl × atr_pct. Emitting the constant would make
future_return a deterministic function of (label, atr_pct) — every
validator P&L metric would degenerate into a pure function of
label-classification accuracy with no price-path risk, inflating
PF/Sharpe (see git history: multi-symbol leakage audit).

``future_return`` is NOT a feature (it lives in _EXCLUDE_COLUMNS) — it
is consumed solely for P&L in the validators.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.logger import get_logger

_log = get_logger(__name__)


def apply_triple_barrier(
    df: pl.DataFrame,
    close_col: str = "close",
    atr_col: str = "atr_pct",        # ATR / price (dimensionless)
    pt_multiplier: float = 1.5,       # profit-taking barrier = 1.5 × ATR
    sl_multiplier: float = 1.0,       # stop-loss barrier   = 1.0 × ATR
    max_holding_bars: int = 6,        # vertical barrier (bars)
    min_atr: float = 0.001,           # atr_pct floor (1 bp) — div-zero guard
) -> pl.DataFrame:
    """Add ``label`` (+1/-1/0) and ``future_return`` columns.

    For each bar ``t``:
      - ``atr_t       = max(atr_pct[t], min_atr)``
      - ``upper       = close[t] × (1 + pt_multiplier × atr_t)``
      - ``lower       = close[t] × (1 - sl_multiplier × atr_t)``
      - scan ``close[t+1 .. t+max_holding_bars]``; first barrier touched
        sets the label and the realized ``future_return``.
      - neither touched → label 0, future_return = vertical-bar return.

    The last ``max_holding_bars`` rows have no full forward window →
    their label/future_return are undefined and the rows are dropped
    (mirrors the previous ``df.head(len - forward_bars)`` behavior).

    Implementation is NumPy-vectorized over the (small) holding horizon;
    strictly causal — bar ``t`` only ever reads ``close[t+1 ..
    t+max_holding_bars]``, never further.
    """
    if close_col not in df.columns or atr_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{close_col}' and '{atr_col}' columns"
        )
    if max_holding_bars < 1:
        raise ValueError("max_holding_bars must be >= 1")

    close = df[close_col].to_numpy().astype(np.float64)
    atr = df[atr_col].to_numpy().astype(np.float64)
    # Guard: NaN/neg ATR → floor; never divide/scale by zero.
    atr = np.where(np.isfinite(atr), atr, min_atr)
    atr = np.maximum(atr, min_atr)

    n = len(close)
    label = np.full(n, np.nan, dtype=np.float64)
    fret = np.full(n, np.nan, dtype=np.float64)
    # AFML Ch.4: each label needs its REAL exit bar t1_i for sample
    # uniqueness weights. Default sentinel -1 marks the tail rows that
    # never form a forward window (dropped below).
    t1 = np.full(n, -1, dtype=np.int64)

    valid_n = n - max_holding_bars
    if valid_n <= 0:
        # Not enough bars to form a single forward window.
        df = df.with_columns([
            pl.Series("label", label),
            pl.Series("future_return", fret),
            pl.Series("t1_bar", t1),
        ])
        return df.head(0)

    entry = close[:valid_n]
    atr_v = atr[:valid_n]
    upper = entry * (1.0 + pt_multiplier * atr_v)
    lower = entry * (1.0 - sl_multiplier * atr_v)

    hit = np.zeros(valid_n, dtype=bool)
    lab = np.zeros(valid_n, dtype=np.float64)
    fr = np.zeros(valid_n, dtype=np.float64)
    # Per-label exit offset (1..max_holding_bars). Initialised to the
    # vertical-barrier offset; overwritten if PT/SL triggers earlier.
    exit_k = np.full(valid_n, max_holding_bars, dtype=np.int64)

    # Sweep the holding horizon k = 1 .. max_holding_bars. The first
    # bar that breaches a barrier wins (``~hit`` mask freezes earlier
    # touches). upper is checked before lower; they cannot both trigger
    # on one bar since lower < entry < upper.
    for k in range(1, max_holding_bars + 1):
        fut = close[k:k + valid_n]
        up = (~hit) & (fut >= upper)
        dn = (~hit) & (~up) & (fut <= lower)
        lab[up] = 1.0
        lab[dn] = -1.0
        # Realized return at the *touch bar* (real close, not the
        # barrier constant). Using ±pt/sl×atr_pct here makes
        # future_return a deterministic function of (label, atr_pct):
        # sign(future_return) ≡ sign(label) and |pnl| is fixed, so any
        # validator P&L (WR/PF/Sharpe) collapses to a pure function of
        # label-classification accuracy with zero price-path risk
        # (tautology → PF/Sharpe explode). The real close at the bar
        # the barrier was first breached restores genuine overshoot
        # variance. Strictly causal: only close[t+k], k ≤ holding.
        fr[up] = (fut[up] - entry[up]) / entry[up]
        fr[dn] = (fut[dn] - entry[dn]) / entry[dn]
        # Record the actual exit offset for newly-hit labels only.
        new_hit = up | dn
        exit_k[new_hit] = k
        hit |= new_hit

    # Vertical barrier: realized return at close[t + max_holding_bars].
    not_hit = ~hit
    vert_close = close[max_holding_bars:max_holding_bars + valid_n]
    lab[not_hit] = 0.0
    fr[not_hit] = (vert_close[not_hit] - entry[not_hit]) / entry[not_hit]
    # exit_k for not_hit already defaults to max_holding_bars.

    label[:valid_n] = lab
    fret[:valid_n] = fr
    # t1_bar is the input-df bar INDEX where the label exits: entry
    # index i + exit_offset k. PT/SL fires give k < max_holding;
    # timeouts give k = max_holding (the vertical barrier).
    t1[:valid_n] = np.arange(valid_n) + exit_k

    df = df.with_columns([
        pl.Series("label", label),
        pl.Series("future_return", fret),
        pl.Series("t1_bar", t1),
    ])
    # Drop the trailing rows with an undefined forward window.
    df = df.head(valid_n)

    stats = label_statistics(df)
    _log.info(
        f"Triple-barrier: {stats['total']:,} rows | "
        f"long={stats['long']} ({stats['long_pct']:.1f}%) | "
        f"short={stats['short']} ({stats['short_pct']:.1f}%) | "
        f"vertical={stats['vertical']} ({stats['vertical_pct']:.1f}%) | "
        f"coverage={stats['coverage']:.1f}%"
    )
    return df


def label_statistics(df: pl.DataFrame, label_col: str = "label") -> dict:
    """Label-distribution diagnostics.

    Returns total / long(+1) / short(-1) / vertical(0) counts, their
    percentages, and ``coverage`` = (long+short)/total in percent.
    """
    if label_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{label_col}' column")

    col = df[label_col].drop_nulls()
    total = int(col.len())
    if total == 0:
        return {
            "total": 0, "long": 0, "short": 0, "vertical": 0,
            "long_pct": 0.0, "short_pct": 0.0,
            "vertical_pct": 0.0, "coverage": 0.0,
        }

    long_n = int((col == 1).sum())
    short_n = int((col == -1).sum())
    vert_n = int((col == 0).sum())
    return {
        "total": total,
        "long": long_n,
        "short": short_n,
        "vertical": vert_n,
        "long_pct": 100.0 * long_n / total,
        "short_pct": 100.0 * short_n / total,
        "vertical_pct": 100.0 * vert_n / total,
        "coverage": 100.0 * (long_n + short_n) / total,
    }
