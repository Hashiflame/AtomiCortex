"""
src/models/temporal_split.py

Honest train/test splitting for single- and multi-symbol datasets.

Background (root cause of ML-018):
  The MULTI dataset is a per-symbol concatenation
  ``[all BTC][all ETH][all SOL]`` — it is NOT globally time-sorted.
  The old split ``df.head(int(n*0.8))`` therefore put the *entire*
  BTC history (including the 2025 OOS period) into the training set
  and used only the SOL tail as "test". An ``--oos-start 2025-01-01``
  validation on BTC then scored the model on rows it had memorised
  → WR 93-98%, PF 13-63 (impossible metrics).

``temporal_split_multi`` cuts every symbol independently by wall-clock
time, so no symbol's future can leak into another symbol's training
window.
"""

from __future__ import annotations

import polars as pl


def compute_default_oos_start_ms(
    df: pl.DataFrame,
    time_col: str = "open_time",
    oos_fraction: float = 0.2,
) -> int:
    """OOS start = last ``oos_fraction`` of the *time* range (not rows).

    Uses the global [t_min, t_max] span so the cutoff is the same
    wall-clock instant for every symbol.
    """
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    if t_min is None or t_max is None:
        raise ValueError(
            f"Cannot compute default OOS start: '{time_col}' has no min/max"
        )
    total_duration = int(t_max) - int(t_min)
    return int(t_min) + int(total_duration * (1.0 - oos_fraction))


def temporal_split_multi(
    df: pl.DataFrame,
    oos_start_ms: int,
    symbol_col: str = "symbol",
    time_col: str = "open_time",
    embargo_bars: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a multi-symbol DataFrame by time, per-symbol.

    For each symbol:
      * ``train`` = rows where ``open_time <  oos_start_ms``
      * ``test``  = rows where ``open_time >= oos_start_ms``

    All train parts and all test parts are concatenated. No symbol's
    future leaks into another symbol's training set.

    ``embargo_bars`` drops the last N rows of each symbol's train part
    (the ones closest to the OOS boundary) so a forward-looking label
    (triple-barrier ``future_return``) cannot peek across the cut.
    """
    if time_col not in df.columns:
        raise ValueError(
            f"temporal_split_multi requires a '{time_col}' column"
        )

    def _embargo(part: pl.DataFrame) -> pl.DataFrame:
        if embargo_bars > 0 and len(part) > embargo_bars:
            return part.head(len(part) - embargo_bars)
        return part

    if symbol_col not in df.columns:
        # Single-symbol fallback — still a temporal split.
        s = df.sort(time_col)
        train = _embargo(s.filter(pl.col(time_col) < oos_start_ms))
        test = s.filter(pl.col(time_col) >= oos_start_ms)
    else:
        train_parts: list[pl.DataFrame] = []
        test_parts: list[pl.DataFrame] = []
        for symbol in sorted(df[symbol_col].unique().to_list()):
            sym_df = df.filter(pl.col(symbol_col) == symbol).sort(time_col)
            train_parts.append(
                _embargo(sym_df.filter(pl.col(time_col) < oos_start_ms))
            )
            test_parts.append(
                sym_df.filter(pl.col(time_col) >= oos_start_ms)
            )
        train = pl.concat(train_parts, how="diagonal").sort(time_col)
        test = pl.concat(test_parts, how="diagonal").sort(time_col)

    # ---- Invariants -----------------------------------------------------
    assert len(train) > 0, "Empty train set"
    assert len(test) > 0, (
        "Empty test set — check oos_start_ms "
        "(--oos-start-date may be past the dataset end)"
    )

    # No symbol may have a training row at or after the OOS boundary.
    if symbol_col in train.columns:
        per_sym_max = train.group_by(symbol_col).agg(
            pl.col(time_col).max().alias("_max_t")
        )
        leaked = per_sym_max.filter(pl.col("_max_t") >= oos_start_ms)
        assert leaked.is_empty(), (
            "Temporal leakage: train rows >= oos_start for symbols "
            f"{leaked[symbol_col].to_list()} "
            f"(e.g. BTC 2025 data leaked into training!)"
        )

    return train, test
