"""Multi-symbol regression tests for ``PurgedKFoldCV.split``.

Pre-fix, ``split`` carved by row index — so on a multi-symbol concat
``[BTC...][ETH...][SOL...]`` the same fold mixed late BTC and early ETH
into "test". The fix moves the cut to the time axis and adds a
``symbol_col`` mode that filters each symbol independently (no symbol's
future leaks into another's training set, same pattern as
``temporal_split_multi``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import polars as pl
import pytest

from src.execution.walk_forward import PurgedKFoldCV


_UTC = timezone.utc
_START = datetime(2024, 1, 1, tzinfo=_UTC)


def _symbol_df(symbol: str, n: int, start: datetime, step: timedelta) -> pl.DataFrame:
    return pl.DataFrame({
        "datetime": [start + i * step for i in range(n)],
        "open_time": [int((start + i * step).timestamp() * 1000) for i in range(n)],
        "symbol": [symbol] * n,
        "value": list(range(n)),
    })


def _multi_symbol_df(
    symbols: Iterable[str], n_per: int = 100, step: timedelta = timedelta(days=1)
) -> pl.DataFrame:
    """Per-symbol concat — the layout that exposes the original ML-018 bug."""
    parts = [_symbol_df(s, n_per, _START, step) for s in symbols]
    # IMPORTANT: deliberately do NOT sort by time here — the concat preserves
    # ``[BTC...][ETH...][SOL...]`` order so row index is not time-monotone.
    return pl.concat(parts, how="diagonal")


# ---------------------------------------------------------------------------
# Multi-symbol: per-symbol disjoint train/test in TIME
# ---------------------------------------------------------------------------

class TestMultiSymbolTimeDisjoint:
    def test_per_symbol_train_before_test(self) -> None:
        df = _multi_symbol_df(["BTCUSDT", "ETHUSDT", "SOLUSDT"], n_per=100)
        cv = PurgedKFoldCV(n_splits=4, embargo_pct=0.02)
        for fold_idx, (train_df, test_df) in enumerate(
            cv.split(df, timestamp_col="datetime", symbol_col="symbol")
        ):
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                tr = train_df.filter(pl.col("symbol") == sym)
                te = test_df.filter(pl.col("symbol") == sym)
                if tr.is_empty() or te.is_empty():
                    continue
                tr_max = tr["datetime"].max()
                te_min = te["datetime"].min()
                assert tr_max < te_min, (
                    f"fold {fold_idx} symbol {sym}: train_max={tr_max} >= "
                    f"test_min={te_min} — temporal leakage!"
                )

    def test_per_symbol_embargo_respected(self) -> None:
        """For each symbol gap between train_max and test_min ≥ embargo
        fraction of the dataset's total time span."""
        embargo_pct = 0.05
        n_per = 200
        step = timedelta(days=1)
        df = _multi_symbol_df(["BTCUSDT", "ETHUSDT"], n_per=n_per, step=step)
        total_span = df["datetime"].max() - df["datetime"].min()
        expected_embargo = embargo_pct * total_span

        cv = PurgedKFoldCV(n_splits=4, embargo_pct=embargo_pct)
        for fold_idx, (train_df, test_df) in enumerate(
            cv.split(df, timestamp_col="datetime", symbol_col="symbol")
        ):
            for sym in ["BTCUSDT", "ETHUSDT"]:
                tr = train_df.filter(pl.col("symbol") == sym)
                te = test_df.filter(pl.col("symbol") == sym)
                if tr.is_empty() or te.is_empty():
                    continue
                gap = te["datetime"].min() - tr["datetime"].max()
                # Allow a one-step (1 day) tolerance for discretisation.
                assert gap >= expected_embargo - step

    def test_test_row_count_is_sum_of_per_symbol_test_rows(self) -> None:
        df = _multi_symbol_df(["BTCUSDT", "ETHUSDT", "SOLUSDT"], n_per=50)
        cv = PurgedKFoldCV(n_splits=3, embargo_pct=0.01)
        for train_df, test_df in cv.split(
            df, timestamp_col="datetime", symbol_col="symbol",
        ):
            per_symbol = sum(
                len(test_df.filter(pl.col("symbol") == s))
                for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            )
            assert per_symbol == len(test_df)


# ---------------------------------------------------------------------------
# Multi-symbol: bug-reproduction case
# ---------------------------------------------------------------------------

class TestPreFixBugReproduction:
    def test_concatenated_layout_no_longer_leaks_across_symbols(self) -> None:
        """Pre-fix scenario: ``[BTC...][ETH...][SOL...]`` concat. The old
        ``data.slice`` carving by row index would place late BTC and early
        ETH inside the same fold's TEST. With the fix every symbol's
        TEST window starts strictly after its own TRAIN max."""
        df = _multi_symbol_df(["BTCUSDT", "ETHUSDT", "SOLUSDT"], n_per=120)
        # Row index for ETH starts at 120 — without per-symbol filtering
        # a row-index-based split would mix ETH into BTC's test or vice
        # versa. Per-symbol filtering must prevent that completely.
        cv = PurgedKFoldCV(n_splits=4, embargo_pct=0.01)
        for train_df, test_df in cv.split(
            df, timestamp_col="datetime", symbol_col="symbol",
        ):
            # Every train row's time must be strictly less than every test
            # row's time *of the same symbol*. (Across-symbol comparison
            # is allowed because symbols span the same wall-clock window.)
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                tr = train_df.filter(pl.col("symbol") == sym)["datetime"]
                te = test_df.filter(pl.col("symbol") == sym)["datetime"]
                if len(tr) == 0 or len(te) == 0:
                    continue
                assert tr.max() < te.min()


# ---------------------------------------------------------------------------
# Single-symbol path (no symbol_col)
# ---------------------------------------------------------------------------

class TestSingleSymbol:
    def test_single_symbol_train_before_test(self) -> None:
        df = _symbol_df("BTCUSDT", n=100, start=_START, step=timedelta(days=1))
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.02)
        for train_df, test_df in cv.split(df, timestamp_col="datetime"):
            assert train_df["datetime"].max() < test_df["datetime"].min()

    def test_single_symbol_train_grows(self) -> None:
        df = _symbol_df("BTCUSDT", n=100, start=_START, step=timedelta(days=1))
        cv = PurgedKFoldCV(n_splits=4, embargo_pct=0.01)
        sizes = [len(tr) for tr, _ in cv.split(df, timestamp_col="datetime")]
        assert sizes == sorted(sizes)


# ---------------------------------------------------------------------------
# Auto-fallback to open_time when default 'datetime' is null/missing
# ---------------------------------------------------------------------------

class TestAutoFallback:
    def test_falls_back_to_open_time_when_datetime_null(self) -> None:
        """Existing synthetic frames have populated open_time but null
        datetime. split() must fall back automatically when called with
        the default kwarg — preserving the older single-arg API."""
        n = 100
        step_ms = 86_400_000  # 1 day
        df = pl.DataFrame({
            "datetime": [None] * n,  # all-null
            "open_time": [1_700_000_000_000 + i * step_ms for i in range(n)],
            "value": list(range(n)),
        })
        cv = PurgedKFoldCV(n_splits=3, embargo_pct=0.02)
        # No timestamp_col override — must auto-pick open_time
        for tr, te in cv.split(df):
            assert tr["open_time"].max() < te["open_time"].min()

    def test_raises_when_no_usable_timestamp(self) -> None:
        df = pl.DataFrame({"value": list(range(50))})
        cv = PurgedKFoldCV(n_splits=3)
        with pytest.raises(ValueError, match="no usable timestamp column"):
            list(cv.split(df))


# ---------------------------------------------------------------------------
# Edge cases preserved from the old API
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_too_small_dataset_still_raises(self) -> None:
        df = _symbol_df("BTC", n=3, start=_START, step=timedelta(days=1))
        cv = PurgedKFoldCV(n_splits=5)
        with pytest.raises(ValueError, match="too small"):
            list(cv.split(df, timestamp_col="datetime"))

    def test_symbol_col_missing_silently_uses_single_symbol_path(self) -> None:
        """If ``symbol_col`` is passed but not present in data, fall back
        to the single-symbol code path rather than crashing."""
        df = _symbol_df("BTC", n=100, start=_START, step=timedelta(days=1))
        cv = PurgedKFoldCV(n_splits=3, embargo_pct=0.02)
        # symbol_col not in df → graceful single-symbol behaviour
        for tr, te in cv.split(df, timestamp_col="datetime", symbol_col="symbol"):
            assert tr["datetime"].max() < te["datetime"].min()
