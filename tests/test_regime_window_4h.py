"""Tests for the 4H feature window length (PR-A).

The bug being fixed
-------------------
``MLStrategyConfig.warmup_bars`` was 300 and drove *both* the trading-readiness
gate and the size of the preloaded 4H buffer.  ``RegimeDetector.detect_all()``
gives every row before ``min_bars`` (= ``hurst_window`` = 300) neutral
constants, so with a 300-bar buffer the row returned by
``build_from_buffer(single_row=True)`` — the one that reaches the model — was
*always* ``regime="range"``, ``hurst=0.5``, ``atr_percentile=0.5``,
``atr_pct=0.0``, ``trend_strength=0.0``, ``regime_confidence=0.0``.

These tests pin the required window length (``WARMUP_ROWS + atr_lookback``),
the deque that has to hold it, the REST/Parquet loaders that have to fetch it,
and the WARNING that has to fire when the buffer is nevertheless too short.
"""
from __future__ import annotations

import math

import polars as pl
import pytest
from loguru import logger as _loguru_logger

from src.execution.strategies.ml_strategy import MLStrategyConfig
from src.features.feature_pipeline import FeaturePipeline
from src.features.live_feature_state import LiveFeatureState
from src.features.regime_detector import RegimeDetector

_BAR_MS_4H = 4 * 3_600_000
_START_CLOSE_MS = 1_600_000_000_000


@pytest.fixture
def loguru_warnings():
    """Capture loguru WARNING records into a list."""
    sink: list[str] = []
    sink_id = _loguru_logger.add(
        lambda msg: sink.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield sink
    finally:
        _loguru_logger.remove(sink_id)


class _NullStore:
    """``build_from_buffer`` never touches the DataStore — this is a stand-in."""


class _FakeBar:
    """Minimal stand-in for a Nautilus Bar (``add_bar`` only reads these)."""

    def __init__(self, ts_event_ms: int, o: float, h: float, lo: float,
                 c: float, v: float) -> None:
        self.ts_event = ts_event_ms * 1_000_000  # ms → ns
        self.open = o
        self.high = h
        self.low = lo
        self.close = c
        self.volume = v


def _ohlcv(n: int) -> list[dict]:
    """Deterministic 4H OHLCV with drift + oscillation.

    No RNG: the offline reference and the live-buffer candidate in
    ``test_tail_row_atr_percentile_matches_offline`` must be bit-identical.
    Drift keeps ADX above zero, the oscillation keeps ATR above zero, and
    both keep ``hurst``/``atr_percentile`` away from their neutral defaults.
    """
    rows: list[dict] = []
    close = 30_000.0
    for i in range(n):
        prev_close = close
        close = prev_close + 25.0 + 60.0 * math.sin(i / 7.0)
        spread = 80.0 + 40.0 * abs(math.cos(i / 5.0))
        rows.append({
            "open_time": _START_CLOSE_MS + i * _BAR_MS_4H,
            "open": prev_close,
            "high": max(prev_close, close) + spread,
            "low": min(prev_close, close) - spread,
            "close": close,
            "volume": 1_000.0 + 25.0 * math.sin(i / 3.0),
        })
    return rows


def _offline_df(n: int) -> pl.DataFrame:
    return pl.DataFrame(_ohlcv(n))


def _seeded_state(n: int) -> LiveFeatureState:
    """Push *n* synthetic bars through the real live path (add_bar → deque)."""
    state = LiveFeatureState()
    for row in _ohlcv(n):
        # add_bar derives open_time as ts_event - bar_duration, so feed it the
        # close time to land on the open_time this row was generated with.
        bar = _FakeBar(
            row["open_time"] + _BAR_MS_4H,
            row["open"], row["high"], row["low"], row["close"], row["volume"],
        )
        state.add_bar(bar, interval="4h")
    return state


def _tail_row(df: pl.DataFrame) -> dict:
    pipeline = FeaturePipeline(_NullStore(), "BTCUSDT", "4h")
    out = pipeline.build_from_buffer(df=df, single_row=True)
    assert len(out) == 1
    return {c: out[c][0] for c in out.columns}


# ═══════════════════════════════════════════════════════════════════════════
# 1-2. The required window and the container that has to hold it
# ═══════════════════════════════════════════════════════════════════════════


class TestRequiredWindow:
    def test_history_bars_covers_warmup_plus_atr_lookback(self) -> None:
        """history_bars must equal the contract in build_from_buffer's docstring.

        ``feature_pipeline.build_from_buffer`` documents that the caller must
        supply ``_WARMUP_ROWS + longest lookback`` rows; for 4H the longest
        lookback is ``atr_lookback``.
        """
        from src.features.window_sizes import (
            ATR_LOOKBACK_4H,
            HISTORY_BARS_4H,
            HURST_WINDOW_4H,
            WARMUP_ROWS,
            required_history_bars,
        )

        assert HISTORY_BARS_4H == required_history_bars(ATR_LOOKBACK_4H)
        assert HISTORY_BARS_4H == WARMUP_ROWS + ATR_LOOKBACK_4H
        assert MLStrategyConfig().history_bars == HISTORY_BARS_4H

        # One source of truth: the detector defaults must come from the same
        # module, otherwise the number silently forks in two places.
        detector = RegimeDetector()
        assert detector.atr_lookback == ATR_LOOKBACK_4H
        assert detector.hurst_window == HURST_WINDOW_4H

        # warmup_bars keeps its own meaning (trading-readiness gate) and must
        # NOT be dragged along with history_bars.
        assert MLStrategyConfig().warmup_bars == 300

    def test_bar_buffer_4h_maxlen_covers_history_bars(self) -> None:
        """A deque shorter than the window silently truncates the buffer."""
        from src.features.window_sizes import HISTORY_BARS_4H

        buf = LiveFeatureState().bar_buffer_4h
        assert buf.maxlen is not None
        assert buf.maxlen >= HISTORY_BARS_4H


# ═══════════════════════════════════════════════════════════════════════════
# 3-4. The row that reaches the model
# ═══════════════════════════════════════════════════════════════════════════


class TestTailRow:
    def test_tail_row_is_not_constant_at_configured_history(self) -> None:
        """The last row must carry real regime values, not neutral defaults."""
        from src.features.window_sizes import HISTORY_BARS_4H, HURST_WINDOW_4H

        # Control: at exactly min_bars rows every row (including the last) is
        # below detect_all's threshold, so the whole set collapses to the
        # documented neutral constants. This proves the assertions below are
        # not vacuous.
        short = _tail_row(_seeded_state(HURST_WINDOW_4H).get_bar_df("4h"))
        assert short["regime_confidence"] == 0.0
        assert short["trend_strength"] == 0.0
        assert short["atr_pct"] == 0.0
        assert short["regime"] == "range"

        df = _seeded_state(HISTORY_BARS_4H).get_bar_df("4h")
        assert len(df) == HISTORY_BARS_4H, "deque truncated the buffer"

        row = _tail_row(df)
        assert row["regime_confidence"] > 0.0
        assert row["trend_strength"] > 0.0
        assert row["atr_pct"] > 0.0
        # Supporting check only: hurst == 0.5 is a legitimate value for a true
        # random walk, so it cannot carry the assertion on its own.
        assert row["hurst"] != 0.5

    def test_tail_row_atr_percentile_matches_offline(self) -> None:
        """The live tail row must equal the offline row at the same index.

        Reference: a long frame run through ``detect_all`` exactly as the
        offline ``build()`` does. Candidate: the first ``HISTORY_BARS_4H`` bars
        pushed through the live path (add_bar → deque → get_bar_df →
        build_from_buffer).

        Comparing ``hurst`` is valid despite the amortisation in detect_all
        (``(i - min_bars) % _RECOMPUTE_EVERY``): that phase depends only on
        ``i`` and ``min_bars``, never on ``n``. Both sides evaluate the same
        ``i = HISTORY_BARS_4H - 1`` with the same ``min_bars``, so they reuse
        the Hurst value at the same lag — the equality survives any future
        change to ``_RECOMPUTE_EVERY``.
        """
        from src.features.window_sizes import HISTORY_BARS_4H

        reference = RegimeDetector().detect_all(_offline_df(HISTORY_BARS_4H + 160))
        target = HISTORY_BARS_4H - 1

        row = _tail_row(_seeded_state(HISTORY_BARS_4H).get_bar_df("4h"))

        assert row["regime"] == reference["regime"][target]
        assert row["atr_percentile"] == pytest.approx(
            reference["atr_percentile"][target], abs=1e-9,
        )
        assert row["hurst"] == pytest.approx(
            reference["hurst"][target], abs=1e-9,
        )
        assert row["atr_pct"] == pytest.approx(
            reference["atr_pct"][target], abs=1e-9,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Fail-soft WARNING
# ═══════════════════════════════════════════════════════════════════════════


class TestShortBufferWarning:
    @staticmethod
    def _short_buffer_warnings(sink: list[str]) -> list[str]:
        return [m for m in sink if "build_from_buffer" in m]

    def test_build_from_buffer_warns_on_short_buffer(self, loguru_warnings) -> None:
        from src.features.window_sizes import ATR_LOOKBACK_4H, HISTORY_BARS_4H

        short_pipeline = FeaturePipeline(_NullStore(), "BTCUSDT", "4h")
        short_df = _seeded_state(HISTORY_BARS_4H // 2).get_bar_df("4h")

        short_pipeline.build_from_buffer(df=short_df, single_row=True)
        warns = self._short_buffer_warnings(loguru_warnings)
        assert len(warns) == 1, warns
        assert f"< {HISTORY_BARS_4H} required" in warns[0]
        assert f"atr_lookback={ATR_LOOKBACK_4H}" in warns[0]
        assert str(len(short_df)) in warns[0]

        # One-shot per pipeline instance — must not spam every inference tick.
        short_pipeline.build_from_buffer(df=short_df, single_row=True)
        assert len(self._short_buffer_warnings(loguru_warnings)) == 1

        # A long-enough buffer must stay silent (fresh instance: the flag above
        # is per-instance).
        long_pipeline = FeaturePipeline(_NullStore(), "BTCUSDT", "4h")
        long_df = _seeded_state(HISTORY_BARS_4H).get_bar_df("4h")
        long_pipeline.build_from_buffer(df=long_df, single_row=True)
        assert len(self._short_buffer_warnings(loguru_warnings)) == 1
