"""Tests for the offline feature-window length in ``FeaturePipeline.build`` (PR-J).

The bug being fixed
-------------------
``build()`` sliced exactly ``_WARMUP_ROWS`` (200) rows off the head of every
feature matrix, but the regime columns it had just computed need more than
that:

* ``RegimeDetector.detect_all(df, min_bars=300)`` gives every row before
  ``min_bars`` the neutral constants ``regime="range"``, ``hurst=0.5``,
  ``atr_pct=0.0``, ``atr_percentile=0.5``, ``trend_strength=0.0``,
  ``regime_confidence=0.0``;
* ``atr_percentile`` is a percentile over the trailing ``atr_lookback``
  (540 for 4H) rows, so rows below that index carry a partial-window value.

With a 200-row trim the first ~340 rows of every training matrix were
therefore either constants or partial-window percentiles — the same defect
PR-A fixed on the live side, where the buffer was raised to
``required_history_bars(ATR_LOOKBACK_4H) == 740``.

These tests pin the offline trim to ``required_history_bars(atr_lookback)``
read from the detector ``build()`` actually constructs, the fail-soft
fallback when the input is shorter than that, and the WARNING that has to
fire instead of a silent degradation.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import polars as pl
import pytest
from loguru import logger as _loguru_logger

from src.features.feature_pipeline import FeaturePipeline
from src.features.regime_detector import RegimeDetector
from src.features.window_sizes import (
    ATR_LOOKBACK_4H,
    WARMUP_ROWS,
    required_history_bars,
)

_BASE_MS = 1_704_067_200_000   # 2024-01-01 00:00 UTC
_BAR_MS_4H = 4 * 3_600_000


# ---------------------------------------------------------------------------
# Fixtures / helpers
#
# The loguru sink and the OHLCV generator are duplicated here on purpose
# (they also exist in tests/test_regime_window_4h.py). Hoisting them into
# conftest.py would drag a third test file into this PR's diff for cosmetic
# reasons only.
# ---------------------------------------------------------------------------


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


def _build_warnings(sink: list[str]) -> list[str]:
    """Only the short-input WARNING from ``build()``.

    The NaN audit in step 12 also logs at WARNING level ("... after build:"),
    hence the ``build[`` prefix rather than a bare substring match.
    """
    return [m for m in sink if "build[" in m]


def _ohlcv(n: int) -> dict[str, list]:
    """Deterministic 4H OHLCV with drift + oscillation.

    No RNG: the reference frame fed straight to ``detect_all`` and the frame
    served through the mock store must be bit-identical. The drift keeps ADX
    above zero, the oscillation keeps ATR above zero and moving, and both
    together keep ``atr_percentile`` away from its neutral 0.5 default.
    """
    open_times: list[int] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    takers: list[float] = []

    close = 30_000.0
    for i in range(n):
        prev_close = close
        close = prev_close + 25.0 + 60.0 * math.sin(i / 7.0)
        spread = 80.0 + 40.0 * abs(math.cos(i / 5.0))
        volume = 1_000.0 + 25.0 * math.sin(i / 3.0)

        open_times.append(_BASE_MS + i * _BAR_MS_4H)
        opens.append(prev_close)
        highs.append(max(prev_close, close) + spread)
        lows.append(min(prev_close, close) - spread)
        closes.append(close)
        volumes.append(volume)
        takers.append(volume * (0.5 + 0.1 * math.sin(i / 11.0)))

    return {
        "open_time": open_times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "taker_buy_volume": takers,
    }


def _klines(n: int) -> pl.DataFrame:
    cols = _ohlcv(n)
    return pl.DataFrame({
        "open_time": pl.Series(cols["open_time"], dtype=pl.Int64),
        "open": pl.Series(cols["open"], dtype=pl.Float64),
        "high": pl.Series(cols["high"], dtype=pl.Float64),
        "low": pl.Series(cols["low"], dtype=pl.Float64),
        "close": pl.Series(cols["close"], dtype=pl.Float64),
        "volume": pl.Series(cols["volume"], dtype=pl.Float64),
        "taker_buy_volume": pl.Series(cols["taker_buy_volume"], dtype=pl.Float64),
        "quote_volume": pl.Series(
            [c * v for c, v in zip(cols["close"], cols["volume"])], dtype=pl.Float64,
        ),
        "trade_count": pl.Series([100] * n, dtype=pl.Int32),
        "close_time": pl.Series(
            [_BASE_MS + (i + 1) * _BAR_MS_4H - 1 for i in range(n)], dtype=pl.Int64,
        ),
        "symbol": pl.Series(["BTCUSDT"] * n, dtype=pl.Utf8),
    })


def _funding(n_bars: int) -> pl.DataFrame:
    """Funding every 8H — one record per two 4H bars."""
    n = max(1, n_bars // 2)
    return pl.DataFrame({
        "fundingTime": pl.Series(
            [_BASE_MS + i * 2 * _BAR_MS_4H for i in range(n)], dtype=pl.Int64,
        ),
        "fundingRate": pl.Series(
            [0.0001 * (1.0 + 0.5 * math.sin(i / 4.0)) for i in range(n)],
            dtype=pl.Float64,
        ),
        "symbol": pl.Series(["BTCUSDT"] * n, dtype=pl.Utf8),
    })


def _metrics(n_bars: int) -> pl.DataFrame:
    """Open-interest metrics — one record per 4H bar."""
    n = max(1, n_bars)
    return pl.DataFrame({
        "create_time": pl.Series(
            [_BASE_MS + i * _BAR_MS_4H for i in range(n)], dtype=pl.Int64,
        ),
        "sum_open_interest_value": pl.Series(
            [5e9 + 2e7 * math.sin(i / 9.0) for i in range(n)], dtype=pl.Float64,
        ),
        "count_long_short_ratio": pl.Series(
            [1.05 + 0.05 * math.sin(i / 6.0) for i in range(n)], dtype=pl.Float64,
        ),
        "sum_taker_long_short_vol_ratio": pl.Series(
            [0.98 + 0.03 * math.cos(i / 5.0) for i in range(n)], dtype=pl.Float64,
        ),
        "symbol": pl.Series(["BTCUSDT"] * n, dtype=pl.Utf8),
    })


class _MockStore:
    """DataStore stand-in serving exactly *n* 4H bars."""

    def __init__(self, n: int) -> None:
        self._n = n

    def get_klines(self, symbol, interval, start, end, columns=None):
        return _klines(self._n)

    def get_funding_rate(self, symbol, start, end):
        return _funding(self._n)

    def get_metrics(self, symbol, start, end):
        return _metrics(self._n)


def _build(n: int, interval: str = "4h") -> pl.DataFrame:
    from datetime import datetime, timezone

    pipeline = FeaturePipeline(
        _MockStore(n),  # type: ignore[arg-type]
        "BTCUSDT",
        interval,
    )
    return pipeline.build(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _regime_reference(n: int) -> pl.DataFrame:
    """``detect_all`` on the raw klines — the frame ``build()`` sees at step 10.

    Steps 4-9 (microstructure + derivatives) only ever add columns: no filter,
    no drop_nulls, no slice, and ``join_asof`` keeps the left row count. So row
    ``i`` of this reference is row ``i`` of the frame ``build()`` trims.
    """
    return RegimeDetector().detect_all(_klines(n))


# ---------------------------------------------------------------------------
# J1-J2. The trim length and the row that survives it
# ---------------------------------------------------------------------------


class TestOfflineTrimLength:
    def test_build_trims_required_history_bars_not_warmup_rows(self) -> None:
        """The trim must be WARMUP_ROWS + atr_lookback, not WARMUP_ROWS alone.

        Expressed through ``required_history_bars`` and the detector's own
        ``atr_lookback`` rather than the literal 740, so the test keeps
        catching a fork between the two numbers instead of pinning a copy.
        """
        n = 1000
        required = required_history_bars(RegimeDetector().atr_lookback)
        assert required == 740, "4H contract changed — update the expectations"

        df = _build(n)

        assert len(df) == n - required
        assert len(df) == 260

    def test_first_surviving_row_is_past_both_windows(self) -> None:
        """The first training row must carry real regime values.

        Three acts: prove the old trim WOULD have produced a constant row
        (otherwise the assertions below are green for the wrong reason),
        assert the new first row is not constant, then pin it to the exact
        reference index so a lucky trim of some other length cannot pass.
        """
        n = 1000
        reference = _regime_reference(n)

        # Act 1 — control. Row WARMUP_ROWS is the row the old trim promoted to
        # first. It sits below detect_all's min_bars (300), where the loop only
        # copies ADX and continues, so every other regime column is the
        # documented neutral constant.
        old_first = {c: reference[c][WARMUP_ROWS] for c in reference.columns}
        assert old_first["regime"] == "range"
        assert old_first["hurst"] == 0.5
        assert old_first["atr_pct"] == 0.0
        assert old_first["atr_percentile"] == 0.5
        assert old_first["trend_strength"] == 0.0
        assert old_first["regime_confidence"] == 0.0

        # Act 2 — the row that actually reaches the model now.
        df = _build(n)
        new_first = {c: df[c][0] for c in df.columns}
        assert new_first["regime_confidence"] > 0.0
        assert new_first["trend_strength"] > 0.0
        assert new_first["atr_pct"] > 0.0
        assert new_first["atr_percentile"] != 0.5

        # Act 3 — it is the reference row at exactly required_history_bars.
        target = required_history_bars(RegimeDetector().atr_lookback)
        assert new_first["regime"] == reference["regime"][target]
        assert new_first["atr_percentile"] == pytest.approx(
            reference["atr_percentile"][target], abs=1e-12,
        )
        assert new_first["hurst"] == pytest.approx(
            reference["hurst"][target], abs=1e-12,
        )
        assert new_first["atr_pct"] == pytest.approx(
            reference["atr_pct"][target], abs=1e-12,
        )


# ---------------------------------------------------------------------------
# J3. Fail-soft on a short input
# ---------------------------------------------------------------------------


class TestShortInputFailSoft:
    def test_short_input_falls_back_to_warmup_rows_and_shouts(
        self, loguru_warnings,
    ) -> None:
        """Too few rows must degrade loudly, never silently and never fatally.

        Fallback contract: trim ``WARMUP_ROWS`` (the pipeline's own rolling
        warm-up, still fully covered) instead of the unreachable requirement,
        and only when even that would empty the frame fall back to keeping the
        single last row.
        """
        n = 300
        required = required_history_bars(RegimeDetector().atr_lookback)

        df = _build(n)

        # Fail-soft: non-empty, trimmed by WARMUP_ROWS.
        assert len(df) == n - WARMUP_ROWS
        assert len(df) == 100

        warns = _build_warnings(loguru_warnings)
        assert len(warns) == 1, warns
        msg = warns[0]
        assert "build[4h]:" in msg
        assert f"{n} bars" in msg
        assert f"< {required} required" in msg
        assert f"WARMUP_ROWS={WARMUP_ROWS}" in msg
        assert f"atr_lookback={ATR_LOOKBACK_4H}" in msg
        # The effective trim and what is left, spelled out — the operator must
        # not have to derive them.
        assert f"trimming {WARMUP_ROWS} rows instead" in msg
        assert f"{n - WARMUP_ROWS} rows remain" in msg

        # Last-resort branch: not even WARMUP_ROWS rows available. Still no
        # empty frame, still a WARNING.
        tiny = 150
        tiny_df = _build(tiny)
        assert len(tiny_df) == 1

        warns = _build_warnings(loguru_warnings)
        assert len(warns) == 2, warns
        tiny_msg = warns[1]
        assert f"{tiny} bars" in tiny_msg
        assert f"trimming {tiny - 1} rows instead" in tiny_msg
        assert "1 rows remain" in tiny_msg


# ---------------------------------------------------------------------------
# J4. The number comes from the detector, not from a constant
# ---------------------------------------------------------------------------


class TestTrimFollowsDetector:
    def test_trim_follows_detector_atr_lookback(self) -> None:
        """Swap the detector's window → the trim must move with it.

        Pins the decision that ``build()`` reads ``atr_lookback`` off the
        detector object it constructed, rather than off ``_WARMUP_ROWS``,
        ``HISTORY_BARS_4H`` or an interval lookup table.
        """
        n = 1000
        short_lookback = 50

        with patch(
            "src.features.feature_pipeline.RegimeDetector",
            lambda: RegimeDetector(atr_lookback=short_lookback),
        ):
            df = _build(n)

        assert len(df) == n - (WARMUP_ROWS + short_lookback)
        assert len(df) == 750


# ---------------------------------------------------------------------------
# J5. The alias stays (guard — green before and after)
# ---------------------------------------------------------------------------


class TestWarmupAliasPreserved:
    def test_warmup_rows_alias_still_exported(self) -> None:
        """``_WARMUP_ROWS`` must survive the trim change.

        The docstrings of ``build_from_buffer`` and
        ``tests/test_regime_window_4h.py`` still name it; deleting it while
        rewriting the trim would break them for no gain.
        """
        from src.features.feature_pipeline import _WARMUP_ROWS

        assert _WARMUP_ROWS == WARMUP_ROWS
        assert _WARMUP_ROWS == 200
