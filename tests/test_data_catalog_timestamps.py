"""Backtest lookahead-bias regression test for ``AtomiCortexCatalog.load_bar_data``.

Pre-fix, ``ts_event`` was set to the bar's ``open_time``. That delivers
the full OHLCV to the strategy at the *start* of the period — one full
bar of lookahead bias, which silently inflated every backtest metric
(WR / PF / Sharpe).

The Nautilus convention for OHLCV bars is ``ts_event = close time``.
Binance's parquet schema gives us ``close_time = open_time + duration - 1ms``
verbatim, so we use that field directly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.execution.data_catalog import AtomiCortexCatalog, _INTERVAL_MAP


_INTERVAL_MS: dict[str, int] = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def _binance_klines_df(interval: str, n_bars: int, start_ms: int) -> pl.DataFrame:
    """Build a polars DF matching the real Binance klines parquet schema."""
    dur = _INTERVAL_MS[interval]
    rows = []
    for i in range(n_bars):
        ot = start_ms + i * dur
        rows.append({
            "open_time": ot,
            "close_time": ot + dur - 1,  # Binance convention
            "open": 50_000.0 + i,
            "high": 50_100.0 + i,
            "low": 49_900.0 + i,
            "close": 50_050.0 + i,
            "volume": 100.0 + i,
        })
    return pl.DataFrame(rows)


@pytest.fixture
def catalog(tmp_path: Path) -> AtomiCortexCatalog:
    return AtomiCortexCatalog(data_dir=tmp_path)


def _load_with_mock(
    catalog: AtomiCortexCatalog,
    interval: str,
    n_bars: int = 5,
    start_ms: int = 1_743_379_200_000,  # 2025-03-31 00:00 UTC
) -> list:
    df = _binance_klines_df(interval, n_bars, start_ms)
    end_ms = start_ms + (n_bars + 1) * _INTERVAL_MS[interval]
    fake_lazy = df.lazy()
    with patch(
        "src.execution.data_catalog.pl.scan_parquet",
        return_value=fake_lazy,
    ):
        return catalog.load_bar_data(
            symbol="BTCUSDT",
            interval=interval,
            start=datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc),
            end=datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc),
        )


# ---------------------------------------------------------------------------
# ts_event = close-time (the canonical fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
def test_ts_event_equals_close_time_ns(
    catalog: AtomiCortexCatalog, interval: str
) -> None:
    """ts_event must equal (open_time + duration − 1ms) in nanoseconds."""
    start_ms = 1_743_379_200_000
    bars = _load_with_mock(catalog, interval, n_bars=3, start_ms=start_ms)
    dur_ms = _INTERVAL_MS[interval]
    for i, bar in enumerate(bars):
        expected_close_ms = start_ms + i * dur_ms + dur_ms - 1
        assert bar.ts_event == expected_close_ms * 1_000_000, (
            f"interval={interval} bar={i}"
        )


@pytest.mark.parametrize("interval", ["15m", "1h", "4h"])
def test_ts_event_is_one_full_bar_later_than_open_time(
    catalog: AtomiCortexCatalog, interval: str
) -> None:
    """Verbatim restatement of the lookahead-bias fix: ts_event must lie
    ~one bar's worth of nanoseconds AFTER open_time (not equal to it)."""
    start_ms = 1_743_379_200_000
    bars = _load_with_mock(catalog, interval, n_bars=1, start_ms=start_ms)
    delta_ns = bars[0].ts_event - start_ms * 1_000_000
    expected = (_INTERVAL_MS[interval] - 1) * 1_000_000
    assert delta_ns == expected, (
        f"interval={interval}: ts_event must be ~+{expected}ns from "
        f"open_time, got +{delta_ns}ns"
    )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval", ["15m", "1h", "4h"])
def test_ts_event_strictly_monotonic(
    catalog: AtomiCortexCatalog, interval: str
) -> None:
    bars = _load_with_mock(catalog, interval, n_bars=10)
    for prev, cur in zip(bars, bars[1:]):
        assert cur.ts_event > prev.ts_event


@pytest.mark.parametrize("interval", ["15m", "1h", "4h"])
def test_consecutive_ts_event_delta_equals_bar_duration(
    catalog: AtomiCortexCatalog, interval: str
) -> None:
    bars = _load_with_mock(catalog, interval, n_bars=5)
    expected_delta_ns = _INTERVAL_MS[interval] * 1_000_000
    for prev, cur in zip(bars, bars[1:]):
        assert cur.ts_event - prev.ts_event == expected_delta_ns


def test_ts_init_is_at_least_ts_event(catalog: AtomiCortexCatalog) -> None:
    """Nautilus invariant: ts_init >= ts_event (init never precedes event)."""
    bars = _load_with_mock(catalog, "4h", n_bars=3)
    for bar in bars:
        assert bar.ts_init >= bar.ts_event


def test_interval_map_covers_test_grid() -> None:
    """Cross-check: every interval we assert on is registered in the map."""
    for itv in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        assert itv in _INTERVAL_MAP
