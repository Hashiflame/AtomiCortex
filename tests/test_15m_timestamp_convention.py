"""PR-F — timestamp-convention regression tests for the 15m strategy.

Mirror defect of A0 (which was fixed for 4H): the 15m preload took
``ts_event`` from kline index 0 (**open** time) while live Nautilus bars
carry the **close** time, and ``_bars_to_df`` copied ``ts_event`` into a
column named ``open_time`` without converting close→open.

Offline the column means the bar's OPEN (``KLINES_SCHEMA`` carries
``open_time`` and ``close_time`` as two separate fields, and
``MTFContextBuilder`` builds its join key as ``open_time + bar_duration``
"(i.e. **close_time**)").  These tests pin that meaning on the live path.

No network: ``requests.get`` is always mocked.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from nautilus_trader.model.data import Bar
from nautilus_trader.model.objects import Price, Quantity

from src.execution.strategies.ml_strategy_15m import (
    MLStrategy15MConfig,
    MLTradingStrategy15M,
)

# ---------------------------------------------------------------------------
# Fixed grid points (UTC).  1_743_511_500_000 ms = 2025-04-01 12:45:00Z —
# the LAST 15m bar of its hour, i.e. the position where a close-time that
# is not converted back to the open would leak into the next bucket.
# ---------------------------------------------------------------------------

_D_MS = 900_000            # 15m bar duration
_HOUR_MS = 3_600_000

_OPEN_1245 = 1_743_511_500_000          # 12:45:00.000
_CLOSE_1245_BINANCE = 1_743_512_399_999  # 12:59:59.999  (open + D - 1)
_CLOSE_1245_BOUNDARY = 1_743_512_400_000  # 13:00:00.000  (open + D)

_HOUR_1200 = 1_743_508_800_000
_HOUR_1300 = 1_743_512_400_000


def _kline(open_ms: int, *, o: float = 100.0, h: float = 101.0,
           lo: float = 99.0, c: float = 100.5, v: float = 10.0) -> list:
    """One Binance futures kline row (12 fields, strings as the API sends)."""
    return [
        open_ms,                 # 0 open time
        str(o),                  # 1 open
        str(h),                  # 2 high
        str(lo),                 # 3 low
        str(c),                  # 4 close
        str(v),                  # 5 volume
        open_ms + _D_MS - 1,     # 6 close time
        "0",                     # 7 quote volume
        0,                       # 8 trade count
        str(v / 2),              # 9 taker buy base volume
        "0",                     # 10 taker buy quote volume
        "0",                     # 11 ignore
    ]


def _strategy() -> MLTradingStrategy15M:
    return MLTradingStrategy15M(config=MLStrategy15MConfig())


def _live_bar(strat: MLTradingStrategy15M, ts_event_ms: int) -> Bar:
    """A bar as Nautilus delivers it live: ts_event is the CLOSE time."""
    ts_ns = ts_event_ms * 1_000_000
    return Bar(
        bar_type=strat._bar_type,
        open=Price(100.0, precision=1),
        high=Price(101.0, precision=1),
        low=Price(99.0, precision=1),
        close=Price(100.5, precision=1),
        volume=Quantity(10.0, precision=3),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


def _run_preload(strat: MLTradingStrategy15M, klines: list) -> MagicMock:
    """Drive ``_preload_historical_bars`` against a mocked REST response."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = klines
    with patch("requests.get", return_value=resp) as getter:
        strat._preload_historical_bars()
    return getter


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


@pytest.fixture
def strat() -> MLTradingStrategy15M:
    return _strategy()


# ═══════════════════════════════════════════════════════════════════════
# ACT 1 — prove preload and live disagree today
# ═══════════════════════════════════════════════════════════════════════

def test_preload_ts_event_comes_from_kline_close_index_6(strat) -> None:
    """Preloaded ts_event must be k[6] (close), not k[0] (open)."""
    _run_preload(strat, [_kline(_OPEN_1245)])

    assert len(strat._bars) == 1
    assert strat._bars[0].ts_event == _CLOSE_1245_BINANCE * 1_000_000, (
        "preload must honor ts_event = bar CLOSE time (kline index 6)"
    )


def test_preload_and_live_bar_for_same_candle_agree_on_ts_event(strat) -> None:
    """The same candle must carry one ts_event whether preloaded or live."""
    _run_preload(strat, [_kline(_OPEN_1245)])
    live = _live_bar(strat, _CLOSE_1245_BINANCE)

    assert strat._bars[0].ts_event == live.ts_event


def test_preload_and_live_bar_for_same_candle_agree_on_open_time(strat) -> None:
    """_bars_to_df must derive one open_time for that same candle."""
    _run_preload(strat, [_kline(_OPEN_1245)])
    preloaded = strat._bars[0]
    live = _live_bar(strat, _CLOSE_1245_BINANCE)

    from_preload = strat._bars_to_df([preloaded])["open_time"][0]
    from_live = strat._bars_to_df([live])["open_time"][0]

    assert from_preload == from_live


# ═══════════════════════════════════════════════════════════════════════
# ACT 2 — the two paths agree on the offline meaning of open_time
# ═══════════════════════════════════════════════════════════════════════

def test_bars_to_df_open_time_matches_kline_open_for_preloaded_bar(
    strat,
) -> None:
    """CONTROL: preloaded bars already map to k[0]; that must not regress."""
    _run_preload(strat, [_kline(_OPEN_1245)])

    df = strat._bars_to_df(strat._bars)
    assert df["open_time"][0] == _OPEN_1245


def test_bars_to_df_open_time_matches_kline_open_for_live_bar(strat) -> None:
    """A live bar's open_time is the candle's OPEN, not its ts_event."""
    df = strat._bars_to_df([_live_bar(strat, _CLOSE_1245_BINANCE)])

    assert df["open_time"][0] == _OPEN_1245


def test_bars_to_df_open_time_grid_is_uniform_across_preload_live_splice(
    strat,
) -> None:
    """No fake gap where the preloaded window meets the live stream."""
    opens = [_OPEN_1245 - 2 * _D_MS, _OPEN_1245 - _D_MS, _OPEN_1245]
    _run_preload(strat, [_kline(o) for o in opens])

    buffer = list(strat._bars) + [
        _live_bar(strat, _OPEN_1245 + i * _D_MS + _D_MS - 1)
        for i in (1, 2, 3)
    ]
    times = strat._bars_to_df(buffer)["open_time"].to_list()
    deltas = [b - a for a, b in zip(times, times[1:])]

    assert deltas == [_D_MS] * 5, f"non-uniform grid: {deltas}"


def test_bars_to_df_handles_boundary_close_convention(strat) -> None:
    """ts_event = open + duration (boundary stamp) resolves to the same open."""
    df = strat._bars_to_df([_live_bar(strat, _CLOSE_1245_BOUNDARY)])

    assert df["open_time"][0] == _OPEN_1245


# ═══════════════════════════════════════════════════════════════════════
# ACT 3 — exact values on the grid
# ═══════════════════════════════════════════════════════════════════════

def test_bars_to_df_open_time_exact_value_1743511500000(strat) -> None:
    """Pin the exact ms: not the close, not one ms before the open."""
    df = strat._bars_to_df([_live_bar(strat, _CLOSE_1245_BINANCE)])
    value = df["open_time"][0]

    assert value == 1_743_511_500_000
    assert value != 1_743_511_499_999, "naive ts_event - duration"
    assert value != 1_743_512_399_999, "raw ts_event passed through"


def test_bars_to_df_open_time_is_multiple_of_900000(strat) -> None:
    """Every open_time in a mixed buffer sits on the 15m grid."""
    _run_preload(strat, [_kline(_OPEN_1245 - _D_MS), _kline(_OPEN_1245)])
    buffer = list(strat._bars) + [
        _live_bar(strat, _OPEN_1245 + _D_MS + _D_MS - 1),
        _live_bar(strat, _OPEN_1245 + 2 * _D_MS + _D_MS - 1),
    ]

    period_ms = int(strat._bar_period_hours() * _HOUR_MS)
    residuals = [t % period_ms for t in strat._bars_to_df(buffer)["open_time"]]

    assert residuals == [0, 0, 0, 0], f"off-grid open_time: {residuals}"


def test_resample_buckets_match_offline_grid(strat) -> None:
    """CONTROL: hourly buckets equal the floor of the true opens."""
    opens = [_HOUR_1200 + i * _D_MS for i in range(8)]
    df15 = pl.DataFrame({
        "open_time": opens,
        "open": [100.0] * 8,
        "high": [101.0] * 8,
        "low": [99.0] * 8,
        "close": [100.5] * 8,
        "volume": [10.0] * 8,
    })

    buckets = strat._resample(df15, _HOUR_MS)["open_time"].to_list()

    assert buckets == [_HOUR_1200, _HOUR_1300]


# ═══════════════════════════════════════════════════════════════════════
# Unclosed-candle filter + preload timestamp diagnostic
# ═══════════════════════════════════════════════════════════════════════

def test_preload_drops_still_forming_candle(strat) -> None:
    """The endpoint's trailing, not-yet-closed candle must not be buffered."""
    forming_open = (_now_ms() // _D_MS) * _D_MS + _D_MS
    klines = [_kline(_OPEN_1245 - _D_MS), _kline(_OPEN_1245),
              _kline(forming_open)]

    _run_preload(strat, klines)

    assert len(strat._bars) == 2
    assert all(
        b.ts_event // 1_000_000 < _now_ms() for b in strat._bars
    ), "a bar whose close time has not passed reached the buffer"


def test_preload_keeps_all_closed_candles(strat) -> None:
    """CONTROL: every already-closed candle survives, in order."""
    opens = [_OPEN_1245 - 2 * _D_MS, _OPEN_1245 - _D_MS, _OPEN_1245]

    _run_preload(strat, [_kline(o) for o in opens])

    assert len(strat._bars) == 3
    stamps = [b.ts_event for b in strat._bars]
    assert stamps == sorted(stamps)


def test_preload_calls_log_preload_timestamp_check_once(strat) -> None:
    """The only runtime observation of the preload path must actually run."""
    strat._log_preload_timestamp_check = MagicMock()
    _run_preload(strat, [_kline(_OPEN_1245 - _D_MS), _kline(_OPEN_1245)])

    strat._log_preload_timestamp_check.assert_called_once_with(
        "binance_api", strat._bars[-1],
    )


def test_preload_timestamp_check_takes_info_branch_for_15m(strat) -> None:
    """Residual is 0 at a 900_000 ms period → the check's INFO branch.

    ``self.log`` is a non-writable Nautilus property, so the branch is
    asserted through the exact condition the parent evaluates:
    ``period_ms = int(self._bar_period_hours() * 3_600_000)`` and
    ``residual = (ts_ms + 1) % period_ms``.
    """
    strat._log_preload_timestamp_check = MagicMock()
    _run_preload(strat, [_kline(_OPEN_1245)])

    source, newest = strat._log_preload_timestamp_check.call_args[0]
    period_ms = int(strat._bar_period_hours() * _HOUR_MS)
    residual = (newest.ts_event // 1_000_000 + 1) % period_ms

    assert source == "binance_api"
    assert period_ms == _D_MS
    assert residual == 0


def test_preload_failure_stays_fail_soft(strat) -> None:
    """CONTROL: a REST failure never escapes and never claims warmup."""
    with patch("requests.get", side_effect=RuntimeError("boom")):
        strat._preload_historical_bars()

    assert strat._bars == []
    assert strat._warmup_complete is False
