"""Tests for live taker_buy_volume handling (Step H1b).

Covers:
- add_bar() with explicit taker_buy_volume stores the real value.
- add_bar() without it falls back to volume*0.5 in get_bar_df() and emits
  a single WARNING.
- CVD is 0 under fallback, non-zero with real values.
- Train/serve consistency: feature pipeline on a buffer-derived DF and on
  an equivalent offline DF produces the same CVD.
- fetch_taker_buy_volume parses Binance kline shape and is fail-soft.
"""
from __future__ import annotations

import polars as pl
import pytest
from loguru import logger as _loguru_logger

from src.features.live_feature_state import LiveFeatureState
from src.features.microstructure import add_cvd_features


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


class FakeBar:
    def __init__(self, ts_event_ms: int, o: float, h: float, l: float,
                 c: float, v: float):
        self.ts_event = ts_event_ms * 1_000_000  # ms → ns
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v


def _seed_bars(state: LiveFeatureState, n: int, *, with_tbv: bool):
    bar_ms = 4 * 3_600_000
    start_close_ms = 1_700_000_000_000
    for i in range(n):
        close_ms = start_close_ms + (i + 1) * bar_ms
        bar = FakeBar(close_ms, 100.0, 101.0, 99.0, 100.5, 1000.0)
        tbv = 700.0 if with_tbv else None  # 700 → cvd = 2*700-1000 = 400
        state.add_bar(bar, interval="4h", taker_buy_volume=tbv)


class TestAddBarStoresTakerBuyVolume:
    def test_explicit_value_stored(self):
        state = LiveFeatureState()
        _seed_bars(state, n=3, with_tbv=True)
        df = state.get_bar_df("4h")
        assert "taker_buy_volume" in df.columns
        assert df["taker_buy_volume"].to_list() == [700.0, 700.0, 700.0]

    def test_missing_value_falls_back_to_half_volume(self, loguru_warnings):
        state = LiveFeatureState()
        _seed_bars(state, n=3, with_tbv=False)
        df = state.get_bar_df("4h")
        assert df["taker_buy_volume"].to_list() == [500.0, 500.0, 500.0]
        warns = [m for m in loguru_warnings if "taker_buy_volume" in m]
        assert len(warns) >= 1

    def test_warning_emitted_only_once(self, loguru_warnings):
        state = LiveFeatureState()
        _seed_bars(state, n=2, with_tbv=False)
        state.get_bar_df("4h")
        state.get_bar_df("4h")
        state.get_bar_df("4h")
        warns = [m for m in loguru_warnings if "taker_buy_volume" in m]
        assert len(warns) == 1


class TestCvdFromBuffer:
    def test_cvd_zero_under_fallback(self):
        state = LiveFeatureState()
        _seed_bars(state, n=5, with_tbv=False)
        df = state.get_bar_df("4h")
        cvd_df = add_cvd_features(df)
        assert cvd_df["cvd"].to_list() == [0.0] * 5

    def test_cvd_nonzero_with_real_taker(self):
        state = LiveFeatureState()
        _seed_bars(state, n=5, with_tbv=True)
        df = state.get_bar_df("4h")
        cvd_df = add_cvd_features(df)
        # cvd = 2*700 - 1000 = 400
        assert all(v == 400.0 for v in cvd_df["cvd"].to_list())

    def test_mixed_bars_preserve_per_bar_values(self):
        state = LiveFeatureState()
        bar_ms = 4 * 3_600_000
        start_close_ms = 1_700_000_000_000
        for i, tbv in enumerate([600.0, None, 800.0]):
            close_ms = start_close_ms + (i + 1) * bar_ms
            state.add_bar(
                FakeBar(close_ms, 100.0, 101.0, 99.0, 100.5, 1000.0),
                interval="4h",
                taker_buy_volume=tbv,
            )
        df = state.get_bar_df("4h")
        assert df["taker_buy_volume"].to_list() == [600.0, 500.0, 800.0]


class TestTrainServeConsistency:
    """add_cvd_features on a buffer-derived DF == on an equivalent offline DF."""

    def test_cvd_matches_offline(self):
        state = LiveFeatureState()
        bar_ms = 4 * 3_600_000
        start_close_ms = 1_700_000_000_000
        rows = []
        for i in range(20):
            close_ms = start_close_ms + (i + 1) * bar_ms
            o = 100.0 + i * 0.1
            h = o + 1.0
            l = o - 1.0
            c = o + 0.5
            v = 1000.0 + i * 10.0
            tbv = 0.4 * v + i * 3.0  # varied, realistic
            state.add_bar(
                FakeBar(close_ms, o, h, l, c, v),
                interval="4h",
                taker_buy_volume=tbv,
            )
            rows.append({
                "open_time": close_ms - bar_ms,
                "open": o, "high": h, "low": l, "close": c,
                "volume": v, "taker_buy_volume": tbv,
            })
        offline_df = pl.DataFrame(rows).sort("open_time")
        live_df = state.get_bar_df("4h")

        offline_cvd = add_cvd_features(offline_df)["cvd"].to_list()
        live_cvd = add_cvd_features(live_df)["cvd"].to_list()
        assert offline_cvd == live_cvd
        assert any(v != 0.0 for v in live_cvd)


class _MockResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _MockSession:
    def __init__(self, resp):
        self._resp = resp
        self.last_params = None

    def get(self, url, params, timeout):
        self.last_params = params
        return self._resp


class TestFetchTakerBuyVolume:
    OPEN_MS = 1_700_000_000_000

    def _kline(self, open_ms, tbv):
        # Binance kline array (12 elements).
        return [open_ms, "100", "101", "99", "100.5", "1000",
                open_ms + 4 * 3_600_000 - 1, "100500", 42,
                str(tbv), "70000", "0"]

    def test_parses_taker_buy_volume(self):
        sess = _MockSession(_MockResp([self._kline(self.OPEN_MS, "725.5")]))
        v = LiveFeatureState.fetch_taker_buy_volume(
            "BTCUSDT", "4h", self.OPEN_MS, session=sess,
        )
        assert v == 725.5
        assert sess.last_params["symbol"] == "BTCUSDT"
        assert sess.last_params["interval"] == "4h"
        assert sess.last_params["startTime"] == self.OPEN_MS
        assert sess.last_params["limit"] == 1

    def test_returns_none_on_timestamp_mismatch(self):
        sess = _MockSession(_MockResp([self._kline(self.OPEN_MS + 1, "725.5")]))
        v = LiveFeatureState.fetch_taker_buy_volume(
            "BTCUSDT", "4h", self.OPEN_MS, session=sess,
        )
        assert v is None

    def test_returns_none_on_http_error(self):
        sess = _MockSession(_MockResp([], status=500))
        v = LiveFeatureState.fetch_taker_buy_volume(
            "BTCUSDT", "4h", self.OPEN_MS, session=sess,
        )
        assert v is None

    def test_returns_none_on_empty_payload(self):
        sess = _MockSession(_MockResp([]))
        v = LiveFeatureState.fetch_taker_buy_volume(
            "BTCUSDT", "4h", self.OPEN_MS, session=sess,
        )
        assert v is None

    def test_returns_none_on_exception(self):
        class _Boom:
            def get(self, *a, **kw):
                raise RuntimeError("network down")
        v = LiveFeatureState.fetch_taker_buy_volume(
            "BTCUSDT", "4h", self.OPEN_MS, session=_Boom(),
        )
        assert v is None


class TestBackwardCompatibility:
    """Existing call sites that don't pass taker_buy_volume keep working."""

    def test_add_bar_without_kwarg(self):
        state = LiveFeatureState()
        bar = FakeBar(1_700_000_000_000 + 4 * 3_600_000,
                      100.0, 101.0, 99.0, 100.5, 1000.0)
        state.add_bar(bar, interval="4h")  # no taker_buy_volume kwarg
        df = state.get_bar_df("4h")
        assert len(df) == 1
        assert df["taker_buy_volume"][0] == 500.0  # fallback applied
