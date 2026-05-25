"""Tests for Step H22 — BinanceRateLimiter singleton."""
from __future__ import annotations

import asyncio
import time

import pytest

from src.execution.binance_rate_limiter import BinanceRateLimiter


@pytest.fixture(autouse=True)
def _reset_singleton():
    BinanceRateLimiter.reset()
    yield
    BinanceRateLimiter.reset()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_instance_returns_same_object(self):
        a = BinanceRateLimiter.instance()
        b = BinanceRateLimiter.instance()
        assert a is b

    def test_reset_creates_fresh_instance(self):
        a = BinanceRateLimiter.instance()
        BinanceRateLimiter.reset()
        b = BinanceRateLimiter.instance()
        assert a is not b

    def test_class_constants(self):
        assert BinanceRateLimiter.MAX_WEIGHT_PER_MINUTE == 1200
        assert BinanceRateLimiter.WINDOW_SECONDS == 60.0


# ---------------------------------------------------------------------------
# acquire — happy path (no waiting)
# ---------------------------------------------------------------------------


class TestAcquireNoWait:
    @pytest.mark.asyncio
    async def test_small_request_does_not_block(self):
        lim = BinanceRateLimiter.instance()
        t0 = time.monotonic()
        await lim.acquire(weight=1)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1
        assert lim.used_weight() == 1

    @pytest.mark.asyncio
    async def test_used_weight_accumulates(self):
        lim = BinanceRateLimiter.instance()
        for _ in range(5):
            await lim.acquire(weight=10)
        assert lim.used_weight() == 50
        assert lim.available_weight() == 1150

    @pytest.mark.asyncio
    async def test_weight_floored_at_one(self):
        lim = BinanceRateLimiter.instance()
        await lim.acquire(weight=0)
        await lim.acquire(weight=-5)
        # Zero / negative coerced to 1.
        assert lim.used_weight() == 2


# ---------------------------------------------------------------------------
# acquire — blocks when budget exhausted
# ---------------------------------------------------------------------------


class TestAcquireBlocks:
    @pytest.mark.asyncio
    async def test_over_budget_acquire_waits_for_window(self, monkeypatch):
        """Fill the bucket, then ask for more — acquire must sleep until
        the oldest entry expires. Patch asyncio.sleep so the test stays
        fast and we can verify it was actually called."""
        lim = BinanceRateLimiter.instance()
        # Burn the full budget so the next request must wait.
        await lim.acquire(weight=1200)
        assert lim.used_weight() == 1200

        sleeps: list[float] = []

        async def _fake_sleep(seconds, *_, **__):
            sleeps.append(seconds)
            # Simulate time passing by mutating the limiter's events.
            # Push every queued event 70s into the past so prune drops them.
            now_now = time.monotonic()
            lim._events = type(lim._events)(
                (ts - 70.0, w) for ts, w in lim._events
            )

        monkeypatch.setattr(
            "src.execution.binance_rate_limiter.asyncio.sleep",
            _fake_sleep,
        )

        await lim.acquire(weight=100)
        assert sleeps, "expected asyncio.sleep to be called"
        # First sleep duration must be positive and ≤ window.
        assert 0 < sleeps[0] <= 60.5
        # After the simulated 70s skip, the bucket cleared and the new
        # 100 weight was admitted.
        assert lim.used_weight() == 100


# ---------------------------------------------------------------------------
# update_from_headers — server is the source of truth
# ---------------------------------------------------------------------------


class TestUpdateFromHeaders:
    @pytest.mark.asyncio
    async def test_header_floor_lifts_used_weight(self):
        lim = BinanceRateLimiter.instance()
        await lim.acquire(weight=10)
        # Server says we've used 800 — local was only 10.
        lim.update_from_headers({"X-MBX-USED-WEIGHT-1M": "800"})
        assert lim.used_weight() == 800
        # Available now reflects the server view.
        assert lim.available_weight() == 400

    def test_lower_header_value_does_not_lower_floor(self):
        lim = BinanceRateLimiter.instance()
        lim.update_from_headers({"X-MBX-USED-WEIGHT-1M": "500"})
        lim.update_from_headers({"X-MBX-USED-WEIGHT-1M": "200"})
        assert lim.used_weight() == 500

    def test_missing_header_silently_ignored(self):
        lim = BinanceRateLimiter.instance()
        lim.update_from_headers({"Content-Type": "application/json"})
        assert lim.used_weight() == 0

    def test_none_headers_silently_ignored(self):
        lim = BinanceRateLimiter.instance()
        lim.update_from_headers(None)
        assert lim.used_weight() == 0

    def test_malformed_header_value_silently_ignored(self):
        lim = BinanceRateLimiter.instance()
        for bad in ("not-a-number", "", "-7", "NaN"):
            lim.update_from_headers({"X-MBX-USED-WEIGHT-1M": bad})
        assert lim.used_weight() == 0

    def test_case_insensitive_header_lookup(self):
        lim = BinanceRateLimiter.instance()
        lim.update_from_headers({"x-mbx-used-weight-1m": "123"})
        assert lim.used_weight() == 123


# ---------------------------------------------------------------------------
# Rolling 60 s window — entries decay
# ---------------------------------------------------------------------------


class TestWindowDecay:
    @pytest.mark.asyncio
    async def test_old_entries_pruned(self, monkeypatch):
        lim = BinanceRateLimiter.instance()
        await lim.acquire(weight=300)
        # Age every queued entry past the 60s window.
        lim._events = type(lim._events)(
            (ts - 70.0, w) for ts, w in lim._events
        )
        # Force a prune via used_weight (which calls _prune).
        assert lim.used_weight() == 0
        assert lim.available_weight() == 1200

    def test_header_floor_persists_within_window(self):
        """Floor stays put while inside the 60-s window."""
        lim = BinanceRateLimiter.instance()
        lim.update_from_headers({"X-MBX-USED-WEIGHT-1M": "700"})
        assert lim.used_weight() == 700
        # Reading twice in quick succession must NOT erase it.
        assert lim.used_weight() == 700

    def test_header_floor_decays_after_window(self):
        """Floor expires when its timestamp ages past the window."""
        lim = BinanceRateLimiter.instance()
        lim.update_from_headers({"X-MBX-USED-WEIGHT-1M": "700"})
        # Backdate the floor timestamp past the 60-s window.
        lim._header_floor_ts -= 90.0
        assert lim.used_weight() == 0


# ---------------------------------------------------------------------------
# Fail-soft on internal errors
# ---------------------------------------------------------------------------


class TestFailSoft:
    @pytest.mark.asyncio
    async def test_acquire_swallows_internal_exception(self, monkeypatch):
        lim = BinanceRateLimiter.instance()
        # Force _acquire to raise.
        async def _boom(weight):
            raise RuntimeError("synthetic")
        monkeypatch.setattr(lim, "_acquire", _boom)
        # Must not propagate.
        await lim.acquire(weight=1)

    def test_update_from_headers_swallows_internal_exception(self, monkeypatch):
        lim = BinanceRateLimiter.instance()
        # Headers obj that raises on .get
        class _Boom:
            def get(self, *a, **kw):
                raise RuntimeError("x")
        lim.update_from_headers(_Boom())  # must not raise


# ---------------------------------------------------------------------------
# Integration: watchdog._signed_get goes through the limiter
# ---------------------------------------------------------------------------


class TestWatchdogIntegration:
    @pytest.mark.asyncio
    async def test_watchdog_get_consumes_weight(self):
        from src.execution.watchdog import Watchdog, WatchdogConfig

        wd = Watchdog(WatchdogConfig(
            trading_mode="testnet",
            binance_api_key="k", binance_api_secret="s",
        ))

        class _Resp:
            status = 200
            headers = {"X-MBX-USED-WEIGHT-1M": "42"}

            async def json(self):
                return [{"symbol": "BTCUSDT", "positionAmt": "0"}]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class _Session:
            def get(self, *a, **kw):
                return _Resp()

        out = await wd._signed_get(_Session(), "/fapi/v2/positionRisk")
        assert out == [{"symbol": "BTCUSDT", "positionAmt": "0"}]
        lim = BinanceRateLimiter.instance()
        # local 5 (positionRisk) raised to 42 by the server header.
        assert lim.used_weight() == 42
