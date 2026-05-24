"""Tests for Step H15 — watchdog cancels orders BEFORE closing positions.

Pre-H15 order was MARKET close → cancel. A resting SL fired during the
close window could open a reverse position (close + SL together = 2×
fill in the same direction). Post-H15: cancel all → sleep 0.5s → close.
"""
from __future__ import annotations

import asyncio

import pytest

from src.execution.watchdog import Watchdog, WatchdogConfig


# ---------------------------------------------------------------------------
# Test rig — replaces aiohttp + the three signed-* helpers with capture stubs
# ---------------------------------------------------------------------------


def _make_watchdog() -> Watchdog:
    return Watchdog(WatchdogConfig(
        trading_mode="testnet",
        binance_api_key="k",
        binance_api_secret="s",
    ))


class _StubSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_session(monkeypatch):
    """Replace `aiohttp.ClientSession` with a no-arg stub the watchdog
    can `async with`."""
    class _ClientSession:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        "aiohttp.ClientSession", _ClientSession,
    )


def _install_capture(wd: Watchdog, monkeypatch):
    """Stub the three signed-* helpers. Returns (events, controls)."""
    events: list[str] = []
    controls: dict = {
        "positions": [
            {"symbol": "BTCUSDT", "positionAmt": "0.5"},
            {"symbol": "ETHUSDT", "positionAmt": "-2.0"},
        ],
        "cancel_returns_none": False,
        "cancel_raises": False,
        "sleep_log": [],
    }

    async def _get(_session, _url, _params=None):
        events.append("get_positions")
        return controls["positions"]

    async def _delete(_session, _url, params):
        events.append(f"cancel:{params['symbol']}")
        if controls["cancel_raises"]:
            raise RuntimeError("synthetic cancel failure")
        if controls["cancel_returns_none"]:
            return None
        return {"ok": True}

    async def _post(_session, _url, params):
        events.append(f"close:{params['symbol']}:{params['side']}:{params['quantity']}")
        return {"orderId": 42}

    async def _sleep(seconds, *_, **__):
        controls["sleep_log"].append(seconds)
        events.append(f"sleep:{seconds}")

    monkeypatch.setattr(wd, "_signed_get", _get)
    monkeypatch.setattr(wd, "_signed_delete", _delete)
    monkeypatch.setattr(wd, "_signed_post", _post)
    monkeypatch.setattr("src.execution.watchdog.asyncio.sleep", _sleep)
    return events, controls


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestCloseSequenceOrdering:
    @pytest.mark.asyncio
    async def test_cancel_before_close_for_all_symbols(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        events, _ = _install_capture(wd, monkeypatch)

        result = await wd.emergency_close_all()

        # Every cancel must appear before every close in the call log.
        cancel_idxs = [i for i, e in enumerate(events) if e.startswith("cancel:")]
        close_idxs = [i for i, e in enumerate(events) if e.startswith("close:")]
        assert cancel_idxs and close_idxs
        assert max(cancel_idxs) < min(close_idxs), (
            f"some close fired before some cancel — events={events}"
        )

        assert result["orders_cancelled"] is True
        assert len(result["positions_closed"]) == 2

    @pytest.mark.asyncio
    async def test_sleep_between_cancel_and_close(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        events, controls = _install_capture(wd, monkeypatch)

        await wd.emergency_close_all()

        # 0.5s sleep was invoked, sitting between the last cancel and
        # the first close.
        assert 0.5 in controls["sleep_log"]
        sleep_idx = events.index("sleep:0.5")
        last_cancel_idx = max(
            i for i, e in enumerate(events) if e.startswith("cancel:")
        )
        first_close_idx = min(
            i for i, e in enumerate(events) if e.startswith("close:")
        )
        assert last_cancel_idx < sleep_idx < first_close_idx

    @pytest.mark.asyncio
    async def test_close_quantity_and_side_for_each_position(self, monkeypatch):
        """Close-side semantics unchanged: positive amt → SELL, negative → BUY."""
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        events, _ = _install_capture(wd, monkeypatch)
        await wd.emergency_close_all()

        # BTC: +0.5 → SELL 0.5 ; ETH: -2.0 → BUY 2.0
        assert "close:BTCUSDT:SELL:0.5" in events
        assert "close:ETHUSDT:BUY:2.0" in events


# ---------------------------------------------------------------------------
# Fail-soft cancel: closes still happen
# ---------------------------------------------------------------------------


class TestCancelFailSoft:
    @pytest.mark.asyncio
    async def test_cancel_returning_none_still_closes(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        events, controls = _install_capture(wd, monkeypatch)
        controls["cancel_returns_none"] = True

        result = await wd.emergency_close_all()

        # Closes still ran.
        close_count = sum(1 for e in events if e.startswith("close:"))
        assert close_count == 2
        # And the failure was surfaced.
        assert any(
            "Cancel orders failed" in e for e in result["errors"]
        ), result["errors"]
        # ``orders_cancelled`` stays True — we *attempted* cancel for all.
        assert result["orders_cancelled"] is True

    @pytest.mark.asyncio
    async def test_cancel_raising_still_closes(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        events, controls = _install_capture(wd, monkeypatch)
        controls["cancel_raises"] = True

        result = await wd.emergency_close_all()

        close_count = sum(1 for e in events if e.startswith("close:"))
        assert close_count == 2
        assert any(
            "Cancel orders raised" in e for e in result["errors"]
        ), result["errors"]


# ---------------------------------------------------------------------------
# Scope filter still respected
# ---------------------------------------------------------------------------


class TestScopeFilter:
    @pytest.mark.asyncio
    async def test_scoped_watchdog_cancels_and_closes_only_its_symbol(
        self, monkeypatch,
    ):
        wd = Watchdog(WatchdogConfig(
            trading_mode="testnet",
            binance_api_key="k",
            binance_api_secret="s",
            symbol="BTCUSDT",
        ))
        _patch_session(monkeypatch)
        events, _ = _install_capture(wd, monkeypatch)

        await wd.emergency_close_all()

        # Only BTCUSDT cancels + closes; ETHUSDT untouched.
        cancel_symbols = [
            e.split(":", 1)[1] for e in events if e.startswith("cancel:")
        ]
        close_symbols = [
            e.split(":", 2)[1] for e in events if e.startswith("close:")
        ]
        assert cancel_symbols == ["BTCUSDT"]
        assert close_symbols == ["BTCUSDT"]


# ---------------------------------------------------------------------------
# No positions → no work
# ---------------------------------------------------------------------------


class TestNoPositions:
    @pytest.mark.asyncio
    async def test_zero_positions_no_cancel_no_close(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        events, controls = _install_capture(wd, monkeypatch)
        controls["positions"] = []

        result = await wd.emergency_close_all()

        assert all(not e.startswith("cancel:") for e in events)
        assert all(not e.startswith("close:") for e in events)
        assert result["positions_closed"] == []
