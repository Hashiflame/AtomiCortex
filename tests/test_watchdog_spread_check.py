"""Tests for Step H16 — Limit-IOC emergency close with MARKET fallback.

Pre-H16: the watchdog sent a naked MARKET reduceOnly to flatten in a
crisis, which can pay 1-5% slippage on a thin book. Post-H16: try
Limit-IOC at markPrice ± 0.3% first; MARKET only if the IOC didn't
fully fill (or markPrice isn't usable).
"""
from __future__ import annotations

import pytest

from src.execution.watchdog import Watchdog, WatchdogConfig


def _make_watchdog(scope: str = "") -> Watchdog:
    return Watchdog(WatchdogConfig(
        trading_mode="testnet",
        binance_api_key="k",
        binance_api_secret="s",
        symbol=scope,
    ))


def _patch_session(monkeypatch):
    class _ClientSession:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("aiohttp.ClientSession", _ClientSession)


def _install_capture(
    wd: Watchdog,
    monkeypatch,
    positions: list[dict],
    *,
    ioc_response: dict | None = None,
    ioc_raises: bool = False,
):
    """Stub signed-* helpers. ``ioc_response`` is returned for LIMIT
    orders; MARKET orders always return a success dict so we can tell
    them apart in assertions."""
    events: list[dict] = []
    sleep_log: list[float] = []

    async def _get(*_a, **_kw):
        return positions

    async def _delete(*_a, **_kw):
        return {"ok": True}

    async def _post(_session, _url, params):
        events.append(dict(params))
        if params.get("type") == "LIMIT":
            if ioc_raises:
                raise RuntimeError("synthetic IOC failure")
            return ioc_response
        # MARKET
        return {"orderId": 99, "status": "FILLED"}

    async def _sleep(seconds, *_, **__):
        sleep_log.append(seconds)

    monkeypatch.setattr(wd, "_signed_get", _get)
    monkeypatch.setattr(wd, "_signed_delete", _delete)
    monkeypatch.setattr(wd, "_signed_post", _post)
    monkeypatch.setattr("src.execution.watchdog.asyncio.sleep", _sleep)
    return events, sleep_log


# ---------------------------------------------------------------------------
# IOC tried first
# ---------------------------------------------------------------------------


class TestIocAttemptedFirst:
    @pytest.mark.asyncio
    async def test_ioc_used_when_filled(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "markPrice": "60000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            ioc_response={"status": "FILLED", "executedQty": "0.5"},
        )
        result = await wd.emergency_close_all()

        order_events = [e for e in events if "type" in e]
        assert order_events, "no order events captured"
        # Only the IOC ran — MARKET was skipped because IOC filled.
        assert order_events[0]["type"] == "LIMIT"
        assert order_events[0]["timeInForce"] == "IOC"
        assert all(e["type"] == "LIMIT" for e in order_events)

        closed = result["positions_closed"]
        assert len(closed) == 1
        assert closed[0]["method"] == "LIMIT_IOC"

    @pytest.mark.asyncio
    async def test_sell_limit_price_is_mark_minus_30bps(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "markPrice": "60000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            ioc_response={"status": "FILLED", "executedQty": "0.5"},
        )
        await wd.emergency_close_all()
        ioc = next(e for e in events if e["type"] == "LIMIT")
        assert ioc["side"] == "SELL"
        # 60000 × 0.997 = 59820.00
        assert float(ioc["price"]) == pytest.approx(59820.00, abs=1e-6)
        assert ioc["reduceOnly"] == "true"

    @pytest.mark.asyncio
    async def test_buy_limit_price_is_mark_plus_30bps(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "ETHUSDT", "positionAmt": "-2.0", "markPrice": "3000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            ioc_response={"status": "FILLED", "executedQty": "2.0"},
        )
        await wd.emergency_close_all()
        ioc = next(e for e in events if e["type"] == "LIMIT")
        assert ioc["side"] == "BUY"
        # 3000 × 1.003 = 3009.00
        assert float(ioc["price"]) == pytest.approx(3009.00, abs=1e-6)


# ---------------------------------------------------------------------------
# Fallback to MARKET
# ---------------------------------------------------------------------------


class TestMarketFallback:
    @pytest.mark.asyncio
    async def test_unfilled_ioc_falls_back_to_market(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "markPrice": "60000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            ioc_response={"status": "EXPIRED", "executedQty": "0"},
        )
        result = await wd.emergency_close_all()

        order_types = [e["type"] for e in events if "type" in e]
        # IOC tried first, then MARKET.
        assert order_types == ["LIMIT", "MARKET"]

        closed = result["positions_closed"]
        assert len(closed) == 1
        assert closed[0]["method"] == "MARKET"

    @pytest.mark.asyncio
    async def test_partial_ioc_falls_back_to_market(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "1.0", "markPrice": "60000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            # Only 0.4 of 1.0 filled.
            ioc_response={"status": "PARTIALLY_FILLED", "executedQty": "0.4"},
        )
        result = await wd.emergency_close_all()
        order_types = [e["type"] for e in events if "type" in e]
        assert order_types == ["LIMIT", "MARKET"]
        # MARKET sends the FULL original qty — easier to over-cancel
        # (reduceOnly clamps) than to leave residual exposure.
        market = next(e for e in events if e["type"] == "MARKET")
        assert market["quantity"] == "1.0"
        assert result["positions_closed"][0]["method"] == "MARKET"

    @pytest.mark.asyncio
    async def test_ioc_none_response_falls_back_to_market(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "markPrice": "60000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions, ioc_response=None,
        )
        result = await wd.emergency_close_all()
        order_types = [e["type"] for e in events if "type" in e]
        assert order_types == ["LIMIT", "MARKET"]
        assert result["positions_closed"][0]["method"] == "MARKET"

    @pytest.mark.asyncio
    async def test_ioc_exception_falls_back_to_market(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "markPrice": "60000.00"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions, ioc_raises=True,
        )
        result = await wd.emergency_close_all()
        order_types = [e["type"] for e in events if "type" in e]
        assert order_types == ["LIMIT", "MARKET"]
        assert result["positions_closed"][0]["method"] == "MARKET"

    @pytest.mark.asyncio
    async def test_missing_markprice_skips_ioc(self, monkeypatch):
        """No markPrice → IOC short-circuits, MARKET runs directly."""
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5"},  # no markPrice
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            ioc_response={"status": "FILLED", "executedQty": "0.5"},
        )
        result = await wd.emergency_close_all()
        # Only MARKET was issued — IOC was never even attempted.
        order_types = [e["type"] for e in events if "type" in e]
        assert order_types == ["MARKET"]
        assert result["positions_closed"][0]["method"] == "MARKET"

    @pytest.mark.asyncio
    async def test_zero_markprice_skips_ioc(self, monkeypatch):
        wd = _make_watchdog()
        _patch_session(monkeypatch)
        positions = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5", "markPrice": "0"},
        ]
        events, _ = _install_capture(
            wd, monkeypatch, positions,
            ioc_response={"status": "FILLED", "executedQty": "0.5"},
        )
        await wd.emergency_close_all()
        order_types = [e["type"] for e in events if "type" in e]
        assert order_types == ["MARKET"]


# ---------------------------------------------------------------------------
# IOC slippage constant
# ---------------------------------------------------------------------------


class TestIocSlippageConfig:
    def test_constant_is_30bps(self):
        assert Watchdog._IOC_SLIPPAGE == 0.003
