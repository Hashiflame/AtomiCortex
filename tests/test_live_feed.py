"""
tests/test_live_feed.py

Unit tests for LiveFeedManager and TickBuffer.
No live network connection — all exchange callbacks are driven by mock objects.
"""

from __future__ import annotations

import time
from io import StringIO
from unittest.mock import MagicMock

import pytest
from loguru import logger as _loguru

from src.ingestion.live_feed import LiveFeedManager, TickBuffer, _to_cf_symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    symbol: str = "BTC-USDT-PERP",
    price: float = 42_000.0,
    amount: float = 0.5,
    side: str = "buy",
    lag_seconds: float = 0.05,
) -> MagicMock:
    """Build a mock Cryptofeed Trade object."""
    trade = MagicMock()
    trade.symbol = symbol
    trade.price = price
    trade.amount = amount
    trade.side = side
    trade.timestamp = time.time() - lag_seconds
    return trade


def _make_funding(
    symbol: str = "BTC-USDT-PERP",
    rate: float = 0.0001,
    mark_price: float = 42_000.0,
) -> MagicMock:
    """Build a mock Cryptofeed Funding object."""
    f = MagicMock()
    f.symbol = symbol
    f.rate = rate
    f.mark_price = mark_price
    return f


def _make_book(
    symbol: str = "BTC-USDT-PERP",
    best_bid: float = 41_990.0,
    best_ask: float = 42_010.0,
) -> MagicMock:
    """Build a mock Cryptofeed OrderBook object with 3 bid/ask levels."""
    book = MagicMock()
    book.symbol = symbol
    book.timestamp = time.time()

    bid_prices = [best_bid - i * 10 for i in range(3)]
    ask_prices = [best_ask + i * 10 for i in range(3)]

    inner = MagicMock()
    inner.bids = {p: 1.0 + i * 0.1 for i, p in enumerate(bid_prices)}
    inner.asks = {p: 0.8 + i * 0.1 for i, p in enumerate(ask_prices)}
    book.book = inner
    return book


# ---------------------------------------------------------------------------
# Test: symbol conversion
# ---------------------------------------------------------------------------

def test_to_cf_symbol_btc() -> None:
    assert _to_cf_symbol("BTCUSDT") == "BTC-USDT-PERP"


def test_to_cf_symbol_eth() -> None:
    assert _to_cf_symbol("ETHUSDT") == "ETH-USDT-PERP"


def test_to_cf_symbol_sol() -> None:
    assert _to_cf_symbol("SOLUSDT") == "SOL-USDT-PERP"


def test_to_cf_symbol_case_insensitive() -> None:
    assert _to_cf_symbol("btcusdt") == "BTC-USDT-PERP"


def test_to_cf_symbol_invalid() -> None:
    with pytest.raises(ValueError, match="USDT-margined"):
        _to_cf_symbol("BTCEUR")


# ---------------------------------------------------------------------------
# Test: TickBuffer — trades
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tick_buffer_add_and_get_trade() -> None:
    buf = TickBuffer(max_trades=10)
    trade = {"price": 42_000.0, "amount": 0.5, "side": "buy", "timestamp": 1.0}
    await buf.add_trade("BTC-USDT-PERP", trade)

    result = buf.get_recent_trades("BTC-USDT-PERP", n=5)
    assert len(result) == 1
    assert result[0]["price"] == 42_000.0
    assert result[0]["side"] == "buy"


@pytest.mark.asyncio
async def test_tick_buffer_respects_max_capacity() -> None:
    buf = TickBuffer(max_trades=3)
    for i in range(5):
        await buf.add_trade("BTC-USDT-PERP", {"price": float(i)})

    result = buf.get_recent_trades("BTC-USDT-PERP", n=10)
    assert len(result) == 3
    assert result[-1]["price"] == 4.0  # most recent


@pytest.mark.asyncio
async def test_tick_buffer_n_limit() -> None:
    buf = TickBuffer(max_trades=100)
    for i in range(10):
        await buf.add_trade("BTC-USDT-PERP", {"price": float(i)})

    result = buf.get_recent_trades("BTC-USDT-PERP", n=3)
    assert len(result) == 3
    assert result[-1]["price"] == 9.0


def test_tick_buffer_missing_symbol_trades() -> None:
    buf = TickBuffer()
    assert buf.get_recent_trades("MISSING", 10) == []


# ---------------------------------------------------------------------------
# Test: TickBuffer — order book
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tick_buffer_update_and_get_book() -> None:
    buf = TickBuffer()
    snap = {
        "bids": [(100.0, 1.0), (99.0, 2.0)],
        "asks": [(101.0, 0.5)],
        "imbalance": 0.5,
        "timestamp": time.time(),
    }
    await buf.update_book("ETH-USDT-PERP", snap)

    result = buf.get_current_book("ETH-USDT-PERP")
    assert result["imbalance"] == 0.5
    assert result["bids"][0] == (100.0, 1.0)


def test_tick_buffer_missing_symbol_book() -> None:
    buf = TickBuffer()
    assert buf.get_current_book("MISSING") == {}


# ---------------------------------------------------------------------------
# Test: TickBuffer — funding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tick_buffer_update_and_get_funding() -> None:
    buf = TickBuffer()
    await buf.update_funding("BTC-USDT-PERP", 0.0001)
    assert buf.get_latest_funding("BTC-USDT-PERP") == 0.0001


@pytest.mark.asyncio
async def test_tick_buffer_funding_overwrites() -> None:
    buf = TickBuffer()
    await buf.update_funding("BTC-USDT-PERP", 0.0001)
    await buf.update_funding("BTC-USDT-PERP", -0.0002)
    assert buf.get_latest_funding("BTC-USDT-PERP") == -0.0002


def test_tick_buffer_missing_symbol_funding() -> None:
    buf = TickBuffer()
    assert buf.get_latest_funding("MISSING") == 0.0


# ---------------------------------------------------------------------------
# Test: LiveFeedManager — health
# ---------------------------------------------------------------------------

def test_is_healthy_initially_false() -> None:
    mgr = LiveFeedManager()
    assert not mgr.is_healthy()


@pytest.mark.asyncio
async def test_is_healthy_after_trade() -> None:
    mgr = LiveFeedManager()
    await mgr._on_trade(_make_trade(), receipt_timestamp=time.time())
    assert mgr.is_healthy()


# ---------------------------------------------------------------------------
# Test: LiveFeedManager — on_trade callback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_trade_buffers_trade() -> None:
    mgr = LiveFeedManager()
    trade = _make_trade(price=43_000.0, amount=1.5, side="sell")
    await mgr._on_trade(trade, receipt_timestamp=time.time())

    trades = mgr.tick_buffer.get_recent_trades("BTC-USDT-PERP", n=1)
    assert len(trades) == 1
    assert trades[0]["price"] == 43_000.0
    assert trades[0]["side"] == "sell"
    assert trades[0]["amount"] == 1.5


@pytest.mark.asyncio
async def test_on_trade_records_latency() -> None:
    mgr = LiveFeedManager()
    trade = _make_trade(lag_seconds=0.2)  # 200ms
    await mgr._on_trade(trade, receipt_timestamp=time.time())

    trades = mgr.tick_buffer.get_recent_trades("BTC-USDT-PERP", n=1)
    assert 150 < trades[0]["latency_ms"] < 400  # allow clock jitter


@pytest.mark.asyncio
async def test_on_trade_high_latency_warns() -> None:
    """A trade older than 1000ms triggers a WARNING log."""
    mgr = LiveFeedManager()
    messages: list[str] = []
    handler_id = _loguru.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
        format="{message}",
    )
    try:
        trade = _make_trade(lag_seconds=2.0)  # 2 000 ms >> 1 000 ms threshold
        await mgr._on_trade(trade, receipt_timestamp=time.time())
    finally:
        _loguru.remove(handler_id)

    assert any("High trade latency" in m for m in messages)


@pytest.mark.asyncio
async def test_on_trade_normal_latency_no_warn() -> None:
    """A trade with sub-threshold latency does not produce a WARNING."""
    mgr = LiveFeedManager()
    messages: list[str] = []
    handler_id = _loguru.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
        format="{message}",
    )
    try:
        trade = _make_trade(lag_seconds=0.05)  # 50 ms << 1 000 ms threshold
        await mgr._on_trade(trade, receipt_timestamp=time.time())
    finally:
        _loguru.remove(handler_id)

    assert not any("High trade latency" in m for m in messages)


# ---------------------------------------------------------------------------
# Test: LiveFeedManager — on_book callback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_book_buffers_snapshot() -> None:
    mgr = LiveFeedManager()
    book = _make_book(best_bid=41_990.0, best_ask=42_010.0)
    await mgr._on_book(book, receipt_timestamp=time.time())

    snap = mgr.tick_buffer.get_current_book("BTC-USDT-PERP")
    assert snap["bids"][0][0] == pytest.approx(41_990.0)
    assert snap["asks"][0][0] == pytest.approx(42_010.0)
    assert -1.0 <= snap["imbalance"] <= 1.0


@pytest.mark.asyncio
async def test_on_book_imbalance_range() -> None:
    mgr = LiveFeedManager()
    book = _make_book()
    await mgr._on_book(book, receipt_timestamp=time.time())

    snap = mgr.tick_buffer.get_current_book("BTC-USDT-PERP")
    assert -1.0 <= snap["imbalance"] <= 1.0


# ---------------------------------------------------------------------------
# Test: LiveFeedManager — on_funding callback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_funding_normal() -> None:
    mgr = LiveFeedManager()
    await mgr._on_funding(_make_funding(rate=0.0001), receipt_timestamp=time.time())
    assert mgr.tick_buffer.get_latest_funding("BTC-USDT-PERP") == pytest.approx(0.0001)


@pytest.mark.asyncio
async def test_on_funding_high_rate_warns() -> None:
    """A funding rate beyond ±0.05 % triggers a WARNING log."""
    mgr = LiveFeedManager()
    messages: list[str] = []
    handler_id = _loguru.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
        format="{message}",
    )
    try:
        await mgr._on_funding(
            _make_funding(rate=0.001),   # 0.1 % >> 0.05 % threshold
            receipt_timestamp=time.time(),
        )
    finally:
        _loguru.remove(handler_id)

    assert any("High funding rate" in m for m in messages)


@pytest.mark.asyncio
async def test_on_funding_normal_rate_no_warn() -> None:
    mgr = LiveFeedManager()
    messages: list[str] = []
    handler_id = _loguru.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
        format="{message}",
    )
    try:
        await mgr._on_funding(
            _make_funding(rate=0.0001),  # 0.01 % << 0.05 % threshold
            receipt_timestamp=time.time(),
        )
    finally:
        _loguru.remove(handler_id)

    assert not any("High funding rate" in m for m in messages)


@pytest.mark.asyncio
async def test_on_funding_negative_high_rate_warns() -> None:
    """Negative high funding rate also triggers the warning."""
    mgr = LiveFeedManager()
    messages: list[str] = []
    handler_id = _loguru.add(
        lambda msg: messages.append(msg.record["message"]),
        level="WARNING",
        format="{message}",
    )
    try:
        await mgr._on_funding(
            _make_funding(rate=-0.001),  # -0.1 %
            receipt_timestamp=time.time(),
        )
    finally:
        _loguru.remove(handler_id)

    assert any("High funding rate" in m for m in messages)
