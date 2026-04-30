"""
src/ingestion/live_feed.py

Live market data via Cryptofeed + Binance Futures WebSocket.

Subscribes to TRADES, L2_BOOK, and FUNDING channels.
Ticks are stored in a TickBuffer (in-memory, per-symbol deque).

Supported symbol format: Binance (e.g. "BTCUSDT") → Cryptofeed ("BTC-USDT-PERP").
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any

from cryptofeed import FeedHandler
from cryptofeed.callback import BookCallback, FundingCallback, TradeCallback
from cryptofeed.defines import BUY, FUNDING, L2_BOOK, TRADES
from cryptofeed.exchanges import BinanceFutures

from src.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------

def _to_cf_symbol(binance_symbol: str) -> str:
    """Convert a Binance symbol to Cryptofeed perpetual format.

    Examples
    --------
    ``"BTCUSDT"``  →  ``"BTC-USDT-PERP"``
    ``"SOLUSDT"``  →  ``"SOL-USDT-PERP"``
    """
    sym = binance_symbol.upper()
    if sym.endswith("USDT"):
        base = sym[:-4]
        return f"{base}-USDT-PERP"
    raise ValueError(
        f"Cannot convert symbol {binance_symbol!r}: only USDT-margined perps are supported"
    )


# ---------------------------------------------------------------------------
# TickBuffer
# ---------------------------------------------------------------------------

class TickBuffer:
    """In-memory ring-buffer for recent market-data ticks.

    Write methods are ``async`` and guard updates with ``asyncio.Lock``.
    Read methods are sync — safe within a single-threaded asyncio loop.

    Parameters
    ----------
    max_trades:
        Maximum number of trade records retained per symbol.
    """

    def __init__(self, max_trades: int = 1000) -> None:
        self._max_trades = max_trades
        self._trades: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max_trades)
        )
        self._books: dict[str, dict[str, Any]] = {}
        self._funding: dict[str, float] = {}
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------
    # Writes (async)
    # ----------------------------------------------------------------

    async def add_trade(self, symbol: str, trade: dict[str, Any]) -> None:
        async with self._lock:
            self._trades[symbol].append(trade)

    async def update_book(self, symbol: str, snapshot: dict[str, Any]) -> None:
        async with self._lock:
            self._books[symbol] = snapshot

    async def update_funding(self, symbol: str, rate: float) -> None:
        async with self._lock:
            self._funding[symbol] = rate

    # ----------------------------------------------------------------
    # Reads (sync)
    # ----------------------------------------------------------------

    def get_recent_trades(self, symbol: str, n: int = 100) -> list[dict[str, Any]]:
        """Return the *n* most recent trades for *symbol* (oldest first)."""
        return list(self._trades.get(symbol, deque()))[-n:]

    def get_current_book(self, symbol: str) -> dict[str, Any]:
        """Return the latest top-5 bid/ask snapshot, or ``{}`` if not yet received."""
        return self._books.get(symbol, {})

    def get_latest_funding(self, symbol: str) -> float:
        """Return the latest funding rate, or ``0.0`` if not yet received."""
        return self._funding.get(symbol, 0.0)


# ---------------------------------------------------------------------------
# LiveFeedManager
# ---------------------------------------------------------------------------

class LiveFeedManager:
    """Manages Cryptofeed WebSocket subscriptions for Binance Futures.

    Channels subscribed: ``TRADES``, ``L2_BOOK``, ``FUNDING``.

    Usage (blocking)
    ----------------
    ::

        mgr = LiveFeedManager()
        mgr.run(["BTCUSDT", "ETHUSDT"], duration=60)

    Ticks are accessible any time via ``mgr.tick_buffer``.
    """

    LATENCY_WARN_MS: float = 1_000.0    # ms — warning threshold for trade latency
    BOOK_LOG_INTERVAL: float = 60.0     # seconds between periodic L2-book log lines
    HEALTH_TIMEOUT: float = 30.0        # seconds without ticks → is_healthy() == False
    HIGH_FUNDING_THRESHOLD: float = 5e-4  # 0.05 %

    def __init__(self, max_trades_buffer: int = 1_000) -> None:
        self.tick_buffer = TickBuffer(max_trades_buffer)
        self._fh: FeedHandler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reconnects: int = 0
        self._last_tick_time: float = 0.0
        self._last_book_log: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public: status
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return ``True`` if a tick arrived within ``HEALTH_TIMEOUT`` seconds."""
        if self._last_tick_time == 0.0:
            return False
        return (time.monotonic() - self._last_tick_time) < self.HEALTH_TIMEOUT

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    @property
    def last_tick_time(self) -> float:
        """Monotonic timestamp of the last received tick (0 if none yet)."""
        return self._last_tick_time

    # ------------------------------------------------------------------
    # Public: run / stop
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str],
        duration: int | None = None,
    ) -> None:
        """Start the live feed.  **Blocking** — returns only when stopped.

        Parameters
        ----------
        symbols:
            Binance symbol names, e.g. ``["BTCUSDT", "ETHUSDT"]``.
        duration:
            Run for this many seconds then stop automatically.
            ``None`` means run until SIGINT / SIGTERM / :meth:`stop`.
        """
        cf_symbols = [_to_cf_symbol(s) for s in symbols]
        _log.info(
            f"LiveFeedManager starting — symbols={symbols} "
            f"cf_symbols={cf_symbols} duration={duration}s"
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        self._fh = FeedHandler(config={"log": {"disabled": True}})
        self._fh.add_feed(
            BinanceFutures(
                symbols=cf_symbols,
                channels=[TRADES, L2_BOOK, FUNDING],
                callbacks={
                    TRADES: TradeCallback(self._on_trade),
                    L2_BOOK: BookCallback(self._on_book),
                    FUNDING: FundingCallback(self._on_funding),
                },
                retries=-1,  # reconnect indefinitely
            )
        )

        if duration is not None:
            loop.call_later(duration, loop.stop)

        # start_loop=False: feeds are attached to the loop but we call run_forever ourselves
        self._fh.run(start_loop=False, install_signal_handlers=True)

        try:
            loop.run_forever()
        except SystemExit:
            _log.info("Signal received — shutting down LiveFeedManager")
        finally:
            self._shutdown(loop)

    def stop(self) -> None:
        """Stop the feed from any context (e.g. from a timer or external thread)."""
        if self._loop and self._loop.is_running():
            self._loop.stop()

    # ------------------------------------------------------------------
    # Private: shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, loop: asyncio.AbstractEventLoop) -> None:
        _log.info("LiveFeedManager: graceful shutdown …")
        try:
            if self._fh is not None:
                shutdown_tasks = self._fh._stop(loop=loop)
                loop.run_until_complete(asyncio.gather(*shutdown_tasks))
        except Exception as exc:
            _log.warning(f"Error during feed shutdown: {exc}")
        finally:
            if not loop.is_closed():
                loop.close()
            _log.info("LiveFeedManager: stopped")

    # ------------------------------------------------------------------
    # Private: callbacks
    # ------------------------------------------------------------------

    async def _on_trade(self, trade: Any, receipt_timestamp: float) -> None:
        self._last_tick_time = time.monotonic()

        latency_ms = (receipt_timestamp - trade.timestamp) * 1_000
        side = "buy" if trade.side == BUY else "sell"

        _log.debug(
            f"TRADE {trade.symbol} {side} "
            f"price={float(trade.price):.4f} qty={float(trade.amount):.6f} "
            f"latency={latency_ms:.0f}ms"
        )

        if latency_ms > self.LATENCY_WARN_MS:
            _log.warning(
                f"High trade latency {latency_ms:.0f}ms for {trade.symbol}"
            )

        await self.tick_buffer.add_trade(
            trade.symbol,
            {
                "price": float(trade.price),
                "amount": float(trade.amount),
                "side": side,
                "timestamp": trade.timestamp,
                "receipt_timestamp": receipt_timestamp,
                "latency_ms": latency_ms,
            },
        )

    async def _on_book(self, book: Any, receipt_timestamp: float) -> None:
        self._last_tick_time = time.monotonic()

        bids = sorted(book.book.bids.keys(), reverse=True)[:5]
        asks = sorted(book.book.asks.keys())[:5]

        bid_qty = sum(float(book.book.bids[p]) for p in bids)
        ask_qty = sum(float(book.book.asks[p]) for p in asks)
        total_qty = bid_qty + ask_qty
        imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else 0.0

        snapshot: dict[str, Any] = {
            "bids": [(float(p), float(book.book.bids[p])) for p in bids],
            "asks": [(float(p), float(book.book.asks[p])) for p in asks],
            "imbalance": imbalance,
            "timestamp": book.timestamp,
        }
        await self.tick_buffer.update_book(book.symbol, snapshot)

        now = time.monotonic()
        if now - self._last_book_log.get(book.symbol, 0.0) >= self.BOOK_LOG_INTERVAL:
            self._last_book_log[book.symbol] = now
            best_bid = float(bids[0]) if bids else 0.0
            best_ask = float(asks[0]) if asks else 0.0
            _log.info(
                f"BOOK {book.symbol} "
                f"bid={best_bid:.4f} ask={best_ask:.4f} "
                f"imbalance={imbalance:+.4f}"
            )

    async def _on_funding(self, funding: Any, receipt_timestamp: float) -> None:
        self._last_tick_time = time.monotonic()

        rate = float(funding.rate) if funding.rate is not None else 0.0
        mark = float(funding.mark_price) if funding.mark_price is not None else None

        _log.info(
            f"FUNDING {funding.symbol} rate={rate:.6f} ({rate:.4%})"
            + (f" mark={mark:.4f}" if mark is not None else "")
        )

        if abs(rate) > self.HIGH_FUNDING_THRESHOLD:
            _log.warning(
                f"High funding rate {rate:.4%} for {funding.symbol} "
                f"(threshold ±{self.HIGH_FUNDING_THRESHOLD:.4%})"
            )

        await self.tick_buffer.update_funding(funding.symbol, rate)
