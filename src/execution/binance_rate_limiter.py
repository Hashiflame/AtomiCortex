"""Centralised Binance Futures REST rate limiter (H22).

Binance Futures imposes 2400 weight per minute per IP. Once exceeded:
HTTP 429 → 2 min IP ban; on repeat, HTTP 418 → up to 3 days. Multiple
modules in AtomiCortex hit the same IP (ml_strategy OI poll / funding
preload / taker_buy_volume fetch / klines preload, reconciler
positionRisk, watchdog emergency endpoints), and they had no
coordination — a synchronous burst could ban the bot mid-session and
lock us out of position management.

Design
------
* Token-bucket per rolling 60 s window — keep weights with their
  timestamps in a deque, prune anything older than 60 s before each
  decision.
* Budget is **half** the Binance limit (``1200``) to stay clear of
  spike-driven bans even when multiple modules wake up together.
* The wire is the source of truth: every Binance response carries
  ``X-MBX-USED-WEIGHT-1M`` — feed it back via ``update_from_headers``
  so our local counter cannot underestimate.
* Async-safe singleton (`asyncio.Lock`).
* Fail-soft: any internal exception is swallowed; the bot continues
  rather than blocking trade-critical paths because of a tracking bug.

Wired into
----------
* ``LiveFeatureState.fetch_taker_buy_volume`` — H1c
* ``Watchdog._signed_*``                       — H15/H16
* ``reconciler.fetch_position_risk``           — H10/this step

Still pending (future step — out of scope for H22 because the task
forbids touching ``ml_strategy.py``):
* OI poll, funding-rate preload, klines preload, per-bar
  taker_buy_volume fetch wrapper — all in ``ml_strategy.py``.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Mapping

from src.logger import get_logger

_log = get_logger(__name__)


class BinanceRateLimiter:
    """Process-wide token-bucket limiter for Binance Futures REST."""

    MAX_WEIGHT_PER_MINUTE: int = 1200   # 50 % of the 2400 hard cap
    WINDOW_SECONDS: float = 60.0

    _instance: "BinanceRateLimiter | None" = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> "BinanceRateLimiter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton — for tests; in production it lives for
        the lifetime of the process."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Each entry: (monotonic_ts_seconds, weight). Pruned lazily on
        # every acquire / current_weight call.
        self._events: deque[tuple[float, int]] = deque()
        # Floor lifted from response headers; locally-tracked weight can
        # never report less than what Binance itself says. Timestamped
        # so it decays with the same 60-s window as local events.
        self._header_floor: int = 0
        self._header_floor_ts: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def acquire(self, weight: int = 1) -> None:
        """Reserve ``weight`` units; wait if the rolling-window budget is
        exhausted. Fail-soft on internal errors (swallows + continues)."""
        try:
            await self._acquire(weight)
        except Exception as exc:  # noqa: BLE001 — fail-soft critical path
            _log.warning(
                "BinanceRateLimiter.acquire failed (continuing): {e}",
                e=exc,
            )

    async def _acquire(self, weight: int) -> None:
        weight = max(1, int(weight))
        async with self._lock:
            # Loop instead of single-sleep — the oldest entry might
            # expire while we hold the lock if the budget is tight.
            for _ in range(8):
                now = time.monotonic()
                self._prune(now)
                used = self._used_weight()
                if used + weight <= self.MAX_WEIGHT_PER_MINUTE:
                    self._events.append((now, weight))
                    return
                # Sleep until the oldest event expires (plus a tiny pad).
                if not self._events:
                    # Nothing to wait for — weight > MAX with empty queue.
                    # Record and return rather than block forever.
                    self._events.append((now, weight))
                    return
                wait = (
                    self._events[0][0] + self.WINDOW_SECONDS - now + 0.01
                )
                wait = max(0.0, wait)
            # Lock is released during sleep so other tasks can update.
                await asyncio.sleep(wait)
            # Defensive: should never spin 8 times in practice. Record
            # and proceed rather than starve.
            now = time.monotonic()
            self._events.append((now, weight))

    def update_from_headers(self, headers: Mapping[str, Any] | None) -> None:
        """Raise the local floor to whatever Binance reports.

        Reads ``X-MBX-USED-WEIGHT-1M``; case-insensitive. Silent on
        missing/malformed values."""
        try:
            if not headers:
                return
            # Accept either dict-like (case-sensitive) or aiohttp's
            # CIMultiDict (case-insensitive); try common spellings.
            val: Any = None
            for key in (
                "X-MBX-USED-WEIGHT-1M",
                "x-mbx-used-weight-1m",
                "X-MBX-USED-WEIGHT-1m",
            ):
                if hasattr(headers, "get"):
                    val = headers.get(key)
                if val is not None:
                    break
            if val is None:
                return
            try:
                reported = int(val)
            except (TypeError, ValueError):
                return
            if reported < 0:
                return
            if reported > self._header_floor:
                self._header_floor = reported
                self._header_floor_ts = time.monotonic()
                _log.debug(
                    "Binance used-weight reported by server: {w}",
                    w=reported,
                )
        except Exception as exc:  # noqa: BLE001
            _log.debug(
                "update_from_headers failed (non-fatal): {e}", e=exc,
            )

    # ------------------------------------------------------------------
    # Introspection (for tests / diagnostics)
    # ------------------------------------------------------------------

    def used_weight(self) -> int:
        """Current rolling-window weight (the larger of local + server)."""
        self._prune(time.monotonic())
        return self._used_weight()

    def available_weight(self) -> int:
        return max(0, self.MAX_WEIGHT_PER_MINUTE - self.used_weight())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.WINDOW_SECONDS
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        # Server-reported floor decays with its own timestamp, on the
        # same 60-s window as local events.
        if (
            self._header_floor > 0
            and self._header_floor_ts < cutoff
        ):
            self._header_floor = 0
            self._header_floor_ts = 0.0

    def _used_weight(self) -> int:
        local = sum(w for _, w in self._events)
        return max(local, self._header_floor)
