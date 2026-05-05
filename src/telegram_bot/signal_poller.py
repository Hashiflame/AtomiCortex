"""
AtomiCortex — Signal Poller.

Polls the shared SQLite database for new trading signals and events
written by the trading bot process, then forwards them to the
Telegram Broadcaster for delivery to subscribers.

Runs as an asyncio background task inside the Telegram bot process.
"""

from __future__ import annotations

import asyncio
import json
try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]
from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.telegram_bot.broadcaster import Broadcaster

_log = get_logger(__name__)


class SignalPoller:
    """Polls shared SQLite for new signals/events every ``poll_interval`` seconds.

    Parameters
    ----------
    db_path:
        Absolute path to the shared SQLite database.
    broadcaster:
        The Telegram Broadcaster instance for sending messages.
    poll_interval:
        Seconds between poll cycles (default: 30).
    """

    def __init__(
        self,
        db_path: str,
        broadcaster: Broadcaster,
        poll_interval: int = 30,
    ) -> None:
        self._db_path = str(db_path)
        self._broadcaster = broadcaster
        self._poll_interval = poll_interval

        # High-water marks to avoid re-processing
        self._last_signal_id: int = 0
        self._last_event_id: int = 0

        # Cached metrics for /health and /stats commands
        self._cached_metrics: dict[str, Any] = {}

        # Background task handle
        self._task: asyncio.Task | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the polling loop as a background asyncio task."""
        if self._running:
            return
        self._running = True
        self._init_high_water_marks()
        self._task = asyncio.create_task(self._poll_loop())
        _log.info(
            "SignalPoller started | interval={i}s | db={p}",
            i=self._poll_interval, p=self._db_path,
        )

    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        _log.info("SignalPoller stopped")

    # ------------------------------------------------------------------
    # Init high-water marks (skip already-existing records on startup)
    # ------------------------------------------------------------------

    def _init_high_water_marks(self) -> None:
        """Set last_signal_id and last_event_id to current max."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM signals_log"
                ).fetchone()
                self._last_signal_id = row[0] if row else 0

                row = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM bot_events"
                ).fetchone()
                self._last_event_id = row[0] if row else 0

                _log.info(
                    "High-water marks | signal_id={s} event_id={e}",
                    s=self._last_signal_id, e=self._last_event_id,
                )
            finally:
                conn.close()
        except Exception as exc:
            _log.error("Failed to init high-water marks: {err}", err=str(exc))

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Main polling loop (runs until stopped)."""
        while self._running:
            try:
                await self._check_new_signals()
                await self._check_new_events()
                await self._update_cached_metrics()
            except Exception as exc:
                _log.error("Poll cycle error: {err}", err=str(exc))

            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Check new signals
    # ------------------------------------------------------------------

    async def _check_new_signals(self) -> None:
        """Find signals with id > last_signal_id and broadcast them."""
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM signals_log "
                    "WHERE id > ? AND result = 'open' "
                    "ORDER BY id ASC",
                    (self._last_signal_id,),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                signal_data = dict(row)
                self._last_signal_id = signal_data["id"]

                _log.info(
                    "New signal detected | id={sid} {dir} {sym}",
                    sid=signal_data["id"],
                    dir=signal_data.get("direction", "?"),
                    sym=signal_data.get("symbol", "?"),
                )

                try:
                    await self._broadcaster.broadcast_signal(signal_data)
                except Exception as exc:
                    _log.error(
                        "Failed to broadcast signal {sid}: {err}",
                        sid=signal_data["id"], err=str(exc),
                    )

            # Also check for newly closed signals
            await self._check_closed_signals()

        except Exception as exc:
            _log.error("_check_new_signals failed: {err}", err=str(exc))

    async def _check_closed_signals(self) -> None:
        """Check for recently closed signals and broadcast close events."""
        try:
            conn = self._connect()
            try:
                # Get signals closed in the last 2 poll intervals
                rows = conn.execute(
                    "SELECT * FROM signals_log "
                    "WHERE result IN ('win', 'loss', 'breakeven') "
                    "AND closed_at IS NOT NULL "
                    "AND datetime(closed_at) > datetime('now', '-2 minutes') "
                    "ORDER BY closed_at ASC",
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                signal = dict(row)
                try:
                    await self._broadcaster.broadcast_signal_closed(signal)
                except Exception as exc:
                    _log.error(
                        "Failed to broadcast signal close {sid}: {err}",
                        sid=signal.get("id"), err=str(exc),
                    )
        except Exception as exc:
            _log.error("_check_closed_signals failed: {err}", err=str(exc))

    # ------------------------------------------------------------------
    # Check new events
    # ------------------------------------------------------------------

    async def _check_new_events(self) -> None:
        """Find events with id > last_event_id and broadcast them."""
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM bot_events "
                    "WHERE id > ? "
                    "ORDER BY id ASC",
                    (self._last_event_id,),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                event = dict(row)
                self._last_event_id = event["id"]

                event_type = event.get("event_type", "")
                message = event.get("message", "")

                _log.info(
                    "New event | id={eid} type={t}",
                    eid=event["id"], t=event_type,
                )

                try:
                    if event_type == "regime_change":
                        data = json.loads(message)
                        await self._broadcaster.broadcast_regime_change(
                            data.get("old", "N/A"),
                            data.get("new", "N/A"),
                        )
                    elif event_type == "circuit_breaker":
                        await self._broadcaster.broadcast_circuit_breaker(message)
                except Exception as exc:
                    _log.error(
                        "Failed to broadcast event {eid}: {err}",
                        eid=event["id"], err=str(exc),
                    )

        except Exception as exc:
            _log.error("_check_new_events failed: {err}", err=str(exc))

    # ------------------------------------------------------------------
    # Update cached metrics
    # ------------------------------------------------------------------

    async def _update_cached_metrics(self) -> None:
        """Read bot_metrics and cache them for /health and /stats."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM bot_metrics WHERE id = 1"
                ).fetchone()
            finally:
                conn.close()

            if row:
                self._cached_metrics = dict(row)
                # Also update broadcaster's cache reference
                self._broadcaster._cached_metrics = self._cached_metrics
        except Exception as exc:
            _log.error("_update_cached_metrics failed: {err}", err=str(exc))

    @property
    def cached_metrics(self) -> dict[str, Any]:
        """Return the latest cached trading metrics."""
        return self._cached_metrics
