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
from typing import Any, Callable

from src.logger import get_logger
from src.telegram_bot.broadcaster import Broadcaster

_log = get_logger(__name__)


class SignalPoller:
    """Polls shared SQLite for new signals/events every ``poll_interval`` seconds.

    Parameters
    ----------
    db_path:
        Absolute path to the shared SQLite database (legacy, single-DB).
    db_paths:
        List of DB paths to poll (multi-timeframe isolation).
        If *db_path* is given, it is prepended to *db_paths*.
    broadcaster:
        The Telegram Broadcaster instance for sending messages.
    poll_interval:
        Seconds between poll cycles (default: 30).
    """

    def __init__(
        self,
        db_path: str | None = None,
        broadcaster: Broadcaster | None = None,
        poll_interval: int = 30,
        *,
        db_paths: list[str] | None = None,
        discover_callback: Callable[[], list[str]] | None = None,
        discover_every_cycles: int = 10,
        recovery_minutes: int = 30,
    ) -> None:
        # Normalise to a list of unique paths
        paths: list[str] = []
        if db_path is not None:
            paths.append(str(db_path))
        if db_paths is not None:
            for p in db_paths:
                if str(p) not in paths:
                    paths.append(str(p))
        if not paths:
            paths = ["data/atomicortex.db"]

        self._db_paths: list[str] = paths
        # Backward-compatible single-path accessor
        self._db_path: str = paths[0]
        self._broadcaster = broadcaster
        self._poll_interval = poll_interval

        # Per-DB high-water marks to avoid re-processing
        self._last_signal_ids: dict[str, int] = {p: 0 for p in paths}
        self._last_event_ids: dict[str, int] = {p: 0 for p in paths}

        # Per-DB set of signal_ids whose close has already been broadcast
        # in this process. Seeded at startup with every currently-closed
        # signal so that, after a restart, we never re-broadcast an
        # already-closed signal even if it falls inside the lookback
        # window. New closes during runtime are added only after a
        # *successful* broadcast (at-least-once semantics + dedup).
        self._broadcasted_close_ids: dict[str, set[int]] = {
            p: set() for p in paths
        }

        # M1: optional callback that re-discovers trading DB paths each
        # cycle. Allows a 15m / 1H bot started AFTER the Telegram bot to
        # appear without restart. None → static path list (legacy).
        self._discover_callback = discover_callback
        self._discover_every_cycles = max(1, int(discover_every_cycles))
        self._cycle_count: int = 0

        # M3: how far back to re-broadcast OPENs after restart.
        # Closed signals are still deduped via _broadcasted_close_ids
        # seed (no spammy re-announcements), but ENTRY signals written
        # in the recovery window before startup are emitted on the
        # first poll. Set to 0 to keep legacy "skip all prior" behaviour.
        self._recovery_minutes = max(0, int(recovery_minutes))

        # Cached metrics for /health and /stats commands
        self._cached_metrics: dict[str, Any] = {}

        # Background task handle
        self._task: asyncio.Task | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _tf_for_db(db_path: str) -> str:
        """Derive the timeframe from the isolated DB filename.

        Authoritative: each strategy writes to its own DB file, so the
        path is the source of truth regardless of whether the row has a
        ``timeframe`` column (keeps 4H trading code untouched).
        """
        name = str(db_path)
        if "_15m" in name:
            return "15m"
        if "_1h" in name:
            return "1h"
        return "4h"

    def _tag_timeframe(self, signal_data: dict[str, Any], db_path: str) -> None:
        """Ensure ``signal_data['timeframe']`` is set.

        Prefers an explicit non-null DB column value; otherwise falls
        back to the path-derived timeframe.
        """
        tf = signal_data.get("timeframe")
        if not tf:
            signal_data["timeframe"] = self._tf_for_db(db_path)

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
            "SignalPoller started | interval={i}s | dbs={p}",
            i=self._poll_interval, p=self._db_paths,
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
        """Set per-DB last_signal_id and last_event_id.

        M3: instead of jumping to MAX(id), step back to the highest id
        whose ``created_at`` is OLDER than the recovery window so any
        ENTRY signal written in the last ``recovery_minutes`` before
        startup gets emitted on the first poll. Closed signals stay
        deduped via the broadcasted-close set (seed includes the full
        history), so no spammy re-announcements.
        """
        for db_path in self._db_paths:
            self._init_marks_for_db(db_path)

    def _init_marks_for_db(self, db_path: str) -> None:
        try:
            conn = self._connect(db_path)
            try:
                if self._recovery_minutes > 0:
                    # Highest id strictly OLDER than the recovery window.
                    row = conn.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM signals_log "
                        "WHERE created_at IS NOT NULL "
                        "AND datetime(created_at) <= "
                        "datetime('now', ?)",
                        (f"-{self._recovery_minutes} minutes",),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM signals_log"
                    ).fetchone()
                self._last_signal_ids[db_path] = row[0] if row else 0

                row = conn.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM bot_events"
                ).fetchone()
                self._last_event_ids[db_path] = row[0] if row else 0

                # Seed the broadcasted-close set with every signal
                # that is already closed at startup — they were
                # either announced in a prior session or are stale,
                # and must never be re-announced after a restart.
                closed = conn.execute(
                    "SELECT id FROM signals_log "
                    "WHERE result IN ('win','loss','breakeven') "
                    "AND closed_at IS NOT NULL"
                ).fetchall()
                self._broadcasted_close_ids[db_path] = {
                    int(r[0]) for r in closed
                }

                _log.info(
                    "High-water marks | db={db} signal_id={s} "
                    "event_id={e} closed_seeded={c} recovery_min={r}",
                    db=db_path,
                    s=self._last_signal_ids[db_path],
                    e=self._last_event_ids[db_path],
                    c=len(self._broadcasted_close_ids[db_path]),
                    r=self._recovery_minutes,
                )
            finally:
                conn.close()
        except Exception as exc:
            _log.error(
                "Failed to init high-water marks for {db}: {err}",
                db=db_path, err=str(exc),
            )

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Main polling loop.

        M1: re-discover DB paths every ``_discover_every_cycles`` to
        pick up isolated trading bots started after this poller.
        M2: fan out per-DB polls with ``asyncio.gather`` so a slow DB
        cannot block the others (worst case N×poll_interval drift).
        """
        while self._running:
            self._cycle_count += 1
            if (
                self._discover_callback is not None
                and self._cycle_count % self._discover_every_cycles == 0
            ):
                self._refresh_db_paths()

            await asyncio.gather(
                *(self._poll_one(p) for p in list(self._db_paths)),
                return_exceptions=True,
            )

            await asyncio.sleep(self._poll_interval)

    async def _poll_one(self, db_path: str) -> None:
        """Per-DB poll bundle. Failures are isolated to this DB."""
        try:
            await self._check_new_signals(db_path)
            await self._check_new_events(db_path)
            await self._update_cached_metrics(db_path)
        except Exception as exc:
            _log.error(
                "Poll cycle error for {db}: {err}",
                db=db_path, err=str(exc),
            )

    def _refresh_db_paths(self) -> None:
        """Re-run the discovery callback and onboard any new DB paths.

        New paths get their own high-water marks + close-set seeded
        from the current DB state, so we don't dump every historic
        signal of a freshly-discovered 15m bot.
        """
        if self._discover_callback is None:
            return
        try:
            discovered = [str(p) for p in self._discover_callback()]
        except Exception as exc:
            _log.warning(
                "DB discovery callback failed (non-fatal): {e}", e=exc,
            )
            return
        for path in discovered:
            if path in self._db_paths:
                continue
            _log.info("Discovered new trading DB: {p}", p=path)
            self._db_paths.append(path)
            # Initialise high-water marks for the new DB — without
            # recovery window so the existing backlog is NOT replayed
            # (it belongs to the strategy that just connected, not us).
            self._last_signal_ids[path] = 0
            self._last_event_ids[path] = 0
            self._broadcasted_close_ids[path] = set()
            saved_recovery = self._recovery_minutes
            try:
                self._recovery_minutes = 0
                self._init_marks_for_db(path)
            finally:
                self._recovery_minutes = saved_recovery

    # ------------------------------------------------------------------
    # Check new signals
    # ------------------------------------------------------------------

    async def _check_new_signals(self, db_path: str) -> None:
        """Find signals with id > last_signal_id and broadcast them.

        M4: drop the ``result='open'`` filter. A signal that opens and
        closes between two poll cycles (real on 15m with a fast SL hit)
        used to be skipped entirely as ENTRY because by the time we
        polled the row's result was already 'win'/'loss'. The close is
        still announced separately via ``_check_closed_signals``.
        """
        try:
            conn = self._connect(db_path)
            try:
                rows = conn.execute(
                    "SELECT * FROM signals_log "
                    "WHERE id > ? "
                    "ORDER BY id ASC",
                    (self._last_signal_ids[db_path],),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                signal_data = dict(row)
                self._last_signal_ids[db_path] = signal_data["id"]
                self._tag_timeframe(signal_data, db_path)

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
            await self._check_closed_signals(db_path)

        except Exception as exc:
            _log.error("_check_new_signals failed: {err}", err=str(exc))

    async def _check_closed_signals(self, db_path: str) -> None:
        """Check for recently closed signals and broadcast close events.

        Deduplication: ``self._broadcasted_close_ids[db_path]`` tracks
        every signal_id whose close has already been broadcast in this
        process. The set is seeded on startup with all already-closed
        signals (see ``_init_high_water_marks``) so a restart never
        re-announces an old close. The 10-minute window is just a query
        bound — correctness comes from the set.

        On broadcast failure the id is *not* added, so the next poll
        cycle retries (at-least-once + idempotent dedup).
        """
        try:
            conn = self._connect(db_path)
            try:
                rows = conn.execute(
                    "SELECT * FROM signals_log "
                    "WHERE result IN ('win', 'loss', 'breakeven') "
                    "AND closed_at IS NOT NULL "
                    "AND datetime(closed_at) > datetime('now', '-10 minutes') "
                    "ORDER BY closed_at ASC",
                ).fetchall()
            finally:
                conn.close()

            seen = self._broadcasted_close_ids.setdefault(db_path, set())

            for row in rows:
                signal = dict(row)
                sid = int(signal["id"])
                if sid in seen:
                    continue

                self._tag_timeframe(signal, db_path)
                try:
                    await self._broadcaster.broadcast_signal_closed(signal)
                except Exception as exc:
                    _log.error(
                        "Failed to broadcast signal close {sid}: {err}",
                        sid=sid, err=str(exc),
                    )
                    # Do not mark as broadcasted — retry on next cycle.
                    continue

                seen.add(sid)
        except Exception as exc:
            _log.error("_check_closed_signals failed: {err}", err=str(exc))

    # ------------------------------------------------------------------
    # Check new events
    # ------------------------------------------------------------------

    async def _check_new_events(self, db_path: str) -> None:
        """Find events with id > last_event_id and broadcast them."""
        try:
            conn = self._connect(db_path)
            try:
                rows = conn.execute(
                    "SELECT * FROM bot_events "
                    "WHERE id > ? "
                    "ORDER BY id ASC",
                    (self._last_event_ids[db_path],),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                event = dict(row)
                self._last_event_ids[db_path] = event["id"]

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

    async def _update_cached_metrics(self, db_path: str) -> None:
        """Read bot_metrics and cache them for /health and /stats."""
        try:
            conn = self._connect(db_path)
            try:
                row = conn.execute(
                    "SELECT * FROM bot_metrics WHERE id = 1"
                ).fetchone()
            finally:
                conn.close()

            if row:
                self._cached_metrics = dict(row)
                # Also update broadcaster's cache reference
                if self._broadcaster is not None:
                    self._broadcaster._cached_metrics = self._cached_metrics
        except Exception as exc:
            _log.error("_update_cached_metrics failed: {err}", err=str(exc))

    @property
    def cached_metrics(self) -> dict[str, Any]:
        """Return the latest cached trading metrics."""
        return self._cached_metrics
