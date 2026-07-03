"""
AtomiCortex — Signal Bridge.

Bridge between the trading bot process and the Telegram bot process.
The trading bot writes signals and events to a shared SQLite database;
the Telegram bot polls for new records and broadcasts them.

Design constraints:
- Synchronous API (called from Nautilus on_bar / on_position_closed)
- Thread-safe via threading.Lock
- All operations wrapped in try/except to never crash the trading bot
- No imports of nautilus or telegram
- Connection-per-call pattern (no persistent connections across processes)
"""

from __future__ import annotations

import json
try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.logger import get_logger

_log = get_logger(__name__)


class SignalBridge:
    """Writes trading signals and events to shared SQLite for the Telegram bot.

    Parameters
    ----------
    db_path:
        Path to the shared SQLite database (same file both processes use).
    """

    def __init__(
        self,
        db_path: str = "data/atomicortex.db",
        default_timeframe: str = "4h",
    ) -> None:
        # ``default_timeframe`` tags every signal from this bridge unless
        # a per-call override is given. The 4H bot uses the '4h' default
        # (its inherited _open_position calls log_signal with no tf), so
        # it is unchanged; the 15m strategy constructs its bridge with
        # default_timeframe='15m'.
        self._db_path = str(db_path)
        self._default_timeframe = default_timeframe
        self._lock = threading.Lock()
        self._init_tables()
        _log.info(
            "SignalBridge initialised | db={p} | tf={tf}",
            p=self._db_path, tf=self._default_timeframe,
        )

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode and foreign keys.

        H9: applies per-connection PRAGMAs so concurrent readers/writers
        (trading strategy + Telegram bot + reconciler) don't fail
        immediately on "database is locked":

        * ``busy_timeout=5000`` — wait up to 5s for the lock instead of
          erroring out at once. Per-connection setting; must be set on
          every connect (not just at init time).
        * ``journal_mode=WAL`` — concurrent reads while a writer holds
          the lock. WAL persists at the DB level but re-asserting on
          every connect is cheap and self-healing if another tool ever
          flips it back.
        * ``synchronous=NORMAL`` — pairs with WAL for a 5-10× write
          speed-up at negligible durability cost.

        Each PRAGMA is wrapped individually — a single bad pragma (e.g.
        a build of sqlite without WAL) must not lose the others.
        """
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        for pragma in (
            "PRAGMA busy_timeout=5000",
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA foreign_keys=ON",
        ):
            try:
                conn.execute(pragma)
            except sqlite3.Error as exc:
                _log.warning(
                    "SignalBridge {pragma} failed (non-fatal): {err}",
                    pragma=pragma, err=str(exc),
                )
        return conn

    # ------------------------------------------------------------------
    # Schema (idempotent)
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        """Create tables if they don't exist (safe to call repeatedly)."""
        try:
            conn = self._connect()
        except Exception as exc:
            _log.error("SignalBridge cannot connect to DB: {err}", err=str(exc))
            return
        try:
            # WAL / busy_timeout already applied in _connect(); proceed
            # straight to schema creation.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT,
                    direction   TEXT,
                    entry_price REAL,
                    stop_loss   REAL,
                    take_profit REAL,
                    confidence  REAL,
                    regime      TEXT,
                    timeframe   TEXT DEFAULT '4h',
                    atr         REAL,
                    funding_rate REAL,
                    position_size REAL,
                    notional    REAL,
                    leverage    REAL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at   TIMESTAMP,
                    close_price REAL,
                    pnl_pct     REAL,
                    result      TEXT DEFAULT 'open'
                );

                CREATE TABLE IF NOT EXISTS bot_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type  TEXT,
                    message     TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS bot_metrics (
                    id              INTEGER PRIMARY KEY DEFAULT 1,
                    equity          REAL,
                    daily_pnl       REAL,
                    regime          TEXT,
                    open_positions  INTEGER,
                    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_signals_result
                    ON signals_log(result);
                CREATE INDEX IF NOT EXISTS idx_signals_created
                    ON signals_log(created_at);
                CREATE INDEX IF NOT EXISTS idx_events_created
                    ON bot_events(created_at);
            """)
            conn.commit()

            # Idempotent migration for DBs created before the timeframe
            # column existed (e.g. the running 4H atomicortex.db). The
            # duplicate-column error is expected and swallowed; no data
            # is rewritten — fully backward compatible.
            try:
                conn.execute(
                    "ALTER TABLE signals_log "
                    "ADD COLUMN timeframe TEXT DEFAULT '4h'"
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already present
        except Exception as exc:
            _log.error("SignalBridge table init failed: {err}", err=str(exc))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Signal operations
    # ------------------------------------------------------------------

    def log_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        regime: str,
        atr: float = 0.0,
        funding_rate: float | None = 0.0,
        position_size: float = 0.0,
        notional: float = 0.0,
        leverage: float = 0.0,
        timeframe: str | None = None,
    ) -> int:
        """Write a new open signal to signals_log. Returns signal_id.

        ``timeframe`` overrides the bridge default for this call; when
        ``None`` the constructor's ``default_timeframe`` is used ('4h'
        for the unchanged 4H bot, '15m' for the 15m strategy's bridge).
        """
        tf = timeframe if timeframe is not None else self._default_timeframe
        with self._lock:
            try:
                conn = self._connect()
                try:
                    now = datetime.now(timezone.utc).isoformat()
                    cursor = conn.execute(
                        """INSERT INTO signals_log (
                            symbol, direction, entry_price,
                            stop_loss, take_profit, confidence,
                            regime, timeframe, atr, funding_rate,
                            position_size, notional, leverage,
                            created_at, result
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                        (
                            symbol, direction, entry_price,
                            stop_loss, take_profit, confidence,
                            regime, tf, atr, funding_rate,
                            position_size, notional, leverage,
                            now,
                        ),
                    )
                    conn.commit()
                    signal_id = cursor.lastrowid or 0
                    _log.info(
                        "Signal logged | id={sid} {dir} {sym} @ ${p:,.2f}",
                        sid=signal_id, dir=direction, sym=symbol, p=entry_price,
                    )
                    return signal_id
                finally:
                    conn.close()
            except Exception as exc:
                _log.error("SignalBridge.log_signal failed: {err}", err=str(exc))
                return 0

    def close_signal(
        self,
        signal_id: int,
        close_price: float,
        pnl_pct: float,
        result: str,
    ) -> None:
        """Update a signal when the position is closed."""
        with self._lock:
            try:
                conn = self._connect()
                try:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """UPDATE signals_log SET
                           closed_at = ?, close_price = ?,
                           pnl_pct = ?, result = ?
                           WHERE id = ?""",
                        (now, close_price, pnl_pct, result, signal_id),
                    )
                    conn.commit()
                    _log.info(
                        "Signal closed | id={sid} result={r} pnl={pnl:+.2f}%",
                        sid=signal_id, r=result, pnl=pnl_pct,
                    )
                finally:
                    conn.close()
            except Exception as exc:
                _log.error(
                    "SignalBridge.close_signal failed: {err}", err=str(exc),
                )

    def mark_rejected(self, signal_id: int, reason: str) -> None:
        """Update a signal that was rejected by the exchange or broker."""
        with self._lock:
            try:
                conn = self._connect()
                try:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """UPDATE signals_log SET
                           result = 'rejected',
                           closed_at = ?,
                           pnl_pct = NULL
                           WHERE id = ?""",
                        (now, signal_id),
                    )
                    conn.commit()
                    _log.info(
                        "Signal rejected | id={sid} reason={r}",
                        sid=signal_id, r=reason,
                    )
                finally:
                    conn.close()
            except Exception as exc:
                _log.error(
                    "SignalBridge.mark_rejected failed: {err}", err=str(exc),
                )

    # ------------------------------------------------------------------
    # Event operations
    # ------------------------------------------------------------------

    def log_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Write a regime change event to bot_events."""
        payload = json.dumps({"old": old_regime, "new": new_regime})
        self._log_event("regime_change", payload)

    def log_circuit_breaker(self, reason: str) -> None:
        """Write a circuit breaker event to bot_events."""
        self._log_event("circuit_breaker", reason)

    def _log_event(self, event_type: str, message: str) -> None:
        """Insert a row into bot_events (internal)."""
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(
                        "INSERT INTO bot_events (event_type, message) "
                        "VALUES (?, ?)",
                        (event_type, message),
                    )
                    conn.commit()
                    _log.info(
                        "Event logged | type={t} msg={m}",
                        t=event_type, m=message[:80],
                    )
                finally:
                    conn.close()
            except Exception as exc:
                _log.error(
                    "SignalBridge._log_event failed: {err}", err=str(exc),
                )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def update_metrics(
        self,
        equity: float,
        daily_pnl: float,
        regime: str,
        open_positions: int,
    ) -> None:
        """Upsert current trading metrics into bot_metrics."""
        with self._lock:
            try:
                conn = self._connect()
                try:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """INSERT OR REPLACE INTO bot_metrics
                           (id, equity, daily_pnl, regime,
                            open_positions, updated_at)
                           VALUES (1, ?, ?, ?, ?, ?)""",
                        (equity, daily_pnl, regime, open_positions, now),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as exc:
                _log.error(
                    "SignalBridge.update_metrics failed: {err}", err=str(exc),
                )
