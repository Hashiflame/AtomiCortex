"""
AtomiCortex — Telegram Bot Database.

SQLite-backed persistence for user management, signal logging,
and bot event tracking.  Uses a connection-per-call pattern for
thread safety (consistent with MetricsCollector).

Phase 7 — Telegram Bot.
"""

from __future__ import annotations

try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger

_log = get_logger(__name__)


class Database:
    """SQLite database for the Telegram bot.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = "data/telegram_bot.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        _log.info("Telegram bot DB initialised | path={p}", p=str(self._db_path))

    # ------------------------------------------------------------------
    # Connection helper  (TG-007: PRAGMA WAL moved to _init_db only)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with row_factory set."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Schema  (TG-012: indexes, TG-013: raise on failure)
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they don't exist.  Raises on failure."""
        conn = self._connect()
        try:
            # TG-007: set WAL once during init (persistent setting)
            conn.execute("PRAGMA journal_mode=WAL")

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id     INTEGER PRIMARY KEY,
                    username    TEXT,
                    first_name  TEXT,
                    role        TEXT DEFAULT 'free',
                    expires_at  TIMESTAMP,
                    joined_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_banned   BOOLEAN DEFAULT FALSE,
                    notes       TEXT
                );

                CREATE TABLE IF NOT EXISTS signals_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT,
                    direction   TEXT,
                    entry_price REAL,
                    stop_loss   REAL,
                    take_profit REAL,
                    confidence  REAL,
                    regime      TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at   TIMESTAMP,
                    pnl_pct     REAL,
                    result      TEXT
                );

                CREATE TABLE IF NOT EXISTS bot_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type  TEXT,
                    message     TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- TG-012: Indexes for frequently queried columns
                CREATE INDEX IF NOT EXISTS idx_users_role
                    ON users(role);
                CREATE INDEX IF NOT EXISTS idx_signals_result
                    ON signals_log(result);
                CREATE INDEX IF NOT EXISTS idx_signals_created
                    ON signals_log(created_at);
                CREATE INDEX IF NOT EXISTS idx_events_created
                    ON bot_events(created_at);

                CREATE TABLE IF NOT EXISTS payments (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         INTEGER NOT NULL,
                    method          TEXT NOT NULL,
                    amount_usd      REAL,
                    stars_amount    INTEGER,
                    days            INTEGER,
                    payload         TEXT,
                    invoice_id      TEXT,
                    status          TEXT DEFAULT 'pending',
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    paid_at         TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_payments_user
                    ON payments(user_id);
                CREATE INDEX IF NOT EXISTS idx_payments_status
                    ON payments(status);
                CREATE INDEX IF NOT EXISTS idx_payments_payload
                    ON payments(payload);
            """)
            conn.commit()
        except Exception as exc:
            # TG-013: Raise instead of silent failure
            _log.error("FATAL: Failed to init telegram bot DB: {err}", err=str(exc))
            raise RuntimeError(f"Telegram bot DB init failed: {exc}") from exc
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Fetch a user by Telegram user_id.

        If the user has an expired premium subscription, automatically
        downgrade them to ``free``.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,),
            ).fetchone()
            if row is None:
                return None

            user = dict(row)

            # Auto-downgrade expired premium (TG-009: always use timezone.utc)
            if (
                user["role"] == "premium"
                and user["expires_at"] is not None
            ):
                try:
                    expires = datetime.fromisoformat(user["expires_at"])
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                    if expires < datetime.now(timezone.utc):
                        conn.execute(
                            "UPDATE users SET role = 'free', expires_at = NULL "
                            "WHERE user_id = ?",
                            (user_id,),
                        )
                        conn.commit()
                        user["role"] = "free"
                        user["expires_at"] = None
                        _log.info(
                            "Auto-downgraded expired premium user {uid}",
                            uid=user_id,
                        )
                except (ValueError, TypeError):
                    pass

            return user
        finally:
            conn.close()

    # TG-014: SQL-level username lookup instead of O(N) Python scan
    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Fetch a user by username (case-insensitive)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE LOWER(username) = LOWER(?)",
                (username,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_user(
        self,
        user_id: int,
        username: str | None = None,
        first_name: str | None = None,
    ) -> None:
        """Create a new user or update username/first_name if exists."""
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO users (user_id, username, first_name)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id)
                   DO UPDATE SET username = excluded.username,
                                 first_name = excluded.first_name""",
                (user_id, username, first_name),
            )
            conn.commit()
        finally:
            conn.close()

    def set_role(
        self,
        user_id: int,
        role: str,
        expires_at: datetime | None = None,
    ) -> None:
        """Set user role.  ``expires_at=None`` means permanent."""
        expires_str = expires_at.isoformat() if expires_at else None
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET role = ?, expires_at = ? WHERE user_id = ?",
                (role, expires_str, user_id),
            )
            conn.commit()
        finally:
            conn.close()

    def ban_user(self, user_id: int) -> None:
        """Ban a user."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET is_banned = TRUE WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def unban_user(self, user_id: int) -> None:
        """Unban a user."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET is_banned = FALSE WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_all_users(self) -> list[dict[str, Any]]:
        """Return all registered users."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY joined_at DESC",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_users_by_role(self, role: str) -> list[dict[str, Any]]:
        """Return users filtered by role."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM users WHERE role = ? AND is_banned = FALSE "
                "ORDER BY joined_at DESC",
                (role,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_non_banned_users(self) -> list[dict[str, Any]]:
        """Return all non-banned users."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM users WHERE is_banned = FALSE "
                "ORDER BY joined_at DESC",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def set_notes(self, user_id: int, notes: str) -> None:
        """Set notes for a user (owner use)."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET notes = ? WHERE user_id = ?",
                (notes, user_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def log_signal(self, signal_data: dict[str, Any]) -> int:
        """Log a new trading signal.  Returns the signal ID."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO signals_log
                   (symbol, direction, entry_price, stop_loss,
                    take_profit, confidence, regime, result)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'open')""",
                (
                    signal_data.get("symbol", ""),
                    signal_data.get("direction", ""),
                    signal_data.get("entry_price", 0.0),
                    signal_data.get("stop_loss", 0.0),
                    signal_data.get("take_profit", 0.0),
                    signal_data.get("confidence", 0.0),
                    signal_data.get("regime", ""),
                ),
            )
            conn.commit()
            signal_id = cursor.lastrowid or 0
            _log.debug("Signal logged | id={sid}", sid=signal_id)
            return signal_id
        finally:
            conn.close()

    def close_signal(
        self,
        signal_id: int,
        pnl_pct: float,
        result: str,
    ) -> None:
        """Close a signal with PnL result ('win' or 'loss')."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE signals_log
                   SET closed_at = ?, pnl_pct = ?, result = ?
                   WHERE id = ?""",
                (now, pnl_pct, result, signal_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_signals_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent signals, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM signals_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_open_signals(self) -> list[dict[str, Any]]:
        """Return signals with result='open'."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM signals_log WHERE result = 'open' "
                "ORDER BY created_at DESC",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Compute aggregate stats from signals_log.

        Returns
        -------
        dict with keys: total_trades, wins, losses, open,
        win_rate, total_pnl_pct.
        """
        conn = self._connect()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM signals_log",
            ).fetchone()[0]

            closed = conn.execute(
                "SELECT COUNT(*) FROM signals_log "
                "WHERE result IN ('win', 'loss')",
            ).fetchone()[0]

            wins = conn.execute(
                "SELECT COUNT(*) FROM signals_log WHERE result = 'win'",
            ).fetchone()[0]

            losses = conn.execute(
                "SELECT COUNT(*) FROM signals_log WHERE result = 'loss'",
            ).fetchone()[0]

            open_count = conn.execute(
                "SELECT COUNT(*) FROM signals_log WHERE result = 'open'",
            ).fetchone()[0]

            total_pnl_row = conn.execute(
                "SELECT COALESCE(SUM(pnl_pct), 0.0) FROM signals_log "
                "WHERE result IN ('win', 'loss')",
            ).fetchone()
            total_pnl = total_pnl_row[0] if total_pnl_row else 0.0

            win_rate = wins / closed if closed > 0 else 0.0

            # 30-day stats (TG-009: always use timezone.utc)
            thirty_days_ago = datetime.now(timezone.utc).isoformat()[:10]
            wins_30d = conn.execute(
                "SELECT COUNT(*) FROM signals_log "
                "WHERE result = 'win' AND created_at >= date(?, '-30 days')",
                (thirty_days_ago,),
            ).fetchone()[0]
            closed_30d = conn.execute(
                "SELECT COUNT(*) FROM signals_log "
                "WHERE result IN ('win', 'loss') "
                "AND created_at >= date(?, '-30 days')",
                (thirty_days_ago,),
            ).fetchone()[0]
            win_rate_30d = wins_30d / closed_30d if closed_30d > 0 else 0.0

            return {
                "total_trades": total,
                "closed_trades": closed,
                "wins": wins,
                "losses": losses,
                "open": open_count,
                "win_rate": win_rate,
                "win_rate_30d": win_rate_30d,
                "total_pnl_pct": total_pnl,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, message: str) -> None:
        """Log a bot event (signal/regime_change/circuit_breaker/error)."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO bot_events (event_type, message) VALUES (?, ?)",
                (event_type, message),
            )
            conn.commit()
        finally:
            conn.close()

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent events, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM bot_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_signals_today_count(self) -> int:
        """Return number of signals created today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM signals_log "
                "WHERE date(created_at) = ?",
                (today,),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Payments
    # ------------------------------------------------------------------

    def create_payment(
        self,
        user_id: int,
        method: str,
        amount_usd: float,
        days: int,
        payload: str = "",
        invoice_id: str = "",
        stars_amount: int = 0,
        status: str = "pending",
    ) -> int:
        """Create a payment record.  Returns the payment ID."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO payments
                   (user_id, method, amount_usd, stars_amount, days,
                    payload, invoice_id, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, method, amount_usd, stars_amount, days,
                 payload, invoice_id, status),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def update_payment_status(
        self,
        payment_id: int,
        status: str,
        paid_at: str | None = None,
    ) -> None:
        """Update payment status (pending → paid / failed)."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE payments SET status = ?, paid_at = ? WHERE id = ?",
                (status, paid_at, payment_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_payment_by_payload(self, payload: str) -> dict[str, Any] | None:
        """Find a payment by its unique payload string."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM payments WHERE payload = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (payload,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_pending_payments(self, method: str | None = None) -> list[dict[str, Any]]:
        """Return all pending payments, optionally filtered by method."""
        conn = self._connect()
        try:
            if method:
                rows = conn.execute(
                    "SELECT * FROM payments WHERE status = 'pending' "
                    "AND method = ? ORDER BY created_at DESC",
                    (method,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM payments WHERE status = 'pending' "
                    "ORDER BY created_at DESC",
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_payments(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent payments, newest first."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM payments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_revenue_stats(self) -> dict[str, Any]:
        """Aggregate revenue statistics."""
        conn = self._connect()
        try:
            total = conn.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM payments "
                "WHERE status = 'paid'",
            ).fetchone()[0]

            this_month = datetime.now(timezone.utc).strftime("%Y-%m")
            month_total = conn.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM payments "
                "WHERE status = 'paid' "
                "AND strftime('%Y-%m', paid_at) = ?",
                (this_month,),
            ).fetchone()[0]

            stars_total = conn.execute(
                "SELECT COALESCE(SUM(stars_amount), 0) FROM payments "
                "WHERE status = 'paid' AND method = 'stars'",
            ).fetchone()[0]

            usdt_total = conn.execute(
                "SELECT COALESCE(SUM(amount_usd), 0) FROM payments "
                "WHERE status = 'paid' AND method = 'usdt'",
            ).fetchone()[0]

            active_premiums = conn.execute(
                "SELECT COUNT(*) FROM users "
                "WHERE role = 'premium'",
            ).fetchone()[0]

            paid_count = conn.execute(
                "SELECT COUNT(*) FROM payments WHERE status = 'paid'",
            ).fetchone()[0]

            return {
                "total_usd": total,
                "this_month_usd": month_total,
                "stars_total": stars_total,
                "usdt_total": usdt_total,
                "active_premiums": active_premiums,
                "total_payments": paid_count,
            }
        finally:
            conn.close()
