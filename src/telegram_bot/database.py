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

    def __init__(
        self,
        db_path: str | Path = "data/telegram_bot.db",
        init_schema: bool = True,
    ) -> None:
        """Attach to a SQLite database.

        ``init_schema=True`` (default, backward compatible) creates the
        Telegram-bot schema (users / signals_log / bot_events / payments
        + indexes) and runs an idempotent ALTER on ``signals_log``.

        ``init_schema=False`` is for *attaching to a database owned by
        another process* (e.g. the trading bot's ``atomicortex.db``).
        No CREATE / ALTER fires. This avoids:
          * cluttering the trading DB with Telegram-only tables;
          * SQLite WAL lock races between the trading writer and the
            Telegram-bot ALTER TABLE on every ``/stats`` request.

        Read methods (``get_recent_signals``, ``get_latest_metrics`` …)
        already swallow ``sqlite3.OperationalError`` for missing tables,
        so they remain safe in this "schema-free" mode.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema = init_schema
        if init_schema:
            self._init_db()
            _log.info("Telegram bot DB initialised | path={p}", p=str(self._db_path))
        else:
            _log.info(
                "Telegram bot DB attached (no-DDL mode) | path={p}",
                p=str(self._db_path),
            )

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
                    timeframe   TEXT DEFAULT '4h',
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

            # Idempotent migration: older DBs (incl. atomicortex*.db written
            # by SignalBridge) predate the `timeframe` column. ADD COLUMN is
            # a no-op once present; the duplicate-column error is expected
            # and swallowed. Backward compatible — never drops/rewrites data.
            try:
                conn.execute(
                    "ALTER TABLE signals_log "
                    "ADD COLUMN timeframe TEXT DEFAULT '4h'"
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists
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

    def get_recent_signals(
        self,
        limit: int = 10,
        timeframe: str | None = None,
        status: str | None = None,
        result_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent signals (newest first), any status by default.

        Single-DB by design — callers merge across the isolated trading
        DBs (mirrors the /stats pattern). ``timeframe`` is only filtered
        when the column exists; ``status`` is ``open`` or ``closed``.

        M6: ``result_filter`` pushes the wins / losses / open selector
        from a Python-side post-filter into the SQL ``WHERE`` clause.
        Accepts ``"wins"`` → ``result='win'``, ``"losses"`` → ``'loss'``,
        ``"open"`` → ``'open'``. Cuts the per-DB row count from a flat
        200 down to exactly ``limit`` after filtering.
        """
        conn = self._connect()
        try:
            cols = {
                r[1] for r in conn.execute(
                    "PRAGMA table_info(signals_log)"
                ).fetchall()
            }
            where: list[str] = []
            params: list[Any] = []
            if timeframe and "timeframe" in cols:
                where.append("COALESCE(timeframe,'4h') = ?")
                params.append(timeframe)
            if status == "open":
                where.append("result = 'open'")
            elif status == "closed":
                where.append("result IN ('win','loss','breakeven')")
            _RESULT_MAP = {"wins": "win", "losses": "loss", "open": "open"}
            if result_filter in _RESULT_MAP:
                where.append("result = ?")
                params.append(_RESULT_MAP[result_filter])
            sql = "SELECT * FROM signals_log"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def get_signals_paginated(
        self,
        page: int,
        per_page: int,
        timeframe: str | None = None,
        result_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """``(page_rows, total_count)`` for single-DB pagination.

        Multi-DB pagination is done by the caller over merged results;
        this is the single-DB primitive. ``result_filter`` ∈
        ``{None, "wins", "losses", "open"}`` (wins→result='win', etc.).
        """
        page = max(1, page)
        conn = self._connect()
        try:
            cols = {
                r[1] for r in conn.execute(
                    "PRAGMA table_info(signals_log)"
                ).fetchall()
            }
            where_parts: list[str] = ["1=1"]
            params: list[Any] = []
            if (
                timeframe
                and timeframe not in ("all", "wins", "losses")
                and "timeframe" in cols
            ):
                where_parts.append("COALESCE(timeframe,'4h') = ?")
                params.append(timeframe)
            if result_filter == "wins":
                where_parts.append("result = 'win'")
            elif result_filter == "losses":
                where_parts.append("result = 'loss'")
            elif result_filter == "open":
                where_parts.append("result = 'open'")
            where = " WHERE " + " AND ".join(where_parts)
            total = conn.execute(
                f"SELECT COUNT(*) FROM signals_log{where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM signals_log{where} "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, per_page, (page - 1) * per_page),
            ).fetchall()
            return [dict(r) for r in rows], int(total)
        except sqlite3.OperationalError:
            return [], 0
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

    def get_trading_stats(
        self,
        symbol: str | None = None,
        days: int = 30,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate trading statistics from this DB's ``signals_log``.

        Single-DB by design (each strategy isolates its own SQLite —
        atomicortex.db / atomicortex_15m.db / atomicortex_1h.db). Use
        :meth:`merge_stats` to combine results across DBs.

        Robust to schema variants: the ``timeframe`` column is only
        filtered when it exists (older SignalBridge DBs lack it); when
        absent every row is treated as ``'4h'``.

        Returns a dict with the keys documented in the task spec, plus
        ``gross_win_pct`` / ``gross_loss_pct`` so :meth:`merge_stats`
        can recompute an aggregate profit factor exactly.
        """
        conn = self._connect()
        try:
            cols = {
                r[1] for r in conn.execute(
                    "PRAGMA table_info(signals_log)"
                ).fetchall()
            }
            has_tf = "timeframe" in cols
            has_conf = "confidence" in cols

            where = [f"created_at >= datetime('now', '-{int(days)} days')"]
            params: list[Any] = []
            if symbol:
                where.append("symbol LIKE ?")
                params.append(f"%{symbol}%")
            if timeframe is not None:
                if has_tf:
                    where.append("COALESCE(timeframe, '4h') = ?")
                    params.append(timeframe)
                elif timeframe != "4h":
                    # No tf column ⇒ all rows are '4h'; a request for any
                    # other tf matches nothing.
                    where.append("1 = 0")
            where_sql = " AND ".join(where)

            conf_expr = "AVG(confidence)" if has_conf else "0.0"
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                          AS total,
                    SUM(CASE WHEN result = 'open'  THEN 1 ELSE 0 END) AS open_cnt,
                    SUM(CASE WHEN result = 'win'   THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN result = 'loss'  THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN result != 'open' THEN pnl_pct ELSE 0 END) AS total_pnl,
                    AVG(CASE WHEN result != 'open' THEN pnl_pct END)  AS avg_pnl,
                    MAX(CASE WHEN result != 'open' THEN pnl_pct END)  AS best,
                    MIN(CASE WHEN result != 'open' THEN pnl_pct END)  AS worst,
                    SUM(CASE WHEN result = 'win'  THEN pnl_pct ELSE 0 END) AS gross_win,
                    SUM(CASE WHEN result = 'loss' THEN pnl_pct ELSE 0 END) AS gross_loss,
                    {conf_expr}                                       AS avg_conf
                FROM signals_log
                WHERE {where_sql}
                """,
                params,
            ).fetchone()
        finally:
            conn.close()

        d = dict(row) if row else {}
        total = int(d.get("total") or 0)
        open_cnt = int(d.get("open_cnt") or 0)
        wins = int(d.get("wins") or 0)
        losses = int(d.get("losses") or 0)
        closed = total - open_cnt
        gross_win = float(d.get("gross_win") or 0.0)
        gross_loss = float(d.get("gross_loss") or 0.0)  # negative-ish
        decided = wins + losses
        return {
            "total_signals": total,
            "open_signals": open_cnt,
            "closed_signals": closed,
            "win_count": wins,
            "loss_count": losses,
            "win_rate": (wins / decided) if decided > 0 else 0.0,
            "total_pnl_pct": float(d.get("total_pnl") or 0.0),
            "avg_pnl_pct": float(d.get("avg_pnl") or 0.0),
            "best_trade_pct": float(d.get("best") or 0.0),
            "worst_trade_pct": float(d.get("worst") or 0.0),
            "profit_factor": (
                gross_win / abs(gross_loss) if gross_loss != 0 else
                (float("inf") if gross_win > 0 else 0.0)
            ),
            "avg_confidence": float(d.get("avg_conf") or 0.0),
            "gross_win_pct": gross_win,
            "gross_loss_pct": gross_loss,
            "period_days": int(days),
        }

    @staticmethod
    def merge_stats(parts: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine per-DB :meth:`get_trading_stats` dicts into a total.

        Counts and PnL sums add; win-rate and profit factor are
        recomputed from the merged gross sums (not averaged) so the
        aggregate is exact.
        """
        agg = {
            "total_signals": 0, "open_signals": 0, "closed_signals": 0,
            "win_count": 0, "loss_count": 0, "total_pnl_pct": 0.0,
            "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
            "gross_win_pct": 0.0, "gross_loss_pct": 0.0,
            "period_days": 0,
        }
        conf_weighted = 0.0
        conf_w = 0
        for p in parts:
            for k in (
                "total_signals", "open_signals", "closed_signals",
                "win_count", "loss_count", "total_pnl_pct",
                "gross_win_pct", "gross_loss_pct",
            ):
                agg[k] += p.get(k, 0)
            agg["best_trade_pct"] = max(
                agg["best_trade_pct"], p.get("best_trade_pct", 0.0)
            )
            agg["worst_trade_pct"] = min(
                agg["worst_trade_pct"], p.get("worst_trade_pct", 0.0)
            )
            agg["period_days"] = max(agg["period_days"], p.get("period_days", 0))
            n = p.get("win_count", 0) + p.get("loss_count", 0)
            conf_weighted += p.get("avg_confidence", 0.0) * max(n, 0)
            conf_w += max(n, 0)

        decided = agg["win_count"] + agg["loss_count"]
        gl = agg["gross_loss_pct"]
        agg["win_rate"] = (agg["win_count"] / decided) if decided > 0 else 0.0
        agg["avg_pnl_pct"] = (
            agg["total_pnl_pct"] / agg["closed_signals"]
            if agg["closed_signals"] > 0 else 0.0
        )
        agg["profit_factor"] = (
            agg["gross_win_pct"] / abs(gl) if gl != 0 else
            (float("inf") if agg["gross_win_pct"] > 0 else 0.0)
        )
        agg["avg_confidence"] = conf_weighted / conf_w if conf_w > 0 else 0.0
        return agg

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

    def get_latest_metrics(self) -> dict[str, Any] | None:
        """Return the latest bot_metrics row written by SignalBridge.

        Returns ``None`` if the table is missing (Database is the
        Telegram-side read API and never creates ``bot_metrics``;
        SignalBridge in the trading process owns its schema) or the
        table is empty (trading bot has not called ``update_metrics``
        yet).  On success returns a dict with keys: equity, daily_pnl,
        regime, open_positions, updated_at.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT equity, daily_pnl, regime, open_positions, updated_at "
                "FROM bot_metrics WHERE id = 1",
            ).fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            # bot_metrics table not created yet — trading bot hasn't started
            return None
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

    def get_payment_by_invoice_id(
        self, invoice_id: str,
    ) -> dict[str, Any] | None:
        """Find a payment by its ``invoice_id`` column.

        Used for idempotent payment processing:
          * Telegram Stars — ``invoice_id`` stores
            ``telegram_payment_charge_id`` (unique per charge).
          * CryptoBot — ``invoice_id`` stores the CryptoBot invoice_id.
        """
        if not invoice_id:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM payments WHERE invoice_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (invoice_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def set_payment_invoice_id(
        self, payment_id: int, invoice_id: str,
    ) -> None:
        """Attach an invoice/charge id to an existing payment row."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE payments SET invoice_id = ? WHERE id = ?",
                (invoice_id, payment_id),
            )
            conn.commit()
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
