"""Phase 5 Step 5.1 — Telegram-bot DB isolation tests.

Pre-fix ``Database.__init__`` always ran ``_init_db()`` which CREATEd
the Telegram schema (users / signals_log / bot_events / payments) and
ran ``ALTER TABLE signals_log`` on whatever DB it was pointed at. The
``/stats`` / ``/signal`` / ``/history`` handlers built a fresh
``Database`` for each call against the *trading* DBs (``atomicortex.db``
etc.) — so every request scribbled into a DB owned by the trading
process, racing the SignalBridge writer's WAL.

The fix adds ``init_schema: bool = True`` (backward compat). Reader-only
attachments to trading DBs now pass ``init_schema=False`` — no DDL,
no schema pollution, no ALTER race.

Tests pin:
* default behaviour creates the schema (legacy callers unchanged);
* ``init_schema=False`` creates no tables on a fresh DB;
* ``init_schema=False`` does not ALTER a pre-existing ``signals_log``;
* read methods continue to work on missing-table DBs (return empty);
* repeat construction stays idempotent;
* the two call-sites that point at trading DBs pass init_schema=False.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.telegram_bot.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _signals_log_columns(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            r[1] for r in conn.execute(
                "PRAGMA table_info(signals_log)"
            ).fetchall()
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Default behaviour preserved (backward compat)
# ---------------------------------------------------------------------------

class TestDefaultInitsSchema:
    def test_default_creates_telegram_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "tg.db"
        Database(path)
        tables = _table_names(path)
        assert {"users", "signals_log", "bot_events", "payments"} <= tables

    def test_default_repeat_construction_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "tg.db"
        Database(path)
        # Second instance must not crash on existing tables.
        Database(path)
        tables = _table_names(path)
        assert "users" in tables

    def test_default_runs_timeframe_alter(self, tmp_path: Path) -> None:
        path = tmp_path / "tg.db"
        Database(path)
        cols = _signals_log_columns(path)
        assert "timeframe" in cols


# ---------------------------------------------------------------------------
# init_schema=False — no DDL, no schema pollution
# ---------------------------------------------------------------------------

class TestInitSchemaFalseDoesNoDDL:
    def test_no_tables_on_fresh_db(self, tmp_path: Path) -> None:
        """init_schema=False on a brand-new DB → file exists (sqlite
        will create it on first connect) but no telegram tables get
        created. Trading DBs owned by another process stay clean."""
        path = tmp_path / "trading.db"
        Database(path, init_schema=False)
        # File was touched (mkdir on parent) but no DDL ran. If sqlite
        # never connects nothing exists; if we connect now there are
        # no tables.
        if path.exists():
            assert _table_names(path) == set()
        else:
            # parent dir at least exists
            assert path.parent.exists()

    def test_does_not_alter_existing_signals_log(self, tmp_path: Path) -> None:
        """Simulate a trading-DB scenario: SignalBridge already created
        a ``signals_log`` table WITHOUT a ``timeframe`` column. The
        Telegram reader must not ALTER it."""
        path = tmp_path / "trading.db"
        # Trading-side schema: pre-existing signals_log without
        # ``timeframe`` (and without bot_metrics etc).
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript("""
                CREATE TABLE signals_log (
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
            """)
            conn.commit()
        finally:
            conn.close()

        cols_before = _signals_log_columns(path)
        Database(path, init_schema=False)
        cols_after = _signals_log_columns(path)
        # Untouched: no timeframe column introduced.
        assert cols_before == cols_after
        assert "timeframe" not in cols_after
        # And no Telegram-only tables sprouted in the trading DB.
        tables = _table_names(path)
        assert "users" not in tables
        assert "payments" not in tables
        assert "bot_events" not in tables


# ---------------------------------------------------------------------------
# Read methods still work in no-DDL mode (existing OperationalError guards)
# ---------------------------------------------------------------------------

class TestReadMethodsOnAttachedDB:
    def test_get_latest_metrics_returns_none_when_missing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "trading.db"
        db = Database(path, init_schema=False)
        # bot_metrics table not created — must NOT crash.
        assert db.get_latest_metrics() is None

    def test_get_recent_signals_returns_empty_when_missing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "trading.db"
        db = Database(path, init_schema=False)
        assert db.get_recent_signals() == []

    def test_get_signals_paginated_returns_empty_when_missing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "trading.db"
        db = Database(path, init_schema=False)
        rows, total = db.get_signals_paginated(page=1, per_page=10)
        assert rows == []
        assert total == 0

    def test_reads_real_data_from_pre_existing_signals_log(
        self, tmp_path: Path
    ) -> None:
        """Trading-DB-style schema with a couple of rows → reader sees
        them through get_recent_signals."""
        path = tmp_path / "trading.db"
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript("""
                CREATE TABLE signals_log (
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
                    result      TEXT DEFAULT 'open'
                );
            """)
            conn.execute(
                "INSERT INTO signals_log (symbol, direction, entry_price, "
                "stop_loss, take_profit, confidence, regime, result) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("BTCUSDT", "long", 50_000, 49_500, 51_000, 0.7,
                 "trend_up", "open"),
            )
            conn.execute(
                "INSERT INTO signals_log (symbol, direction, entry_price, "
                "stop_loss, take_profit, confidence, regime, result) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("ETHUSDT", "short", 3_000, 3_050, 2_950, 0.65,
                 "high_vol", "win"),
            )
            conn.commit()
        finally:
            conn.close()

        db = Database(path, init_schema=False)
        signals = db.get_recent_signals(limit=10)
        assert len(signals) == 2
        symbols = {s["symbol"] for s in signals}
        assert symbols == {"BTCUSDT", "ETHUSDT"}


# ---------------------------------------------------------------------------
# Call-site wiring: handlers + bot pass init_schema=False for trading DBs
# ---------------------------------------------------------------------------

class TestCallSiteWiring:
    def test_resolve_stat_dbs_uses_no_ddl_mode(self, tmp_path: Path) -> None:
        """handlers_free._resolve_stat_dbs builds Database instances for
        every trading DB on each /stats call. They must use
        init_schema=False so the DDL never runs."""
        from unittest.mock import MagicMock

        from src.telegram_bot.handlers_free import _resolve_stat_dbs

        # Use 3 different paths (4h / 1h / 15m), all empty trading DBs.
        path_4h = tmp_path / "atomicortex.db"
        path_1h = tmp_path / "atomicortex_1h.db"
        path_15m = tmp_path / "atomicortex_15m.db"
        for p in (path_4h, path_1h, path_15m):
            p.touch()

        context = MagicMock()
        context.bot_data = {
            "shared_db_paths": [path_4h, path_1h, path_15m],
        }
        out = _resolve_stat_dbs(context)
        assert len(out) == 3
        tfs = {tf for tf, _ in out}
        assert tfs == {"4h", "1h", "15m"}
        # No telegram tables created on any of these trading DBs.
        for path in (path_4h, path_1h, path_15m):
            assert "users" not in _table_names(path)
            assert "payments" not in _table_names(path)
            assert "bot_events" not in _table_names(path)

    def test_shared_db_init_signature_includes_init_schema_false(self) -> None:
        """The bot.py wiring passes init_schema=False to the shared_db
        Database constructor (textual contract — guards against a
        regression that drops the keyword)."""
        src = open(
            "src/telegram_bot/bot.py", encoding="utf-8",
        ).read()
        # The exact wiring line must include init_schema=False.
        assert "shared_db_path, init_schema=False" in src

    def test_handlers_resolve_stat_dbs_uses_init_schema_false(self) -> None:
        src = open(
            "src/telegram_bot/handlers_free.py", encoding="utf-8",
        ).read()
        assert "init_schema=False" in src


# ---------------------------------------------------------------------------
# Idempotence — DDL mode + no-DDL mode can co-exist on different paths
# ---------------------------------------------------------------------------

class TestModesCoexist:
    def test_tg_own_db_and_trading_db_isolated(self, tmp_path: Path) -> None:
        tg_path = tmp_path / "telegram_bot.db"
        trading_path = tmp_path / "atomicortex.db"
        Database(tg_path)                                 # full schema
        Database(trading_path, init_schema=False)         # no DDL
        # TG DB has the telegram schema.
        assert "users" in _table_names(tg_path)
        # Trading DB stays empty (no telegram tables).
        if trading_path.exists():
            assert "users" not in _table_names(trading_path)
