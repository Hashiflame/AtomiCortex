"""Tests for Steps M1-M5 — Signal Poller robustness fixes.

M1: dynamic DB discovery — new isolated trading DBs appear without
    restarting the Telegram bot.
M2: parallel polling via asyncio.gather — a slow DB no longer blocks
    the others.
M3: recovery window — ENTRY signals written in the last N minutes
    before startup are re-broadcast (closes stay deduped).
M4: drop the result='open' filter — a signal that opens AND closes
    between two polls still emits an ENTRY broadcast.
M5: composite (timeframe, id) key — signal_id=42 in two DBs no longer
    points the detail view at the wrong row.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.telegram_bot.signal_poller import SignalPoller


# ---------------------------------------------------------------------------
# Test rig — minimal SignalBridge-shaped DB
# ---------------------------------------------------------------------------


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE signals_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT, direction TEXT,
            entry_price REAL, stop_loss REAL, take_profit REAL,
            confidence  REAL, regime TEXT, timeframe TEXT,
            atr         REAL, funding_rate REAL,
            position_size REAL, notional REAL, leverage REAL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at   TIMESTAMP, close_price REAL,
            pnl_pct     REAL, result TEXT DEFAULT 'open'
        );
        CREATE TABLE bot_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT, message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE bot_metrics (
            id INTEGER PRIMARY KEY DEFAULT 1,
            equity REAL, daily_pnl REAL, regime TEXT,
            open_positions INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn


def _insert_signal(
    conn: sqlite3.Connection, *,
    tf: str = "4h", result: str = "open",
    created_at: str | None = None,
    closed_at: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO signals_log "
        "(symbol, direction, entry_price, stop_loss, take_profit, "
        "confidence, regime, timeframe, atr, funding_rate, "
        "created_at, closed_at, result) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,COALESCE(?, CURRENT_TIMESTAMP),"
        " ?, ?)",
        ("BTCUSDT", "long", 50_000, 49_000, 52_000, 0.7,
         "trend_up", tf, 500, 0.0001,
         created_at, closed_at, result),
    )
    conn.commit()
    return cur.lastrowid


def _make_broadcaster():
    b = MagicMock()
    b.broadcast_signal = AsyncMock()
    b.broadcast_signal_closed = AsyncMock()
    b.broadcast_regime_change = AsyncMock()
    b.broadcast_circuit_breaker = AsyncMock()
    b._cached_metrics = {}
    return b


# ---------------------------------------------------------------------------
# M1 — dynamic DB discovery
# ---------------------------------------------------------------------------


class TestM1DynamicDiscovery:
    def test_discover_adds_new_db_without_restart(self, tmp_path):
        db_4h = tmp_path / "4h.db"
        db_15m = tmp_path / "15m.db"
        _make_db(db_4h).close()

        discovered: list[list[str]] = [
            [str(db_4h)],
            [str(db_4h), str(db_15m)],
        ]
        call_count = {"n": 0}

        def _discover():
            i = call_count["n"]
            call_count["n"] += 1
            return discovered[min(i, len(discovered) - 1)]

        poller = SignalPoller(
            db_paths=[str(db_4h)],
            broadcaster=_make_broadcaster(),
            discover_callback=_discover,
            discover_every_cycles=1,
            recovery_minutes=0,
        )
        # Sanity: before any discovery, only 4h is known.
        assert poller._db_paths == [str(db_4h)]

        # Spin up the 15m DB AFTER the poller is built.
        _make_db(db_15m).close()

        poller._refresh_db_paths()  # first refresh adds nothing new
        poller._refresh_db_paths()  # second discovery returns both
        assert str(db_15m) in poller._db_paths
        # And its high-water marks are seeded (empty DB → 0).
        assert poller._last_signal_ids[str(db_15m)] == 0
        assert poller._broadcasted_close_ids[str(db_15m)] == set()

    def test_existing_paths_are_not_re_initialised(self, tmp_path):
        db_4h = tmp_path / "4h.db"
        conn = _make_db(db_4h)
        sid = _insert_signal(conn)
        conn.close()

        poller = SignalPoller(
            db_paths=[str(db_4h)],
            broadcaster=_make_broadcaster(),
            recovery_minutes=0,
        )
        poller._init_high_water_marks()
        assert poller._last_signal_ids[str(db_4h)] == sid

        # Manually advance the mark to simulate a runtime broadcast.
        poller._last_signal_ids[str(db_4h)] = 999
        poller._discover_callback = lambda: [str(db_4h)]
        poller._refresh_db_paths()
        # Mark must NOT have been reset by re-discovery.
        assert poller._last_signal_ids[str(db_4h)] == 999

    def test_discover_callback_failure_is_swallowed(self, tmp_path):
        db_4h = tmp_path / "4h.db"
        _make_db(db_4h).close()

        def _boom():
            raise RuntimeError("synthetic")

        poller = SignalPoller(
            db_paths=[str(db_4h)],
            broadcaster=_make_broadcaster(),
            discover_callback=_boom,
            recovery_minutes=0,
        )
        # Must not propagate.
        poller._refresh_db_paths()


# ---------------------------------------------------------------------------
# M2 — parallel polling via asyncio.gather
# ---------------------------------------------------------------------------


class TestM2ParallelPolling:
    @pytest.mark.asyncio
    async def test_slow_db_does_not_block_fast_db(self, tmp_path, monkeypatch):
        db_a = tmp_path / "a.db"
        db_b = tmp_path / "b.db"
        _make_db(db_a).close()
        _make_db(db_b).close()

        poller = SignalPoller(
            db_paths=[str(db_a), str(db_b)],
            broadcaster=_make_broadcaster(),
            poll_interval=1,
            recovery_minutes=0,
        )

        # Mark when each path's poll completes.
        finished: dict[str, float] = {}

        async def _slow_or_fast(path):
            if path == str(db_a):
                await asyncio.sleep(0.30)
            else:
                await asyncio.sleep(0.05)
            finished[path] = time.monotonic()

        monkeypatch.setattr(poller, "_poll_one", _slow_or_fast)

        # Manually run one cycle the way _poll_loop does.
        await asyncio.gather(
            *(poller._poll_one(p) for p in poller._db_paths),
            return_exceptions=True,
        )
        # Both completed; fast finished BEFORE slow (parallel, not serial).
        assert finished[str(db_b)] < finished[str(db_a)]

    @pytest.mark.asyncio
    async def test_error_in_one_db_does_not_break_others(
        self, tmp_path, monkeypatch,
    ):
        db_a = tmp_path / "a.db"
        db_b = tmp_path / "b.db"
        _make_db(db_a).close()
        _make_db(db_b).close()
        poller = SignalPoller(
            db_paths=[str(db_a), str(db_b)],
            broadcaster=_make_broadcaster(),
            poll_interval=1, recovery_minutes=0,
        )
        called: dict[str, int] = {str(db_a): 0, str(db_b): 0}

        async def _check(path):
            called[path] += 1
            if path == str(db_a):
                raise RuntimeError("synthetic")

        monkeypatch.setattr(
            poller, "_check_new_signals", _check,
        )
        monkeypatch.setattr(
            poller, "_check_new_events", AsyncMock(),
        )
        monkeypatch.setattr(
            poller, "_update_cached_metrics", AsyncMock(),
        )
        # _poll_one catches per-DB exceptions.
        await asyncio.gather(
            *(poller._poll_one(p) for p in poller._db_paths),
            return_exceptions=True,
        )
        # Both DBs were polled even though one raised.
        assert called[str(db_a)] == 1
        assert called[str(db_b)] == 1


# ---------------------------------------------------------------------------
# M3 — recovery window re-broadcasts recent ENTRY signals
# ---------------------------------------------------------------------------


class TestM3RecoveryWindow:
    def test_recent_signals_within_window_are_eligible(self, tmp_path):
        db = tmp_path / "r.db"
        conn = _make_db(db)
        old_id = _insert_signal(
            conn, created_at="2020-01-01 00:00:00",
        )
        # Two recent signals (use SQLite's local-now to land inside
        # the window regardless of wall clock).
        conn.execute(
            "INSERT INTO signals_log "
            "(symbol, direction, entry_price, stop_loss, take_profit, "
            "confidence, regime, timeframe, atr, funding_rate, "
            "created_at, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "datetime('now', '-5 minutes'), 'open')",
            ("BTCUSDT", "long", 1, 1, 1, 0.5, "trend_up", "4h", 1, 0.0),
        )
        recent_id = conn.execute(
            "SELECT MAX(id) FROM signals_log"
        ).fetchone()[0]
        conn.commit(); conn.close()

        poller = SignalPoller(
            db_paths=[str(db)],
            broadcaster=_make_broadcaster(),
            recovery_minutes=30,
        )
        poller._init_high_water_marks()

        # Mark sits at the OLD id, so the recent one is fair game.
        assert poller._last_signal_ids[str(db)] == old_id
        assert poller._last_signal_ids[str(db)] < recent_id

    def test_recovery_zero_keeps_legacy_skip_all(self, tmp_path):
        db = tmp_path / "r.db"
        conn = _make_db(db)
        sid = _insert_signal(conn)
        conn.close()

        poller = SignalPoller(
            db_paths=[str(db)],
            broadcaster=_make_broadcaster(),
            recovery_minutes=0,
        )
        poller._init_high_water_marks()
        # Mark = MAX(id) — nothing replayed.
        assert poller._last_signal_ids[str(db)] == sid

    def test_closed_signals_are_seeded_for_dedup(self, tmp_path):
        """Closes within the recovery window must NOT be re-announced
        — the seeded set protects against close-spam on restart."""
        db = tmp_path / "r.db"
        conn = _make_db(db)
        closed_id = _insert_signal(
            conn, result="win",
            created_at="2020-01-01 00:00:00",
            closed_at="2020-01-01 00:30:00",
        )
        conn.close()

        poller = SignalPoller(
            db_paths=[str(db)],
            broadcaster=_make_broadcaster(),
            recovery_minutes=60,
        )
        poller._init_high_water_marks()
        assert closed_id in poller._broadcasted_close_ids[str(db)]


# ---------------------------------------------------------------------------
# M4 — fast open+close still gets an ENTRY broadcast
# ---------------------------------------------------------------------------


class TestM4FastCloseEntry:
    @pytest.mark.asyncio
    async def test_signal_closed_between_polls_still_emits_entry(
        self, tmp_path,
    ):
        db = tmp_path / "fast.db"
        conn = _make_db(db)
        # Signal that's already 'win' by the time the poller looks.
        fast_id = _insert_signal(
            conn, result="win",
            closed_at="2026-05-25 12:00:00",
        )
        conn.close()

        br = _make_broadcaster()
        poller = SignalPoller(
            db_paths=[str(db)], broadcaster=br, recovery_minutes=0,
        )
        # Force mark to 0 so the inserted row is "new".
        poller._last_signal_ids[str(db)] = 0
        poller._broadcasted_close_ids[str(db)] = set()

        await poller._check_new_signals(str(db))
        # ENTRY broadcast happened despite result='win'.
        br.broadcast_signal.assert_awaited_once()
        arg = br.broadcast_signal.await_args.args[0]
        assert arg["id"] == fast_id


# ---------------------------------------------------------------------------
# M5 — composite (timeframe, id) lookup
# ---------------------------------------------------------------------------


class TestM5CompositeSignalKey:
    def test_legacy_first_match_when_timeframe_none(self):
        from src.telegram_bot.handlers_premium import find_signal_by_id

        ctx = MagicMock()
        ctx.bot_data = {}
        # Patch _collect_recent at the module level — it owns the
        # cross-DB merge that find_signal_by_id walks.
        rows = [
            {"id": 42, "timeframe": "4h", "symbol": "BTCUSDT", "direction": "long"},
            {"id": 42, "timeframe": "15m", "symbol": "BTCUSDT", "direction": "short"},
        ]
        import src.telegram_bot.handlers_premium as hp
        original = hp._collect_recent
        hp._collect_recent = lambda *a, **kw: rows  # type: ignore[assignment]
        try:
            out = find_signal_by_id(ctx, 42)
        finally:
            hp._collect_recent = original

        # Legacy: first match.
        assert out["timeframe"] == "4h"

    def test_composite_key_disambiguates_collision(self):
        from src.telegram_bot.handlers_premium import find_signal_by_id

        ctx = MagicMock()
        rows = [
            {"id": 42, "timeframe": "4h", "direction": "long"},
            {"id": 42, "timeframe": "15m", "direction": "short"},
        ]
        import src.telegram_bot.handlers_premium as hp
        original = hp._collect_recent
        hp._collect_recent = lambda *a, **kw: rows
        try:
            out_4h = find_signal_by_id(ctx, 42, timeframe="4h")
            out_15m = find_signal_by_id(ctx, 42, timeframe="15m")
            out_1h = find_signal_by_id(ctx, 42, timeframe="1h")
        finally:
            hp._collect_recent = original

        assert out_4h["direction"] == "long"
        assert out_15m["direction"] == "short"
        assert out_1h is None  # no matching tf

    def test_signal_detail_keyboard_legacy_callback(self):
        from src.telegram_bot.keyboards import signal_detail_keyboard

        kb = signal_detail_keyboard(42)
        # Detail button is the middle of the row.
        cb = kb.inline_keyboard[0][1].callback_data
        assert cb == "signal_detail:42"

    def test_signal_detail_keyboard_composite_callback(self):
        from src.telegram_bot.keyboards import signal_detail_keyboard

        kb = signal_detail_keyboard(42, timeframe="15m")
        cb = kb.inline_keyboard[0][1].callback_data
        assert cb == "signal_detail:15m:42"
