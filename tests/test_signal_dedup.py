"""
Phase 5.4 — Signal close-notification deduplication.

The poller's _check_closed_signals previously re-emitted the same close
on every cycle inside the lookback window (~4× per close at 30s poll /
2min window). These tests pin the new behaviour: each close is emitted
exactly once, across restarts, across symbols, across DBs, and after
broadcast failures.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.migrate_db_v3 import migrate
from src.execution.signal_bridge import SignalBridge
from src.telegram_bot.signal_poller import SignalPoller


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_db(path) -> str:
    """Create an empty atomicortex schema DB and return its path."""
    p = str(path)
    SignalBridge(p)     # creates signals_log schema
    migrate(p)          # applies any pending migrations
    return p


def _insert_signal(
    db_path: str,
    *,
    signal_id: int | None = None,
    symbol: str = "BTCUSDT-PERP.BINANCE",
    result: str = "open",
    created_at: datetime | None = None,
    closed_at: datetime | None = None,
    timeframe: str = "4h",
) -> int:
    """Insert a signals_log row and return its id."""
    created_at = created_at or datetime.now(timezone.utc)
    conn = sqlite3.connect(db_path)
    try:
        cols = [
            "symbol", "direction", "entry_price", "stop_loss",
            "take_profit", "confidence", "regime", "timeframe",
            "created_at", "closed_at", "pnl_pct", "result",
        ]
        vals = [
            symbol, "long", 100.0, 95.0, 110.0,
            0.7, "trend_up", timeframe,
            created_at.isoformat(),
            closed_at.isoformat() if closed_at else None,
            1.2 if result != "open" else None,
            result,
        ]
        if signal_id is not None:
            cols = ["id", *cols]
            vals = [signal_id, *vals]
        placeholders = ", ".join("?" * len(cols))
        cur = conn.execute(
            f"INSERT INTO signals_log ({', '.join(cols)}) "
            f"VALUES ({placeholders})",
            vals,
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def _close_signal(
    db_path: str, signal_id: int, *, when: datetime | None = None,
    result: str = "win",
) -> None:
    when = when or datetime.now(timezone.utc)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE signals_log SET result = ?, closed_at = ?, pnl_pct = ? "
            "WHERE id = ?",
            (result, when.isoformat(), 1.5, signal_id),
        )
        conn.commit()
    finally:
        conn.close()


def _broadcaster_mock() -> MagicMock:
    b = MagicMock()
    b.broadcast_signal = AsyncMock()
    b.broadcast_signal_closed = AsyncMock()
    b.broadcast_regime_change = AsyncMock()
    b.broadcast_circuit_breaker = AsyncMock()
    return b


# ---------------------------------------------------------------------------
# 1. Single close → single notification (no spam)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_close_emits_once_across_poll_cycles(tmp_path):
    db = _make_db(tmp_path / "atomicortex.db")
    b = _broadcaster_mock()
    poller = SignalPoller(db_path=db, broadcaster=b)
    poller._init_high_water_marks()

    # A signal opens after startup, then closes — must broadcast exactly once.
    sid = _insert_signal(db, result="open")
    _close_signal(db, sid, when=datetime.now(timezone.utc))

    for _ in range(5):
        await poller._check_closed_signals(db)

    assert b.broadcast_signal_closed.await_count == 1
    arg = b.broadcast_signal_closed.await_args[0][0]
    assert arg["id"] == sid


# ---------------------------------------------------------------------------
# 2. Restart skips already-closed signals (seeded from DB on init)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_restart_does_not_replay_old_closes(tmp_path):
    db = _make_db(tmp_path / "atomicortex.db")

    # Close already in DB BEFORE the poller starts (simulating crash/restart
    # while close is still inside the 10-minute window).
    sid = _insert_signal(db, result="open")
    _close_signal(db, sid, when=datetime.now(timezone.utc))

    b = _broadcaster_mock()
    poller = SignalPoller(db_path=db, broadcaster=b)
    poller._init_high_water_marks()

    for _ in range(3):
        await poller._check_closed_signals(db)

    assert b.broadcast_signal_closed.await_count == 0
    assert sid in poller._broadcasted_close_ids[db]


# ---------------------------------------------------------------------------
# 3. Restart, then NEW close → must broadcast
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_new_close_after_restart_is_broadcast(tmp_path):
    db = _make_db(tmp_path / "atomicortex.db")
    # Old close (already known) and an open signal that will close later.
    old = _insert_signal(db, result="open")
    _close_signal(db, old, when=datetime.now(timezone.utc))
    new_open = _insert_signal(db, result="open")

    b = _broadcaster_mock()
    poller = SignalPoller(db_path=db, broadcaster=b)
    poller._init_high_water_marks()

    # First cycle: nothing new.
    await poller._check_closed_signals(db)
    assert b.broadcast_signal_closed.await_count == 0

    # Now the new signal closes.
    _close_signal(db, new_open, when=datetime.now(timezone.utc))
    await poller._check_closed_signals(db)
    await poller._check_closed_signals(db)   # would have spammed before

    assert b.broadcast_signal_closed.await_count == 1
    arg = b.broadcast_signal_closed.await_args[0][0]
    assert arg["id"] == new_open


# ---------------------------------------------------------------------------
# 4. Multi-symbol — BTC + ETH close in the same window → 2 notifications
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_symbol_close_each_emits_once(tmp_path):
    db = _make_db(tmp_path / "atomicortex.db")
    b = _broadcaster_mock()
    poller = SignalPoller(db_path=db, broadcaster=b)
    poller._init_high_water_marks()

    btc = _insert_signal(db, symbol="BTCUSDT-PERP.BINANCE", result="open")
    eth = _insert_signal(db, symbol="ETHUSDT-PERP.BINANCE", result="open")

    now = datetime.now(timezone.utc)
    _close_signal(db, btc, when=now)
    _close_signal(db, eth, when=now + timedelta(seconds=1))

    for _ in range(4):
        await poller._check_closed_signals(db)

    assert b.broadcast_signal_closed.await_count == 2
    seen_ids = {
        call.args[0]["id"]
        for call in b.broadcast_signal_closed.await_args_list
    }
    assert seen_ids == {btc, eth}


# ---------------------------------------------------------------------------
# 5. Multi-DB (e.g. 4h + 15m strategies) — ids collide across DBs, must dedup per-DB
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_db_dedup_is_per_db(tmp_path):
    db_4h = _make_db(tmp_path / "atomicortex.db")
    db_15m = _make_db(tmp_path / "atomicortex_15m.db")

    b = _broadcaster_mock()
    poller = SignalPoller(
        db_paths=[db_4h, db_15m], broadcaster=b,
    )
    poller._init_high_water_marks()

    # Each DB gets an open signal that will close. Ids may collide
    # (each DB has its own autoincrement). Dedup must not bleed across.
    s4 = _insert_signal(db_4h, result="open", timeframe="4h")
    s15 = _insert_signal(db_15m, result="open", timeframe="15m")
    now = datetime.now(timezone.utc)
    _close_signal(db_4h, s4, when=now)
    _close_signal(db_15m, s15, when=now)

    for _ in range(3):
        await poller._check_closed_signals(db_4h)
        await poller._check_closed_signals(db_15m)

    assert b.broadcast_signal_closed.await_count == 2
    timeframes = {
        call.args[0]["timeframe"]
        for call in b.broadcast_signal_closed.await_args_list
    }
    assert timeframes == {"4h", "15m"}


# ---------------------------------------------------------------------------
# 6. Broadcast failure → retried next cycle, dedupes after success
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_broadcast_failure_is_retried_then_dedup(tmp_path):
    db = _make_db(tmp_path / "atomicortex.db")
    b = _broadcaster_mock()

    calls = {"n": 0}

    async def flaky_broadcast(signal):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Telegram timeout")

    b.broadcast_signal_closed = AsyncMock(side_effect=flaky_broadcast)

    poller = SignalPoller(db_path=db, broadcaster=b)
    poller._init_high_water_marks()

    sid = _insert_signal(db, result="open")
    _close_signal(db, sid, when=datetime.now(timezone.utc))

    # Cycle 1: broadcast raises → id not marked as seen → retried.
    await poller._check_closed_signals(db)
    assert sid not in poller._broadcasted_close_ids[db]

    # Cycle 2: succeeds, id is now marked seen.
    await poller._check_closed_signals(db)
    assert sid in poller._broadcasted_close_ids[db]

    # Cycle 3+: no more calls.
    await poller._check_closed_signals(db)
    await poller._check_closed_signals(db)

    assert b.broadcast_signal_closed.await_count == 2   # one fail + one ok
