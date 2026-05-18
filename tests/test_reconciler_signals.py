"""Tests for the signal reconciler (ledger-correction service)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from scripts.migrate_db_v3 import migrate
from src.execution.reconciler_signals import SignalReconciler
from src.execution.signal_bridge import SignalBridge

_NOW = datetime.now(timezone.utc)


class FakeSource:
    """Deterministic price source: fixed (ts_ms, high, low) bars."""

    def __init__(self, bars):
        self._bars = bars

    def bars(self, symbol, interval, start, end):
        return list(self._bars)


def _mk_db(tmp_path) -> str:
    db = str(tmp_path / "atomicortex.db")
    SignalBridge(db)          # create canonical signals_log schema
    migrate(db)               # add v3 columns + daily_stats/perf_cache
    return db


def _insert_open(
    db: str, *, direction: str, entry: float, sl: float, tp: float,
    created: datetime, tf: str = "4h",
) -> int:
    conn = sqlite3.connect(db)
    try:
        cur = conn.execute(
            """INSERT INTO signals_log
                 (symbol, direction, entry_price, stop_loss, take_profit,
                  confidence, regime, timeframe, created_at, result)
               VALUES (?,?,?,?,?,?,?,?,?, 'open')""",
            ("BTCUSDT-PERP.BINANCE", direction, entry, sl, tp, 0.7,
             "trend_up", tf, created.isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _row(db: str, sid: int) -> dict:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return dict(conn.execute(
            "SELECT * FROM signals_log WHERE id=?", (sid,)
        ).fetchone())
    finally:
        conn.close()


def _bar(created: datetime, hours: float, high: float, low: float):
    ts = int((created + timedelta(hours=hours)).timestamp() * 1000)
    return (ts, high, low)


def _rec(db, bars, **kw):
    return SignalReconciler(
        db_path=db, price_source=FakeSource(bars),
        skip_recent_bars=kw.pop("skip_recent_bars", 0), **kw,
    )


# ── 1-4: directional SL/TP ────────────────────────────────────────────

def test_long_tp_hit_closes_as_win(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    _rec(db, [_bar(c, 8, 111, 99)]).reconcile()
    r = _row(db, sid)
    assert r["result"] == "win" and r["close_price"] == 110


def test_long_sl_hit_closes_as_loss(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    _rec(db, [_bar(c, 8, 101, 94)]).reconcile()
    r = _row(db, sid)
    assert r["result"] == "loss" and r["close_price"] == 95


def test_short_tp_hit_closes_as_win(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="short", entry=100, sl=105, tp=90, created=c)
    _rec(db, [_bar(c, 8, 101, 89)]).reconcile()
    r = _row(db, sid)
    assert r["result"] == "win" and r["close_price"] == 90


def test_short_sl_hit_closes_as_loss(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="short", entry=100, sl=105, tp=90, created=c)
    _rec(db, [_bar(c, 8, 106, 99)]).reconcile()
    r = _row(db, sid)
    assert r["result"] == "loss" and r["close_price"] == 105


# ── 5: tie rule ───────────────────────────────────────────────────────

def test_sl_wins_when_both_hit_same_bar(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    _rec(db, [_bar(c, 8, 111, 94)]).reconcile()  # both touched
    assert _row(db, sid)["result"] == "loss"


# ── 6: skip recent ────────────────────────────────────────────────────

def test_recent_signal_skipped(tmp_path):
    db = _mk_db(tmp_path)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110,
                       created=_NOW, tf="4h")
    res = SignalReconciler(
        db_path=db, price_source=FakeSource([_bar(_NOW, 1, 200, 1)]),
        skip_recent_bars=2, bar_hours=4.0,
    ).reconcile()
    assert res["skipped_recent"] == 1
    assert _row(db, sid)["result"] == "open"


# ── 7: no touch ───────────────────────────────────────────────────────

def test_open_signal_unchanged_if_not_hit(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    res = _rec(db, [_bar(c, 8, 108, 97)]).reconcile()  # inside band
    assert res["still_open"] == 1
    assert _row(db, sid)["result"] == "open"


# ── 8: dry-run ────────────────────────────────────────────────────────

def test_dry_run_does_not_write(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    res = _rec(db, [_bar(c, 8, 111, 99)], dry_run=True).reconcile()
    assert res["closed_win"] == 1
    assert _row(db, sid)["result"] == "open"  # unchanged


# ── 9: idempotent ─────────────────────────────────────────────────────

def test_idempotent_second_run(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    bars = [_bar(c, 8, 111, 99)]
    _rec(db, bars).reconcile()
    res2 = _rec(db, bars).reconcile()
    assert res2["checked"] == 0 and res2["closed_win"] == 0


# ── 10: pnl ───────────────────────────────────────────────────────────

def test_pnl_calculated_correctly(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="short", entry=100, sl=105, tp=90, created=c)
    _rec(db, [_bar(c, 8, 101, 89)]).reconcile()  # short TP win
    # short win at tp=90: pnl = -((90-100)/100*100) = +10
    assert abs(_row(db, sid)["pnl_pct"] - 10.0) < 1e-6


# ── 11: duration ──────────────────────────────────────────────────────

def test_duration_minutes_set(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    sid = _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    _rec(db, [_bar(c, 8, 111, 99)]).reconcile()
    # 8h window; created_at carries μs while bar ts is integer-ms, so
    # the int-minute duration is 479 or 480 depending on truncation.
    assert _row(db, sid)["duration_minutes"] in (479, 480)


# ── 12: daily_stats ───────────────────────────────────────────────────

def test_daily_stats_updated_after_reconcile(tmp_path):
    db = _mk_db(tmp_path)
    c = _NOW - timedelta(days=10)
    _insert_open(db, direction="long", entry=100, sl=95, tp=110, created=c)
    _rec(db, [_bar(c, 8, 111, 99)]).reconcile()
    conn = sqlite3.connect(db)
    try:
        n = conn.execute("SELECT COUNT(*) FROM daily_stats").fetchone()[0]
    finally:
        conn.close()
    assert n >= 1
