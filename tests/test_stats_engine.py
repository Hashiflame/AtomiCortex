"""Tests for the StatsEngine metrics calculator."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from scripts.migrate_db_v3 import migrate
from src.analytics.stats_engine import StatsEngine
from src.execution.signal_bridge import SignalBridge

_NOW = datetime.now(timezone.utc)


def _mk_db(tmp_path, name="atomicortex.db") -> str:
    db = str(tmp_path / name)
    SignalBridge(db)
    migrate(db)
    return db


def _add(db, *, result, pnl, created, closed, tf="4h", rr=1.5,
         dur=240, conf=0.7):
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """INSERT INTO signals_log
                 (symbol, direction, entry_price, stop_loss, take_profit,
                  confidence, regime, timeframe, created_at, closed_at,
                  pnl_pct, result, rr_ratio, duration_minutes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("BTCUSDT-PERP.BINANCE", "long", 100, 95, 110, conf, "trend_up",
             tf, created.isoformat(),
             closed.isoformat() if closed else None,
             pnl, result, rr, dur),
        )
        conn.commit()
    finally:
        conn.close()


def test_win_rate_correct(tmp_path):
    db = _mk_db(tmp_path)
    for i in range(3):
        _add(db, result="win", pnl=1.0, created=_NOW - timedelta(days=5 + i),
             closed=_NOW - timedelta(days=4 + i))
    _add(db, result="loss", pnl=-0.5, created=_NOW - timedelta(days=2),
         closed=_NOW - timedelta(days=1))
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert p["win_count"] == 3 and p["loss_count"] == 1
    assert abs(p["win_rate"] - 0.75) < 1e-9


def test_profit_factor_correct(tmp_path):
    db = _mk_db(tmp_path)
    _add(db, result="win", pnl=3.0, created=_NOW - timedelta(days=5),
         closed=_NOW - timedelta(days=5))
    _add(db, result="win", pnl=1.0, created=_NOW - timedelta(days=4),
         closed=_NOW - timedelta(days=4))
    _add(db, result="loss", pnl=-2.0, created=_NOW - timedelta(days=3),
         closed=_NOW - timedelta(days=3))
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert abs(p["profit_factor"] - (4.0 / 2.0)) < 1e-9


def test_sharpe_ratio_formula(tmp_path):
    db = _mk_db(tmp_path)
    # ≥10 closed signals so the min-sample guard does not null the ratio.
    pnls = [1.0, -0.5, 2.0, 0.7, -1.2, 1.5, -0.3, 2.2, 0.4, -0.8, 1.1]
    for i, x in enumerate(pnls):
        d = _NOW - timedelta(days=20 - i)
        _add(db, result="win" if x > 0 else "loss", pnl=x,
             created=d, closed=d)
    daily = [x / 100.0 for x in pnls]
    mean = sum(daily) / len(daily)
    var = sum((r - mean) ** 2 for r in daily) / (len(daily) - 1)
    expected = mean / math.sqrt(var) * math.sqrt(365)  # H8: crypto annualisation factor
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert p["sharpe_ratio"] is not None
    assert abs(p["sharpe_ratio"] - round(expected, 4)) < 1e-3


def test_sortino_only_downside(tmp_path):
    db = _mk_db(tmp_path)
    pnls = [2.0, -1.0, 3.0, -0.5, 1.2, -0.8, 2.5, -0.4, 1.7, -1.1, 0.9]
    for i, x in enumerate(pnls):
        d = _NOW - timedelta(days=20 - i)
        _add(db, result="win" if x > 0 else "loss", pnl=x, created=d, closed=d)
    daily = [x / 100.0 for x in pnls]
    mean = sum(daily) / len(daily)
    downs = [r for r in daily if r < 0]
    dmean = sum(downs) / len(downs)
    dvar = sum((r - dmean) ** 2 for r in downs) / (len(downs) - 1)
    expected = mean / math.sqrt(dvar) * math.sqrt(365)  # H8: crypto annualisation factor
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert p["sortino_ratio"] is not None
    assert abs(p["sortino_ratio"] - round(expected, 4)) < 1e-3
    assert p["sortino_ratio"] != p["sharpe_ratio"]


def test_sortino_falls_back_to_sharpe_when_no_downside(tmp_path):
    """Bug 1: zero losing days ⇒ Sortino == Sharpe (not 0.0)."""
    db = _mk_db(tmp_path)
    for i in range(11):  # all winners, distinct days, ≥10 sample
        d = _NOW - timedelta(days=20 - i)
        _add(db, result="win", pnl=1.0 + 0.1 * i, created=d, closed=d)
    p = StatsEngine([db]).compute_performance(period_days=60, use_cache=False)
    assert p["sharpe_ratio"] is not None
    assert p["sortino_ratio"] == p["sharpe_ratio"]
    assert p["sortino_ratio"] != 0.0


def test_low_sample_ratios_are_none(tmp_path):
    """Bug 2: < 10 closed signals ⇒ Sharpe/Sortino/Calmar = None."""
    db = _mk_db(tmp_path)
    for i in range(5):
        d = _NOW - timedelta(days=8 - i)
        _add(db, result="win", pnl=1.0, created=d, closed=d)
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert p["closed_signals"] == 5
    assert p["sharpe_ratio"] is None
    assert p["sortino_ratio"] is None
    assert p["calmar_ratio"] is None
    # Non-ratio metrics still computed.
    assert p["win_rate"] == 1.0 and p["total_signals"] == 5


def test_max_drawdown_correct(tmp_path):
    db = _mk_db(tmp_path)
    # +10% then -20% → equity 10000→11000→8800; peak 11000 → DD -20%
    d1 = _NOW - timedelta(days=5)
    d2 = _NOW - timedelta(days=4)
    _add(db, result="win", pnl=10.0, created=d1, closed=d1)
    _add(db, result="loss", pnl=-20.0, created=d2, closed=d2)
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert abs(p["max_drawdown"] - (-20.0)) < 1e-6


def test_equity_curve_compounding(tmp_path):
    db = _mk_db(tmp_path)
    d1 = _NOW - timedelta(days=5)
    d2 = _NOW - timedelta(days=4)
    _add(db, result="win", pnl=10.0, created=d1, closed=d1)
    _add(db, result="win", pnl=5.0, created=d2, closed=d2)
    curve = StatsEngine([db]).compute_equity_curve()
    assert len(curve) == 2
    assert abs(curve[-1]["equity"] - 10000 * 1.10 * 1.05) < 1e-2


def test_empty_signals_returns_zeros(tmp_path):
    db = _mk_db(tmp_path)
    p = StatsEngine([db]).compute_performance(period_days=30, use_cache=False)
    assert p["total_signals"] == 0
    assert p["win_rate"] == 0.0 and p["total_pnl_pct"] == 0.0
    # 0 closed < min sample ⇒ ratios are None, not 0.0.
    assert p["sharpe_ratio"] is None and p["sortino_ratio"] is None


def test_performance_cache_written(tmp_path):
    db = _mk_db(tmp_path)
    d = _NOW - timedelta(days=3)
    _add(db, result="win", pnl=1.5, created=d, closed=d)
    eng = StatsEngine([db])
    eng.compute_performance(timeframe="all", period_days=30, use_cache=False)
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT win_rate FROM performance_cache "
            "WHERE timeframe='all' AND period_days=30 AND symbol='all'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    cached = eng.compute_performance(timeframe="all", period_days=30)
    assert "updated_at" in cached


def test_monthly_stats_correct(tmp_path):
    db = _mk_db(tmp_path)
    apr = datetime(2026, 4, 10, tzinfo=timezone.utc)
    may = datetime(2026, 5, 10, tzinfo=timezone.utc)
    _add(db, result="win", pnl=2.0, created=apr, closed=apr)
    _add(db, result="loss", pnl=-1.0, created=may, closed=may)
    _add(db, result="win", pnl=3.0, created=may, closed=may)
    m = StatsEngine([db]).compute_monthly()
    by = {x["month"]: x for x in m}
    assert by["2026-04"]["wins"] == 1 and by["2026-04"]["losses"] == 0
    assert by["2026-05"]["wins"] == 1 and by["2026-05"]["losses"] == 1
    assert abs(by["2026-05"]["pnl_pct"] - 2.0) < 1e-9


def test_multi_db_merge(tmp_path):
    db1 = _mk_db(tmp_path, "atomicortex.db")
    db2 = _mk_db(tmp_path, "atomicortex_15m.db")
    d = _NOW - timedelta(days=3)
    _add(db1, result="win", pnl=1.0, created=d, closed=d)
    _add(db1, result="win", pnl=2.0, created=d, closed=d)
    _add(db2, result="loss", pnl=-1.0, created=d, closed=d, tf="15m")
    p = StatsEngine([db1, db2]).compute_performance(
        timeframe="all", period_days=30, use_cache=False
    )
    assert p["total_signals"] == 3
    assert p["win_count"] == 2 and p["loss_count"] == 1
