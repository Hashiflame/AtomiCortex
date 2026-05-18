"""Tests for the FastAPI stats API."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from scripts.migrate_db_v3 import migrate
from src.api.main import app
from src.execution.signal_bridge import SignalBridge

_NOW = datetime.now(timezone.utc)


def _seed(db: str, rows: list[dict]) -> None:
    SignalBridge(db)
    migrate(db)
    conn = sqlite3.connect(db)
    try:
        for r in rows:
            conn.execute(
                """INSERT INTO signals_log
                     (symbol, direction, entry_price, stop_loss,
                      take_profit, confidence, regime, timeframe,
                      created_at, closed_at, pnl_pct, result)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("BTCUSDT-PERP.BINANCE", r.get("dir", "long"), 100, 95, 110,
                 0.7, "trend_up", r.get("tf", "4h"),
                 r["created"].isoformat(),
                 r["closed"].isoformat() if r.get("closed") else None,
                 r.get("pnl"), r["result"]),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "atomicortex.db")
    rows = [
        {"result": "win", "pnl": 2.0,
         "created": _NOW - timedelta(days=5),
         "closed": _NOW - timedelta(days=5)},
        {"result": "loss", "pnl": -1.0,
         "created": _NOW - timedelta(days=4),
         "closed": _NOW - timedelta(days=4)},
        {"result": "win", "pnl": 1.5,
         "created": _NOW - timedelta(days=3),
         "closed": _NOW - timedelta(days=3)},
        {"result": "open", "created": _NOW - timedelta(days=1)},
        {"result": "open", "created": _NOW - timedelta(hours=2)},
    ]
    _seed(db, rows)
    monkeypatch.setenv("ATOMICORTEX_DB_PATHS", db)
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "4h" in body["bots"]


def test_stats_endpoint_returns_correct_fields(client):
    r = client.get("/api/v1/stats?period=30&timeframe=all")
    assert r.status_code == 200
    b = r.json()
    for k in (
        "win_rate", "profit_factor", "expected_value", "total_pnl_pct",
        "max_drawdown", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "total_signals", "open_signals", "closed_signals",
        "win_count", "loss_count", "avg_rr_ratio", "avg_duration_h",
        "live_since", "days_tracked", "updated_at",
    ):
        assert k in b, f"missing {k}"
    assert b["total_signals"] == 5
    assert b["open_signals"] == 2
    assert b["win_count"] == 2 and b["loss_count"] == 1


def test_stats_by_timeframe(client):
    r = client.get("/api/v1/stats/4h?period=30")
    assert r.status_code == 200
    assert r.json()["timeframe"] == "4h"


def test_signals_pagination(client):
    p1 = client.get("/api/v1/signals?limit=2&offset=0").json()
    p2 = client.get("/api/v1/signals?limit=2&offset=2").json()
    assert p1["count"] == 2 and p2["count"] == 2
    ids1 = {s["id"] for s in p1["signals"]}
    ids2 = {s["id"] for s in p2["signals"]}
    assert ids1.isdisjoint(ids2)


def test_open_signals_only_open(client):
    b = client.get("/api/v1/signals/open").json()
    assert b["count"] == 2
    assert all(s["result"] == "open" for s in b["signals"])


def test_equity_curve_sorted_by_date(client):
    curve = client.get("/api/v1/equity-curve?period=3650").json()
    assert isinstance(curve, list) and len(curve) >= 2
    dates = [pt["date"] for pt in curve]
    assert dates == sorted(dates)


def test_live_endpoint_has_regime(client):
    b = client.get("/api/v1/live").json()
    assert "market_regime" in b
    assert b["open_positions"] == 2
    assert "4h" in b["bots_status"]


def test_cors_headers_present(client):
    r = client.get("/api/v1/health", headers={"Origin": "https://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"
