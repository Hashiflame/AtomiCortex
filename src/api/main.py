"""
AtomiCortex REST API for website integration.

FastAPI app serving trading statistics from the isolated signal DBs
(read-only). Stats come from :class:`StatsEngine` (cache-first);
signals are paginated straight from ``signals_log``.

DB discovery:
  * ``ATOMICORTEX_DB_PATHS`` env (comma-separated) — explicit override
    (used by tests);
  * else every ``data/atomicortex*.db`` that exists, 4H base first.

Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.analytics.stats_engine import StatsEngine

_ROOT = Path(__file__).resolve().parent.parent.parent


def get_db_paths() -> list[str]:
    """Resolve the trading DBs to read (env override → discovery)."""
    env = os.getenv("ATOMICORTEX_DB_PATHS")
    if env:
        return [p for p in env.split(",") if p.strip()]
    base = _ROOT / "data"
    out: list[str] = []
    for name in ("atomicortex.db", "atomicortex_15m.db", "atomicortex_1h.db"):
        p = base / name
        if p.exists():
            out.append(str(p))
    return out or [str(base / "atomicortex.db")]


def _tf_of(db_path: str) -> str:
    n = Path(db_path).name
    return "15m" if "_15m" in n else "1h" if "_1h" in n else "4h"


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _query_signals(
    status: str, timeframe: str, limit: int, offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for db in get_db_paths():
        if not Path(db).exists():
            continue
        if timeframe != "all" and _tf_of(db) != timeframe:
            continue
        conn = _connect(db)
        try:
            where = []
            if status == "open":
                where.append("result = 'open'")
            elif status == "closed":
                where.append("result IN ('win','loss','breakeven')")
            sql = "SELECT * FROM signals_log"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY created_at DESC"
            for r in conn.execute(sql).fetchall():
                d = dict(r)
                d.setdefault("timeframe", _tf_of(db))
                d["_db_tf"] = _tf_of(db)
                rows.append(d)
        except sqlite3.OperationalError:
            continue
        finally:
            conn.close()
    rows.sort(key=lambda s: str(s.get("created_at") or ""), reverse=True)
    return rows[offset: offset + limit]


def _engine() -> StatsEngine:
    return StatsEngine(get_db_paths())


app = FastAPI(title="AtomiCortex API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict:
    bots: dict[str, str] = {}
    for db in get_db_paths():
        bots[_tf_of(db)] = "running" if Path(db).exists() else "missing"
    return {"status": "ok", "bots": bots}


@app.get("/api/v1/stats")
def stats(period: int = Query(30), timeframe: str = Query("all")) -> dict:
    return _engine().compute_performance(
        timeframe=timeframe, period_days=period, symbol="all",
    )


@app.get("/api/v1/stats/{timeframe}")
def stats_by_tf(timeframe: str, period: int = Query(30)) -> dict:
    return _engine().compute_performance(
        timeframe=timeframe, period_days=period, symbol="all",
    )


@app.get("/api/v1/signals")
def signals(
    status: str = Query("all"),
    timeframe: str = Query("all"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    items = _query_signals(status, timeframe, limit, offset)
    return {
        "count": len(items), "limit": limit, "offset": offset,
        "status": status, "timeframe": timeframe, "signals": items,
    }


@app.get("/api/v1/signals/open")
def signals_open(timeframe: str = Query("all")) -> dict:
    items = _query_signals("open", timeframe, 500, 0)
    return {"count": len(items), "signals": items}


@app.get("/api/v1/equity-curve")
def equity_curve(
    timeframe: str = Query("all"), period: int = Query(90),
) -> list[dict]:
    return _engine().compute_equity_curve(
        timeframe=timeframe, symbol="all", period_days=period,
    )


@app.get("/api/v1/monthly-stats")
def monthly_stats(timeframe: str = Query("all")) -> list[dict]:
    return _engine().compute_monthly(timeframe=timeframe, symbol="all")


@app.get("/api/v1/live")
def live() -> dict:
    """Real-time snapshot: regime / equity / open positions / last signal."""
    regime = "UNKNOWN"
    equity = 10_000.0
    daily_pnl = 0.0
    last_signal: dict | None = None
    open_positions = 0
    bots: dict[str, str] = {}

    for db in get_db_paths():
        tf = _tf_of(db)
        bots[tf] = "running" if Path(db).exists() else "missing"
        if not Path(db).exists():
            continue
        conn = _connect(db)
        try:
            try:
                m = conn.execute(
                    "SELECT equity, daily_pnl, regime FROM bot_metrics "
                    "WHERE id = 1"
                ).fetchone()
                if m and db.endswith("atomicortex.db"):
                    if m["regime"]:
                        regime = str(m["regime"])
                    if m["equity"] is not None:
                        equity = float(m["equity"])
                    if m["daily_pnl"] is not None:
                        daily_pnl = float(m["daily_pnl"])
            except sqlite3.OperationalError:
                pass

            open_positions += conn.execute(
                "SELECT COUNT(*) FROM signals_log WHERE result='open'"
            ).fetchone()[0]

            row = conn.execute(
                "SELECT * FROM signals_log ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row:
                cand = dict(row)
                cand.setdefault("timeframe", tf)
                if (
                    last_signal is None
                    or str(cand.get("created_at"))
                    > str(last_signal.get("created_at"))
                ):
                    last_signal = cand
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    return {
        "market_regime": regime,
        "last_signal": last_signal,
        "open_positions": open_positions,
        "equity": equity,
        "daily_pnl_pct": daily_pnl,
        "bots_status": bots,
    }
