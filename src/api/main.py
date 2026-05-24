"""
AtomiCortex REST API for website integration.

FastAPI app serving trading statistics from the isolated signal DBs
(read-only). Stats come from :class:`StatsEngine` (cache-first);
signals are paginated straight from ``signals_log``.

Security (Phase 5.2):
  * X-API-Key header required on all endpoints except /health.
  * Key read from ``ATOMICORTEX_API_KEY``; if absent, an ephemeral key
    is generated at startup and logged (rotate by setting the env var).
  * CORS allowlist via ``API_CORS_ORIGINS`` (comma-separated); wildcard
    ``*`` is never returned.
  * In-memory per-IP sliding-window rate limiter
    (``API_RATE_LIMIT_PER_MINUTE``, default 60).

DB discovery:
  * ``ATOMICORTEX_DB_PATHS`` env (comma-separated) — explicit override
    (used by tests);
  * else every ``data/atomicortex*.db`` that exists, 4H base first.

Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from src.analytics.stats_engine import StatsEngine

_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG = logging.getLogger(__name__)


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


# ----------------------------------------------------------------------
# Security: API key, CORS allowlist, rate limiter
# ----------------------------------------------------------------------
_GENERATED_API_KEY: str | None = None


def _get_api_key() -> str:
    """Return the active API key. Generate + log one if env is empty."""
    key = os.getenv("ATOMICORTEX_API_KEY", "").strip()
    if key:
        return key
    global _GENERATED_API_KEY
    if _GENERATED_API_KEY is None:
        _GENERATED_API_KEY = secrets.token_urlsafe(32)
        _LOG.warning(
            "ATOMICORTEX_API_KEY not set — generated ephemeral key for this "
            "process: %s  (set ATOMICORTEX_API_KEY in .env to persist)",
            _GENERATED_API_KEY,
        )
    return _GENERATED_API_KEY


def _allowed_origins() -> set[str]:
    raw = os.getenv("API_CORS_ORIGINS", "http://localhost,http://127.0.0.1")
    return {o.strip() for o in raw.split(",") if o.strip() and o.strip() != "*"}


def _rate_limit_per_minute() -> int:
    try:
        return max(1, int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60")))
    except ValueError:
        return 60


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided: str | None = Depends(_api_key_header),
) -> None:
    if provided is None or provided == "":
        raise HTTPException(status_code=401, detail="Missing API key")
    if not secrets.compare_digest(provided, _get_api_key()):
        raise HTTPException(status_code=403, detail="Invalid API key")


_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _reset_rate_buckets() -> None:
    """Test helper — clear in-memory rate-limit state."""
    _rate_buckets.clear()


app = FastAPI(title="AtomiCortex API", version="1.0.0")


@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_buckets[ip]
    cutoff = now - 60.0
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= _rate_limit_per_minute():
        return JSONResponse(
            {"detail": "Rate limit exceeded"}, status_code=429,
        )
    bucket.append(now)
    return await call_next(request)


@app.middleware("http")
async def _cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    allowed = _allowed_origins()

    if request.method == "OPTIONS" and origin:
        response = JSONResponse({}, status_code=200)
    else:
        response = await call_next(request)

    if origin and origin in allowed:
        response.headers["access-control-allow-origin"] = origin
        response.headers["access-control-allow-credentials"] = "true"
        response.headers["access-control-allow-methods"] = (
            "GET, OPTIONS"
        )
        response.headers["access-control-allow-headers"] = (
            "X-API-Key, Content-Type"
        )
        response.headers["vary"] = "Origin"
    return response


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/api/v1/health")
def health() -> dict:
    bots: dict[str, str] = {}
    for db in get_db_paths():
        bots[_tf_of(db)] = "running" if Path(db).exists() else "missing"
    return {"status": "ok", "bots": bots}


@app.get("/api/v1/stats", dependencies=[Depends(require_api_key)])
def stats(period: int = Query(30), timeframe: str = Query("all")) -> dict:
    return _engine().compute_performance(
        timeframe=timeframe, period_days=period, symbol="all",
    )


@app.get(
    "/api/v1/stats/{timeframe}", dependencies=[Depends(require_api_key)],
)
def stats_by_tf(timeframe: str, period: int = Query(30)) -> dict:
    return _engine().compute_performance(
        timeframe=timeframe, period_days=period, symbol="all",
    )


@app.get("/api/v1/signals", dependencies=[Depends(require_api_key)])
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


@app.get(
    "/api/v1/signals/open", dependencies=[Depends(require_api_key)],
)
def signals_open(timeframe: str = Query("all")) -> dict:
    items = _query_signals("open", timeframe, 500, 0)
    return {"count": len(items), "signals": items}


@app.get(
    "/api/v1/equity-curve", dependencies=[Depends(require_api_key)],
)
def equity_curve(
    timeframe: str = Query("all"), period: int = Query(90),
) -> list[dict]:
    return _engine().compute_equity_curve(
        timeframe=timeframe, symbol="all", period_days=period,
    )


@app.get(
    "/api/v1/monthly-stats", dependencies=[Depends(require_api_key)],
)
def monthly_stats(timeframe: str = Query("all")) -> list[dict]:
    return _engine().compute_monthly(timeframe=timeframe, symbol="all")


@app.get("/api/v1/live", dependencies=[Depends(require_api_key)])
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
