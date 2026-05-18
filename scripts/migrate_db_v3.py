#!/usr/bin/env python3
"""
Database migration v3: stats-engine schema.

Idempotent — safe to run any number of times. Adds the columns the
reconciler / stats engine need to ``signals_log`` and creates the
``daily_stats`` and ``performance_cache`` tables. Never drops or
rewrites data; ``ADD COLUMN`` duplicates are caught and skipped.

Run:
    python scripts/migrate_db_v3.py
    python scripts/migrate_db_v3.py --db data/atomicortex_15m.db
    python scripts/migrate_db_v3.py --all      # every atomicortex*.db
"""

from __future__ import annotations

import argparse
import glob
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# (column, type/default clause)
_SIGNALS_COLUMNS: list[tuple[str, str]] = [
    ("timeframe", "TEXT DEFAULT '4h'"),
    ("market_type", "TEXT DEFAULT 'futures_perp'"),
    ("rr_ratio", "REAL"),
    ("duration_minutes", "INTEGER"),
    ("mae_pct", "REAL"),
    ("mfe_pct", "REAL"),
]

_DAILY_STATS_DDL = """
CREATE TABLE IF NOT EXISTS daily_stats (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    date           DATE NOT NULL,
    timeframe      TEXT NOT NULL DEFAULT 'all',
    symbol         TEXT NOT NULL DEFAULT 'BTCUSDT',
    equity         REAL DEFAULT 10000.0,
    daily_pnl_pct  REAL DEFAULT 0.0,
    signals_count  INTEGER DEFAULT 0,
    wins           INTEGER DEFAULT 0,
    losses         INTEGER DEFAULT 0,
    drawdown_pct   REAL DEFAULT 0.0,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, timeframe, symbol)
);
"""

_PERF_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS performance_cache (
    id             INTEGER PRIMARY KEY,
    timeframe      TEXT NOT NULL DEFAULT 'all',
    period_days    INTEGER NOT NULL DEFAULT 30,
    symbol         TEXT NOT NULL DEFAULT 'all',
    win_rate       REAL,
    profit_factor  REAL,
    expected_value REAL,
    total_pnl_pct  REAL,
    max_drawdown   REAL,
    sharpe_ratio   REAL,
    sortino_ratio  REAL,
    calmar_ratio   REAL,
    avg_rr_ratio   REAL,
    total_signals  INTEGER,
    closed_signals INTEGER,
    win_count      INTEGER,
    loss_count     INTEGER,
    avg_duration_h REAL,
    live_since     DATE,
    days_tracked   INTEGER,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(timeframe, period_days, symbol)
);
"""


def migrate(db_path: str) -> dict:
    """Apply v3 schema to one DB. Returns a summary dict."""
    p = Path(db_path)
    if not p.exists():
        return {"db": db_path, "status": "missing", "added": [], "tables": []}

    added: list[str] = []
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")

        # signals_log must exist (created by SignalBridge). If not, skip
        # column adds gracefully — tables below are still created.
        has_signals = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='signals_log'"
        ).fetchone()

        if has_signals:
            existing = {
                r[1]
                for r in conn.execute("PRAGMA table_info(signals_log)").fetchall()
            }
            for col, decl in _SIGNALS_COLUMNS:
                if col in existing:
                    continue
                try:
                    conn.execute(
                        f"ALTER TABLE signals_log ADD COLUMN {col} {decl}"
                    )
                    added.append(col)
                except sqlite3.OperationalError:
                    pass  # raced / duplicate — idempotent

        conn.executescript(_DAILY_STATS_DDL)
        conn.executescript(_PERF_CACHE_DDL)
        conn.commit()
    finally:
        conn.close()

    return {
        "db": db_path,
        "status": "ok",
        "added": added,
        "tables": ["daily_stats", "performance_cache"],
        "signals_log": bool(has_signals),
    }


def _discover_dbs() -> list[str]:
    base = _ROOT / "data"
    return sorted(str(p) for p in base.glob("atomicortex*.db"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stats-engine schema migration (v3)")
    ap.add_argument("--db", default="data/atomicortex.db", help="DB path")
    ap.add_argument("--all", action="store_true", help="Migrate every atomicortex*.db")
    args = ap.parse_args()

    targets = _discover_dbs() if args.all else [args.db]
    if not targets:
        print("No atomicortex*.db found under data/")
        return

    for db in targets:
        res = migrate(db)
        if res["status"] == "missing":
            print(f"⚠️  {db}: not found — skipped")
            continue
        cols = ", ".join(res["added"]) if res["added"] else "(none new)"
        print(
            f"✅ {db}: signals_log cols added: {cols} | "
            f"tables ensured: {', '.join(res['tables'])}"
        )


if __name__ == "__main__":
    main()
