"""Tests for Step H10 — StatsEngine writes performance_cache into an
isolated stats_cache.db, not into the trading DB it reads signals from.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.analytics.stats_engine import StatsEngine
from src.execution.signal_bridge import SignalBridge


def _seed_trading_db(path: Path, n_days: int = 12) -> None:
    """Populate a trading DB with enough closed signals to exit the
    _MIN_RATIO_SAMPLE gate so StatsEngine actually computes + caches."""
    SignalBridge(str(path))  # creates signals_log schema
    now = datetime.now(timezone.utc).replace(
        hour=12, minute=0, second=0, microsecond=0,
    )
    conn = sqlite3.connect(path)
    for i in range(n_days):
        day = now - timedelta(days=n_days - i)
        pnl = 0.5 if i % 2 == 0 else -0.3
        result = "win" if pnl > 0 else "loss"
        conn.execute(
            "INSERT INTO signals_log "
            "(symbol, direction, entry_price, stop_loss, take_profit, "
            "confidence, regime, timeframe, atr, funding_rate, "
            "created_at, closed_at, pnl_pct, result) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("BTCUSDT", "long", 50_000, 49_000, 52_000, 0.7,
             "trend_up", "4h", 500, 0.0001,
             day.isoformat(), day.isoformat(), pnl, result),
        )
    conn.commit()
    conn.close()


def _has_table(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cache DB path resolution
# ---------------------------------------------------------------------------


class TestCacheDbResolution:
    def test_default_is_sibling_to_first_trading_db(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        SignalBridge(str(trading))
        eng = StatsEngine([str(trading)])
        assert eng._cache_db == str(tmp_path / "stats_cache.db")

    def test_explicit_override(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        SignalBridge(str(trading))
        custom = tmp_path / "custom" / "my_cache.db"
        eng = StatsEngine([str(trading)], cache_db_path=str(custom))
        assert eng._cache_db == str(custom)
        # Parent directory was created.
        assert custom.parent.exists()

    def test_no_trading_dbs_means_no_cache(self):
        eng = StatsEngine([])
        assert eng._cache_db is None


# ---------------------------------------------------------------------------
# Cache DB is auto-created with the right schema
# ---------------------------------------------------------------------------


class TestCacheDbAutoCreation:
    def test_cache_db_created_on_init(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        SignalBridge(str(trading))
        cache = tmp_path / "stats_cache.db"
        assert not cache.exists()
        StatsEngine([str(trading)])
        assert cache.exists()

    def test_performance_cache_table_present_in_cache_db(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        SignalBridge(str(trading))
        StatsEngine([str(trading)])
        assert _has_table(str(tmp_path / "stats_cache.db"),
                          "performance_cache")

    def test_init_is_idempotent(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        SignalBridge(str(trading))
        # Re-init twice — must not error and must keep schema.
        StatsEngine([str(trading)])
        StatsEngine([str(trading)])
        assert _has_table(str(tmp_path / "stats_cache.db"),
                          "performance_cache")


# ---------------------------------------------------------------------------
# Isolation: trading DB never gains a performance_cache table
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_trading_db_does_not_get_cache_table(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        _seed_trading_db(trading)
        eng = StatsEngine([str(trading)])
        eng.compute_performance(use_cache=False)  # forces _write_cache too
        assert not _has_table(str(trading), "performance_cache")

    def test_write_path_targets_cache_db_only(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        _seed_trading_db(trading)
        eng = StatsEngine([str(trading)])
        eng.compute_performance(use_cache=False)
        # Cache DB has rows.
        conn = sqlite3.connect(str(tmp_path / "stats_cache.db"))
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM performance_cache",
            ).fetchone()[0]
        finally:
            conn.close()
        assert n >= 1

    def test_signals_still_read_from_trading_db(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        _seed_trading_db(trading, n_days=14)
        eng = StatsEngine([str(trading)])
        out = eng.compute_performance(
            timeframe="all", period_days=30, symbol="all", use_cache=False,
        )
        # Real computation happened — both decisions (win + loss) seen.
        assert out["closed_signals"] >= 10
        assert out["win_count"] > 0
        assert out["loss_count"] > 0


# ---------------------------------------------------------------------------
# Round-trip: write → read returns the same cached row
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_second_call_serves_from_cache(self, tmp_path):
        trading = tmp_path / "atomicortex.db"
        _seed_trading_db(trading, n_days=14)
        eng = StatsEngine([str(trading)])
        a = eng.compute_performance(use_cache=False)
        b = eng.compute_performance(use_cache=True)
        # Same metric values (cache hit returns the just-written row).
        for k in ("win_rate", "total_signals", "total_pnl_pct"):
            assert a[k] == b[k], k


# ---------------------------------------------------------------------------
# Fail-soft: locked cache DB doesn't break read of signals
# ---------------------------------------------------------------------------


class TestFailSoft:
    def test_uncacheable_cache_path_still_returns_stats(self, tmp_path, caplog):
        trading = tmp_path / "atomicortex.db"
        _seed_trading_db(trading, n_days=14)
        # Make the cache path a directory so creation fails.
        bad_cache = tmp_path / "blocked"
        bad_cache.mkdir()
        # Pass the directory as the cache "file" — sqlite3.connect will
        # error. StatsEngine should log + carry on.
        eng = StatsEngine([str(trading)], cache_db_path=str(bad_cache))
        out = eng.compute_performance(use_cache=False)
        assert out["closed_signals"] >= 10
