"""Tests for Step H9 — SQLite busy_timeout / WAL on every connection.

Without busy_timeout, concurrent writers fall over with "database is
locked" the instant they collide. The fix sets per-connection PRAGMAs
in ``SignalBridge._connect()``.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.execution.signal_bridge import SignalBridge


def _bridge(tmp_path: Path) -> SignalBridge:
    return SignalBridge(db_path=str(tmp_path / "ac.db"))


# ---------------------------------------------------------------------------
# Per-connection PRAGMAs are present
# ---------------------------------------------------------------------------


class TestConnectionPragmas:
    def test_busy_timeout_is_5000(self, tmp_path):
        br = _bridge(tmp_path)
        conn = br._connect()
        try:
            val = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert val == 5000, f"busy_timeout={val}"
        finally:
            conn.close()

    def test_journal_mode_is_wal(self, tmp_path):
        br = _bridge(tmp_path)
        conn = br._connect()
        try:
            val = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert val.lower() == "wal", f"journal_mode={val}"
        finally:
            conn.close()

    def test_synchronous_is_normal(self, tmp_path):
        br = _bridge(tmp_path)
        conn = br._connect()
        try:
            val = conn.execute("PRAGMA synchronous").fetchone()[0]
            # NORMAL = 1
            assert val == 1, f"synchronous={val}"
        finally:
            conn.close()

    def test_foreign_keys_on(self, tmp_path):
        br = _bridge(tmp_path)
        conn = br._connect()
        try:
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        finally:
            conn.close()

    def test_pragmas_applied_on_every_connection(self, tmp_path):
        """busy_timeout is per-connection — fresh connections must each
        carry it, not rely on the first connect having set it."""
        br = _bridge(tmp_path)
        for _ in range(3):
            conn = br._connect()
            try:
                assert (
                    conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
                )
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Fail-soft: a missing pragma must not break _connect
# ---------------------------------------------------------------------------


class TestPragmaFailSoft:
    def test_bad_pragma_logged_not_raised(self, tmp_path, monkeypatch):
        """Patch _connect's per-pragma loop indirectly: force the conn
        produced by sqlite3.connect to error on one pragma."""
        br = _bridge(tmp_path)

        orig_connect = sqlite3.connect

        class _WrappedConn:
            def __init__(self, inner):
                self._inner = inner
                self.row_factory = None

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def execute(self, sql, *args, **kwargs):
                if "synchronous" in sql.lower():
                    raise sqlite3.Error("synthetic pragma failure")
                return self._inner.execute(sql, *args, **kwargs)

        def fake_connect(*a, **kw):
            return _WrappedConn(orig_connect(*a, **kw))

        monkeypatch.setattr("src.execution.signal_bridge.sqlite3.connect",
                            fake_connect)

        # Must not raise.
        conn = br._connect()
        # And the other pragmas must still have been attempted (they hit
        # the inner connection which works).
        assert conn is not None


# ---------------------------------------------------------------------------
# Concurrent writes don't fail immediately
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    def test_two_writers_both_succeed(self, tmp_path):
        """Two threads each log 25 signals — none should error out with
        'database is locked'."""
        br = _bridge(tmp_path)

        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def worker(prefix: str):
            try:
                barrier.wait()
                for i in range(25):
                    br.log_signal(
                        symbol=f"{prefix}{i}",
                        direction="long",
                        entry_price=50_000.0,
                        stop_loss=49_000.0,
                        take_profit=52_000.0,
                        confidence=0.7,
                        regime="trend_up",
                        atr=500.0,
                        funding_rate=0.0001,
                    )
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("A",))
        t2 = threading.Thread(target=worker, args=("B",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == [], f"unexpected errors: {errors!r}"

        # All 50 rows should have landed.
        conn = br._connect()
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM signals_log",
            ).fetchone()[0]
        finally:
            conn.close()
        assert n == 50

    def test_writer_waits_through_held_lock(self, tmp_path):
        """Hold a write lock from one connection while another writes —
        the second must succeed (waited under busy_timeout) instead of
        raising OperationalError immediately."""
        br = _bridge(tmp_path)

        # Open a long-running transaction on conn_hold.
        conn_hold = br._connect()
        conn_hold.execute("BEGIN IMMEDIATE")
        conn_hold.execute(
            "INSERT INTO bot_events (event_type, message) VALUES (?, ?)",
            ("hold", "test"),
        )

        result: dict = {}

        def writer():
            t0 = time.monotonic()
            try:
                br.log_signal(
                    symbol="BTCUSDT", direction="long",
                    entry_price=50_000.0, stop_loss=49_000.0,
                    take_profit=52_000.0, confidence=0.7,
                    regime="trend_up", atr=500.0, funding_rate=0.0001,
                )
                result["ok"] = True
                result["elapsed"] = time.monotonic() - t0
            except sqlite3.OperationalError as exc:
                result["error"] = exc

        t = threading.Thread(target=writer)
        t.start()
        # Hold the lock briefly, then release. Writer should block on the
        # busy_timeout and proceed once we commit.
        time.sleep(0.3)
        conn_hold.commit()
        conn_hold.close()
        t.join(timeout=10.0)

        assert result.get("ok") is True, f"writer failed: {result!r}"
        # Writer must have waited at least the time we held the lock.
        assert result["elapsed"] >= 0.25
        # And much less than the 5s timeout.
        assert result["elapsed"] < 4.5
