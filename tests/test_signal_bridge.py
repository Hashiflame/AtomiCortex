"""
Tests for SignalBridge and SignalPoller.

Covers:
- SignalBridge CRUD: log_signal, close_signal, log_regime_change,
  log_circuit_breaker, update_metrics
- Error resilience: DB errors don't crash
- Thread safety: concurrent writes
- SignalPoller: new signal detection, dedup via last_id,
  event parsing (regime_change, circuit_breaker),
  metrics caching, stop lifecycle, broadcast invocation

Total ≥ 18 tests.
"""

from __future__ import annotations

import asyncio
import json
try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.signal_bridge import SignalBridge
from src.telegram_bot.signal_poller import SignalPoller


# ── Fixtures ──

@pytest.fixture
def bridge(tmp_path):
    db_path = str(tmp_path / "test_bridge.db")
    return SignalBridge(db_path=db_path)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_poller.db")


@pytest.fixture
def shared_bridge(db_path):
    """Bridge + its db_path for poller tests."""
    return SignalBridge(db_path=db_path)


@pytest.fixture
def mock_broadcaster():
    b = MagicMock()
    b.broadcast_signal = AsyncMock()
    b.broadcast_signal_closed = AsyncMock()
    b.broadcast_regime_change = AsyncMock()
    b.broadcast_circuit_breaker = AsyncMock()
    b._cached_metrics = {}
    return b


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: log_signal
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeLogSignal:
    def test_log_signal_creates_record(self, bridge) -> None:
        """log_signal inserts a row into signals_log."""
        sid = bridge.log_signal(
            symbol="BTCUSDT-PERP.BINANCE",
            direction="long",
            entry_price=100_000.0,
            stop_loss=98_000.0,
            take_profit=105_000.0,
            confidence=0.73,
            regime="trend",
        )
        assert sid > 0

        conn = sqlite3.connect(bridge._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals_log WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row is not None
        assert row["symbol"] == "BTCUSDT-PERP.BINANCE"
        assert row["direction"] == "long"
        assert row["result"] == "open"
        assert row["confidence"] == pytest.approx(0.73)

    def test_log_signal_returns_correct_id(self, bridge) -> None:
        """Successive calls return incrementing IDs."""
        id1 = bridge.log_signal("A", "long", 1, 0.9, 1.1, 0.5, "trend")
        id2 = bridge.log_signal("B", "short", 2, 2.1, 1.9, 0.6, "high_vol")
        assert id2 == id1 + 1

    def test_log_signal_stores_extra_fields(self, bridge) -> None:
        """atr, funding_rate, position_size, notional, leverage stored."""
        sid = bridge.log_signal(
            symbol="ETH", direction="long",
            entry_price=3000, stop_loss=2900, take_profit=3200,
            confidence=0.8, regime="trend",
            atr=100, funding_rate=0.0005,
            position_size=0.5, notional=1500, leverage=5,
        )
        conn = sqlite3.connect(bridge._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals_log WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["atr"] == pytest.approx(100)
        assert row["funding_rate"] == pytest.approx(0.0005)
        assert row["position_size"] == pytest.approx(0.5)
        assert row["notional"] == pytest.approx(1500)
        assert row["leverage"] == pytest.approx(5)


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: close_signal
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeCloseSignal:
    def test_close_signal_updates_result(self, bridge) -> None:
        """close_signal updates result, pnl_pct, and closed_at."""
        sid = bridge.log_signal("BTC", "long", 100, 95, 110, 0.7, "trend")
        bridge.close_signal(sid, close_price=105, pnl_pct=5.0, result="win")

        conn = sqlite3.connect(bridge._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals_log WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["result"] == "win"
        assert row["pnl_pct"] == pytest.approx(5.0)
        assert row["close_price"] == pytest.approx(105)
        assert row["closed_at"] is not None


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: log_regime_change
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeRegimeChange:
    def test_log_regime_change_creates_event(self, bridge) -> None:
        """log_regime_change inserts a bot_events row with JSON payload."""
        bridge.log_regime_change("trend", "high_vol")

        conn = sqlite3.connect(bridge._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM bot_events WHERE event_type = 'regime_change'"
        ).fetchone()
        conn.close()
        assert row is not None
        data = json.loads(row["message"])
        assert data["old"] == "trend"
        assert data["new"] == "high_vol"


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: log_circuit_breaker
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeCircuitBreaker:
    def test_log_circuit_breaker_creates_event(self, bridge) -> None:
        """log_circuit_breaker inserts a bot_events row."""
        bridge.log_circuit_breaker("Daily loss limit exceeded")

        conn = sqlite3.connect(bridge._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM bot_events WHERE event_type = 'circuit_breaker'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert "Daily loss" in row["message"]


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: update_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeMetrics:
    def test_update_metrics_creates_record(self, bridge) -> None:
        """update_metrics inserts into bot_metrics."""
        bridge.update_metrics(
            equity=12_000, daily_pnl=0.015,
            regime="trend", open_positions=2,
        )

        conn = sqlite3.connect(bridge._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM bot_metrics WHERE id = 1").fetchone()
        conn.close()
        assert row is not None
        assert row["equity"] == pytest.approx(12_000)
        assert row["daily_pnl"] == pytest.approx(0.015)
        assert row["regime"] == "trend"
        assert row["open_positions"] == 2

    def test_update_metrics_idempotent(self, bridge) -> None:
        """Repeated update_metrics calls overwrite (upsert, not duplicate)."""
        bridge.update_metrics(10_000, 0.01, "trend", 1)
        bridge.update_metrics(11_000, 0.02, "high_vol", 2)

        conn = sqlite3.connect(bridge._db_path)
        rows = conn.execute("SELECT COUNT(*) FROM bot_metrics").fetchone()
        conn.close()
        assert rows[0] == 1  # only one row ever


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: error resilience
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeErrorResilience:
    def test_db_error_does_not_crash(self, tmp_path) -> None:
        """Operations on a bad path return gracefully, not raise."""
        # Use a path that is a directory, not a file — should fail
        bad_path = str(tmp_path / "bad_dir")
        Path(bad_path).mkdir()
        bridge = SignalBridge(db_path=str(Path(bad_path) / "nonexistent" / "db.sqlite"))
        # Should not raise
        result = bridge.log_signal("X", "long", 1, 0, 2, 0.5, "trend")
        assert result == 0  # error returns 0


# ═══════════════════════════════════════════════════════════════════════════
# SignalBridge: thread safety
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalBridgeThreadSafety:
    def test_concurrent_writes(self, bridge) -> None:
        """Multiple threads writing signals concurrently don't lose data."""
        errors = []
        n_threads = 5
        writes_per_thread = 10

        def _writer(thread_id: int) -> None:
            for i in range(writes_per_thread):
                try:
                    bridge.log_signal(
                        f"SYM_{thread_id}_{i}", "long",
                        100 + i, 90, 110, 0.5, "trend",
                    )
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=_writer, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        conn = sqlite3.connect(bridge._db_path)
        count = conn.execute("SELECT COUNT(*) FROM signals_log").fetchone()[0]
        conn.close()
        assert count == n_threads * writes_per_thread


# ═══════════════════════════════════════════════════════════════════════════
# SignalPoller: _check_new_signals
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalPollerNewSignals:
    @pytest.mark.asyncio
    async def test_finds_new_signals(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Poller detects new signals and calls broadcast_signal."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)
        poller._init_high_water_marks()

        # Write a signal AFTER initialising marks
        shared_bridge.log_signal("BTC", "long", 100, 90, 110, 0.7, "trend")

        await poller._check_new_signals(db_path)

        mock_broadcaster.broadcast_signal.assert_awaited_once()
        call_data = mock_broadcaster.broadcast_signal.call_args[0][0]
        assert call_data["symbol"] == "BTC"
        assert call_data["direction"] == "long"

    @pytest.mark.asyncio
    async def test_does_not_duplicate(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Same signal is NOT broadcast twice."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)
        poller._init_high_water_marks()

        shared_bridge.log_signal("ETH", "short", 3000, 3100, 2900, 0.6, "high_vol")

        await poller._check_new_signals(db_path)
        await poller._check_new_signals(db_path)  # second poll

        # Only one broadcast
        assert mock_broadcaster.broadcast_signal.await_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# SignalPoller: _check_new_events
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalPollerNewEvents:
    @pytest.mark.asyncio
    async def test_parses_regime_change(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Poller parses regime_change event and calls broadcast_regime_change."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)
        poller._init_high_water_marks()

        shared_bridge.log_regime_change("trend", "high_vol")

        await poller._check_new_events(db_path)

        mock_broadcaster.broadcast_regime_change.assert_awaited_once_with(
            "trend", "high_vol",
        )

    @pytest.mark.asyncio
    async def test_parses_circuit_breaker(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Poller parses circuit_breaker event and calls broadcast."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)
        poller._init_high_water_marks()

        shared_bridge.log_circuit_breaker("Max drawdown")

        await poller._check_new_events(db_path)

        mock_broadcaster.broadcast_circuit_breaker.assert_awaited_once_with(
            "Max drawdown",
        )


# ═══════════════════════════════════════════════════════════════════════════
# SignalPoller: _update_cached_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalPollerMetrics:
    @pytest.mark.asyncio
    async def test_caches_metrics(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Poller reads bot_metrics and caches them."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)

        shared_bridge.update_metrics(
            equity=12_000, daily_pnl=0.015,
            regime="trend", open_positions=2,
        )

        await poller._update_cached_metrics(db_path)

        assert poller.cached_metrics["equity"] == pytest.approx(12_000)
        assert poller.cached_metrics["regime"] == "trend"
        # Broadcaster should also have the cache
        assert mock_broadcaster._cached_metrics["equity"] == pytest.approx(12_000)


# ═══════════════════════════════════════════════════════════════════════════
# SignalPoller: lifecycle (start/stop)
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalPollerLifecycle:
    @pytest.mark.asyncio
    async def test_stop_cancels_loop(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """stop() cleanly cancels the background task."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=60)
        await poller.start()
        assert poller._running is True
        assert poller._task is not None

        await poller.stop()
        assert poller._running is False

    @pytest.mark.asyncio
    async def test_broadcast_called_on_new_signal(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Full integration: start poller → write signal → verify broadcast."""
        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=0.1)
        poller._init_high_water_marks()

        # Write signal
        shared_bridge.log_signal(
            "SOL", "long", 180, 170, 200, 0.75, "trend",
        )

        # Run a single poll cycle manually
        await poller._check_new_signals(db_path)

        mock_broadcaster.broadcast_signal.assert_awaited_once()
        assert mock_broadcaster.broadcast_signal.call_args[0][0]["symbol"] == "SOL"

        # Cleanup
        await poller.stop()


# ═══════════════════════════════════════════════════════════════════════════
# SignalPoller: high-water marks init
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalPollerHighWaterMarks:
    @pytest.mark.asyncio
    async def test_skips_existing_signals_on_startup(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Signals that existed before poller start are NOT broadcast."""
        # Write signals before poller init
        shared_bridge.log_signal("BTC", "long", 100, 90, 110, 0.7, "trend")
        shared_bridge.log_signal("ETH", "short", 3000, 3100, 2900, 0.6, "high_vol")

        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)
        poller._init_high_water_marks()

        await poller._check_new_signals(db_path)

        # No broadcast — both signals existed before init
        mock_broadcaster.broadcast_signal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_only_new_signals_after_startup(
        self, shared_bridge, db_path, mock_broadcaster,
    ) -> None:
        """Only signals written after init are broadcast."""
        shared_bridge.log_signal("BTC", "long", 100, 90, 110, 0.7, "trend")

        poller = SignalPoller(db_path, mock_broadcaster, poll_interval=1)
        poller._init_high_water_marks()

        # This one is new
        shared_bridge.log_signal("SOL", "short", 180, 185, 170, 0.8, "high_vol")

        await poller._check_new_signals(db_path)

        assert mock_broadcaster.broadcast_signal.await_count == 1
        assert mock_broadcaster.broadcast_signal.call_args[0][0]["symbol"] == "SOL"
