"""Tests for 4H strategy dead-man's-switch heartbeat wiring.

The 4H ``MLTradingStrategy`` must:
  * create a ``HeartbeatManager`` in ``on_start`` (via ``_start_heartbeat``)
  * use an isolated Redis key (``atomicortex:heartbeat``) distinct from the
    15m / 1H bots so a dead 15m never silences the 4H watchdog
  * fail-soft on every error path (no event loop, Redis down, init crash)
    so a monitoring failure never stops the bot from trading
  * schedule ``HeartbeatManager.stop()`` in ``on_stop`` so the watchdog sees
    a clean shutdown rather than a crash
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> MLStrategyConfig:
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        dry_run=True,
    )


@pytest.fixture
def strategy(cfg: MLStrategyConfig) -> MLTradingStrategy:
    # Nautilus exposes ``log`` as a Cython property tied to the runtime; it
    # works fine outside an engine for our purposes, so no mocking needed.
    return MLTradingStrategy(config=cfg)


# ---------------------------------------------------------------------------
# Config-level isolation
# ---------------------------------------------------------------------------

class TestKeyIsolation:
    def test_4h_default_key(self) -> None:
        assert MLStrategyConfig().heartbeat_key == "atomicortex:heartbeat"

    def test_4h_and_15m_keys_differ(self) -> None:
        """Critical: a dead 15m bot must not silence the 4H watchdog."""
        from src.execution.strategies.ml_strategy_15m import MLStrategy15MConfig
        assert MLStrategyConfig().heartbeat_key != MLStrategy15MConfig().heartbeat_key

    def test_4h_and_1h_keys_differ(self) -> None:
        from src.configs.strategy_1h import MLStrategyConfig1H
        assert MLStrategyConfig().heartbeat_key != MLStrategyConfig1H().heartbeat_key


# ---------------------------------------------------------------------------
# __init__ — heartbeat slot
# ---------------------------------------------------------------------------

class TestHeartbeatSlot:
    def test_heartbeat_starts_as_none(self, strategy: MLTradingStrategy) -> None:
        """Strategy carries a heartbeat slot from construction (None until start)."""
        assert hasattr(strategy, "_heartbeat")
        assert strategy._heartbeat is None


# ---------------------------------------------------------------------------
# _start_heartbeat — fail-soft on every error path
# ---------------------------------------------------------------------------

class TestStartHeartbeat:
    def test_no_event_loop_fail_soft(self, strategy: MLTradingStrategy) -> None:
        """No running asyncio loop → leave _heartbeat=None, do not raise."""
        # Called from sync context — no running loop. Must not crash the bot.
        strategy._start_heartbeat()
        assert strategy._heartbeat is None

    def test_redis_down_does_not_block_bot(self, strategy: MLTradingStrategy) -> None:
        """HeartbeatManager handles Redis failure internally — bot keeps going.

        Verifies the manager is created (with the right key) even when Redis
        is unreachable; the background loop retries on its own.
        """
        async def _runner():
            with patch("redis.asyncio.Redis") as mock_redis_cls:
                mock_client = MagicMock()
                # ping() raises → _connect_redis returns None, bg loop retries
                async def _ping_fail():
                    raise ConnectionError("redis down")
                mock_client.ping = _ping_fail
                mock_redis_cls.return_value = mock_client

                strategy._start_heartbeat()
                # Let the scheduled start() task run far enough to try connecting
                await asyncio.sleep(0.05)

                assert strategy._heartbeat is not None
                assert strategy._heartbeat._heartbeat_key == "atomicortex:heartbeat"
                # Stop the bg task so the test exits cleanly
                await strategy._heartbeat.stop()

        asyncio.run(_runner())

    def test_init_crash_fail_soft(self, strategy: MLTradingStrategy) -> None:
        """If HeartbeatManager() itself raises → log warning, do not crash bot."""
        async def _runner():
            with patch(
                "src.execution.heartbeat.HeartbeatManager",
                side_effect=RuntimeError("boom"),
            ):
                # Inside a running loop so the RuntimeError comes from the
                # constructor path, not the missing-loop path.
                strategy._start_heartbeat()
            assert strategy._heartbeat is None

        asyncio.run(_runner())

    def test_uses_config_heartbeat_key(self, strategy: MLTradingStrategy) -> None:
        """Manager is constructed with the key from MLStrategyConfig."""
        async def _runner():
            with patch(
                "src.execution.heartbeat.HeartbeatManager"
            ) as mock_hm:
                mock_instance = MagicMock()
                async def _noop():
                    return None
                mock_instance.start = _noop
                mock_hm.return_value = mock_instance

                strategy._start_heartbeat()
                await asyncio.sleep(0)  # let create_task schedule
                kwargs = mock_hm.call_args.kwargs
                assert kwargs["heartbeat_key"] == "atomicortex:heartbeat"

        asyncio.run(_runner())


# ---------------------------------------------------------------------------
# on_stop — heartbeat shutdown
# ---------------------------------------------------------------------------

class TestOnStop:
    def test_on_stop_without_heartbeat_is_safe(
        self, strategy: MLTradingStrategy
    ) -> None:
        """If heartbeat was never started, on_stop must not raise."""
        strategy._heartbeat = None
        strategy.cancel_all_orders = MagicMock()
        strategy.close_all_positions = MagicMock()
        # Should not raise
        strategy.on_stop()

    def test_on_stop_schedules_heartbeat_stop(
        self, strategy: MLTradingStrategy
    ) -> None:
        """When heartbeat exists and loop runs, on_stop schedules stop()."""
        async def _runner():
            mock_hb = MagicMock()
            stop_called = asyncio.Event()

            async def _stop():
                stop_called.set()

            mock_hb.stop = _stop
            strategy._heartbeat = mock_hb
            strategy.cancel_all_orders = MagicMock()
            strategy.close_all_positions = MagicMock()

            strategy.on_stop()
            # Yield so create_task actually runs the stop coroutine
            await asyncio.wait_for(stop_called.wait(), timeout=1.0)

        asyncio.run(_runner())

    def test_on_stop_heartbeat_failure_does_not_block_shutdown(
        self, strategy: MLTradingStrategy
    ) -> None:
        """A broken heartbeat must not prevent position close on shutdown."""
        # heartbeat is set but no running loop → get_running_loop raises
        strategy._heartbeat = MagicMock()
        strategy.cancel_all_orders = MagicMock()
        strategy.close_all_positions = MagicMock()
        # Should not raise — the warning branch handles it.
        strategy.on_stop()


# ---------------------------------------------------------------------------
# Inheritance — MetaMLTradingStrategy keeps the 4H key
# ---------------------------------------------------------------------------

class TestMetaStrategyInheritsKey:
    def test_meta_strategy_uses_4h_key(self) -> None:
        from src.execution.strategies.meta_strategy import MetaMLStrategyConfig
        assert MetaMLStrategyConfig().heartbeat_key == "atomicortex:heartbeat"
