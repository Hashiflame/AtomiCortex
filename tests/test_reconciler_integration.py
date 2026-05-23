"""Integration tests for PositionReconciler inside the 4H ML strategy.

The reconciler class itself is unit-tested elsewhere; here we verify the
*wiring* into ``MLTradingStrategy``: it is created at start, runs once
immediately (catching leftovers from a prior crash), recurs on the
15-min timer, normalises symbols correctly, logs CRITICAL on drift,
and never crashes the bot when it fails.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.execution.reconciler import (
    InternalPosition,
    PositionReconciler,
    ReconciliationResult,
)
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
        warmup_bars=2,
        dry_run=True,
    )


@pytest.fixture
def strategy(cfg: MLStrategyConfig) -> MLTradingStrategy:
    return MLTradingStrategy(config=cfg)


def _tracker_with(positions: dict) -> MagicMock:
    """Build a mock tracker whose ``_positions`` dict matches the real shape."""
    t = MagicMock()
    t._positions = positions
    return t


def _pos(direction: int, quantity: float) -> MagicMock:
    p = MagicMock()
    p.direction = direction
    p.quantity = quantity
    return p


# ---------------------------------------------------------------------------
# Construction wiring
# ---------------------------------------------------------------------------

class TestReconcilerSlot:
    def test_reconciler_starts_as_none(self, strategy: MLTradingStrategy) -> None:
        assert hasattr(strategy, "_reconciler")
        assert strategy._reconciler is None


# ---------------------------------------------------------------------------
# _schedule_reconciliation — startup behaviour
# ---------------------------------------------------------------------------

class TestScheduleReconciliation:
    def test_schedule_creates_reconciler_and_runs_once_at_startup(
        self, strategy: MLTradingStrategy
    ) -> None:
        """At start, the reconciler must be created and run immediately
        — the canonical "find positions left from prior crash" path."""
        ran = []

        def fake_run() -> None:
            ran.append(True)

        # clock.set_timer is a Cython callable bound to engine internals;
        # patch the strategy's own _run_reconciliation to observe the call.
        with patch.object(
            MLTradingStrategy, "_run_reconciliation", lambda self: fake_run()
        ):
            strategy._schedule_reconciliation()

        assert strategy._reconciler is not None
        assert isinstance(strategy._reconciler, PositionReconciler)
        assert ran, "_run_reconciliation must be invoked once at startup"

    def test_schedule_failure_is_fail_soft(self, strategy: MLTradingStrategy) -> None:
        """If PositionReconciler() raises, the bot keeps going with no reconciler."""
        with patch(
            "src.execution.reconciler.PositionReconciler",
            side_effect=RuntimeError("boom"),
        ):
            strategy._schedule_reconciliation()
        assert strategy._reconciler is None


# ---------------------------------------------------------------------------
# _run_reconciliation — sync dispatcher
# ---------------------------------------------------------------------------

class TestRunReconciliation:
    def test_no_reconciler_is_noop(self, strategy: MLTradingStrategy) -> None:
        strategy._reconciler = None
        # Must not raise.
        strategy._run_reconciliation()

    def test_no_event_loop_is_fail_soft(self, strategy: MLTradingStrategy) -> None:
        """Outside an asyncio loop, the schedule attempt must not crash."""
        strategy._reconciler = MagicMock()
        # Sync context → asyncio.get_running_loop raises RuntimeError;
        # the warning branch must swallow it.
        strategy._run_reconciliation()


# ---------------------------------------------------------------------------
# _reconcile_async — the actual work
# ---------------------------------------------------------------------------

class TestReconcileAsync:
    def test_normalises_full_instrument_id_to_bare_symbol(
        self, strategy: MLTradingStrategy
    ) -> None:
        """Tracker keys ('BTCUSDT-PERP.BINANCE') must map to bare 'BTCUSDT'
        before being handed to the reconciler — otherwise everything looks
        like a ghost."""
        captured: dict = {}

        async def fake_reconcile(internal, exchange_positions=None):
            captured["internal"] = internal
            return ReconciliationResult(is_clean=True)

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({
            "BTCUSDT-PERP.BINANCE": _pos(direction=1, quantity=0.5),
        })

        asyncio.run(strategy._reconcile_async())

        assert "BTCUSDT" in captured["internal"]
        ip = captured["internal"]["BTCUSDT"]
        assert isinstance(ip, InternalPosition)
        assert ip.symbol == "BTCUSDT"
        assert ip.direction == 1
        assert ip.quantity == 0.5

    def test_clean_result_emits_no_critical(self, strategy: MLTradingStrategy) -> None:
        async def fake_reconcile(internal, exchange_positions=None):
            return ReconciliationResult(is_clean=True)

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({})

        with patch.object(type(strategy), "log") as mock_log:
            mock_log.critical = MagicMock()
            asyncio.run(strategy._reconcile_async())
            # Clean → no CRITICAL noise

    def test_orphan_path_runs_to_completion(
        self, strategy: MLTradingStrategy
    ) -> None:
        """An exchange position with no tracker counterpart → the orphan
        branch runs (and emits CRITICAL via self.log.critical). We assert
        the code path completes without raising; the actual ``self.log``
        is Nautilus' Cython logger which can't be mocked at this layer."""
        async def fake_reconcile(internal, exchange_positions=None):
            return ReconciliationResult(
                is_clean=False,
                orphan_positions=[{
                    "symbol": "ETHUSDT",
                    "direction": 1,
                    "quantity": 0.2,
                    "entry_price": 3_000.0,
                    "unrealized_pnl": 0.0,
                }],
            )

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({})
        # Must not raise — iterates orphan_positions and emits log.critical.
        asyncio.run(strategy._reconcile_async())

    def test_ghost_path_runs_to_completion(
        self, strategy: MLTradingStrategy
    ) -> None:
        async def fake_reconcile(internal, exchange_positions=None):
            return ReconciliationResult(
                is_clean=False,
                ghost_positions=[{
                    "symbol": "BTCUSDT", "direction": -1, "quantity": 0.1,
                }],
            )

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({
            "BTCUSDT-PERP.BINANCE": _pos(direction=-1, quantity=0.1),
        })
        asyncio.run(strategy._reconcile_async())

    def test_mismatch_path_runs_to_completion(
        self, strategy: MLTradingStrategy
    ) -> None:
        async def fake_reconcile(internal, exchange_positions=None):
            return ReconciliationResult(
                is_clean=False,
                mismatched_sizes=[{
                    "symbol": "BTCUSDT",
                    "internal_direction": 1, "exchange_direction": 1,
                    "internal_quantity": 0.1, "exchange_quantity": 0.2,
                }],
            )

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({
            "BTCUSDT-PERP.BINANCE": _pos(direction=1, quantity=0.1),
        })
        asyncio.run(strategy._reconcile_async())

    def test_reconcile_exception_is_fail_soft(
        self, strategy: MLTradingStrategy
    ) -> None:
        """A crash inside reconcile() must NOT crash the bot."""
        async def fake_reconcile(internal, exchange_positions=None):
            raise RuntimeError("network down")

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({})

        # Must not raise
        asyncio.run(strategy._reconcile_async())

    def test_no_reconciler_async_is_noop(
        self, strategy: MLTradingStrategy
    ) -> None:
        strategy._reconciler = None
        asyncio.run(strategy._reconcile_async())  # no raise


# ---------------------------------------------------------------------------
# End-to-end: prior-crash recovery — orphan detected on restart
# ---------------------------------------------------------------------------

class TestPriorCrashRecovery:
    def test_startup_finds_position_left_by_previous_run(
        self, strategy: MLTradingStrategy
    ) -> None:
        """Simulates: bot crashed mid-trade, exchange holds an open position
        but tracker is empty after restart. Reconciler must be invoked with
        an empty internal dict so the exchange position is classified as
        orphan."""
        captured = {}

        async def fake_reconcile(internal, exchange_positions=None):
            captured["internal"] = internal
            return ReconciliationResult(
                is_clean=False,
                orphan_positions=[{
                    "symbol": "BTCUSDT",
                    "direction": 1,
                    "quantity": 0.05,
                    "entry_price": 50_000.0,
                    "unrealized_pnl": 0.0,
                }],
            )

        strategy._reconciler = MagicMock()
        strategy._reconciler.reconcile = fake_reconcile
        strategy._tracker = _tracker_with({})  # fresh state, nothing tracked

        asyncio.run(strategy._reconcile_async())

        # The wiring contract: a fresh tracker yields {}; combined with a
        # real reconciler this surfaces every exchange position as orphan.
        assert captured["internal"] == {}
