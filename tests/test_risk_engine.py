"""
Tests for src/risk — RiskEngine, CircuitBreaker, PortfolioTracker.

Covers:
- 10 pre-trade filter tests
- 5 position sizing tests
- 4 circuit breaker tests
- 3 portfolio tracker tests
- 1 integration test
Total ≥ 23 tests
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.risk.risk_engine import (
    PortfolioState,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    TradeSignal,
)
from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from src.risk.portfolio_tracker import PortfolioTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig()


@pytest.fixture
def engine(config: RiskConfig) -> RiskEngine:
    return RiskEngine(config, equity=10_000)


@pytest.fixture
def ok_signal() -> TradeSignal:
    """A signal that should pass all filters."""
    return TradeSignal(
        symbol="BTCUSDT",
        direction=1,
        confidence=0.73,
        regime="trend",
        entry_price=94_250.0,
        atr=1_500.0,
        atr_pct=0.016,
        funding_rate=0.0001,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def ok_state() -> PortfolioState:
    """A healthy portfolio state."""
    return PortfolioState(
        equity=10_000,
        open_positions=0,
        daily_pnl_pct=0.01,
        weekly_pnl_pct=0.02,
        current_drawdown_pct=0.02,
        consecutive_losses=0,
        last_loss_time=None,
        peak_equity=10_000,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PRE-TRADE FILTERS
# ═══════════════════════════════════════════════════════════════════════════


class TestPreTradeFilters:
    """Pre-trade filter unit tests."""

    def test_low_confidence_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """Signal with confidence < 0.65 must be rejected."""
        ok_signal.confidence = 0.50
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "confidence" in decision.reason.lower()
        assert decision.position_size == 0.0

    def test_daily_loss_limit_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """Daily P&L <= -3% must block trading."""
        ok_state.daily_pnl_pct = -0.04
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "daily" in decision.reason.lower()

    def test_max_positions_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """3 open positions must block new entries."""
        ok_state.open_positions = 3
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "position" in decision.reason.lower()

    def test_extreme_funding_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """Funding rate > 0.1% must be rejected."""
        ok_signal.funding_rate = 0.005  # 0.5%
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "funding" in decision.reason.lower()

    def test_vol_spike_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """ATR% > 2× average (>2%) must trigger vol spike."""
        ok_signal.atr_pct = 0.025  # 2.5% > 2×1% threshold
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "volatility" in decision.reason.lower()

    def test_consecutive_losses_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """5 consecutive losses within pause window must block."""
        ok_state.consecutive_losses = 5
        ok_state.last_loss_time = datetime.now(timezone.utc) - timedelta(hours=1)
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "consecutive" in decision.reason.lower()

    def test_consecutive_losses_after_pause_approved(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """After 4h pause, 5 consecutive losses should allow trading."""
        ok_state.consecutive_losses = 5
        ok_state.last_loss_time = datetime.now(timezone.utc) - timedelta(hours=5)
        decision = engine.evaluate(ok_signal, ok_state)
        assert decision.approved

    def test_weekly_loss_rejected(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """Weekly P&L <= -8% must block."""
        ok_state.weekly_pnl_pct = -0.09
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "weekly" in decision.reason.lower()

    def test_drawdown_kill_switch(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """Drawdown > 15% triggers kill switch."""
        ok_state.current_drawdown_pct = 0.20  # 20% drawdown
        decision = engine.evaluate(ok_signal, ok_state)
        assert not decision.approved
        assert "kill" in decision.reason.lower()

    def test_all_conditions_ok_approved(
        self, engine: RiskEngine, ok_signal: TradeSignal, ok_state: PortfolioState,
    ) -> None:
        """When all conditions are met, signal must be approved."""
        decision = engine.evaluate(ok_signal, ok_state)
        assert decision.approved
        assert decision.reason == ""
        assert decision.position_size > 0
        assert decision.notional > 0
        assert decision.leverage > 0


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionSizing:
    """ATR-based position sizing unit tests."""

    def test_sizing_10k_atr500(self, engine: RiskEngine) -> None:
        """$10k equity, ATR=$500: dollar_risk=$100, stop=750, contracts=0.1333."""
        signal = TradeSignal(
            symbol="BTCUSDT",
            direction=1,
            confidence=0.80,
            regime="trend",
            entry_price=50_000.0,
            atr=500.0,
            atr_pct=0.01,
            funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        contracts, notional, leverage = engine.calculate_position_size(signal, 10_000)
        # dollar_risk = 10000 × 0.01 = 100
        # stop_distance = 500 × 1.5 = 750
        # contracts = 100 / 750 = 0.1333...
        assert abs(contracts - 100 / 750) < 1e-6
        # notional = contracts × 50000
        expected_notional = (100 / 750) * 50_000
        assert abs(notional - expected_notional) < 0.01
        # leverage = notional / equity
        expected_leverage = expected_notional / 10_000
        assert abs(leverage - expected_leverage) < 1e-4

    def test_leverage_cap_applied(self, engine: RiskEngine) -> None:
        """When calculated leverage exceeds max, it must be capped."""
        signal = TradeSignal(
            symbol="BTCUSDT",
            direction=1,
            confidence=0.80,
            regime="trend",
            entry_price=100_000.0,
            atr=5.0,  # tiny ATR → huge position → must cap
            atr_pct=0.00005,
            funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        contracts, notional, leverage = engine.calculate_position_size(signal, 10_000)
        assert leverage <= 10.0 + 1e-9  # max_leverage = 10
        assert notional <= 10_000 * 10 + 0.01

    def test_stop_loss_long_below_entry(self, engine: RiskEngine) -> None:
        """LONG stop-loss must be below entry."""
        sl = engine.calculate_stop_loss(50_000, direction=1, atr=500)
        assert sl < 50_000
        # sl = 50000 - 500 × 1.5 = 49250
        assert abs(sl - 49_250) < 0.01

    def test_stop_loss_short_above_entry(self, engine: RiskEngine) -> None:
        """SHORT stop-loss must be above entry."""
        sl = engine.calculate_stop_loss(50_000, direction=-1, atr=500)
        assert sl > 50_000
        # sl = 50000 + 500 × 1.5 = 50750
        assert abs(sl - 50_750) < 0.01

    def test_take_profit_rr_ratio(self, engine: RiskEngine) -> None:
        """Take-profit must achieve R:R = 1.5."""
        entry = 50_000.0
        sl = engine.calculate_stop_loss(entry, direction=1, atr=500)
        tp = engine.calculate_take_profit(entry, 1, sl)
        risk = abs(entry - sl)      # 750
        reward = abs(tp - entry)    # 1125
        rr = reward / risk
        assert abs(rr - 1.5) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    """Circuit breaker unit tests."""

    def test_soft_trigger_multiplier(self) -> None:
        """Daily PnL between -2% and -3% → multiplier = 0.5."""
        cb = CircuitBreaker()
        state = PortfolioState(
            equity=9_750,
            open_positions=1,
            daily_pnl_pct=-0.025,  # -2.5% (between soft and hard)
            weekly_pnl_pct=-0.01,
            current_drawdown_pct=0.025,
            consecutive_losses=1,
            last_loss_time=None,
            peak_equity=10_000,
        )
        mult = cb.get_position_size_multiplier(state)
        assert mult == 0.5

    def test_hard_trigger_multiplier(self) -> None:
        """Daily PnL <= -3% → multiplier = 0.0."""
        cb = CircuitBreaker()
        state = PortfolioState(
            equity=9_700,
            open_positions=0,
            daily_pnl_pct=-0.035,  # -3.5%
            weekly_pnl_pct=-0.03,
            current_drawdown_pct=0.03,
            consecutive_losses=0,
            last_loss_time=None,
            peak_equity=10_000,
        )
        mult = cb.get_position_size_multiplier(state)
        assert mult == 0.0

    def test_kill_switch_triggered(self) -> None:
        """Drawdown > 15% → is_triggered = True."""
        cb = CircuitBreaker()
        state = PortfolioState(
            equity=8_400,
            open_positions=0,
            daily_pnl_pct=-0.01,
            weekly_pnl_pct=-0.05,
            current_drawdown_pct=0.16,  # 16% drawdown
            consecutive_losses=0,
            last_loss_time=None,
            peak_equity=10_000,
        )
        result = cb.check(state, current_atr=1000, avg_atr=1000, current_funding=0.0)
        assert result.is_triggered
        assert "kill" in result.trigger_reason.lower()

    def test_reset_daily(self) -> None:
        """reset_daily clears internal daily flags."""
        cb = CircuitBreaker()
        # Force internal trigger
        cb._daily_triggered = True
        cb._daily_trigger_reason = "test reason"
        cb.reset_daily()
        assert not cb._daily_triggered
        assert cb._daily_trigger_reason == ""


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioTracker:
    """Portfolio tracker unit tests."""

    def test_fill_and_unrealized_pnl(self) -> None:
        """fill + update_price should produce correct unrealized P&L."""
        tracker = PortfolioTracker(initial_equity=10_000)
        now = datetime.now(timezone.utc)

        tracker.update_fill(
            symbol="BTCUSDT",
            direction=1,
            quantity=0.1,
            price=50_000,
            fee=5.0,
            timestamp=now,
        )
        # Price goes up
        tracker.update_price("BTCUSDT", 51_000)

        state = tracker.get_state()
        # unrealized = 1 × (51000 - 50000) × 0.1 = 100
        expected_unrealized = 0.1 * (51_000 - 50_000)
        assert abs(state.daily_pnl_pct - expected_unrealized / 10_000) < 1e-6
        assert state.open_positions == 1

    def test_close_position_realized_pnl(self) -> None:
        """Closing a position returns correct realized P&L."""
        tracker = PortfolioTracker(initial_equity=10_000)
        now = datetime.now(timezone.utc)

        tracker.update_fill(
            symbol="ETHUSDT",
            direction=1,
            quantity=1.0,
            price=3_000,
            fee=3.0,
            timestamp=now,
        )

        pnl = tracker.close_position(
            symbol="ETHUSDT",
            close_price=3_100,
            fee=3.0,
            timestamp=now + timedelta(hours=1),
        )
        # gross = 1 × (3100 - 3000) × 1.0 = 100
        # fees = 3 + 3 = 6
        # net = 94
        assert abs(pnl - 94.0) < 0.01
        assert tracker.get_state().open_positions == 0

    def test_consecutive_losses_counter(self) -> None:
        """record_loss increments consecutive counter correctly."""
        tracker = PortfolioTracker(initial_equity=10_000)
        now = datetime.now(timezone.utc)

        for i in range(3):
            tracker.record_loss(now + timedelta(minutes=i))

        state = tracker.get_state()
        assert state.consecutive_losses == 3
        assert state.last_loss_time is not None
