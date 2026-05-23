"""Cash-accounting invariants for ``PortfolioTracker``.

Background
----------
This is a perpetual-futures bot. The codebase models equity as

    equity = cash + sum(unrealized_pnl_of_open_positions)

where ``unrealized_pnl = direction × (mark - entry) × qty`` is the
mark-to-market PnL *relative to entry*, not the position's full mark value.
That is the standard futures model: opening a position does not move cash
(only fees do); closing realises the PnL into cash.

Bug history (pre-fix)
---------------------
``close_position`` added ``quantity × entry_price`` (the notional) back into
cash on close, even though ``update_fill`` never deducted it on open. Each
zero-PnL round-trip therefore inflated equity by one full notional.

The ``test_round_trip_zero_pnl_invariant`` test is the canonical regression
check: N buy+sell cycles at the same price must leave equity unchanged
(modulo fees).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.risk.portfolio_tracker import PortfolioTracker
from src.risk.risk_engine import PortfolioState


_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# THE INVARIANT — this is the test that proves the bug existed
# ---------------------------------------------------------------------------

class TestRoundTripInvariant:
    def test_single_round_trip_zero_pnl_zero_fee(self) -> None:
        """Buy + sell at the same price with no fees → equity unchanged.

        Pre-fix: equity = 10_000 + 5_000 (notional re-added) = 15_000.
        """
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_NOW)
        t.close_position("BTCUSDT", 50_000, fee=0.0, timestamp=_NOW + timedelta(hours=1))
        assert t.get_state().equity == pytest.approx(10_000, abs=1e-6)

    def test_ten_round_trips_zero_pnl_zero_fee(self) -> None:
        """10 round-trips with zero PnL → equity unchanged.

        Pre-fix: equity grew by 10 × notional. This is the catastrophic
        case that broke daily-loss / sizing / drawdown calculations.
        """
        t = PortfolioTracker(initial_equity=10_000)
        for i in range(10):
            ts = _NOW + timedelta(hours=i)
            t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=ts)
            t.close_position(
                "BTCUSDT", 50_000, fee=0.0,
                timestamp=ts + timedelta(minutes=30),
            )
        assert t.get_state().equity == pytest.approx(10_000, abs=1e-6)

    def test_round_trip_invariant_holds_for_short(self) -> None:
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", -1, 0.1, 50_000, fee=0.0, timestamp=_NOW)
        t.close_position("BTCUSDT", 50_000, fee=0.0, timestamp=_NOW + timedelta(hours=1))
        assert t.get_state().equity == pytest.approx(10_000, abs=1e-6)


# ---------------------------------------------------------------------------
# Open / close mechanics
# ---------------------------------------------------------------------------

class TestOpenClose:
    def test_open_position_only_deducts_fee(self) -> None:
        """Futures model: opening a position moves cash only by the fee."""
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=5.0, timestamp=_NOW)
        # equity = cash + unrealized (0 at entry) = cash
        assert t.get_state().equity == pytest.approx(9_995.0, abs=1e-6)
        assert t.get_state().open_positions == 1

    def test_close_with_profit(self) -> None:
        """Close with gain → equity = initial + gross_pnl − total_fees."""
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=5.0, timestamp=_NOW)
        pnl = t.close_position(
            "BTCUSDT", 51_000, fee=5.0,
            timestamp=_NOW + timedelta(hours=1),
        )
        # gross = 0.1 × (51000 − 50000) = 100; fees = 10; net = 90
        assert pnl == pytest.approx(90.0, abs=1e-6)
        assert t.get_state().equity == pytest.approx(10_000 + 90.0, abs=1e-6)
        assert t.get_state().open_positions == 0

    def test_close_with_loss(self) -> None:
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=5.0, timestamp=_NOW)
        pnl = t.close_position(
            "BTCUSDT", 49_500, fee=5.0,
            timestamp=_NOW + timedelta(hours=1),
        )
        # gross = 0.1 × −500 = −50; fees = 10; net = −60
        assert pnl == pytest.approx(-60.0, abs=1e-6)
        assert t.get_state().equity == pytest.approx(10_000 - 60.0, abs=1e-6)

    def test_close_short_with_profit(self) -> None:
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", -1, 0.1, 50_000, fee=5.0, timestamp=_NOW)
        pnl = t.close_position(
            "BTCUSDT", 49_000, fee=5.0,
            timestamp=_NOW + timedelta(hours=1),
        )
        # short gross = −1 × (49000 − 50000) × 0.1 = 100; fees = 10; net = 90
        assert pnl == pytest.approx(90.0, abs=1e-6)
        assert t.get_state().equity == pytest.approx(10_090.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Unrealized PnL still flows into equity while position is open
# ---------------------------------------------------------------------------

class TestMarkToMarket:
    def test_unrealized_pnl_lifts_equity_during_open_position(self) -> None:
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_NOW)
        t.update_price("BTCUSDT", 51_000)
        # Unrealized = 100, no fees → equity = 10_100
        assert t.get_state().equity == pytest.approx(10_100.0, abs=1e-6)

    def test_unrealized_collapses_back_to_cash_on_close(self) -> None:
        """Closing at the same mark used for update_price must keep equity stable."""
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_NOW)
        t.update_price("BTCUSDT", 51_000)
        equity_before = t.get_state().equity
        t.close_position(
            "BTCUSDT", 51_000, fee=0.0,
            timestamp=_NOW + timedelta(hours=1),
        )
        assert t.get_state().equity == pytest.approx(equity_before, abs=1e-6)


# ---------------------------------------------------------------------------
# Downstream consumers — drawdown / get_state / risk_engine wiring
# ---------------------------------------------------------------------------

class TestDownstreamConsumers:
    def test_drawdown_reflects_real_losses_not_phantom_growth(self) -> None:
        """Pre-fix the inflated equity raised peak_equity, hiding real drawdown.

        With the fix: a losing trade after a flat trade produces a real
        drawdown reading.
        """
        t = PortfolioTracker(initial_equity=10_000)
        # Flat round-trip — must NOT raise peak above initial
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_NOW)
        t.close_position("BTCUSDT", 50_000, fee=0.0, timestamp=_NOW + timedelta(hours=1))
        # Loss round-trip
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_NOW + timedelta(hours=2))
        t.close_position(
            "BTCUSDT", 49_500, fee=0.0,
            timestamp=_NOW + timedelta(hours=3),
        )
        # equity = 10_000 - 50 = 9_950; peak = 10_000; dd = 0.005
        assert t.get_state().equity == pytest.approx(9_950.0, abs=1e-6)
        assert t.get_drawdown() == pytest.approx(0.005, abs=1e-6)

    def test_get_state_returns_correct_equity(self) -> None:
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=5.0, timestamp=_NOW)
        t.close_position(
            "BTCUSDT", 50_500, fee=5.0,
            timestamp=_NOW + timedelta(hours=1),
        )
        state = t.get_state()
        assert isinstance(state, PortfolioState)
        # gross = 50, fees = 10, net = 40
        assert state.equity == pytest.approx(10_040.0, abs=1e-6)

    def test_risk_engine_sees_corrected_equity_for_sizing(self) -> None:
        """RiskEngine.calculate_position_size uses ``equity`` from get_state.

        After a flat round-trip equity must be unchanged, so the next
        trade is sized off the real account, not a phantom-inflated one.
        """
        from src.risk.risk_engine import RiskConfig, RiskEngine, TradeSignal

        t = PortfolioTracker(initial_equity=10_000)
        # 10 flat round-trips
        for i in range(10):
            ts = _NOW + timedelta(hours=i)
            t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=ts)
            t.close_position(
                "BTCUSDT", 50_000, fee=0.0,
                timestamp=ts + timedelta(minutes=30),
            )

        engine = RiskEngine(
            RiskConfig(risk_per_trade=0.01, atr_stop_multiplier=1.5),
            equity=10_000,
        )
        equity_now = t.get_state().equity
        # atr=500$ → stop_dist = 1.5 × 500 = 750
        # dollar_risk = equity × 0.01
        # contracts = dollar_risk / 750; notional = contracts × 50_000
        signal = TradeSignal(
            symbol="BTCUSDT",
            direction=1,
            confidence=0.7,
            regime="trend_up",
            entry_price=50_000,
            atr=500.0,
            atr_pct=0.01,
            funding_rate=0.0,
            timestamp=_NOW,
        )
        _, notional_correct, _ = engine.calculate_position_size(signal, equity_now)
        _, notional_pre_fix, _ = engine.calculate_position_size(signal, 60_000.0)
        # The corrected sizing must use the real ~10k equity, NOT the
        # phantom 60k that the buggy tracker would have surfaced.
        assert notional_correct == pytest.approx(notional_pre_fix / 6.0, rel=0.05)
