"""Tests for Step H6 — single source of truth for equity.

Nautilus PortfolioFacade is authoritative (exchange-confirmed balance);
PortfolioTracker.sync_equity() aligns the tracker so risk decisions
(sizing / drawdown / circuit breaker) read the same number.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.risk.portfolio_tracker import PortfolioTracker


T0 = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def _new(equity=10_000.0):
    return PortfolioTracker(initial_equity=equity)


class TestSyncEquity:
    def test_sync_adjusts_cash_with_no_positions(self):
        pt = _new(10_000)
        pt.sync_equity(11_000.0)
        assert pt._get_equity() == pytest.approx(11_000.0)
        assert pt.get_state().equity == pytest.approx(11_000.0)
        assert pt._cash == pytest.approx(11_000.0)

    def test_sync_with_open_position_preserves_unrealized(self):
        pt = _new(10_000)
        # Open + mark to +500 unrealized.
        pt.update_fill("BTCUSDT", direction=1, quantity=1.0, price=50_000.0,
                       fee=0.0, timestamp=T0)
        pt.update_price("BTCUSDT", 50_500.0)
        assert pt._positions["BTCUSDT"].unrealized_pnl == pytest.approx(500.0)

        # Nautilus says total equity is 11_000. cash must become
        # 11_000 - 500 = 10_500 so invariant equity = cash + unrealized holds.
        pt.sync_equity(11_000.0)
        assert pt._cash == pytest.approx(10_500.0)
        assert pt._get_equity() == pytest.approx(11_000.0)
        # Unrealized must not be clobbered.
        assert pt._positions["BTCUSDT"].unrealized_pnl == pytest.approx(500.0)

    def test_sync_bumps_peak_equity(self):
        pt = _new(10_000)
        pt.sync_equity(12_345.0)
        assert pt._peak_equity == pytest.approx(12_345.0)

    def test_sync_does_not_lower_peak(self):
        pt = _new(10_000)
        pt._peak_equity = 15_000.0
        pt.sync_equity(11_000.0)
        assert pt._peak_equity == pytest.approx(15_000.0)

    def test_sync_does_not_touch_daily_counters(self):
        pt = _new(10_000)
        pt._daily_realized_pnl = 250.0
        pt._weekly_realized_pnl = 500.0
        pt._consecutive_losses = 3
        pt._day_start_equity = 9_500.0
        pt.sync_equity(11_000.0)
        assert pt._daily_realized_pnl == 250.0
        assert pt._weekly_realized_pnl == 500.0
        assert pt._consecutive_losses == 3
        assert pt._day_start_equity == 9_500.0

    def test_sync_drops_nan_silently(self):
        pt = _new(10_000)
        pre_cash = pt._cash
        pt.sync_equity(float("nan"))
        assert pt._cash == pre_cash

    def test_sync_drops_inf_silently(self):
        pt = _new(10_000)
        pre_cash = pt._cash
        pt.sync_equity(float("inf"))
        assert pt._cash == pre_cash

    def test_sync_drops_non_numeric_silently(self):
        pt = _new(10_000)
        pre_cash = pt._cash
        pt.sync_equity("oops")  # type: ignore[arg-type]
        assert pt._cash == pre_cash

    def test_drift_disappears_after_sync(self):
        """Simulate tracker drift from Nautilus (e.g. unaccounted funding):
        before sync the two diverge; after sync they agree."""
        pt = _new(10_000)
        pt.update_fill("BTCUSDT", direction=1, quantity=1.0, price=50_000.0,
                       fee=20.0, timestamp=T0)
        pt.update_price("BTCUSDT", 50_300.0)
        tracker_eq_pre = pt._get_equity()

        # Nautilus reports a slightly different total (funding paid in).
        nautilus_eq = tracker_eq_pre + 12.34
        assert pt._get_equity() != pytest.approx(nautilus_eq)

        pt.sync_equity(nautilus_eq)
        assert pt._get_equity() == pytest.approx(nautilus_eq)


class TestRiskEngineSeesSyncedEquity:
    def test_get_state_after_sync_returns_synced_value(self):
        pt = _new(10_000)
        pt.sync_equity(13_000.0)
        st = pt.get_state()
        assert st.equity == pytest.approx(13_000.0)
        # peak follows up so a fresh peak is reflected in drawdown denom.
        assert st.peak_equity == pytest.approx(13_000.0)
        assert st.current_drawdown_pct == 0.0

    def test_drawdown_correct_after_sync_then_loss(self):
        pt = _new(10_000)
        pt.sync_equity(12_000.0)  # peak → 12k
        pt.sync_equity(11_400.0)  # 5% drop
        st = pt.get_state()
        assert st.peak_equity == pytest.approx(12_000.0)
        assert st.current_drawdown_pct == pytest.approx(0.05)


class TestRecordEquityCallsSync:
    """End-to-end-ish: _record_equity feeds Nautilus equity into tracker."""

    def _strategy_with_position(self):
        from src.execution.strategies.ml_strategy import (
            MLStrategyConfig, MLTradingStrategy,
        )
        cfg = MLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
            warmup_bars=10, dry_run=True,
        )
        strat = MLTradingStrategy(config=cfg)
        strat._tracker = PortfolioTracker(initial_equity=10_000.0)
        return strat

    def test_record_equity_invokes_sync(self, monkeypatch):
        strat = self._strategy_with_position()
        # Stub Nautilus read via the testable helper.
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 11_500.0)

        calls: list[float] = []
        monkeypatch.setattr(
            strat._tracker, "sync_equity", lambda v: calls.append(float(v)),
        )

        strat._record_equity(ts_ns=1)
        assert calls == [11_500.0]

    def test_record_equity_fails_soft_when_nautilus_returns_none(self, monkeypatch):
        strat = self._strategy_with_position()
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: None)

        called: list[float] = []
        monkeypatch.setattr(
            strat._tracker, "sync_equity",
            lambda v: called.append(float(v)),
        )
        pre_cash = strat._tracker._cash
        strat._record_equity(ts_ns=1)
        assert called == []
        assert strat._tracker._cash == pre_cash
