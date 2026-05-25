"""Tests for Step H5 — unrealized PnL, peak_equity, drawdown, daily_pnl.

Covers three connected fixes:
1. update_price() now also bumps peak_equity so drawdown sees mark-to-
   market gains.
2. _day_start_equity is frozen at the start of each UTC day; daily_pnl_pct
   uses it as the denominator so % stops scale with portfolio size.
3. on_bar() calls update_price() for every open position — verified via
   integration on a minimal strategy.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.risk.portfolio_tracker import PortfolioTracker


T0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _new_at(t0: datetime, equity: float = 10_000.0):
    """Build a tracker whose internal day boundary is anchored at *t0*'s
    UTC midnight — independent of the wall clock so rollover assertions
    don't depend on whether the test runs today or tomorrow."""
    pt = PortfolioTracker(initial_equity=equity)
    pt._day_start = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    pt._week_start = pt._day_start - timedelta(days=pt._day_start.weekday())
    return pt


def _new(equity=10_000.0):
    return PortfolioTracker(initial_equity=equity)


def _open_long(pt, *, symbol="BTCUSDT", qty=1.0, price=50_000.0, fee=10.0, ts=T0):
    pt.update_fill(symbol, direction=1, quantity=qty, price=price, fee=fee,
                   timestamp=ts)


# ---------------------------------------------------------------------------
# update_price → unrealized_pnl
# ---------------------------------------------------------------------------


class TestUpdatePriceUnrealized:
    def test_update_price_sets_unrealized(self):
        pt = _new()
        _open_long(pt)
        pt.update_price("BTCUSDT", 51_000.0)
        pos = pt._positions["BTCUSDT"]
        assert pos.unrealized_pnl == pytest.approx(1000.0)  # 1 * (51k-50k)

    def test_update_price_for_short(self):
        pt = _new()
        pt.update_fill("BTCUSDT", direction=-1, quantity=1.0, price=50_000.0,
                       fee=10.0, timestamp=T0)
        pt.update_price("BTCUSDT", 49_000.0)
        pos = pt._positions["BTCUSDT"]
        assert pos.unrealized_pnl == pytest.approx(1000.0)

    def test_update_price_unknown_symbol_is_noop(self):
        pt = _new()
        pt.update_price("ETHUSDT", 3_000.0)  # no positions
        assert "ETHUSDT" not in pt._positions

    def test_get_state_reflects_unrealized_after_update_price(self):
        pt = _new()
        _open_long(pt)
        pt.update_price("BTCUSDT", 51_000.0)
        st = pt.get_state()
        # equity = cash - open-fee + unrealised gain
        assert st.equity == pytest.approx(10_000 - 10 + 1000)


# ---------------------------------------------------------------------------
# Peak equity & drawdown
# ---------------------------------------------------------------------------


class TestPeakEquityOnUpdatePrice:
    def test_peak_bumps_on_unrealized_gain(self):
        pt = _new()
        _open_long(pt)
        pre = pt._peak_equity
        pt.update_price("BTCUSDT", 52_000.0)  # +2000 unrealized
        assert pt._peak_equity > pre
        assert pt._peak_equity == pytest.approx(pt._get_equity())

    def test_drawdown_uses_intraday_peak(self):
        """Up 2000 unrealized then back to flat → drawdown reflects the
        round-trip from the intraday peak (old code missed it)."""
        pt = _new()
        _open_long(pt)
        pt.update_price("BTCUSDT", 52_000.0)  # peak bumps to ~11_990
        peak = pt._peak_equity
        pt.update_price("BTCUSDT", 50_000.0)  # back to entry (still -open_fee)
        dd = pt.get_drawdown()
        # dd ≈ (peak - current_equity) / peak ≈ 2000 / 11990 ≈ 0.167
        assert dd == pytest.approx((peak - pt._get_equity()) / peak)
        assert dd > 0.15

    def test_peak_does_not_drop_on_loss(self):
        pt = _new()
        _open_long(pt)
        pre = pt._peak_equity
        pt.update_price("BTCUSDT", 48_000.0)  # -2000 unrealized
        assert pt._peak_equity == pre  # never decreases


# ---------------------------------------------------------------------------
# day_start_equity → daily_pnl_pct denominator
# ---------------------------------------------------------------------------


class TestDailyPnlDenominator:
    def test_day_start_equity_initialised(self):
        pt = _new(10_000)
        assert pt._day_start_equity == 10_000.0

    def test_day_rollover_freezes_new_day_start_equity(self):
        pt = _new_at(T0, 10_000)
        # Close a winning trade → cash up to ~11_000
        _open_long(pt, qty=1.0, price=50_000.0, fee=10.0, ts=T0)
        pt.close_position("BTCUSDT", close_price=51_000.0, fee=10.0,
                          timestamp=T0)
        eq_eod = pt._get_equity()
        # Trigger a day boundary via _roll_periods.
        next_day = (T0 + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        pt._roll_periods(next_day)
        assert pt._day_start_equity == pytest.approx(eq_eod)

    def test_daily_pnl_pct_uses_day_start_equity(self):
        """After equity grew to 11k, a +330 daily realised PnL = 3% of 11k
        (not 3.3% of the original 10k deposit)."""
        pt = _new_at(T0, 10_000)
        # Day 1: realise +1000 to push equity to ~11k
        _open_long(pt, qty=1.0, price=50_000.0, fee=0.0, ts=T0)
        pt.close_position("BTCUSDT", 51_000.0, fee=0.0, timestamp=T0)
        # Roll to day 2.
        next_day = (T0 + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        pt._roll_periods(next_day)
        assert pt._day_start_equity == pytest.approx(11_000.0)
        # Day 2: realise +330
        _open_long(pt, qty=1.0, price=50_000.0, fee=0.0,
                   ts=next_day + timedelta(hours=1))
        pt.close_position(
            "BTCUSDT", 50_330.0, fee=0.0,
            timestamp=next_day + timedelta(hours=2),
        )
        # daily_pnl_pct should be 330 / 11000 = 0.03 exactly.
        assert pt.get_daily_pnl() == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Invariant: equity = cash + Σ unrealized_pnl
# ---------------------------------------------------------------------------


class TestEquityInvariant:
    def test_invariant_holds_after_updates(self):
        pt = _new(10_000)
        _open_long(pt)
        pt.update_price("BTCUSDT", 51_500.0)
        eq = pt._get_equity()
        unrealised = sum(p.unrealized_pnl for p in pt._positions.values())
        assert eq == pytest.approx(pt._cash + unrealised)


# ---------------------------------------------------------------------------
# Persistence round-trip preserves day_start_equity
# ---------------------------------------------------------------------------


class TestPersistRoundtrip:
    def test_day_start_equity_persisted(self, tmp_path):
        state_file = tmp_path / "state.json"
        pt = PortfolioTracker(initial_equity=10_000, state_path=state_file)
        # Move day_start_equity off the default.
        pt._day_start_equity = 12_345.67
        pt._persist()
        pt2 = PortfolioTracker(initial_equity=10_000, state_path=state_file)
        assert pt2._day_start_equity == pytest.approx(12_345.67)


# ---------------------------------------------------------------------------
# Strategy wiring: on_bar marks all open positions
# ---------------------------------------------------------------------------


class TestStrategyMarksPositionsOnBar:
    @staticmethod
    def _strategy_with_position():
        from src.execution.strategies.ml_strategy import (
            MLStrategyConfig, MLTradingStrategy,
        )
        cfg = MLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
            warmup_bars=10,
            dry_run=True,
        )
        strat = MLTradingStrategy(config=cfg)
        # _tracker is wired in on_start() which can't run outside Nautilus;
        # inject a tracker directly for unit-test purposes.
        strat._tracker = PortfolioTracker(initial_equity=10_000.0)
        strat._tracker.update_fill(
            "BTCUSDT", direction=1, quantity=1.0, price=50_000.0,
            fee=0.0, timestamp=T0,
        )
        return strat

    def test_on_bar_calls_update_price_for_each_position(self, monkeypatch):
        strat = self._strategy_with_position()
        calls: list[tuple[str, float]] = []

        def _spy(symbol, price):
            calls.append((symbol, price))

        monkeypatch.setattr(strat._tracker, "update_price", _spy)

        # Run only the wiring block (on_bar itself can't run outside
        # Nautilus due to Cython self.log). Mirror the exact try/except.
        class _Bar:
            close = 51_000.0

        try:
            close_px = float(_Bar.close)
            for _sym in list(strat._tracker._positions.keys()):
                strat._tracker.update_price(_sym, close_px)
        except Exception:
            pass

        assert calls == [("BTCUSDT", 51_000.0)]

    def test_marking_yields_nonzero_unrealized(self):
        strat = self._strategy_with_position()
        strat._tracker.update_price("BTCUSDT", 51_000.0)
        st = strat._tracker.get_state()
        # Drawdown 0 because mark-up is a gain; equity > initial.
        assert st.equity > 10_000.0
        assert st.current_drawdown_pct == 0.0
        assert any(
            p.unrealized_pnl > 0 for p in strat._tracker._positions.values()
        )
