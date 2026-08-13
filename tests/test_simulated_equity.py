"""Tests for PR-C — simulated equity in modes where orders are not sent.

Background (incident 2026-08-09)
--------------------------------
``run_live.py --mode paper --dry-run`` connected to a mainnet futures
account holding 0 USDT. ``_record_equity`` synced that 0 into the
tracker while ``peak_equity`` stayed at the configured 10_000, so
``get_drawdown()`` returned 1.0 and the circuit breaker fired a
100.00% drawdown kill switch on the very first bar — before regime
detection, before features, before any model confidence was measured.

Contract established here
-------------------------
``dry_run`` is the switch: it is the same flag that gates order
submission (``on_bar``: ``if not self._config.dry_run``). When it is
True the bot cannot move the exchange balance, so the exchange balance
must not move the bot's risk decisions. When it is False (testnet /
live — real orders) the exchange stays authoritative, exactly as before.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
)
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.portfolio_tracker import PortfolioTracker


INITIAL_EQUITY = 10_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(dry_run: bool, trading_mode: str = "paper") -> MLStrategyConfig:
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=INITIAL_EQUITY,
        warmup_bars=2,
        dry_run=dry_run,
        trading_mode=trading_mode,
    )


def _strategy(dry_run: bool, trading_mode: str = "paper") -> MLTradingStrategy:
    """Strategy with a REAL PortfolioTracker (not a mock).

    The existing tests/test_circuit_breaker_integration.py stubs both
    ``_record_equity`` and ``_tracker``, which is precisely why the
    incident slipped through: the equity → drawdown → breaker chain was
    never exercised end to end. Here it is.
    """
    strat = MLTradingStrategy(config=_cfg(dry_run, trading_mode))
    strat._tracker = PortfolioTracker(initial_equity=INITIAL_EQUITY)
    # Never touch the network from a unit test.
    strat._fetch_taker_buy_volume_for_bar = MagicMock(return_value=None)
    return strat


def _make_bar() -> MagicMock:
    bar = MagicMock()
    bar.open.as_double.return_value = 50_000.0
    bar.high.as_double.return_value = 50_500.0
    bar.low.as_double.return_value = 49_500.0
    bar.close.as_double.return_value = 50_000.0
    bar.volume.as_double.return_value = 1_000.0
    bar.ts_event = int(
        datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9
    )
    bar.bar_type = MagicMock()
    return bar


def _primed_for_on_bar(strat: MLTradingStrategy) -> MLTradingStrategy:
    """Bypass warmup and short-circuit step 2 so the breaker gate is the
    only thing that can stop the bar."""
    strat._warmup_complete = True
    strat._breaker = CircuitBreaker()
    strat._detect_regime = MagicMock(return_value=None)
    return strat


# ---------------------------------------------------------------------------
# Group 1 — RED before the fix
# ---------------------------------------------------------------------------

class TestSimulatedModeSkipsExchangeEquity:
    """Test 1 — the exchange value must never reach the tracker."""

    def test_dry_run_does_not_sync_exchange_equity(self, monkeypatch):
        strat = _strategy(dry_run=True)
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 0.0)

        calls: list[float] = []
        monkeypatch.setattr(
            strat._tracker, "sync_equity", lambda v: calls.append(float(v)),
        )

        strat._record_equity(ts_ns=1)

        assert calls == []
        assert strat._tracker._get_equity() == pytest.approx(INITIAL_EQUITY)
        assert strat._tracker.get_drawdown() == pytest.approx(0.0)


class TestIncidentReproduction:
    """Test 2 — the incident itself: simulated mode + 0 USDT balance
    must NOT trip the circuit breaker, and the bar must reach step 2."""

    def test_dry_run_zero_balance_does_not_trip_breaker(self, monkeypatch):
        strat = _primed_for_on_bar(_strategy(dry_run=True))
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 0.0)

        strat.on_bar(_make_bar())

        # Reached step 2 — the breaker did not swallow the bar.
        strat._detect_regime.assert_called_once()
        assert strat._tracker.get_drawdown() == pytest.approx(0.0)


class TestEquityCurveInSimulatedMode:
    """Test 4 — the curve records the tracker's own equity, not the
    exchange's, so the curve and the risk layer never disagree."""

    def test_dry_run_equity_curve_records_tracker_equity(self, monkeypatch):
        strat = _strategy(dry_run=True)
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 0.0)

        strat._record_equity(ts_ns=7)

        assert strat._equity_curve == [(7, pytest.approx(INITIAL_EQUITY))]

    def test_dry_run_equity_curve_survives_missing_tracker(self, monkeypatch):
        """_tracker is created in on_start; unit paths may leave it None."""
        strat = _strategy(dry_run=True)
        strat._tracker = None
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 0.0)

        strat._record_equity(ts_ns=7)  # must not raise

        assert strat._equity_curve == []


class TestExchangeNotConsulted:
    """Test 5 — in simulated mode the exchange is not read at all."""

    def test_dry_run_does_not_read_exchange_at_all(self, monkeypatch):
        def _boom():
            raise RuntimeError("exchange must not be consulted in dry-run")

        strat = _strategy(dry_run=True)
        monkeypatch.setattr(strat, "_read_nautilus_equity", _boom)

        strat._record_equity(ts_ns=1)  # must not raise

        assert strat._equity_curve == [(1, pytest.approx(INITIAL_EQUITY))]


class TestOneShotNotices:
    """Tests 7 and 8 — both notices fire once per process, not per bar."""

    def test_simulated_equity_notice_logged_once(self, monkeypatch):
        strat = _strategy(dry_run=True)
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 0.0)

        assert strat._simulated_equity_logged is False

        strat._record_equity(ts_ns=1)
        assert strat._simulated_equity_logged is True

        strat._record_equity(ts_ns=2)
        assert strat._simulated_equity_logged is True
        # Flag guards the log line only — the method still runs in full.
        assert len(strat._equity_curve) == 2

    @pytest.mark.parametrize("balance", [0.0, 1e-12])
    def test_zero_balance_error_logged_once_in_live_mode(self, balance, monkeypatch):
        """1e-12 is an effective zero: it still yields a 99.999…% drawdown,
        so an exact ``== 0.0`` check would stay silent where it matters most."""
        strat = _strategy(dry_run=False, trading_mode="testnet")
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: balance)

        assert strat._zero_balance_logged is False

        strat._record_equity(ts_ns=1)
        assert strat._zero_balance_logged is True

        strat._record_equity(ts_ns=2)
        assert strat._zero_balance_logged is True
        assert len(strat._equity_curve) == 2

    def test_zero_balance_error_not_logged_for_healthy_balance(self, monkeypatch):
        strat = _strategy(dry_run=False, trading_mode="testnet")
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 9_800.0)

        strat._record_equity(ts_ns=1)

        assert strat._zero_balance_logged is False


# ---------------------------------------------------------------------------
# Group 2 — GREEN before AND after (guards against an over-broad fix)
# ---------------------------------------------------------------------------

class TestRealModesUnchanged:
    """Test 3 — testnet / live keep today's behaviour exactly."""

    @pytest.mark.parametrize("mode", ["testnet", "live"])
    def test_live_mode_zero_balance_still_trips_breaker(self, mode, monkeypatch):
        strat = _primed_for_on_bar(_strategy(dry_run=False, trading_mode=mode))
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 0.0)

        strat.on_bar(_make_bar())

        # Kill switch fires → bar never reaches regime detection.
        strat._detect_regime.assert_not_called()
        assert strat._tracker.get_drawdown() == pytest.approx(1.0)

    @pytest.mark.parametrize("mode", ["testnet", "live"])
    def test_not_dry_run_still_syncs(self, mode, monkeypatch):
        """Test 6 — H6 invariant: Nautilus stays authoritative when the
        bot actually sends orders."""
        strat = _strategy(dry_run=False, trading_mode=mode)
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: 11_500.0)

        calls: list[float] = []
        monkeypatch.setattr(
            strat._tracker, "sync_equity", lambda v: calls.append(float(v)),
        )

        strat._record_equity(ts_ns=1)

        assert calls == [11_500.0]
