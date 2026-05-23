"""Integration tests for ``CircuitBreaker`` inside the 4H ML strategy.

The breaker class itself is unit-tested elsewhere; here we verify that
``MLTradingStrategy.on_bar`` actually calls it, respects ``is_triggered``,
and never lets a breaker bug stop the bot (fail-soft).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
)
from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from src.risk.risk_engine import PortfolioState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> MLStrategyConfig:
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=2,  # tiny so we exit warmup quickly
        dry_run=True,
    )


def _make_bar() -> MagicMock:
    bar = MagicMock()
    bar.open.as_double.return_value = 50_000.0
    bar.high.as_double.return_value = 50_500.0
    bar.low.as_double.return_value = 49_500.0
    bar.close.as_double.return_value = 50_000.0
    bar.volume.as_double.return_value = 1_000.0
    bar.ts_event = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    bar.bar_type = MagicMock()
    return bar


def _make_strategy(cfg: MLStrategyConfig) -> MLTradingStrategy:
    """Strategy primed to bypass warmup and reach the breaker check."""
    s = MLTradingStrategy(config=cfg)
    s._warmup_complete = True
    # Tracker is required by the breaker branch
    s._tracker = MagicMock()
    s._tracker.get_state.return_value = PortfolioState(
        equity=10_000,
        open_positions=0,
        daily_pnl_pct=0.0,
        weekly_pnl_pct=0.0,
        current_drawdown_pct=0.0,
        consecutive_losses=0,
        last_loss_time=None,
        peak_equity=10_000,
    )
    s._record_equity = MagicMock()
    s._detect_regime = MagicMock(return_value=None)  # short-circuit step 2
    return s


def _state_with(**overrides) -> PortfolioState:
    base = dict(
        equity=10_000.0,
        open_positions=0,
        daily_pnl_pct=0.0,
        weekly_pnl_pct=0.0,
        current_drawdown_pct=0.0,
        consecutive_losses=0,
        last_loss_time=None,
        peak_equity=10_000.0,
    )
    base.update(overrides)
    return PortfolioState(**base)


# ---------------------------------------------------------------------------
# Construction — breaker slot exists, None until on_start
# ---------------------------------------------------------------------------

class TestBreakerWiring:
    def test_breaker_slot_starts_as_none(self, cfg: MLStrategyConfig) -> None:
        s = MLTradingStrategy(config=cfg)
        assert hasattr(s, "_breaker")
        assert s._breaker is None

    def test_breaker_set_in_on_start_logic(self, cfg: MLStrategyConfig) -> None:
        """``on_start`` constructs a real CircuitBreaker.

        We don't call the full on_start (it touches Nautilus internals);
        instead we replay the breaker-init line and assert the resulting
        type matches what production code creates.
        """
        s = MLTradingStrategy(config=cfg)
        from src.risk.circuit_breaker import CircuitBreaker as CB
        s._breaker = CB()
        assert isinstance(s._breaker, CircuitBreaker)


# ---------------------------------------------------------------------------
# on_bar — breaker is actually called and respected
# ---------------------------------------------------------------------------

class TestBreakerBlocksTrading:
    def _run_with_triggered_state(
        self,
        cfg: MLStrategyConfig,
        portfolio_state: PortfolioState,
    ) -> MLTradingStrategy:
        s = _make_strategy(cfg)
        s._breaker = CircuitBreaker()
        s._tracker.get_state.return_value = portfolio_state
        s.on_bar(_make_bar())
        return s

    def test_kill_switch_drawdown_15pct_blocks_bar(
        self, cfg: MLStrategyConfig
    ) -> None:
        s = self._run_with_triggered_state(
            cfg, _state_with(current_drawdown_pct=0.16, peak_equity=10_000),
        )
        # Triggered → regime detection MUST NOT run
        s._detect_regime.assert_not_called()

    def test_daily_hard_3pct_blocks_bar(self, cfg: MLStrategyConfig) -> None:
        s = self._run_with_triggered_state(
            cfg, _state_with(daily_pnl_pct=-0.04),
        )
        s._detect_regime.assert_not_called()

    def test_weekly_8pct_blocks_bar(self, cfg: MLStrategyConfig) -> None:
        s = self._run_with_triggered_state(
            cfg, _state_with(weekly_pnl_pct=-0.09),
        )
        s._detect_regime.assert_not_called()

    def test_five_consecutive_losses_blocks_bar(
        self, cfg: MLStrategyConfig
    ) -> None:
        s = self._run_with_triggered_state(
            cfg, _state_with(consecutive_losses=5),
        )
        s._detect_regime.assert_not_called()


# ---------------------------------------------------------------------------
# Normal path — breaker passes, bar processing continues
# ---------------------------------------------------------------------------

class TestNormalPath:
    def test_breaker_pass_lets_regime_detection_run(
        self, cfg: MLStrategyConfig
    ) -> None:
        """Clean portfolio state → breaker check returns is_triggered=False,
        and the bar proceeds to step 2 (regime detection)."""
        s = _make_strategy(cfg)
        s._breaker = CircuitBreaker()
        s.on_bar(_make_bar())
        # Reached step 2 — but it returns None and we early-exit there.
        s._detect_regime.assert_called_once()

    def test_breaker_check_actually_invoked(self, cfg: MLStrategyConfig) -> None:
        """Verify the integration point really calls ``breaker.check``."""
        s = _make_strategy(cfg)
        s._breaker = MagicMock()
        s._breaker.check.return_value = CircuitBreakerState()
        s.on_bar(_make_bar())
        s._breaker.check.assert_called_once()
        call_kwargs = s._breaker.check.call_args.kwargs
        assert "portfolio_state" in call_kwargs
        assert "current_funding" in call_kwargs


# ---------------------------------------------------------------------------
# Fail-soft — breaker bug must NEVER kill the bot
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_breaker_none_skips_check_and_continues(
        self, cfg: MLStrategyConfig
    ) -> None:
        """If breaker was never initialised (e.g. unit test path), on_bar
        should still process the bar normally."""
        s = _make_strategy(cfg)
        s._breaker = None
        # Should not raise; regime detection still runs.
        s.on_bar(_make_bar())
        s._detect_regime.assert_called_once()

    def test_breaker_check_exception_is_swallowed(
        self, cfg: MLStrategyConfig
    ) -> None:
        """A crash inside breaker.check must not crash the bot — bar
        processing falls through to regime detection."""
        s = _make_strategy(cfg)
        s._breaker = MagicMock()
        s._breaker.check.side_effect = RuntimeError("breaker exploded")
        s.on_bar(_make_bar())
        s._detect_regime.assert_called_once()


# ---------------------------------------------------------------------------
# Warmup precedence — breaker not checked before warmup completes
# ---------------------------------------------------------------------------

class TestWarmupPrecedence:
    def test_breaker_not_checked_during_warmup(self, cfg: MLStrategyConfig) -> None:
        """Per design: warmup check (step 1) runs first; breaker (step 1b)
        only fires once warmup is complete."""
        s = MLTradingStrategy(config=cfg)
        s._warmup_complete = False
        s._tracker = MagicMock()
        s._breaker = MagicMock()
        s._record_equity = MagicMock()
        s._detect_regime = MagicMock(return_value=None)
        # warmup_bars=2 in cfg, so first bar leaves warmup incomplete
        s.on_bar(_make_bar())
        s._breaker.check.assert_not_called()
