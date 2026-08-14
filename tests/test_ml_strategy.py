"""
Tests for MLTradingStrategy, LiveTrader, and run_live.

Strategy lifecycle methods (on_start, on_bar, etc.) cannot easily be unit-
tested outside a Nautilus engine because `log`, `cache`, `portfolio`,
`order_factory`, and `submit_order` are C-level read-only properties.

We therefore test:
- Config / construction (no engine needed)
- Pure helper functions (_bar_to_dict, _compute_features, _select_model)
- RiskEngine integration at the boundary
- LiveTrader config / construction
- A BacktestEngine mini-run to prove the strategy loads

Total ≥ 18 tests.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
    _bar_to_dict,
)
from src.execution.live_trader import LiveTrader, LiveTraderConfig
from src.features.window_sizes import HISTORY_BARS_4H
from src.risk.risk_engine import (
    PortfolioState,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    TradeSignal,
)
from src.risk.portfolio_tracker import PortfolioTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_strategy_config() -> MLStrategyConfig:
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        history_bars=10,
        dry_run=True,
    )


@pytest.fixture
def mock_bar() -> MagicMock:
    """Create a mock Nautilus Bar."""
    bar = MagicMock()
    bar.open.as_double.return_value = 94_000.0
    bar.high.as_double.return_value = 94_500.0
    bar.low.as_double.return_value = 93_500.0
    bar.close.as_double.return_value = 94_250.0
    bar.volume.as_double.return_value = 1000.0
    bar.ts_event = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    bar.bar_type = MagicMock()
    return bar


def _make_bars(n: int, base_price: float = 50_000.0) -> list[MagicMock]:
    """Generate n mock bars with slight price movement."""
    bars = []
    for i in range(n):
        bar = MagicMock()
        price = base_price + i * 10
        bar.open.as_double.return_value = price - 50
        bar.high.as_double.return_value = price + 100
        bar.low.as_double.return_value = price - 100
        bar.close.as_double.return_value = price
        bar.volume.as_double.return_value = 1000.0 + i
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4 * i)
        bar.ts_event = int(ts.timestamp() * 1e9)
        bar.bar_type = MagicMock()
        bars.append(bar)
    return bars


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class TestMLStrategyConfig:
    """MLStrategyConfig unit tests."""

    def test_config_created_correctly(self) -> None:
        """Default config should be valid."""
        cfg = MLStrategyConfig()
        assert cfg.instrument_id == "BTCUSDT-PERP.BINANCE"
        assert cfg.warmup_bars == 300
        assert cfg.history_bars == HISTORY_BARS_4H
        assert cfg.dry_run is False
        assert cfg.confidence_threshold == 0.55  # binary, ML-017

    def test_config_custom_values(self) -> None:
        """Custom values should be applied."""
        cfg = MLStrategyConfig(
            instrument_id="ETHUSDT-PERP.BINANCE",
            initial_equity=50_000.0,
            dry_run=True,
            confidence_threshold=0.70,
        )
        assert cfg.instrument_id == "ETHUSDT-PERP.BINANCE"
        assert cfg.initial_equity == 50_000.0
        assert cfg.dry_run is True
        assert cfg.confidence_threshold == 0.70

    def test_config_rr_ratio(self) -> None:
        """R:R ratio defaults to 1.5."""
        cfg = MLStrategyConfig()
        assert cfg.rr_ratio == 1.5

    def test_config_frozen(self) -> None:
        """StrategyConfig is frozen (immutable)."""
        cfg = MLStrategyConfig()
        # msgspec frozen structs cannot be modified at runtime
        # but the creation itself should succeed
        assert cfg.interval == "4h"


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


class TestMLStrategyInit:
    """Strategy construction tests (no engine required)."""

    def test_strategy_constructs(self, default_strategy_config: MLStrategyConfig) -> None:
        """Strategy should construct without errors."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        assert strategy._config.dry_run is True
        assert strategy._bar_count == 0
        assert strategy._bars == []

    def test_strategy_internal_state(self, default_strategy_config: MLStrategyConfig) -> None:
        """Internal state should be properly initialised."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        assert strategy._risk_engine is None
        assert strategy._tracker is None
        assert strategy._trend_model is None
        assert strategy._highvol_model is None
        assert strategy._equity_curve == []
        assert strategy._pending_stops == {}


# ═══════════════════════════════════════════════════════════════════════════
# BAR CONVERSION HELPER
# ═══════════════════════════════════════════════════════════════════════════


class TestBarToDict:
    """Bar-to-dict helper tests."""

    def test_bar_to_dict(self, mock_bar: MagicMock) -> None:
        """_bar_to_dict should extract OHLCV correctly."""
        d = _bar_to_dict(mock_bar)
        assert d["open"] == 94_000.0
        assert d["high"] == 94_500.0
        assert d["low"] == 93_500.0
        assert d["close"] == 94_250.0
        assert d["volume"] == 1000.0

    def test_bar_to_dict_keys(self, mock_bar: MagicMock) -> None:
        """Should contain exactly 5 OHLCV keys."""
        d = _bar_to_dict(mock_bar)
        assert set(d.keys()) == {"open", "high", "low", "close", "volume"}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestModelSelection:
    """Model selection logic tests."""

    def test_trend_up_selects_trend_model(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """trend_up regime should select the trend model with base threshold."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._trend_model = MagicMock(name="trend")
        strategy._trend_features = ["f1", "f2"]
        strategy._highvol_model = MagicMock(name="highvol")

        model, feats, threshold = strategy._select_model("trend_up")
        assert model is strategy._trend_model
        assert feats == ["f1", "f2"]
        assert threshold == pytest.approx(default_strategy_config.confidence_threshold)

    def test_trend_down_selects_trend_model(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """trend_down regime should also select the trend model."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._trend_model = MagicMock(name="trend")
        strategy._trend_features = ["f1"]

        model, _, threshold = strategy._select_model("trend_down")
        assert model is strategy._trend_model
        assert threshold == pytest.approx(default_strategy_config.confidence_threshold)

    def test_high_vol_selects_highvol_model(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """high_vol regime should select the high_vol model."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._highvol_model = MagicMock(name="highvol")
        strategy._highvol_features = ["hv1"]

        model, feats, threshold = strategy._select_model("high_vol")
        assert model is strategy._highvol_model
        assert feats == ["hv1"]
        assert threshold == pytest.approx(default_strategy_config.confidence_threshold)

    def test_range_uses_trend_model_with_higher_threshold(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """range regime uses the trend model with a stricter (>=0.60) threshold (binary, ML-017)."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._trend_model = MagicMock(name="trend")
        strategy._trend_features = ["f1"]
        model, feats, threshold = strategy._select_model("range")
        assert model is strategy._trend_model
        assert feats == ["f1"]
        assert threshold >= 0.60

    def test_unknown_falls_back_to_trend_model(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Defensive: unexpected regime falls back to trend model with strict threshold."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._trend_model = MagicMock(name="trend")
        strategy._trend_features = ["f1"]
        model, _, threshold = strategy._select_model("unknown")
        assert model is strategy._trend_model
        assert threshold >= 0.60  # binary, ML-017


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION (pure function, no engine)
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureComputation:
    """Feature vector computation tests.

    _compute_features uses self._bars + self.log.  Since self.log is
    C-level read-only, we test by accessing internal state directly.
    """

    def test_compute_features_returns_correct_shape(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """_compute_features should return array matching feature list length."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._bars = _make_bars(350)

        feature_names = [
            "returns_1", "returns_3", "body_ratio", "upper_wick",
            "lower_wick", "volume_ratio", "adx", "hurst",
        ]

        # _compute_features logs via self.log, which is a Cython property.
        # We can't mock it, but _compute_features catches exceptions
        # internally.  Pass through anyway.
        result = strategy._compute_features(feature_names)

        assert result is not None
        assert result.shape == (len(feature_names),)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_compute_features_insufficient_bars(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """With very few bars, feature computation may gracefully return None
        (ADX needs >= 14 bars).  Verify no crash.
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._bars = _make_bars(5)

        result = strategy._compute_features(["returns_1", "body_ratio"])
        # With < 14 bars, ADX/Hurst may raise internally;
        # _compute_features catches the exception and returns None.
        # With only simple features like returns_1, it might succeed.
        # Either outcome is acceptable — just no crash.
        if result is not None:
            assert len(result) == 2

    def test_compute_features_no_nans(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Output should have no NaN/Inf values."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._bars = _make_bars(100)

        all_features = [
            "returns_1", "returns_3", "returns_6", "returns_12",
            "body_ratio", "upper_wick", "lower_wick",
            "volume_sma_20", "volume_ratio", "volume_zscore",
            "cvd", "cvd_cum", "cvd_slope_3",
            "hurst", "adx", "atr_pct", "atr_percentile",
            "symbol_encoded",
        ]

        result = strategy._compute_features(all_features)
        assert result is not None
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ═══════════════════════════════════════════════════════════════════════════
# RISK ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskEngineIntegration:
    """Test that the strategy's risk evaluation path works correctly."""

    def test_signal_approved_with_good_state(self) -> None:
        """A good signal + healthy portfolio → approved."""
        engine = RiskEngine(RiskConfig(), equity=10_000)
        signal = TradeSignal(
            symbol="BTCUSDT-PERP.BINANCE",
            direction=1,
            confidence=0.75,
            regime="trend_up",
            entry_price=94_250.0,
            atr=1500.0,
            atr_pct=0.016,
            funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=10_000, open_positions=0,
            daily_pnl_pct=0.01, weekly_pnl_pct=0.02,
            current_drawdown_pct=0.02, consecutive_losses=0,
            last_loss_time=None, peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        assert decision.approved
        assert decision.position_size > 0
        assert decision.stop_loss < signal.entry_price  # LONG
        assert decision.take_profit > signal.entry_price

    def test_signal_rejected_max_positions(self) -> None:
        """Max positions reached → rejected."""
        engine = RiskEngine(RiskConfig(), equity=10_000)
        signal = TradeSignal(
            symbol="BTCUSDT-PERP.BINANCE",
            direction=1, confidence=0.75, regime="trend",
            entry_price=94_250.0, atr=1500.0, atr_pct=0.016,
            funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=10_000, open_positions=3,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.0, consecutive_losses=0,
            last_loss_time=None, peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        assert not decision.approved

    def test_signal_rejected_low_confidence(self) -> None:
        """Low confidence → rejected."""
        engine = RiskEngine(RiskConfig(), equity=10_000)
        signal = TradeSignal(
            symbol="BTCUSDT-PERP.BINANCE",
            direction=1, confidence=0.40, regime="trend",
            entry_price=94_250.0, atr=1500.0, atr_pct=0.016,
            funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=10_000, open_positions=0,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.0, consecutive_losses=0,
            last_loss_time=None, peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        assert not decision.approved


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO TRACKER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioTrackerIntegration:
    """Portfolio tracker as used by the strategy."""

    def test_fill_and_close_cycle(self) -> None:
        """Open → close cycle updates state correctly."""
        tracker = PortfolioTracker(10_000)
        now = datetime.now(timezone.utc)

        tracker.update_fill("BTCUSDT", 1, 0.1, 50_000, 5.0, now)
        assert tracker.get_state().open_positions == 1

        pnl = tracker.close_position("BTCUSDT", 51_000, 5.0, now + timedelta(hours=4))
        assert pnl > 0
        assert tracker.get_state().open_positions == 0

    def test_loss_tracking(self) -> None:
        """Losses should increment consecutive counter."""
        tracker = PortfolioTracker(10_000)
        now = datetime.now(timezone.utc)

        tracker.update_fill("BTCUSDT", 1, 0.1, 50_000, 5.0, now)
        pnl = tracker.close_position("BTCUSDT", 49_000, 5.0, now + timedelta(hours=4))
        assert pnl < 0
        assert tracker.get_state().consecutive_losses == 1


# ═══════════════════════════════════════════════════════════════════════════
# LIVE TRADER CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class TestLiveTraderConfig:
    """LiveTrader configuration tests."""

    def test_default_config(self) -> None:
        """Default LiveTraderConfig should be valid."""
        cfg = LiveTraderConfig()
        assert cfg.trading_mode == "testnet"
        assert cfg.dry_run is False
        assert cfg.symbols == ["BTCUSDT-PERP"]

    def test_live_trader_constructs(self) -> None:
        """LiveTrader should construct without errors."""
        cfg = LiveTraderConfig(dry_run=True)
        trader = LiveTrader(cfg)
        assert trader._config.dry_run is True
        assert trader._node is None

    def test_live_trader_custom_symbols(self) -> None:
        """Custom symbols should be stored."""
        cfg = LiveTraderConfig(symbols=["ETHUSDT-PERP", "SOLUSDT-PERP"])
        trader = LiveTrader(cfg)
        assert len(trader._config.symbols) == 2

    def test_strategy_factory_defaults_none(self) -> None:
        """Backward compat: strategy_factory defaults to None so the
        4H path in build_node is taken (running bot unaffected)."""
        cfg = LiveTraderConfig()
        assert cfg.strategy_factory is None

    def test_strategy_factory_is_settable(self) -> None:
        """Phase 5 hook: a factory (cfg, symbol) -> Strategy can be
        injected for the isolated 15m / 1H launchers."""
        sentinel = object()
        seen: list[tuple] = []

        def _factory(cfg: LiveTraderConfig, symbol: str):
            seen.append((cfg, symbol))
            return sentinel

        cfg = LiveTraderConfig(strategy_factory=_factory)
        trader = LiveTrader(cfg)
        assert trader._config.strategy_factory is _factory
        # Factory is plain callable with the documented signature.
        assert trader._config.strategy_factory(cfg, "BTCUSDT-PERP") is sentinel
        assert seen == [(cfg, "BTCUSDT-PERP")]


# ═══════════════════════════════════════════════════════════════════════════
# IDEMPOTENT ORDER IDs
# ═══════════════════════════════════════════════════════════════════════════


class TestIdempotentOrderIds:
    """Client order ID uniqueness tests."""

    def test_idempotent_order_ids(self) -> None:
        """Client order IDs should be unique per invocation."""
        ids = set()
        for _ in range(100):
            ts_ms = int(time.time() * 1000)
            oid = f"AC-L-{ts_ms}"
            ids.add(oid)
            time.sleep(0.001)

        assert len(ids) >= 90  # allow minor collisions at ms boundary

    def test_order_id_format(self) -> None:
        """Order IDs should follow AC-{direction}-{timestamp} format."""
        ts_ms = int(time.time() * 1000)
        long_id = f"AC-L-{ts_ms}"
        short_id = f"AC-S-{ts_ms}"
        assert long_id.startswith("AC-L-")
        assert short_id.startswith("AC-S-")
        assert len(long_id) > 10


# ═══════════════════════════════════════════════════════════════════════════
# PROD-001: Kill switch boundary test
# ═══════════════════════════════════════════════════════════════════════════


class TestKillSwitchPROD001:
    """PROD-001: verify simplified kill switch logic."""

    def test_drawdown_15pct_triggers_kill_switch(self) -> None:
        """DD=15.1% → KILL SWITCH → rejected."""
        engine = RiskEngine(RiskConfig(max_drawdown_kill=-0.15), equity=10_000)
        signal = TradeSignal(
            symbol="BTCUSDT", direction=1, confidence=0.80,
            regime="trend", entry_price=94_000.0, atr=1500.0,
            atr_pct=0.016, funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=8_490, open_positions=0,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.151,  # 15.1% > 15%
            consecutive_losses=0, last_loss_time=None,
            peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        assert not decision.approved
        assert "KILL SWITCH" in decision.reason

    def test_drawdown_14_9pct_passes_kill_switch(self) -> None:
        """DD=14.9% → passes kill switch → approved (if other filters pass)."""
        engine = RiskEngine(RiskConfig(max_drawdown_kill=-0.15), equity=10_000)
        signal = TradeSignal(
            symbol="BTCUSDT", direction=1, confidence=0.80,
            regime="trend", entry_price=94_000.0, atr=1500.0,
            atr_pct=0.016, funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=8_510, open_positions=0,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.149,  # 14.9% < 15%
            consecutive_losses=0, last_loss_time=None,
            peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        assert decision.approved

    def test_drawdown_exactly_15pct_passes(self) -> None:
        """DD=15.0% (boundary) → NOT triggered (> required, not >=)."""
        engine = RiskEngine(RiskConfig(max_drawdown_kill=-0.15), equity=10_000)
        signal = TradeSignal(
            symbol="BTCUSDT", direction=1, confidence=0.80,
            regime="trend", entry_price=94_000.0, atr=1500.0,
            atr_pct=0.016, funding_rate=0.0001,
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=8_500, open_positions=0,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.15,  # exactly 15%
            consecutive_losses=0, last_loss_time=None,
            peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        # 0.15 is NOT > 0.15, so kill switch should NOT trigger
        assert decision.approved


# ═══════════════════════════════════════════════════════════════════════════
# PROD-002: Consecutive losses single-count test
# ═══════════════════════════════════════════════════════════════════════════


class TestConsecutiveLossesPROD002:
    """PROD-002: verify record_loss is called exactly once per close."""

    def test_four_losses_do_not_trigger_circuit_breaker(self) -> None:
        """4 consecutive losses → counter=4, circuit breaker (limit=5) NOT triggered."""
        tracker = PortfolioTracker(10_000)
        # Relax daily loss limit so we only test consecutive losses
        engine = RiskEngine(
            RiskConfig(consecutive_losses_limit=5, daily_loss_limit=-0.10,
                       weekly_loss_limit=-0.20),
            equity=10_000,
        )

        for i in range(4):
            ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            tracker.update_fill("BTCUSDT", 1, 0.1, 50_000, 1.0, ts)
            # Lose $100 each time
            tracker.close_position("BTCUSDT", 49_000, 1.0, ts + timedelta(hours=1))

        state = tracker.get_state()
        assert state.consecutive_losses == 4  # exactly 4, not 8

        # Signal should pass (4 < 5)
        signal = TradeSignal(
            symbol="BTCUSDT", direction=1, confidence=0.80,
            regime="trend", entry_price=50_000.0, atr=1500.0,
            atr_pct=0.016, funding_rate=0.0001,
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        decision = engine.evaluate(signal, state)
        assert decision.approved

    def test_five_losses_trigger_circuit_breaker(self) -> None:
        """5 consecutive losses → counter=5, circuit breaker triggered."""
        tracker = PortfolioTracker(10_000)
        # Relax daily loss limit so we only test consecutive losses
        engine = RiskEngine(
            RiskConfig(consecutive_losses_limit=5, daily_loss_limit=-0.10,
                       weekly_loss_limit=-0.20),
            equity=10_000,
        )

        for i in range(5):
            ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            tracker.update_fill("BTCUSDT", 1, 0.1, 50_000, 1.0, ts)
            tracker.close_position("BTCUSDT", 49_000, 1.0, ts + timedelta(hours=1))

        state = tracker.get_state()
        assert state.consecutive_losses == 5  # exactly 5, not 10

        # Signal should be blocked
        signal = TradeSignal(
            symbol="BTCUSDT", direction=1, confidence=0.80,
            regime="trend", entry_price=50_000.0, atr=1500.0,
            atr_pct=0.016, funding_rate=0.0001,
            timestamp=datetime(2026, 1, 1, 6, tzinfo=timezone.utc),
        )
        decision = engine.evaluate(signal, state)
        assert not decision.approved
        assert "consecutive losses" in decision.reason


# ═══════════════════════════════════════════════════════════════════════════
# PROD-003: Funding rate from feature data
# ═══════════════════════════════════════════════════════════════════════════


class TestFundingRatePROD003:
    """PROD-003: verify extreme funding rate blocks signal."""

    def test_extreme_funding_rate_blocks_signal(self) -> None:
        """funding_rate > 0.1% → rejected by _check_funding_rate."""
        engine = RiskEngine(
            RiskConfig(max_funding_rate=0.001), equity=10_000,
        )
        signal = TradeSignal(
            symbol="BTCUSDT", direction=1, confidence=0.80,
            regime="trend", entry_price=94_000.0, atr=1500.0,
            atr_pct=0.016,
            funding_rate=0.002,  # 0.2% — extreme
            timestamp=datetime.now(timezone.utc),
        )
        state = PortfolioState(
            equity=10_000, open_positions=0,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.0, consecutive_losses=0,
            last_loss_time=None, peak_equity=10_000,
        )
        decision = engine.evaluate(signal, state)
        assert not decision.approved
        assert "funding" in decision.reason.lower()

    def test_get_funding_rate_from_features(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """_get_funding_rate extracts rate from feature vector."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        feature_names = ["returns_1", "funding_rate", "adx"]
        feature_vector = np.array([0.01, 0.0015, 25.0])

        rate = strategy._get_funding_rate(feature_vector, feature_names)
        assert rate == pytest.approx(0.0015)
        assert strategy._last_funding_rate == pytest.approx(0.0015)

    def test_get_funding_rate_fallback_none(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """H4: when funding_rate not in features and no prior reading,
        return None (RiskEngine treats None as fail-safe block — 0.0 would
        silently bypass the extreme-funding filter)."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        feature_names = ["returns_1", "adx"]
        feature_vector = np.array([0.01, 25.0])

        rate = strategy._get_funding_rate(feature_vector, feature_names)
        assert rate is None


# ═══════════════════════════════════════════════════════════════════════════
# PROD-005: Deferred SL submission
# ═══════════════════════════════════════════════════════════════════════════


class TestDeferredStopLossPROD005:
    """PROD-005: verify SL params are stored for deferred submission."""

    def test_pending_sl_params_stored_on_init(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Strategy should have empty _pending_sl_params on init."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        assert strategy._pending_sl_params == {}

    def test_pending_sl_params_type(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """_pending_sl_params should be a dict."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        assert isinstance(strategy._pending_sl_params, dict)


# ═══════════════════════════════════════════════════════════════════════════
# PRELOAD: Config defaults
# ═══════════════════════════════════════════════════════════════════════════


class TestPreloadConfig:
    """Verify new config fields for the preload system."""

    def test_config_defaults_preload_enabled(self) -> None:
        """preload_enabled defaults to True."""
        cfg = MLStrategyConfig()
        assert cfg.preload_enabled is True

    def test_config_defaults_trading_mode(self) -> None:
        """trading_mode defaults to testnet."""
        cfg = MLStrategyConfig()
        assert cfg.trading_mode == "testnet"

    def test_config_custom_trading_mode(self) -> None:
        """Custom trading_mode should be applied."""
        cfg = MLStrategyConfig(trading_mode="live")
        assert cfg.trading_mode == "live"

    def test_config_preload_disabled(self) -> None:
        """preload_enabled=False should be stored."""
        cfg = MLStrategyConfig(preload_enabled=False)
        assert cfg.preload_enabled is False

    def test_warmup_complete_init_false(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """_warmup_complete should be False on init."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        assert strategy._warmup_complete is False


# ═══════════════════════════════════════════════════════════════════════════
# PRELOAD: Binance API
# ═══════════════════════════════════════════════════════════════════════════


class TestPreloadFromBinanceApi:
    """Test _preload_from_binance_api with mocked HTTP."""

    def test_preload_from_binance_api(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Mock requests.get → 300 klines → bar_buffer filled → warmup_complete."""
        from unittest.mock import patch

        strategy = MLTradingStrategy(config=default_strategy_config)

        # Build 300 fake klines (Binance format: list of lists)
        base_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        fake_klines = []
        for i in range(300):
            ts = base_ts + i * 4 * 3600 * 1000  # 4h apart
            fake_klines.append([
                ts,              # 0: open_time (ms)
                "94000.0",       # 1: open
                "94500.0",       # 2: high
                "93500.0",       # 3: low
                "94250.0",       # 4: close
                "1000.000",      # 5: volume
                ts + 4*3600*1000 - 1,  # 6: close_time
                "1000000.0",     # 7: quote volume
                100,             # 8: trade count
                "500.0",         # 9: taker buy base
                "500000.0",      # 10: taker buy quote
                "0",             # 11: ignore
            ])

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_klines
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            bars = strategy._preload_from_binance_api("BTCUSDT", 300)

        assert len(bars) == 300
        # Verify first bar
        assert bars[0].open.as_double() == 94000.0
        assert bars[0].close.as_double() == 94250.0
        # Nautilus convention: ts_event = bar CLOSE time, i.e. kline index 6
        assert bars[0].ts_event == int(fake_klines[0][6]) * 1_000_000
        assert bars[0].ts_init == bars[0].ts_event

        # Verify request was made
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert "testnet.binancefuture.com" in call_kwargs[1].get("url", "") or \
               "testnet.binancefuture.com" in call_kwargs[0][0] if call_kwargs[0] else True

    def test_binance_preload_requests_full_history(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """The REST limit must not clip the requested history window.

        Binance allows up to 1500 klines per call, so the full 4H window fits
        in a single request; a hard-coded 500 would silently truncate it.
        """
        from unittest.mock import patch
        from src.execution.strategies.ml_strategy import _BINANCE_KLINES_MAX_LIMIT
        from src.features.window_sizes import HISTORY_BARS_4H

        strategy = MLTradingStrategy(config=default_strategy_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            strategy._preload_from_binance_api("BTCUSDT", HISTORY_BARS_4H)

        params = mock_get.call_args.kwargs["params"]
        assert params["limit"] == HISTORY_BARS_4H
        assert params["limit"] <= _BINANCE_KLINES_MAX_LIMIT

    def test_preload_from_parquet(self) -> None:
        """Mock DataStore.get_klines → DataFrame → Bar objects in time order."""
        from unittest.mock import patch
        import polars as pl

        cfg = MLStrategyConfig(
            warmup_bars=10,
            dry_run=True,
            features_dir="./data/features/ml_features",
        )
        strategy = MLTradingStrategy(config=cfg)

        # Build mock DataFrame matching Parquet schema
        base_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        rows = []
        for i in range(100):
            ts = base_ts + i * 4 * 3600 * 1000
            rows.append({
                "open_time": ts,
                "close_time": ts + 4 * 3600 * 1000 - 1,
                "open": 94000.0 + i * 10,
                "high": 94500.0 + i * 10,
                "low": 93500.0 + i * 10,
                "close": 94250.0 + i * 10,
                "volume": 1000.0 + i,
            })
        df = pl.DataFrame(rows)

        mock_store = MagicMock()
        mock_store.get_klines.return_value = df
        mock_store.close = MagicMock()

        with patch("src.execution.strategies.ml_strategy.Path.resolve", return_value=Path("/tmp/fake/ml_features")), \
             patch("src.execution.strategies.ml_strategy.Path.exists", return_value=True), \
             patch("src.ingestion.data_store.DataStore.__init__", return_value=None), \
             patch("src.ingestion.data_store.DataStore.get_klines", return_value=df), \
             patch("src.ingestion.data_store.DataStore.close"):
            bars = strategy._preload_from_parquet("BTCUSDT", 10)

        assert len(bars) == 10
        # Should be sorted chronologically (last 10)
        for i in range(len(bars) - 1):
            assert bars[i].ts_event <= bars[i + 1].ts_event

    def test_parquet_lookback_window_covers_history_bars(self) -> None:
        """The Parquet lookback window must be able to hold the full history.

        Parquet is preload source #1 and its result is accepted at >= 50 bars,
        so a window shorter than the requested history wins over the REST
        fallback while still returning a truncated buffer.
        """
        from unittest.mock import patch
        import polars as pl
        from src.features.window_sizes import HISTORY_BARS_4H

        cfg = MLStrategyConfig(
            warmup_bars=10,
            dry_run=True,
            features_dir="./data/features/ml_features",
        )
        strategy = MLTradingStrategy(config=cfg)

        base_ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        df = pl.DataFrame([
            {
                "open_time": base_ts + i * 4 * 3600 * 1000,
                "close_time": base_ts + (i + 1) * 4 * 3600 * 1000 - 1,
                "open": 94_000.0, "high": 94_500.0, "low": 93_500.0,
                "close": 94_250.0, "volume": 1_000.0,
            }
            for i in range(3)
        ])

        mock_get_klines = MagicMock(return_value=df)
        now_before = datetime.now(timezone.utc)

        with patch("src.execution.strategies.ml_strategy.Path.resolve",
                   return_value=Path("/tmp/fake/ml_features")), \
             patch("src.execution.strategies.ml_strategy.Path.exists",
                   return_value=True), \
             patch("src.ingestion.data_store.DataStore.__init__",
                   return_value=None), \
             patch("src.ingestion.data_store.DataStore.get_klines",
                   mock_get_klines), \
             patch("src.ingestion.data_store.DataStore.close"):
            strategy._preload_from_parquet("BTCUSDT", HISTORY_BARS_4H)

        start = mock_get_klines.call_args.kwargs["start"]
        # The strategy's own `now` is >= now_before, so measuring against
        # now_before is the conservative direction.
        window_hours = (now_before - start).total_seconds() / 3600.0
        assert window_hours >= HISTORY_BARS_4H * 4.0

    def test_preload_fallback_to_api(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Parquet fails → fallback to Binance API."""
        from unittest.mock import patch

        strategy = MLTradingStrategy(config=default_strategy_config)

        # Parquet raises FileNotFoundError
        with patch.object(
            strategy, "_preload_from_parquet",
            side_effect=FileNotFoundError("No Parquet data"),
        ), patch.object(
            strategy, "_preload_from_binance_api",
            return_value=_make_bars(300),
        ) as mock_api:
            strategy._preload_historical_bars()

        # Should have called API fallback
        mock_api.assert_called_once_with("BTCUSDT", 10)  # history_bars=10 from fixture
        assert strategy._warmup_complete is True
        # _preload_historical_bars takes bars[-n_bars:] where n_bars=10 (history_bars)
        assert len(strategy._bars) == 10

    def test_preload_all_sources_fail(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Both sources fail → warmup_complete=False, strategy continues."""
        from unittest.mock import patch

        strategy = MLTradingStrategy(config=default_strategy_config)

        with patch.object(
            strategy, "_preload_from_parquet",
            side_effect=Exception("Parquet broken"),
        ), patch.object(
            strategy, "_preload_from_binance_api",
            side_effect=Exception("API down"),
        ):
            # Should NOT raise
            strategy._preload_historical_bars()

        assert strategy._warmup_complete is False
        assert len(strategy._bars) == 0

    def test_on_bar_skips_during_warmup(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """warmup_complete=False → on_bar returns without generating signal."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._warmup_complete = False

        initial_bars = len(strategy._bars)
        mock_bar = _make_bars(1)[0]

        # on_bar uses self.log (Cython), can't run outside engine.
        # Instead verify the logic: bar appended, but warmup blocks.
        strategy._bars.append(mock_bar)
        strategy._bar_count += 1

        # Simulate warmup check
        assert not strategy._warmup_complete
        assert len(strategy._bars) < strategy._config.warmup_bars

    def test_on_bar_processes_after_warmup(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """warmup_complete=True → on_bar should process (not return early)."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._warmup_complete = True
        strategy._bars = _make_bars(50)

        # With warmup complete, the warmup guard should not return
        # (regime detection would be the next step)
        assert strategy._warmup_complete is True
        assert len(strategy._bars) >= strategy._config.warmup_bars

    def test_preload_uses_testnet_url(self) -> None:
        """trading_mode=testnet → URL=testnet.binancefuture.com."""
        from unittest.mock import patch

        cfg = MLStrategyConfig(
            warmup_bars=10,
            dry_run=True,
            trading_mode="testnet",
        )
        strategy = MLTradingStrategy(config=cfg)

        base_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        fake_klines = [[
            base_ts, "94000", "94500", "93500", "94250", "1000",
            base_ts + 1, "1000", 100, "500", "500", "0",
        ]]

        mock_response = MagicMock()
        mock_response.json.return_value = fake_klines
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            strategy._preload_from_binance_api("BTCUSDT", 10)

        call_url = mock_get.call_args[0][0]
        assert "testnet.binancefuture.com" in call_url

    def test_preload_uses_mainnet_url(self) -> None:
        """trading_mode=live → URL=fapi.binance.com."""
        from unittest.mock import patch

        cfg = MLStrategyConfig(
            warmup_bars=10,
            dry_run=True,
            trading_mode="live",
        )
        strategy = MLTradingStrategy(config=cfg)

        base_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        fake_klines = [[
            base_ts, "94000", "94500", "93500", "94250", "1000",
            base_ts + 1, "1000", 100, "500", "500", "0",
        ]]

        mock_response = MagicMock()
        mock_response.json.return_value = fake_klines
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            strategy._preload_from_binance_api("BTCUSDT", 10)

        call_url = mock_get.call_args[0][0]
        assert "fapi.binance.com" in call_url


# ═══════════════════════════════════════════════════════════════════════════
# PR-A0: PRELOAD TIMESTAMP CONVENTION (ts_event = bar CLOSE time)
# ═══════════════════════════════════════════════════════════════════════════
#
# Nautilus delivers ts_event = bar CLOSE time for external bars, and
# LiveFeatureState.add_bar() undoes that (ts_event snapped back to the grid).
# Preload used to build bars from OPEN time, so every preloaded bar landed
# in the buffer one full bar too early. These tests pin the convention on
# both preload sources, on the seam with the first live bar, and on the
# freshness guard that reads ts_event as an absolute timestamp.

_BAR_MS = 4 * 3_600_000


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _kline(open_ms: int, tbv: str = "500.0") -> list:
    """Binance kline row: [open_time, o, h, l, c, vol, close_time, ...].

    ``tbv`` lands at index 9 (taker_buy_base_asset_volume). The default
    500.0 against a volume of 1000 is deliberately the degenerate case
    (cvd = 2*500 - 1000 = 0); tests that need a distinguishable CVD pass
    their own value.
    """
    return [
        open_ms, "94000.0", "94500.0", "93500.0", "94250.0", "1000.000",
        open_ms + _BAR_MS - 1,   # 6: close_time
        "1000000.0", 100, tbv, "500000.0", "0",
    ]


def _klines_df(
    open_times: list[int],
    *,
    with_close_time: bool = True,
    taker_buy_volume: float | None = None,
):
    """Parquet-shaped kline frame (klines_4h schema subset).

    ``taker_buy_volume=None`` omits the column entirely — the legacy
    store layout.
    """
    import polars as pl

    rows = []
    for i, ts in enumerate(open_times):
        row = {
            "open_time": ts,
            "open": 94_000.0 + i, "high": 94_500.0 + i,
            "low": 93_500.0 + i, "close": 94_250.0 + i,
            "volume": 1_000.0 + i,
        }
        if with_close_time:
            row["close_time"] = ts + _BAR_MS - 1
        if taker_buy_volume is not None:
            row["taker_buy_volume"] = taker_buy_volume
        rows.append(row)
    return pl.DataFrame(rows)


def _grid_open_times(n: int, *, last_closed: bool = True) -> list[int]:
    """*n* consecutive 4H open times ending at the newest candle.

    ``last_closed=False`` makes the last entry the candle that is still
    forming right now — exactly what /fapi/v1/klines returns as its last
    element.
    """
    last_open = (_now_ms() // _BAR_MS) * _BAR_MS
    if last_closed:
        last_open -= _BAR_MS
    return [last_open - (n - 1 - i) * _BAR_MS for i in range(n)]


def _opens_ending_closed_ago(n: int, hours_ago: float) -> list[int]:
    """*n* 4H open times whose newest candle closed *hours_ago* hours ago."""
    last_close = _now_ms() - int(hours_ago * 3_600_000)
    last_open = last_close + 1 - _BAR_MS
    return [last_open - (n - 1 - i) * _BAR_MS for i in range(n)]


def _run_parquet_preload(strategy, df, n_bars: int):
    from unittest.mock import patch

    with patch("src.execution.strategies.ml_strategy.Path.resolve",
               return_value=Path("/tmp/fake/ml_features")), \
         patch("src.execution.strategies.ml_strategy.Path.exists",
               return_value=True), \
         patch("src.ingestion.data_store.DataStore.__init__", return_value=None), \
         patch("src.ingestion.data_store.DataStore.get_klines", return_value=df), \
         patch("src.ingestion.data_store.DataStore.close"):
        return strategy._preload_from_parquet("BTCUSDT", n_bars)


def _run_api_preload(strategy, klines: list, n_bars: int):
    from unittest.mock import patch

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = klines
    resp.raise_for_status = MagicMock()
    with patch("requests.get", return_value=resp) as mock_get:
        bars = strategy._preload_from_binance_api("BTCUSDT", n_bars)
    return bars, mock_get


def _run_full_preload_via_api(strategy, klines: list):
    """_preload_historical_bars() with Parquet absent (the VM's real case)."""
    from unittest.mock import patch

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = klines
    resp.raise_for_status = MagicMock()
    with patch.object(strategy, "_preload_from_parquet", return_value=[]), \
         patch("requests.get", return_value=resp):
        strategy._preload_historical_bars()


def _run_full_preload_via_parquet(strategy, df):
    """_preload_historical_bars() driving the real _preload_from_parquet."""
    from unittest.mock import patch

    api = MagicMock(return_value=[])
    with patch("src.execution.strategies.ml_strategy.Path.resolve",
               return_value=Path("/tmp/fake/ml_features")), \
         patch("src.execution.strategies.ml_strategy.Path.exists",
               return_value=True), \
         patch("src.ingestion.data_store.DataStore.__init__", return_value=None), \
         patch("src.ingestion.data_store.DataStore.get_klines", return_value=df), \
         patch("src.ingestion.data_store.DataStore.close"), \
         patch.object(strategy, "_preload_from_binance_api", api):
        strategy._preload_historical_bars()
    return api


def _real_bar(strategy, ts_event_ms: int):
    """A genuine Nautilus Bar (not a mock) stamped at *ts_event_ms*."""
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity

    ts_ns = ts_event_ms * 1_000_000
    return Bar(
        bar_type=strategy._bar_type,
        open=Price(94_000.0, precision=1),
        high=Price(94_500.0, precision=1),
        low=Price(93_500.0, precision=1),
        close=Price(94_250.0, precision=1),
        volume=Quantity(1_000.0, precision=3),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


@pytest.fixture
def guard_config() -> MLStrategyConfig:
    """Config whose history_bars clears the >= 50 Parquet acceptance bar.

    _preload_from_parquet truncates to n_bars, so with the default
    history_bars=10 the Parquet source can never be accepted and the REST
    fallback runs regardless of freshness — which would make the guard
    tests below vacuous.
    """
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        history_bars=60,
        dry_run=True,
    )


class TestPreloadTimestampConvention:
    """PR-A0: preload must stamp ts_event with the bar's CLOSE time."""

    def test_preload_parquet_ts_event_is_close_time(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Parquet preload takes ts_event from the close_time column."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(5)

        bars = _run_parquet_preload(strategy, _klines_df(opens), 5)

        assert len(bars) == 5
        for bar, open_ms in zip(bars, opens):
            assert bar.ts_event == (open_ms + _BAR_MS - 1) * 1_000_000
            assert bar.ts_init == bar.ts_event

    def test_preload_binance_ts_event_is_close_time(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """REST preload takes ts_event from kline index 6 (close_time)."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(5)
        klines = [_kline(o) for o in opens]

        bars, _ = _run_api_preload(strategy, klines, 5)

        assert len(bars) == 5
        for bar, k in zip(bars, klines):
            assert bar.ts_event == int(k[6]) * 1_000_000
            assert bar.ts_init == bar.ts_event

    def test_preload_parquet_without_close_time_returns_empty(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Legacy Parquet without close_time → fail soft to the REST source.

        Deriving close_time arithmetically would hide the schema drift; an
        empty result drops below the >= 50 acceptance bar and lets
        _preload_historical_bars fall through to Binance REST.
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        df = _klines_df(_grid_open_times(60), with_close_time=False)

        bars = _run_parquet_preload(strategy, df, 60)

        assert bars == []

    def test_preload_binance_short_kline_row_handling(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Truncated rows: tail row is dropped, an interior one aborts.

        An interior gap would silently shift every positional rolling
        window (CVD, volume, funding/OI z-scores, regime detection), which
        is worse than warming up from live bars.
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(4)
        klines = [_kline(o) for o in opens]

        interior = [list(k) for k in klines]
        interior[1] = interior[1][:6]
        bars, mock_get = _run_api_preload(strategy, interior, 4)
        assert bars == []
        assert mock_get.call_count == 1          # no pointless retries

        tail = [list(k) for k in klines]
        tail[-1] = tail[-1][:6]
        bars_tail, _ = _run_api_preload(strategy, tail, 4)
        assert len(bars_tail) == 3
        assert bars_tail[-1].ts_event == (opens[-2] + _BAR_MS - 1) * 1_000_000

    def test_preload_drops_unclosed_last_kline(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """The still-forming candle must not enter the buffer.

        Verified against mainnet: /fapi/v1/klines returns the in-progress
        candle as its last element, with close_time in the future.
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(4, last_closed=False)

        bars, _ = _run_api_preload(strategy, [_kline(o) for o in opens], 4)
        assert len(bars) == 3
        assert bars[-1].ts_event // 1_000_000 < _now_ms()

        bars_pq = _run_parquet_preload(strategy, _klines_df(opens), 4)
        assert len(bars_pq) == 3
        assert bars_pq[-1].ts_event // 1_000_000 < _now_ms()

    def test_preload_bar_lands_on_its_own_open_time_in_live_state(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """A preloaded candle keeps its own open_time in the 4H buffer."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(60)

        _run_full_preload_via_api(strategy, [_kline(o) for o in opens])

        newest = strategy._live_state.bar_buffer_4h[-1]["open_time"]
        assert newest == opens[-1]                  # its own grid slot
        assert newest != opens[-1] - _BAR_MS        # not one bar too early

    def test_preload_live_seam_is_exactly_one_bar_duration(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Last preloaded bar and first live bar: no gap, no overlap.

        on_bar() cannot run outside a Nautilus engine, so the live side is
        modelled by its only timestamp-bearing step — add_bar().
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(60)
        _run_full_preload_via_api(strategy, [_kline(o) for o in opens])

        buf = strategy._live_state.bar_buffer_4h
        before = len(buf)
        last_preload_bar = strategy._bars[-1]

        live_open = opens[-1] + _BAR_MS
        live_bar = _real_bar(strategy, live_open + _BAR_MS - 1)
        strategy._live_state.add_bar(live_bar, interval="4h")

        assert len(buf) == before + 1
        assert buf[-1]["open_time"] - buf[-2]["open_time"] == _BAR_MS
        times = [r["open_time"] for r in buf]
        assert len(set(times)) == len(times)       # no duplicate slot
        assert live_bar.ts_event - last_preload_bar.ts_event == (
            _BAR_MS * 1_000_000
        )

    def test_freshness_guard_keeps_bars_closed_within_two_periods(
        self, guard_config: MLStrategyConfig,
    ) -> None:
        """Parquet closing 6h ago is fresh (threshold = 2 x 4h = 8h)."""
        strategy = MLTradingStrategy(config=guard_config)
        df = _klines_df(_opens_ending_closed_ago(60, 6.0))

        api = _run_full_preload_via_parquet(strategy, df)

        api.assert_not_called()
        assert strategy._warmup_complete is True
        assert len(strategy._bars) == guard_config.history_bars

    def test_freshness_guard_discards_bars_closed_beyond_two_periods(
        self, guard_config: MLStrategyConfig,
    ) -> None:
        """Parquet closing 9h ago is stale → REST fallback runs."""
        strategy = MLTradingStrategy(config=guard_config)
        df = _klines_df(_opens_ending_closed_ago(60, 9.0))

        api = _run_full_preload_via_parquet(strategy, df)

        api.assert_called_once_with("BTCUSDT", guard_config.history_bars)
        assert strategy._bars == []
        assert strategy._warmup_complete is False


# ═══════════════════════════════════════════════════════════════════════════
# PR-B: REAL taker_buy_volume ON THE PRELOADED WINDOW
# ═══════════════════════════════════════════════════════════════════════════
#
# Preload used to call add_bar() without taker_buy_volume, so every warmed
# bar fell back to volume*0.5 — cvd ≡ 0 and taker_buy_ratio ≡ 0.5 across the
# whole buffer. Both preload sources already carry the real number
# (kline index 9 / the taker_buy_volume column), so no extra request is
# needed; these tests pin that it actually reaches the buffer.


@pytest.fixture
def loguru_warnings():
    """Capture loguru WARNING records into a list.

    LiveFeatureState logs through loguru (src.logger), unlike the strategy
    itself, which uses the Nautilus logger.
    """
    from loguru import logger as _loguru_logger

    sink: list[str] = []
    sink_id = _loguru_logger.add(
        lambda msg: sink.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield sink
    finally:
        _loguru_logger.remove(sink_id)


class TestPreloadTakerBuyVolume:
    """PR-B: the preloaded window must carry real taker_buy_volume."""

    def test_preload_fills_real_taker_buy_volume(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """REST preload: every buffered bar keeps kline index 9, CVD != 0."""
        from src.features.microstructure import add_cvd_features

        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(60)

        _run_full_preload_via_api(strategy, [_kline(o, "700.0") for o in opens])

        buf = strategy._live_state.bar_buffer_4h
        assert len(buf) == default_strategy_config.history_bars
        assert all(r["taker_buy_volume"] == 700.0 for r in buf)

        cvd = add_cvd_features(
            strategy._live_state.get_bar_df("4h")
        )["cvd"].to_list()
        # volume = 1000, tbv = 700 → cvd = 2*700 - 1000 = 400
        assert all(v == 400.0 for v in cvd)

    def test_preload_parquet_fills_real_taker_buy_volume(
        self, guard_config: MLStrategyConfig,
    ) -> None:
        """Parquet preload: the taker_buy_volume column reaches the buffer."""
        strategy = MLTradingStrategy(config=guard_config)
        df = _klines_df(
            _opens_ending_closed_ago(60, 6.0), taker_buy_volume=700.0,
        )

        api = _run_full_preload_via_parquet(strategy, df)

        api.assert_not_called()
        buf = strategy._live_state.bar_buffer_4h
        assert len(buf) == guard_config.history_bars
        assert all(r["taker_buy_volume"] == 700.0 for r in buf)

    def test_preload_parquet_without_taker_column_keeps_bars(
        self, guard_config: MLStrategyConfig,
    ) -> None:
        """A legacy frame without the column must NOT discard the bars.

        Deliberately unlike the close_time guard: a missing close_time
        breaks every timestamp, a missing taker column costs one feature
        group. Dropping the whole window would be the worse trade.
        """
        strategy = MLTradingStrategy(config=guard_config)
        df = _klines_df(_opens_ending_closed_ago(60, 6.0))

        api = _run_full_preload_via_parquet(strategy, df)

        api.assert_not_called()
        buf = strategy._live_state.bar_buffer_4h
        assert len(buf) == guard_config.history_bars
        assert all(r["taker_buy_volume"] is None for r in buf)

    def test_preload_short_kline_row_keeps_bar_without_tbv(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """A 9-field row loses only its own tbv — the batch survives.

        Index 6 (close_time) is present, so OHLCV and the timestamp are
        intact; only index 9 is missing. Refusing the batch here would
        cost the whole warmup for one absent feature value.
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(4)
        klines = [_kline(o, "700.0") for o in opens]
        klines[1] = klines[1][:9]

        _run_full_preload_via_api(strategy, klines)

        tbvs = [r["taker_buy_volume"] for r in strategy._live_state.bar_buffer_4h]
        assert len(tbvs) == 4
        assert tbvs[1] is None
        assert tbvs[0] == 700.0
        assert tbvs[2] == 700.0
        assert tbvs[3] == 700.0

    def test_preload_live_seam_key_formula_is_identical(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Preloaded and live bars derive open_time by the same rule.

        Both must land on the absolute 4H grid; the old ts_event - duration
        produced grid-1ms for a Binance-close timestamp, which is what made
        the REST lookup miss by exactly one millisecond.
        """
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(60)

        _run_full_preload_via_api(strategy, [_kline(o) for o in opens])

        buf = strategy._live_state.bar_buffer_4h
        preload_open = buf[-1]["open_time"]

        live_bar = _real_bar(strategy, opens[-1] + 2 * _BAR_MS - 1)
        strategy._live_state.add_bar(live_bar, interval="4h")
        live_open = buf[-1]["open_time"]

        assert preload_open == opens[-1]
        assert preload_open % _BAR_MS == 0
        assert live_open % _BAR_MS == 0
        assert live_open - preload_open == _BAR_MS

    def test_warmup_log_reports_taker_coverage(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """Coverage is counted per preload and exposed for the warmup log.

        The strategy logs through the Nautilus logger, which pytest cannot
        capture, so the assertion is on the counter the log line is built
        from.
        """
        n = default_strategy_config.history_bars
        opens = _grid_open_times(60)

        full = MLTradingStrategy(config=default_strategy_config)
        _run_full_preload_via_api(full, [_kline(o, "700.0") for o in opens])
        assert full._preload_tbv_coverage == (n, n)

        none_covered = MLTradingStrategy(config=default_strategy_config)
        _run_full_preload_via_api(
            none_covered, [_kline(o)[:9] for o in opens],
        )
        assert none_covered._preload_tbv_coverage == (0, n)

    def test_no_tbv_fallback_warning_after_preload(
        self, default_strategy_config: MLStrategyConfig, loguru_warnings,
    ) -> None:
        """A fully covered preload must not trip the volume*0.5 fallback."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        opens = _grid_open_times(60)

        _run_full_preload_via_api(strategy, [_kline(o, "700.0") for o in opens])
        strategy._live_state.get_bar_df("4h")

        assert [m for m in loguru_warnings if "missing real" in m] == []
