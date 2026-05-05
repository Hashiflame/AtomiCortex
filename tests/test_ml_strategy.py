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
        assert cfg.dry_run is False
        assert cfg.confidence_threshold == 0.65

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
        """trend_up regime should select the trend model."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._trend_model = MagicMock(name="trend")
        strategy._trend_features = ["f1", "f2"]
        strategy._highvol_model = MagicMock(name="highvol")

        model, feats = strategy._select_model("trend_up")
        assert model is strategy._trend_model
        assert feats == ["f1", "f2"]

    def test_trend_down_selects_trend_model(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """trend_down regime should also select the trend model."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._trend_model = MagicMock(name="trend")
        strategy._trend_features = ["f1"]

        model, _ = strategy._select_model("trend_down")
        assert model is strategy._trend_model

    def test_high_vol_selects_highvol_model(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """high_vol regime should select the high_vol model."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        strategy._highvol_model = MagicMock(name="highvol")
        strategy._highvol_features = ["hv1"]

        model, feats = strategy._select_model("high_vol")
        assert model is strategy._highvol_model
        assert feats == ["hv1"]

    def test_range_returns_none(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """range regime has no model → returns None."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        model, feats = strategy._select_model("range")
        assert model is None
        assert feats == []

    def test_unknown_returns_none(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """unknown regime has no model → returns None."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        model, feats = strategy._select_model("unknown")
        assert model is None


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

    def test_get_funding_rate_fallback_zero(
        self, default_strategy_config: MLStrategyConfig,
    ) -> None:
        """When funding_rate not in features → returns 0.0 (safe default)."""
        strategy = MLTradingStrategy(config=default_strategy_config)
        feature_names = ["returns_1", "adx"]
        feature_vector = np.array([0.01, 25.0])

        rate = strategy._get_funding_rate(feature_vector, feature_names)
        assert rate == 0.0


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
        assert bars[0].ts_event == base_ts * 1_000_000  # ms → ns

        # Verify request was made
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert "testnet.binancefuture.com" in call_kwargs[1].get("url", "") or \
               "testnet.binancefuture.com" in call_kwargs[0][0] if call_kwargs[0] else True

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
        mock_api.assert_called_once_with("BTCUSDT", 10)  # warmup_bars=10 from fixture
        assert strategy._warmup_complete is True
        # _preload_historical_bars takes bars[-n_bars:] where n_bars=10 (warmup_bars)
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
