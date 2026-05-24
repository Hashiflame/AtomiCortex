"""Tests for Step H4 — fail-safe handling of unknown funding_rate.

Old behavior: missing/error funding silently fell back to 0.0, which was
indistinguishable from a legitimate neutral-market reading and bypassed
the extreme-funding filter.

New behavior:
- _get_funding_rate() returns None when no real reading is available.
- _last_funding_rate is initialised to None.
- RiskEngine._check_funding_rate(None) blocks the signal (fail-safe).
- 0.0 still passes (legitimate neutral market).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.risk.risk_engine import (
    PortfolioState, RiskConfig, RiskEngine, TradeSignal,
)


# ---------------------------------------------------------------------------
# RiskEngine._check_funding_rate
# ---------------------------------------------------------------------------


def _signal(funding_rate):
    return TradeSignal(
        symbol="BTCUSDT",
        direction=1,
        confidence=0.7,
        regime="trend_up",
        entry_price=50_000.0,
        atr=500.0,
        atr_pct=0.01,
        funding_rate=funding_rate,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _engine():
    return RiskEngine(RiskConfig(max_funding_rate=0.001), equity=10_000.0)


class TestCheckFundingRate:
    def test_none_blocks(self):
        eng = _engine()
        ok, reason = eng._check_funding_rate(_signal(None))
        assert ok is False
        assert "unknown" in reason.lower()

    def test_zero_passes(self):
        """0.0 is a legitimate neutral-market reading — must NOT block."""
        eng = _engine()
        ok, reason = eng._check_funding_rate(_signal(0.0))
        assert ok is True
        assert reason == ""

    def test_negative_extreme_blocks(self):
        eng = _engine()
        ok, reason = eng._check_funding_rate(_signal(-0.002))
        assert ok is False
        assert "Extreme" in reason

    def test_positive_extreme_blocks(self):
        eng = _engine()
        ok, reason = eng._check_funding_rate(_signal(0.002))
        assert ok is False
        assert "Extreme" in reason

    def test_boundary_at_threshold_passes(self):
        """abs(rate) > threshold blocks; equal to threshold passes."""
        eng = _engine()
        ok, _ = eng._check_funding_rate(_signal(0.001))
        assert ok is True

    def test_typical_passes(self):
        eng = _engine()
        ok, _ = eng._check_funding_rate(_signal(0.0001))
        assert ok is True


class TestEvaluateBlocksOnNone:
    """End-to-end through evaluate() — confirms None propagates to a
    blocked decision rather than crashing somewhere downstream."""

    def test_evaluate_rejects_none_funding(self):
        eng = _engine()
        port = PortfolioState(
            equity=10_000.0, open_positions=0,
            daily_pnl_pct=0.0, weekly_pnl_pct=0.0,
            current_drawdown_pct=0.0, consecutive_losses=0,
            last_loss_time=None, peak_equity=10_000.0,
        )
        decision = eng.evaluate(_signal(None), port)
        assert decision.approved is False
        assert "unknown" in decision.reason.lower()


# ---------------------------------------------------------------------------
# Strategy._get_funding_rate
# ---------------------------------------------------------------------------


def _strategy():
    from src.execution.strategies.ml_strategy import (
        MLStrategyConfig, MLTradingStrategy,
    )
    cfg = MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        warmup_bars=10,
        dry_run=True,
    )
    return MLTradingStrategy(config=cfg)


class TestGetFundingRate:
    def test_initial_last_funding_is_none(self):
        s = _strategy()
        assert s._last_funding_rate is None

    def test_empty_feature_vector_returns_none(self):
        s = _strategy()
        rate = s._get_funding_rate(feature_vector=None, feature_names=[])
        assert rate is None
        assert s._last_funding_rate is None

    def test_missing_column_returns_none(self):
        s = _strategy()
        rate = s._get_funding_rate(
            feature_vector=np.array([1.0, 2.0]),
            feature_names=["foo", "bar"],
        )
        assert rate is None

    def test_nan_value_returns_last_known(self):
        s = _strategy()
        # No prior reading → still None.
        rate = s._get_funding_rate(
            feature_vector=np.array([float("nan")]),
            feature_names=["funding_rate"],
        )
        assert rate is None

    def test_real_value_updates_last_known(self):
        s = _strategy()
        rate = s._get_funding_rate(
            feature_vector=np.array([0.0003]),
            feature_names=["funding_rate"],
        )
        assert rate == pytest.approx(0.0003)
        assert s._last_funding_rate == pytest.approx(0.0003)

    def test_after_good_then_bad_returns_last_known(self):
        """A bad reading after a good one should fall back to last good,
        not jump back to None."""
        s = _strategy()
        s._get_funding_rate(
            feature_vector=np.array([0.00025]),
            feature_names=["funding_rate"],
        )
        rate = s._get_funding_rate(feature_vector=None, feature_names=[])
        assert rate == pytest.approx(0.00025)

    def test_zero_is_a_valid_reading(self):
        """0.0 from the feature vector is a legitimate value, not 'no data'."""
        s = _strategy()
        rate = s._get_funding_rate(
            feature_vector=np.array([0.0]),
            feature_names=["funding_rate"],
        )
        assert rate == 0.0
        assert s._last_funding_rate == 0.0
