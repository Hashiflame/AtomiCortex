"""
Phase 5.5 — Meta-gate wiring tests.

Pins the integration of MetaSignalGate inside MetaMLTradingStrategy:
  * The gate runs *after* RiskEngine approval and *before* the order
    is actually submitted (i.e. inside ``_open_position`` — the single
    site where a risk-approved decision becomes a live trade).
  * Below threshold → no position opened.
  * Above threshold → position opened, sized by ``size_multiplier``.
  * Missing / corrupt bundle at on_start → fail-soft (degrade to base).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution.strategies.meta_strategy import (
    MetaDecision,
    MetaMLStrategyConfig,
    MetaMLTradingStrategy,
    MetaSignalGate,
)
from src.execution.strategies.ml_strategy import MLTradingStrategy
from src.risk.risk_engine import RiskDecision, TradeSignal


_META_BUNDLE = Path("data/features/models/v3/meta_model_v3.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_strategy(meta_enabled: bool = True, **overrides) -> MetaMLTradingStrategy:
    cfg = MetaMLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        dry_run=True,
        meta_enabled=meta_enabled,
        **overrides,
    )
    return MetaMLTradingStrategy(config=cfg)


def _make_signal() -> TradeSignal:
    return TradeSignal(
        symbol="BTCUSDT-PERP.BINANCE",
        direction=1,
        confidence=0.72,
        regime="trend_up",
        entry_price=50_000.0,
        atr=750.0,
        atr_pct=0.015,
        funding_rate=0.0001,
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )


def _make_decision() -> RiskDecision:
    return RiskDecision(
        approved=True,
        reason="",
        position_size=0.10,
        stop_loss=49_250.0,
        take_profit=51_125.0,
        notional=5_000.0,
        leverage=0.5,
        expected_fee_bps=4.0,
        risk_reward_ratio=1.5,
    )


def _stub_gate(meta_proba: float, threshold: float = 0.60) -> MagicMock:
    """A drop-in MetaSignalGate mock with deterministic .evaluate()."""
    take = meta_proba >= threshold
    denom = max(1.0 - threshold, 1e-9)
    size_mult = (
        max(0.0, min(1.0, (meta_proba - threshold) / denom)) if take else 0.0
    )
    gate = MagicMock(spec=MetaSignalGate)
    gate.evaluate = MagicMock(
        return_value=MetaDecision(
            take=take, meta_proba=meta_proba, size_multiplier=size_mult,
        )
    )
    return gate


# ---------------------------------------------------------------------------
# 1. __init__ no longer eagerly loads the bundle (fail-soft prerequisite)
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_init_does_not_load_gate(self):
        s = _make_strategy(meta_enabled=True)
        assert s._gate is None

    def test_init_works_when_model_missing(self):
        """Pointing at a nonexistent bundle must not raise from __init__."""
        s = _make_strategy(
            meta_enabled=True,
            meta_model_path="/nonexistent/path/meta.pkl",
        )
        assert s._gate is None


# ---------------------------------------------------------------------------
# 2. on_start loads the gate (and fails soft)
# ---------------------------------------------------------------------------
class TestOnStartLoading:
    def test_on_start_loads_real_bundle(self):
        if not _META_BUNDLE.exists():
            pytest.skip("meta_model_v3.pkl not present")
        s = _make_strategy(meta_enabled=True)
        with patch.object(MLTradingStrategy, "on_start", lambda self: None):
            s.on_start()
        assert s._gate is not None
        assert isinstance(s._gate, MetaSignalGate)
        assert s._gate.threshold == 0.60

    def test_on_start_fail_soft_when_bundle_missing(self):
        """Missing model → degrade to base, do not crash."""
        s = _make_strategy(
            meta_enabled=True,
            meta_model_path="/tmp/definitely-not-here.pkl",
        )
        with patch.object(MLTradingStrategy, "on_start", lambda self: None):
            s.on_start()
        assert s._gate is None

    def test_on_start_respects_meta_disabled(self):
        s = _make_strategy(meta_enabled=False)
        with patch.object(MLTradingStrategy, "on_start", lambda self: None):
            s.on_start()
        assert s._gate is None


# ---------------------------------------------------------------------------
# 3. _open_position calls the gate after risk approval
# ---------------------------------------------------------------------------
class TestGateInvocation:
    def test_gate_consulted_when_active(self):
        s = _make_strategy(meta_enabled=True)
        s._gate = _stub_gate(meta_proba=0.80)

        with patch.object(MLTradingStrategy, "_open_position") as base_open:
            s._open_position(_make_decision(), _make_signal())

        s._gate.evaluate.assert_called_once()
        # The context contains signal-derived columns.
        ctx_arg = s._gate.evaluate.call_args.kwargs.get("context") or \
                  s._gate.evaluate.call_args.args[2]
        assert "atr_pct" in ctx_arg
        assert "regime_trend_up" in ctx_arg
        # Super (real submit) is invoked.
        base_open.assert_called_once()

    def test_no_gate_falls_back_to_base(self):
        """No bundle loaded → super()._open_position called directly."""
        s = _make_strategy(meta_enabled=True)
        s._gate = None

        with patch.object(MLTradingStrategy, "_open_position") as base_open:
            s._open_position(_make_decision(), _make_signal())

        base_open.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Below-threshold proba → no position opened
# ---------------------------------------------------------------------------
class TestBelowThreshold:
    def test_low_proba_blocks_position(self):
        s = _make_strategy(meta_enabled=True)
        s._gate = _stub_gate(meta_proba=0.45)   # below 0.60

        decision = _make_decision()
        original_size = decision.position_size

        with patch.object(MLTradingStrategy, "_open_position") as base_open:
            s._open_position(decision, _make_signal())

        base_open.assert_not_called()
        # Decision is left untouched when rejected (defensive).
        assert decision.position_size == original_size

    def test_exact_threshold_treated_as_take(self):
        """proba == threshold is a take (>= per gate semantics)."""
        s = _make_strategy(meta_enabled=True)
        s._gate = _stub_gate(meta_proba=0.60)

        with patch.object(MLTradingStrategy, "_open_position") as base_open:
            s._open_position(_make_decision(), _make_signal())

        base_open.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Above-threshold proba → position opened, size scales with proba
# ---------------------------------------------------------------------------
class TestAboveThreshold:
    def test_high_proba_opens_position(self):
        s = _make_strategy(meta_enabled=True)
        s._gate = _stub_gate(meta_proba=0.95)

        with patch.object(MLTradingStrategy, "_open_position") as base_open:
            s._open_position(_make_decision(), _make_signal())

        base_open.assert_called_once()

    def test_size_scales_with_meta_proba_monotonic(self):
        """Higher meta_proba ⇒ larger position size passed to submit."""
        sizes: list[float] = []
        notionals: list[float] = []
        for proba in (0.70, 0.85, 0.99):
            s = _make_strategy(meta_enabled=True)
            s._gate = _stub_gate(meta_proba=proba)
            decision = _make_decision()
            with patch.object(MLTradingStrategy, "_open_position") as base_open:
                s._open_position(decision, _make_signal())
            assert base_open.call_count == 1
            submitted = base_open.call_args.args[0]
            sizes.append(submitted.position_size)
            notionals.append(submitted.notional)

        assert sizes == sorted(sizes)
        assert notionals == sorted(notionals)
        # Top proba scales close to (but not above) the original.
        assert sizes[-1] <= 0.10 + 1e-9
        # Bottom proba is strictly shrunk.
        assert sizes[0] < 0.10

    def test_size_mult_one_leaves_decision_untouched(self):
        """meta_proba=1.0 → size_mult=1.0 → decision passes through verbatim."""
        s = _make_strategy(meta_enabled=True)
        s._gate = _stub_gate(meta_proba=1.0)

        decision = _make_decision()
        original_size = decision.position_size
        original_notional = decision.notional

        with patch.object(MLTradingStrategy, "_open_position") as base_open:
            s._open_position(decision, _make_signal())

        base_open.assert_called_once()
        assert decision.position_size == pytest.approx(original_size)
        assert decision.notional == pytest.approx(original_notional)


# ---------------------------------------------------------------------------
# 6. Context construction
# ---------------------------------------------------------------------------
class TestContextBuilding:
    def test_context_includes_signal_features(self):
        s = _make_strategy(meta_enabled=False)
        ctx = s._build_meta_context(_make_signal())
        assert ctx["atr_pct"] == pytest.approx(0.015)
        assert ctx["funding_rate"] == pytest.approx(0.0001)
        assert ctx["regime_trend_up"] == 1.0
        assert "hour_sin" in ctx and "hour_cos" in ctx

    def test_context_handles_unusual_regime(self):
        s = _make_strategy(meta_enabled=False)
        sig = _make_signal()
        sig.regime = "high_vol"
        ctx = s._build_meta_context(sig)
        assert ctx.get("regime_high_vol") == 1.0
        assert "regime_trend_up" not in ctx
