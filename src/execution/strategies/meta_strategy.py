"""
src/execution/strategies/meta_strategy.py

Block 4 — Meta-labeling execution layer.

Two classes
-----------
MetaSignalGate
    Pure, Nautilus-free wrapper around the trained meta-model. Given a
    base signal (direction + confidence) and the surrounding 4H context,
    returns ``(take, size_multiplier)``. Fully unit-testable.

MetaMLTradingStrategy
    Thin subclass of MLTradingStrategy that interposes MetaSignalGate
    between the base ML signal and risk evaluation. The production 4H
    bot keeps running MLTradingStrategy unchanged — this is a parallel
    strategy class intended for paper trading / experimentation.

The gate's feature schema must match what build_meta_dataset.py emitted
and train_meta_model.py consumed; the meta bundle's `feature_columns`
list is the canonical contract.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger
from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
    _safe_float,
)
from src.risk.risk_engine import RiskDecision, TradeSignal

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure meta gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetaDecision:
    """Result of evaluating one base signal against the meta-model."""

    take: bool
    meta_proba: float
    size_multiplier: float  # in [0, 1]; multiply base risk_per_trade by this


class MetaSignalGate:
    """Take/skip gate built on a trained meta-labeling booster.

    The booster expects features in a specific order (saved in the
    bundle). Callers pass a `context` dict — the gate looks up each
    expected column by name, defaults missing fields to 0.0. This
    keeps the live integration tolerant of feature renames / missing
    sources without silently passing the wrong column index.

    Position-size scaling
    ---------------------
    When the meta-proba is at the threshold, the trade is "barely
    approved" → small size. When proba == 1.0, full size. The default
    scaling is linear above the threshold:

        size_mult = clip( (proba - threshold) / (1 - threshold), 0, 1 )

    At proba=threshold → 0× (skip-equivalent), at proba=1.0 → 1×.
    Callers wanting a softer floor can pass ``min_size`` (e.g. 0.25)
    so even "borderline takes" trade at a meaningful fraction.
    """

    def __init__(
        self,
        bundle_path: Path,
        threshold: float = 0.60,
        min_size: float = 0.0,
    ) -> None:
        self._bundle_path = Path(bundle_path)
        with open(self._bundle_path, "rb") as f:
            bundle = pickle.load(f)
        self._booster = bundle["booster"]
        self._feature_columns: list[str] = bundle["feature_columns"]
        self._threshold = float(threshold)
        if not 0.0 <= min_size <= 1.0:
            raise ValueError("min_size must be in [0, 1]")
        self._min_size = float(min_size)
        _log.info(
            f"MetaSignalGate loaded: {self._bundle_path.name} | "
            f"features={len(self._feature_columns)} | "
            f"threshold={self._threshold} | min_size={self._min_size}"
        )

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Vector assembly
    # ------------------------------------------------------------------

    def build_feature_vector(self, context: dict[str, float]) -> np.ndarray:
        """Project `context` onto the booster's column order.

        Missing / non-finite keys become NaN so LightGBM uses its
        native missing-value routing — matching what the trainer saw
        during fit. Replacing them with 0.0 would inject train/serve
        skew (booster would read "no data" as a real measurement of
        zero). Returns shape (1, n_features).
        """
        vec = np.array(
            [_safe_float(context.get(c)) for c in self._feature_columns],
            dtype=np.float64,
        ).reshape(1, -1)
        return np.where(np.isfinite(vec), vec, np.nan)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def evaluate(
        self,
        base_direction: int,
        base_confidence: float,
        context: dict[str, float],
    ) -> MetaDecision:
        """Score one base signal; return whether to take it + size mult.

        `base_direction` and `base_confidence` are folded into the
        context (under the canonical names used by build_meta_dataset)
        before scoring so callers don't have to remember the schema.
        """
        if base_direction == 0:
            # No base signal → nothing to gate. Bypass meta entirely.
            return MetaDecision(take=False, meta_proba=0.0, size_multiplier=0.0)

        ctx = dict(context)
        ctx.setdefault("base_direction", float(base_direction))
        ctx.setdefault("base_confidence", float(base_confidence))
        if "base_proba_up" not in ctx:
            # Reconstruct P(UP) from direction+confidence: direction=+1
            # means P(UP)=confidence; direction=-1 means P(UP)=1-confidence.
            ctx["base_proba_up"] = (
                float(base_confidence) if base_direction > 0
                else 1.0 - float(base_confidence)
            )

        X = self.build_feature_vector(ctx)
        proba = float(self._booster.predict(X)[0])

        take = proba >= self._threshold
        if not take:
            return MetaDecision(take=False, meta_proba=proba, size_multiplier=0.0)

        # Linear ramp from threshold → 1.0, clipped to [min_size, 1.0].
        denom = max(1.0 - self._threshold, 1e-9)
        raw = (proba - self._threshold) / denom
        size_mult = max(self._min_size, min(1.0, raw))
        return MetaDecision(take=True, meta_proba=proba, size_multiplier=size_mult)


# ---------------------------------------------------------------------------
# Nautilus strategy wrapper
# ---------------------------------------------------------------------------

class MetaMLStrategyConfig(MLStrategyConfig, frozen=True):
    """Extends MLStrategyConfig with meta-gate parameters."""

    meta_model_path: str = "./data/features/models/v3/meta_model_v3.pkl"
    meta_threshold: float = 0.60
    meta_min_size: float = 0.25  # never trade below 25% of base risk
    # Disable meta gate at runtime (lets the same class run as plain
    # MLTradingStrategy for control comparisons in paper trading).
    meta_enabled: bool = True


class MetaMLTradingStrategy(MLTradingStrategy):
    """4H ML strategy with a meta-labeling gate layered on top.

    The base signal path is inherited unchanged. After
    :func:`LGBMTrainer.get_signal` returns a non-zero direction, we
    consult the meta-gate. If it approves, we scale the strategy's
    `risk_per_trade` for THIS trade (and only this trade) so position
    size ∝ meta_proba. Risk-engine evaluation, SL/TP placement, and
    order management are entirely inherited.

    Production safety: the live 4H bot uses MLTradingStrategy. This
    subclass is opt-in and writes into the same signal_db_path /
    heartbeat_key only if you wire it up — by default treat it as
    paper-trading only.
    """

    def __init__(self, config: MetaMLStrategyConfig) -> None:
        super().__init__(config)
        self._meta_config: MetaMLStrategyConfig = config
        # Defer gate loading to on_start so a missing/corrupt bundle
        # degrades gracefully (we keep trading on the base signal)
        # instead of crashing the strategy at construction time.
        self._gate: MetaSignalGate | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Set up base strategy, then load the meta-gate (fail-soft)."""
        super().on_start()
        if not self._meta_config.meta_enabled:
            self.log.info("Meta gate disabled by config — base strategy only")
            return
        try:
            self._gate = MetaSignalGate(
                bundle_path=Path(self._meta_config.meta_model_path),
                threshold=self._meta_config.meta_threshold,
                min_size=self._meta_config.meta_min_size,
            )
            self.log.info(
                f"Meta gate active | threshold={self._meta_config.meta_threshold} "
                f"min_size={self._meta_config.meta_min_size}"
            )
        except Exception as exc:
            # Fail-soft: degrade to base strategy. No silent corruption —
            # log loudly so the operator can fix it.
            self.log.error(
                f"Meta gate failed to load ({exc}); "
                f"continuing as base MLTradingStrategy"
            )
            self._gate = None

    # ------------------------------------------------------------------
    # Gate evaluation
    # ------------------------------------------------------------------

    def _apply_meta_gate(
        self,
        direction: int,
        confidence: float,
        context: dict[str, float],
    ) -> tuple[bool, float]:
        """Return (take, size_multiplier). Bypass cleanly when gate off."""
        if self._gate is None or direction == 0:
            return True, 1.0
        decision = self._gate.evaluate(direction, confidence, context)
        self.log.info(
            f"meta-gate | proba={decision.meta_proba:.3f} "
            f"thr={self._meta_config.meta_threshold} | "
            f"take={decision.take} size_mult={decision.size_multiplier:.3f}"
        )
        return decision.take, decision.size_multiplier

    def _build_meta_context(self, signal: TradeSignal) -> dict[str, float]:
        """Best-effort feature dict for the meta gate.

        Populates only the columns we can read directly from the live
        ``TradeSignal``; the gate's ``build_feature_vector`` defaults
        any unknown / missing column to 0.0, which LightGBM handles
        natively. This keeps the integration robust to future schema
        evolution without silently mis-aligning features.
        """
        ctx: dict[str, float] = {
            "atr_pct": float(signal.atr_pct or 0.0),
            "funding_rate": float(signal.funding_rate or 0.0),
        }
        ts = getattr(signal, "timestamp", None)
        if ts is not None:
            try:
                hour = float(ts.hour) + float(ts.minute) / 60.0
                ctx["hour_sin"] = math.sin(2.0 * math.pi * hour / 24.0)
                ctx["hour_cos"] = math.cos(2.0 * math.pi * hour / 24.0)
            except Exception:
                pass
        regime = getattr(signal, "regime", None)
        if isinstance(regime, str) and regime:
            ctx[f"regime_{regime}"] = 1.0
        return ctx

    # ------------------------------------------------------------------
    # Integration point: meta-gate fires AFTER risk approval, BEFORE
    # the order is actually submitted. Rejection here drops the trade;
    # approval may shrink the position size by ``size_multiplier``.
    # ------------------------------------------------------------------

    def _open_position(
        self,
        decision: RiskDecision,
        signal: TradeSignal,
    ) -> None:
        if self._gate is None:
            super()._open_position(decision, signal)
            return

        context = self._build_meta_context(signal)
        take, size_mult = self._apply_meta_gate(
            direction=signal.direction,
            confidence=signal.confidence,
            context=context,
        )

        if not take:
            self.log.info(
                f"META-GATE REJECTED | regime={signal.regime} "
                f"dir={signal.direction} conf={signal.confidence:.3f} | "
                f"holding for next bar"
            )
            return

        if size_mult < 1.0:
            decision.position_size *= size_mult
            decision.notional *= size_mult
            decision.leverage *= size_mult
            self.log.info(
                f"META-GATE SCALED | mult={size_mult:.3f} → "
                f"size={decision.position_size:.6f} "
                f"notional=${decision.notional:.2f}"
            )

        super()._open_position(decision, signal)
