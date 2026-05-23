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

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger
from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
)

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

        Missing keys default to 0.0 (booster handles by-design via
        LightGBM's NaN/zero handling). Returns shape (1, n_features).
        """
        vec = np.array(
            [float(context.get(c, 0.0)) for c in self._feature_columns],
            dtype=np.float64,
        ).reshape(1, -1)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

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
        self._gate: MetaSignalGate | None = None
        if config.meta_enabled:
            self._gate = MetaSignalGate(
                bundle_path=Path(config.meta_model_path),
                threshold=config.meta_threshold,
                min_size=config.meta_min_size,
            )

    # ------------------------------------------------------------------
    # Override the single point where direction/confidence become a
    # trade. We hook the risk engine: before evaluate(), we either drop
    # the signal (meta says skip) or shrink risk_per_trade in proportion
    # to meta_proba. Restored afterwards so other regimes/symbols aren't
    # affected.
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
