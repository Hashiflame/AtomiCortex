"""
AtomiCortex — Paper Trading Strategy.

A thin wrapper around MLTradingStrategy that intercepts order submission
and routes fills through PaperTrader instead of real exchange execution.
All ML logic, risk filtering, and regime detection are identical.

Phase 5 — Paper Trading.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from nautilus_trader.model.data import Bar
from nautilus_trader.model.events import OrderFilled, PositionClosed, PositionOpened

from src.execution.paper_trader import PaperTrader, PaperTraderConfig
from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
    _bar_to_dict,
)
from src.logger import get_logger
from src.monitoring.metrics_collector import MetricsCollector
from src.risk.risk_engine import RiskDecision, TradeSignal

_log = get_logger(__name__)


class PaperTradingStrategy(MLTradingStrategy):
    """MLTradingStrategy variant that uses PaperTrader for simulation.

    Inherits all ML signal generation, regime detection, and risk
    filtering from the parent.  Overrides only _open_position to route
    through PaperTrader instead of exchange order submission.
    """

    def __init__(
        self,
        config: MLStrategyConfig,
        paper_config: PaperTraderConfig | None = None,
        metrics_db: str = "data/metrics.db",
    ) -> None:
        # Force dry_run=True so parent doesn't submit real orders
        super().__init__(config)
        self._paper_trader = PaperTrader(paper_config or PaperTraderConfig(
            initial_equity=config.initial_equity,
        ))
        self._metrics = MetricsCollector(
            db_path=metrics_db,
            initial_equity=config.initial_equity,
        )
        self._signals_total: int = 0
        self._signals_traded: int = 0
        self._signals_filtered: int = 0

    def on_bar(self, bar: Bar) -> None:
        """Override on_bar to log detailed paper signals and save to SQLite."""
        self._bars.append(bar)
        self._bar_count += 1

        if self._bar_count <= self._config.warmup_bars:
            if self._bar_count % 50 == 0:
                self.log.debug(
                    f"Warmup: {self._bar_count}/{self._config.warmup_bars}"
                )
            return

        # All the ML signal generation logic
        if not self._regime_detector:
            return

        bars_data = [_bar_to_dict(b) for b in self._bars]
        try:
            regime_state = self._regime_detector.classify_regime(bars_data)
        except Exception as exc:
            self.log.warning(f"Regime detection error: {exc}")
            return

        regime_label = regime_state.regime

        # Model selection
        model, feature_names = self._select_model(regime_label)
        if model is None:
            return

        # Feature computation
        feature_vector = self._compute_features(feature_names)
        if feature_vector is None:
            return

        # ML prediction
        try:
            proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
            confidence = float(np.max(proba))
            direction = 1 if np.argmax(proba) == 1 else -1
        except Exception as exc:
            self.log.warning(f"Prediction error: {exc}")
            return

        # Confidence threshold
        if confidence < self._config.confidence_threshold:
            return

        self._signals_total += 1

        # Build signal
        current_price = bar.close.as_double()
        atr_dollar = regime_state.atr_pct * current_price
        now_utc = datetime.fromtimestamp(bar.ts_event / 1e9, tz=timezone.utc)
        funding_rate = self._get_funding_rate(feature_vector, feature_names)

        signal = TradeSignal(
            symbol=str(self._instrument_id),
            direction=direction,
            confidence=confidence,
            regime=regime_label,
            entry_price=current_price,
            atr=atr_dollar,
            atr_pct=regime_state.atr_pct,
            funding_rate=funding_rate,
            timestamp=now_utc,
        )

        # Risk evaluation
        portfolio_state = self._tracker.get_state()
        decision = self._risk_engine.evaluate(signal, portfolio_state)

        # Save signal to SQLite
        self._metrics.save_signal_to_db(
            symbol=signal.symbol,
            direction=direction,
            confidence=confidence,
            regime=regime_label,
            entry_price=current_price,
            approved=decision.approved,
            reason=decision.reason if not decision.approved else "",
        )

        if not decision.approved:
            self._signals_filtered += 1
            self.log.info(
                f"[PAPER] Signal BLOCKED | {regime_label} | "
                f"dir={direction} conf={confidence:.3f} | "
                f"reason={decision.reason}"
            )
            return

        # Execute through PaperTrader
        self._signals_traded += 1
        fill = self._paper_trader.simulate_fill(
            symbol=signal.symbol,
            direction=direction,
            quantity=decision.position_size,
            current_price=current_price,
        )

        self.log.info(
            f"[PAPER] Trade EXECUTED | {regime_label} "
            f"dir={direction} conf={confidence:.3f} | "
            f"size={decision.position_size:.6f} "
            f"notional=${decision.notional:.2f} | "
            f"fill=${fill.fill_price:.2f} fee=${fill.fee:.4f}"
        )

        # Status summary
        self.log.info(
            f"[PAPER] Signals: {self._signals_total} total | "
            f"{self._signals_traded} traded | "
            f"{self._signals_filtered} filtered"
        )
