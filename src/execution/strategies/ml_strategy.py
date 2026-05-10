"""
AtomiCortex — ML Trading Strategy.

Main live/backtest trading strategy that combines:
- ML regime-specific signal generation (trend + high_vol LightGBM models)
- Pre-trade risk filtering via RiskEngine
- ATR-based position sizing
- Exchange-side STOP_MARKET for risk management

Phase 4 — Step 4.4.
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AggregationSource,
    BarAggregation,
    OrderSide,
    PriceType,
    TimeInForce,
    TriggerType,
)
from nautilus_trader.model.events import OrderFilled, PositionClosed, PositionOpened
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId, Venue
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from src.logger import get_logger
from src.risk.risk_engine import (
    PortfolioState,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    TradeSignal,
)
from src.risk.portfolio_tracker import PortfolioTracker

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy config
# ---------------------------------------------------------------------------

class MLStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for the ML-driven trading strategy."""

    instrument_id: str = "BTCUSDT-PERP.BINANCE"
    bar_type: str = "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
    interval: str = "4h"
    confidence_threshold: float = 0.55
    models_dir: str = "./data/features/models"
    features_dir: str = "./data/features/ml_features"
    risk_per_trade: float = 0.01
    max_leverage: int = 10
    max_open_positions: int = 3
    initial_equity: float = 10_000.0
    warmup_bars: int = 300
    dry_run: bool = False
    rr_ratio: float = 1.5
    preload_enabled: bool = True
    trading_mode: str = "testnet"  # testnet / paper / live


# ---------------------------------------------------------------------------
# Helper: bar → OHLCV dict
# ---------------------------------------------------------------------------

def _bar_to_dict(bar: Bar) -> dict[str, float]:
    """Extract OHLCV from a Nautilus Bar object."""
    return {
        "open": bar.open.as_double(),
        "high": bar.high.as_double(),
        "low": bar.low.as_double(),
        "close": bar.close.as_double(),
        "volume": bar.volume.as_double(),
    }


# ---------------------------------------------------------------------------
# ML Trading Strategy
# ---------------------------------------------------------------------------

class MLTradingStrategy(Strategy):
    """
    Live trading strategy that:
    1. Collects 4H bars
    2. Detects market regime (ADX-first)
    3. Selects regime-appropriate LightGBM model
    4. Generates ML signal → confidence
    5. Passes through RiskEngine pre-trade filters
    6. Opens positions with exchange-side stop orders
    """

    def __init__(self, config: MLStrategyConfig) -> None:
        super().__init__(config)
        self._config = config
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)
        self._venue = Venue(self._instrument_id.venue.value)

        # Will be initialised in on_start
        self._risk_engine: RiskEngine | None = None
        self._tracker: PortfolioTracker | None = None
        self._trend_model: Any = None
        self._highvol_model: Any = None
        self._trend_features: list[str] = []
        self._highvol_features: list[str] = []
        self._regime_detector: Any = None

        # Bar buffer
        self._bars: list[Bar] = []
        self._bar_count: int = 0
        self._warmup_complete: bool = False

        # Equity curve tracking
        self._equity_curve: list[tuple[int, float]] = []

        # Track pending SL/TP per instrument to avoid duplicate orders
        self._pending_stops: dict[str, str] = {}  # instrument_id → client_order_id

        # Last known funding rate (updated from feature data)
        self._last_funding_rate: float = 0.0

        # Pending SL params: entry client_order_id -> {decision, signal} for
        # deferred stop-loss submission (placed in on_order_filled, not _open_position)
        self._pending_sl_params: dict[str, dict[str, Any]] = {}

        # Signal bridge for Telegram integration
        self._signal_bridge: Any = None  # SignalBridge (lazy init in on_start)
        self._pending_signal_ids: dict[str, int] = {}  # symbol -> signal_id
        self._last_regime: str = ""  # for detecting regime changes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Initialise components and subscribe to bars."""
        self.log.info("MLTradingStrategy starting...")

        # 1. Risk Engine
        risk_cfg = RiskConfig(
            risk_per_trade=self._config.risk_per_trade,
            max_leverage=self._config.max_leverage,
            max_open_positions=self._config.max_open_positions,
            confidence_threshold=self._config.confidence_threshold,
        )
        self._risk_engine = RiskEngine(risk_cfg, equity=self._config.initial_equity)

        # 2. Portfolio Tracker
        self._tracker = PortfolioTracker(self._config.initial_equity)

        # 3. Regime Detector
        from src.features.regime_detector import RegimeDetector
        self._regime_detector = RegimeDetector()

        # 4. Load ML models
        self._load_models()

        # 5. Subscribe to bars
        self.subscribe_bars(self._bar_type)
        self.log.info(
            f"Subscribed to {self._bar_type} | "
            f"dry_run={self._config.dry_run} | "
            f"warmup={self._config.warmup_bars} bars"
        )

        # 6. Preload historical bars
        if self._config.preload_enabled:
            self._preload_historical_bars()

        # 7. Signal Bridge for Telegram integration
        try:
            from src.execution.signal_bridge import SignalBridge
            # __file__ = src/execution/strategies/ml_strategy.py
            # project root = 4 levels up
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            db_path = str(project_root / "data" / "atomicortex.db")
            self._signal_bridge = SignalBridge(db_path=db_path)
            self.log.info(f"SignalBridge initialised | db={db_path}")
        except Exception as exc:
            self.log.warning(f"SignalBridge init failed (non-fatal): {exc}")
            self._signal_bridge = None

        if self._warmup_complete:
            self.log.info(
                f"Strategy ready | "
                f"bars={len(self._bars)} | "
                f"warmup=COMPLETE"
            )
        else:
            self.log.info(
                f"Strategy started | warmup=IN_PROGRESS | "
                f"need {self._config.warmup_bars} bars"
            )

    def on_stop(self) -> None:
        """Graceful shutdown: close all positions, cancel pending orders."""
        self.log.info("MLTradingStrategy stopping — closing positions...")
        if not self._config.dry_run:
            self.cancel_all_orders(self._instrument_id)
            self.close_all_positions(self._instrument_id)
        self.log.info(
            f"Strategy stopped | bars_processed={self._bar_count} | "
            f"equity_points={len(self._equity_curve)}"
        )

    # ------------------------------------------------------------------
    # Bar handler
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """Core logic: regime → model → signal → risk → execute."""
        self.log.info(
            f"on_bar called | {bar.bar_type} | close={bar.close}"
        )

        try:
            self.log.info(
                f"on_bar step 1: adding to buffer, size={len(self._bars)}"
            )
            self._bars.append(bar)
            self._bar_count += 1

            # Record equity
            self._record_equity(bar.ts_event)

            # 1. Warmup check
            if not self._warmup_complete:
                if len(self._bars) >= self._config.warmup_bars:
                    self._warmup_complete = True
                    self.log.info(
                        f"Warmup complete via live bars | "
                        f"bars={len(self._bars)}"
                    )
                else:
                    remaining = (
                        self._config.warmup_bars - len(self._bars)
                    )
                    self.log.info(
                        f"on_bar: warmup not complete "
                        f"({len(self._bars)}/{self._config.warmup_bars}, "
                        f"{remaining} remaining)"
                    )
                    return

            # 2. Detect regime
            self.log.info("on_bar step 2: detecting regime")
            regime_state = self._detect_regime()
            if regime_state is None:
                self.log.info("on_bar: regime_state is None — skipping")
                return

            regime_label = regime_state.regime.value  # e.g. "trend_up", "high_vol"

            # Compute last_return for diagnostics
            last_return = 0.0
            if len(self._bars) >= 2:
                prev_close = self._bars[-2].close.as_double()
                cur_close = self._bars[-1].close.as_double()
                if prev_close > 0:
                    last_return = (cur_close - prev_close) / prev_close

            self.log.info(
                f"on_bar step 3: regime={regime_label} | "
                f"adx={regime_state.adx:.1f} | "
                f"atr_pct={regime_state.atr_percentile:.2f} | "
                f"last_return={last_return:.4f}"
            )

            # 3. Select model + per-regime confidence threshold
            model, features_list, conf_threshold = self._select_model(regime_label)
            if model is None:
                self.log.warning(
                    f"on_bar: no model loaded for regime '{regime_label}' — skipping"
                )
                return

            # 4. Compute features
            self.log.info(
                f"on_bar step 4: computing features "
                f"(n={len(features_list)})"
            )
            feature_vector = self._compute_features(features_list)
            if feature_vector is None:
                self.log.info("on_bar: feature_vector is None — skipping")
                return

            # 5. Get ML signal
            self.log.info("on_bar step 5: ML prediction")
            from src.models.lgbm_trainer import LGBMTrainer
            direction, confidence = LGBMTrainer.get_signal(
                None,  # static-ish: only uses model.predict
                model,
                feature_vector,
                confidence_threshold=conf_threshold,
            )
            self.log.info(
                f"on_bar step 6: dir={direction} confidence={confidence:.3f}"
            )

            if direction == 0:
                self.log.info(
                    f"on_bar: no signal | regime={regime_label} | "
                    f"confidence={confidence:.3f}"
                )
                return

            # 6. Build TradeSignal
            current_price = bar.close.as_double()
            atr_dollar = regime_state.atr_pct * current_price
            now_utc = datetime.fromtimestamp(bar.ts_event / 1e9, tz=timezone.utc)

            # Read funding rate from feature data (PROD-003 fix)
            funding_rate = self._get_funding_rate(feature_vector, features_list)

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

            # 7. Risk evaluation
            self.log.info("on_bar step 7: risk evaluation")
            portfolio_state = self._tracker.get_state()
            decision = self._risk_engine.evaluate(signal, portfolio_state)

            if not decision.approved:
                self.log.info(
                    f"Signal BLOCKED | {regime_label} | "
                    f"dir={direction} conf={confidence:.3f} | "
                    f"reason={decision.reason}"
                )
                return

            # 8. Execute
            self.log.info(
                f"Signal APPROVED | {regime_label} dir={direction} "
                f"conf={confidence:.3f} | size={decision.position_size:.6f} "
                f"notional=${decision.notional:.2f}"
            )
            if not self._config.dry_run:
                self._open_position(decision, signal)
            else:
                self.log.info(
                    f"[DRY RUN] Would open {signal.direction} "
                    f"{decision.position_size:.6f} @ ${signal.entry_price:.2f} | "
                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
                )

            # Regime change detection
            if self._last_regime and regime_label != self._last_regime:
                if self._signal_bridge:
                    try:
                        self._signal_bridge.log_regime_change(
                            self._last_regime, regime_label,
                        )
                    except Exception:
                        pass
            self._last_regime = regime_label

            # Periodic metrics update (every 6 bars = ~24H)
            if self._bar_count % 6 == 0 and self._signal_bridge and self._tracker:
                try:
                    state = self._tracker.get_state()
                    self._signal_bridge.update_metrics(
                        equity=state.equity,
                        daily_pnl=state.daily_pnl_pct,
                        regime=regime_label,
                        open_positions=state.open_positions,
                    )
                except Exception:
                    pass

        except Exception as exc:
            import traceback
            self.log.error(f"on_bar EXCEPTION: {exc}")
            self.log.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Order events
    # ------------------------------------------------------------------

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order fills: entry fills update tracker + submit SL,
        exit fills are ignored (handled by on_position_closed).

        PROD-009 fix: distinguish entry vs exit fills via order tags.
        PROD-005 fix: SL is submitted here (after confirmed fill), not
        in _open_position, to guarantee no position without a stop.
        """
        fill_price = event.last_px.as_double()
        fill_qty = event.last_qty.as_double()
        commission = event.commission.as_double() if event.commission else 0.0
        is_buy = event.is_buy
        now_utc = datetime.fromtimestamp(event.ts_event / 1e9, tz=timezone.utc)
        client_oid = str(event.client_order_id)

        self.log.info(
            f"ORDER FILLED | {event.order_side} {fill_qty} "
            f"@ {fill_price} | commission={commission} | oid={client_oid}"
        )

        # Distinguish entry vs exit fill:
        # - Entry fills have pending SL params stored by _open_position
        # - Exit fills (SL hit, manual close) do NOT have pending params
        is_entry_fill = client_oid in self._pending_sl_params

        if is_entry_fill and self._tracker:
            # Entry fill → update tracker + place SL
            direction = 1 if is_buy else -1
            self._tracker.update_fill(
                symbol=str(event.instrument_id),
                direction=direction,
                quantity=fill_qty,
                price=fill_price,
                fee=commission,
                timestamp=now_utc,
            )

            # Submit deferred stop-loss (PROD-005 fix: SL after confirmed entry)
            sl_params = self._pending_sl_params.pop(client_oid)
            self._submit_stop_loss_with_retry(
                decision=sl_params["decision"],
                signal=sl_params["signal"],
                fill_qty=fill_qty,
            )
        else:
            # Exit fill (SL or manual close) → fees tracked via close_position
            self.log.debug(
                f"Exit fill (SL/close) | {event.order_side} {fill_qty} @ {fill_price}"
            )

    def on_position_opened(self, event: PositionOpened) -> None:
        """Log position open."""
        self.log.info(
            f"POSITION OPENED | {event.instrument_id} | "
            f"side={event.entry} qty={event.quantity}"
        )

    def on_position_closed(self, event: PositionClosed) -> None:
        """Handle position close: update PnL via tracker.

        PROD-002 fix: do NOT call record_loss() here — close_position()
        already calls it internally when realized_pnl < 0.
        """
        realized_pnl = event.realized_pnl.as_double() if event.realized_pnl else 0.0
        now_utc = datetime.fromtimestamp(event.ts_event / 1e9, tz=timezone.utc)

        self.log.info(
            f"POSITION CLOSED | {event.instrument_id} | "
            f"realized_pnl={realized_pnl:.4f} | "
            f"return={event.realized_return:.4f}"
        )

        # Close in tracker (record_loss is handled inside close_position)
        if self._tracker:
            self._tracker.close_position(
                symbol=str(event.instrument_id),
                close_price=event.avg_px_close.as_double(),
                fee=0.0,  # fees already accounted in on_order_filled
                timestamp=now_utc,
            )

        # Close signal in bridge for Telegram notification
        symbol = str(event.instrument_id)
        signal_id = self._pending_signal_ids.pop(symbol, None)
        if signal_id and self._signal_bridge:
            try:
                self._signal_bridge.close_signal(
                    signal_id=signal_id,
                    close_price=event.avg_px_close.as_double(),
                    pnl_pct=event.realized_return * 100 if event.realized_return else 0.0,
                    result="win" if realized_pnl > 0 else "loss",
                )
            except Exception as exc:
                self.log.warning(f"SignalBridge close_signal failed: {exc}")

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        decision: RiskDecision,
        signal: TradeSignal,
    ) -> None:
        """Submit market entry order.

        PROD-005 fix: stop-loss is now submitted in on_order_filled()
        after the entry fill is confirmed, not here.  This eliminates
        the crash-between-entry-and-SL window.
        """
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self.log.error(f"Instrument {self._instrument_id} not found in cache")
            return

        # Direction → OrderSide
        entry_side = OrderSide.BUY if signal.direction == 1 else OrderSide.SELL

        # Quantity (round to instrument precision)
        qty = instrument.make_qty(decision.position_size)

        # Client order ID (idempotent)
        ts_ms = int(time.time() * 1000)
        dir_str = "L" if signal.direction == 1 else "S"
        entry_tag = f"AC-{dir_str}-{ts_ms}"

        # Market entry
        entry_order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=entry_side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
            tags=[entry_tag],
        )

        # Store SL params for deferred submission (on_order_filled will use these)
        self._pending_sl_params[str(entry_order.client_order_id)] = {
            "decision": decision,
            "signal": signal,
        }

        self.submit_order(entry_order)
        self.log.info(
            f"ENTRY submitted | {entry_side} {qty} | "
            f"tag={entry_tag} | SL deferred to fill confirmation"
        )

        # Log signal in bridge for Telegram notification
        if self._signal_bridge:
            try:
                signal_id = self._signal_bridge.log_signal(
                    symbol=signal.symbol,
                    direction="long" if signal.direction == 1 else "short",
                    entry_price=decision.entry_price if hasattr(decision, 'entry_price') else signal.entry_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    confidence=signal.confidence,
                    regime=signal.regime,
                    atr=signal.atr,
                    funding_rate=signal.funding_rate,
                    position_size=decision.position_size,
                    notional=decision.notional,
                    leverage=decision.leverage,
                )
                self._pending_signal_ids[signal.symbol] = signal_id
            except Exception as exc:
                self.log.warning(f"SignalBridge log_signal failed: {exc}")

    def _submit_stop_loss_with_retry(
        self,
        decision: RiskDecision,
        signal: TradeSignal,
        fill_qty: float,
        max_retries: int = 3,
    ) -> None:
        """Submit exchange-side STOP_MARKET with retry logic.

        PROD-005 fix: called from on_order_filled after entry is confirmed.
        Retries up to ``max_retries`` times.  Logs CRITICAL if all fail.
        """
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self.log.critical(
                f"CRITICAL: Cannot place SL — instrument {self._instrument_id} "
                f"not in cache! Position is UNPROTECTED!"
            )
            return

        exit_side = OrderSide.SELL if signal.direction == 1 else OrderSide.BUY
        sl_price = instrument.make_price(decision.stop_loss)
        qty = instrument.make_qty(fill_qty)

        for attempt in range(1, max_retries + 1):
            try:
                stop_order = self.order_factory.stop_market(
                    instrument_id=self._instrument_id,
                    order_side=exit_side,
                    quantity=qty,
                    trigger_price=sl_price,
                    trigger_type=TriggerType.LAST_PRICE,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                    tags=[f"SL-attempt-{attempt}"],
                )
                self.submit_order(stop_order)
                self.log.info(
                    f"STOP LOSS submitted | {exit_side} {qty} "
                    f"@ trigger={sl_price} | attempt={attempt}"
                )
                return  # success
            except Exception as exc:
                self.log.error(
                    f"SL submission failed (attempt {attempt}/{max_retries}): {exc}"
                )

        # All retries exhausted
        self.log.critical(
            f"CRITICAL: Failed to place SL after {max_retries} attempts! "
            f"Position is UNPROTECTED! Manual intervention required."
        )

    def _get_funding_rate(
        self,
        feature_vector: np.ndarray | None,
        feature_names: list[str],
    ) -> float:
        """Extract funding rate from feature vector.

        PROD-003 fix: read actual funding_rate from feature data instead
        of hardcoded 0.0001.  Falls back to 0.0 (safe default that will
        not bypass the extreme-funding filter).
        """
        if feature_vector is not None and "funding_rate" in feature_names:
            idx = feature_names.index("funding_rate")
            rate = float(feature_vector[idx])
            if not np.isnan(rate) and not np.isinf(rate):
                self._last_funding_rate = rate
                return rate

        # Fallback: use last known rate or 0.0 (safe default)
        return self._last_funding_rate

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(self, feature_names: list[str]) -> np.ndarray | None:
        """Build a feature vector from the current bar buffer.

        Uses the last 50 bars to compute microstructure features,
        then extracts only the features the model was trained on.
        """
        try:
            import polars as pl

            # Convert bars to DataFrame
            lookback = min(len(self._bars), 540)
            recent_bars = self._bars[-lookback:]
            records = [_bar_to_dict(b) for b in recent_bars]
            df = pl.DataFrame(records)

            # Add basic derived features inline (lightweight version)
            close = df["close"].to_numpy()
            high = df["high"].to_numpy()
            low = df["low"].to_numpy()
            volume = df["volume"].to_numpy()

            # Returns
            returns = np.diff(np.log(close), prepend=np.log(close[0]))

            feature_dict: dict[str, float] = {}

            # Price features
            for period in [1, 3, 6, 12, 24]:
                key = f"returns_{period}"
                if key in feature_names:
                    if len(close) > period:
                        feature_dict[key] = float(
                            (close[-1] - close[-1 - period]) / close[-1 - period]
                        )
                    else:
                        feature_dict[key] = 0.0

            # Body / wick ratios
            last_bar = recent_bars[-1]
            o, h, l, c = (
                last_bar.open.as_double(),
                last_bar.high.as_double(),
                last_bar.low.as_double(),
                last_bar.close.as_double(),
            )
            rng = h - l if h > l else 1e-10
            feature_dict["body_ratio"] = abs(c - o) / rng
            feature_dict["upper_wick"] = (h - max(o, c)) / rng
            feature_dict["lower_wick"] = (min(o, c) - l) / rng
            feature_dict["gap"] = 0.0

            # Volume features
            vol_sma_20 = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
            feature_dict["volume_sma_20"] = vol_sma_20
            feature_dict["volume_ratio"] = float(volume[-1]) / vol_sma_20 if vol_sma_20 > 0 else 1.0
            vol_std = float(np.std(volume[-20:])) if len(volume) >= 20 else 1.0
            feature_dict["volume_zscore"] = (float(volume[-1]) - vol_sma_20) / vol_std if vol_std > 0 else 0.0
            feature_dict["large_volume"] = 1.0 if feature_dict["volume_ratio"] > 2.0 else 0.0

            # CVD approximations
            buy_volume = volume * np.where(close > np.roll(close, 1), 1, 0.5)
            sell_volume = volume - buy_volume
            cvd = np.cumsum(buy_volume - sell_volume)
            feature_dict["cvd"] = float(buy_volume[-1] - sell_volume[-1])
            feature_dict["cvd_cum"] = float(cvd[-1])
            for slope_n in [3, 6, 12]:
                key = f"cvd_slope_{slope_n}"
                if key in feature_names and len(cvd) >= slope_n:
                    feature_dict[key] = float(cvd[-1] - cvd[-slope_n]) / slope_n
                else:
                    feature_dict[key] = 0.0
            feature_dict["taker_buy_ratio"] = float(
                buy_volume[-1] / volume[-1]
            ) if volume[-1] > 0 else 0.5

            # VWAP
            cum_vp = np.cumsum(close * volume)
            cum_v = np.cumsum(volume)
            vwap = cum_vp[-1] / cum_v[-1] if cum_v[-1] > 0 else close[-1]
            feature_dict["vwap_4h"] = float(vwap)
            feature_dict["price_to_vwap"] = float(close[-1] / vwap) if vwap > 0 else 1.0

            # Derivative placeholders (will be updated with live data)
            for key in [
                "funding_rate", "funding_abs", "funding_zscore_7d",
                "funding_zscore_30d", "funding_extreme", "funding_positive",
                "funding_cum_24h", "oi_value", "oi_delta_4h", "oi_delta_12h",
                "oi_zscore", "oi_quadrant", "ls_ratio", "ls_ratio_zscore",
                "taker_vol_ratio", "basis_approx", "basis_extreme",
            ]:
                if key in feature_names and key not in feature_dict:
                    feature_dict[key] = 0.0

            # Regime features
            from src.features.regime_detector import (
                calculate_adx,
                calculate_atr_percentile,
                calculate_hurst_exponent,
            )
            hurst = calculate_hurst_exponent(close[-300:])
            adx_arr = calculate_adx(high, low, close)
            adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
            _, atr_pctl = calculate_atr_percentile(high, low, close)

            feature_dict["hurst"] = hurst
            feature_dict["adx"] = adx_val
            feature_dict["atr_pct"] = float(
                (high[-1] - low[-1]) / close[-1]
            ) if close[-1] > 0 else 0.0
            feature_dict["atr_percentile"] = atr_pctl
            feature_dict["trend_strength"] = min(adx_val / 50.0, 1.0)
            feature_dict["regime_confidence"] = min(adx_val / 50.0, 1.0)

            # symbol_encoded (always last feature)
            sym_str = str(self._instrument_id)
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            # Extract base symbol from instrument_id
            base = sym_str.split("-")[0] if "-" in sym_str else sym_str.split(".")[0]
            feature_dict["symbol_encoded"] = float(sym_map.get(base, -1))

            # Build vector in correct order
            vector = np.array(
                [feature_dict.get(f, 0.0) for f in feature_names],
                dtype=np.float64,
            )
            return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as exc:
            self.log.error(f"Feature computation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(self) -> Any:
        """Detect current market regime from bar buffer."""
        try:
            import polars as pl
            from src.features.regime_detector import RegimeState, MarketRegime

            lookback = min(len(self._bars), 540)
            recent = self._bars[-lookback:]
            records = [_bar_to_dict(b) for b in recent]
            df = pl.DataFrame(records)

            state = self._regime_detector.detect(df, idx=-1)
            self.log.debug(
                f"Regime: {state.regime.value} | ADX={state.adx:.1f} "
                f"| ATR%={state.atr_pct:.4f}"
            )
            return state

        except Exception as exc:
            self.log.error(f"Regime detection failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load pickled LightGBM model bundles."""
        models_dir = Path(self._config.models_dir)
        for regime in ["trend", "high_vol"]:
            path = models_dir / f"{regime}_model.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                booster = bundle["booster"]
                features = bundle.get("feature_columns", [])
                if regime == "trend":
                    self._trend_model = booster
                    self._trend_features = features
                else:
                    self._highvol_model = booster
                    self._highvol_features = features
                self.log.info(
                    f"Loaded {regime} model from {path} "
                    f"({len(features)} features)"
                )
            else:
                self.log.warning(f"Model not found: {path}")

    def _select_model(
        self, regime_label: str,
    ) -> tuple[Any, list[str], float]:
        """Select model, feature list, and confidence threshold by regime.

        Returns
        -------
        (model, feature_names, confidence_threshold)

        Mapping:
        Mapping (binary model — random baseline 0.50, ML-017):
        * trend_up / trend_down → trend model, threshold = config default (0.55)
        * range                 → trend model, threshold = 0.60 (stricter)
        * high_vol              → high-vol model, threshold = config default
        * anything else (should never happen) → trend model, threshold = 0.60
        """
        base_threshold = self._config.confidence_threshold
        if regime_label in ("trend_up", "trend_down"):
            return self._trend_model, self._trend_features, base_threshold
        if regime_label == "range":
            return self._trend_model, self._trend_features, max(base_threshold, 0.60)
        if regime_label == "high_vol":
            return self._highvol_model, self._highvol_features, base_threshold
        # Defensive fallback — RegimeDetector no longer produces "unknown"
        self.log.warning(
            f"Unexpected regime '{regime_label}' — falling back to trend model"
        )
        return self._trend_model, self._trend_features, max(base_threshold, 0.60)

    # ------------------------------------------------------------------
    # Historical bar preloading
    # ------------------------------------------------------------------

    def _preload_historical_bars(self) -> None:
        """Preload historical bars from Parquet or Binance API.

        Tries sources in order:
        1. Local Parquet via DataStore
        2. Binance REST API (testnet or mainnet)

        Fills self._bars and sets self._warmup_complete = True on success.
        All exceptions are caught — preload never crashes the strategy.
        """
        symbol_clean = "BTCUSDT"  # without -PERP
        n_bars = self._config.warmup_bars  # 300

        bars: list[Bar] = []
        source = "none"

        # Attempt 1: local Parquet
        try:
            bars = self._preload_from_parquet(symbol_clean, n_bars)
            if len(bars) >= 50:  # minimum viable warmup
                source = "parquet"
                self.log.info(
                    f"Preloaded {len(bars)} bars from Parquet"
                )
        except Exception as e:
            self.log.warning(f"Parquet preload failed: {e}")

        # Attempt 2: Binance REST API
        if len(bars) < 50:
            try:
                bars = self._preload_from_binance_api(
                    symbol_clean, n_bars,
                )
                if bars:
                    source = "binance_api"
                    self.log.info(
                        f"Preloaded {len(bars)} bars from Binance API"
                    )
            except Exception as e:
                self.log.warning(f"Binance API preload failed: {e}")

        # Fill bar buffer
        if bars:
            for bar in bars[-n_bars:]:
                self._bars.append(bar)
            self._warmup_complete = True
            self.log.info(
                f"Warmup complete via {source} | "
                f"bars={len(self._bars)}"
            )
        else:
            self.log.warning(
                "Preload failed — waiting for live bars warmup"
            )
            self._warmup_complete = False

    def _preload_from_parquet(
        self, symbol: str, n_bars: int,
    ) -> list[Bar]:
        """Load last *n_bars* from the local Parquet store.

        Uses ``DataStore.get_klines()`` with a generous lookback
        window (100 days) and returns the tail.
        """
        # Lazy import — avoid crash if DuckDB / data is missing
        from src.ingestion.data_store import DataStore

        # Derive data root from features_dir: features_dir → ../../
        features_path = Path(self._config.features_dir).resolve()
        data_root = features_path.parent  # data/features

        if not data_root.exists():
            self.log.warning(f"Parquet data root not found: {data_root}")
            return []

        store = DataStore(data_root)
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=100)
            df = store.get_klines(
                symbol=symbol,
                interval="4h",
                start=start,
                end=now,
            )

            if df.is_empty():
                self.log.warning("Parquet returned empty DataFrame")
                return []

            bars: list[Bar] = []
            for row in df.iter_rows(named=True):
                ts_ns = int(row["open_time"]) * 1_000_000  # ms → ns
                bar = Bar(
                    bar_type=self._bar_type,
                    open=Price(float(row["open"]), precision=1),
                    high=Price(float(row["high"]), precision=1),
                    low=Price(float(row["low"]), precision=1),
                    close=Price(float(row["close"]), precision=1),
                    volume=Quantity(float(row["volume"]), precision=3),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
                bars.append(bar)

            # Sort by timestamp and take last n_bars
            bars.sort(key=lambda b: b.ts_event)
            return bars[-n_bars:]
        finally:
            store.close()

    def _preload_from_binance_api(
        self, symbol: str, n_bars: int,
    ) -> list[Bar]:
        """Load last *n_bars* via Binance Futures REST API.

        Uses synchronous ``requests`` because ``on_start()`` is called
        before the Nautilus event loop starts.

        Retries up to 3 times with 2-second delays.
        """
        import requests  # lazy — only needed for preload

        mode = self._config.trading_mode.lower()
        if mode == "testnet":
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"

        url = f"{base_url}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": "4h",
            "limit": min(n_bars, 500),
        }

        for attempt in range(1, 4):  # 3 retries
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                raw_klines = resp.json()

                bars: list[Bar] = []
                for k in raw_klines:
                    ts_ns = int(k[0]) * 1_000_000  # open_time ms → ns
                    bar = Bar(
                        bar_type=self._bar_type,
                        open=Price(float(k[1]), precision=1),
                        high=Price(float(k[2]), precision=1),
                        low=Price(float(k[3]), precision=1),
                        close=Price(float(k[4]), precision=1),
                        volume=Quantity(float(k[5]), precision=3),
                        ts_event=ts_ns,
                        ts_init=ts_ns,
                    )
                    bars.append(bar)

                self.log.info(
                    f"Binance API: fetched {len(bars)} klines "
                    f"from {base_url} (attempt {attempt})"
                )
                return bars

            except Exception as exc:
                self.log.warning(
                    f"Binance API attempt {attempt}/3 failed: {exc}"
                )
                if attempt < 3:
                    time.sleep(2)

        return []

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    def _record_equity(self, ts_ns: int) -> None:
        """Record equity snapshot for curve plotting."""
        account = self.portfolio.account(self._venue)
        if account is None:
            return
        try:
            balance = account.balance_total(USDT)
            upnl = self.portfolio.unrealized_pnl(self._instrument_id)
            equity = balance.as_double() + (
                upnl.as_double() if upnl is not None else 0.0
            )
            self._equity_curve.append((ts_ns, equity))
        except Exception:
            pass
