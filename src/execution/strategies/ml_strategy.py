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
    # Isolation — consistent with MLStrategyConfig1H / MLStrategyConfig15M
    signal_db_path: str = "data/atomicortex.db"
    heartbeat_key: str = "atomicortex:heartbeat"


# ---------------------------------------------------------------------------
# Helper: bar → OHLCV dict
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float:
    """Convert a feature value to a float that LightGBM treats correctly.

    Returns ``NaN`` for any value LightGBM cannot interpret as a real
    measurement — ``None``, non-numeric, or ±inf. ``NaN`` is the only
    "missing" signal the booster understands; ``0.0`` would be read as
    a real measurement of zero and would erase the distinction between
    "data unavailable" (e.g. a warm-up rolling feature) and "value is
    actually zero" (e.g. funding rate genuinely at 0 %).
    """
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(f):
        return float("nan")
    return f


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

        # Live feature state (Phase 6 — train/serve skew fix)
        from src.features.live_feature_state import LiveFeatureState
        self._live_state = LiveFeatureState()
        self._pipeline: Any = None  # FeaturePipeline, init in on_start

        # Consecutive feature failure counter (Phase 6 observability)
        self._consecutive_feature_failures: int = 0

        # Dead-man's-switch heartbeat (isolated key — atomicortex:heartbeat
        # for the 4H bot; distinct from bot_15m_heartbeat / bot_1h_heartbeat).
        # Fail-soft: stays None if Redis / event loop unavailable so the bot
        # keeps trading even when monitoring is degraded.
        self._heartbeat: Any = None

        # Multi-level circuit breaker (master-doc thresholds: -2/-3% daily,
        # -8% weekly, -15% drawdown kill, 5 consecutive losses). Created in
        # on_start; stays None in unit tests that never call on_start.
        self._breaker: Any = None

        # Position reconciler — detects orphan positions on the exchange
        # (entry filled but bot crashed before SL) and ghosts in tracker
        # state. Report-only (log.critical); operator decides remediation.
        self._reconciler: Any = None

        # Crash-safe mirror of _pending_sl_params. Stays None if the store
        # can't be created; the bot then runs with in-memory only (a
        # degraded but non-fatal mode — same behaviour as before this fix).
        self._pending_store: Any = None

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

        # Resolve a shared crash-safe state file. Same dir as the signal DB
        # so all 4H runtime state co-locates (pending_sl_4h.json lives here
        # too). Fail-soft: if path resolution blows up, both classes accept
        # state_path=None and fall back to in-memory only.
        _risk_state_path = None
        try:
            _proj_root = Path(__file__).resolve().parents[3]
            _db_path = Path(self._config.signal_db_path)
            if not _db_path.is_absolute():
                _db_path = _proj_root / _db_path
            _risk_state_path = _db_path.parent / "risk_state_4h.json"
        except Exception as exc:
            self.log.warning(f"Risk state path resolution failed: {exc}")

        # 2. Portfolio Tracker
        self._tracker = PortfolioTracker(
            self._config.initial_equity, state_path=_risk_state_path,
        )

        # 2b. Circuit Breaker — multi-level trading-halt guard. Hard-coded
        # master-doc thresholds; evaluated each bar before regime detection.
        # Shares the same state file as PortfolioTracker so the persisted
        # day_start stamp keeps daily-trigger semantics consistent.
        from src.risk.circuit_breaker import CircuitBreaker
        self._breaker = CircuitBreaker(state_path=_risk_state_path)

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
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            db_path = str(project_root / self._config.signal_db_path)
            self._signal_bridge = SignalBridge(db_path=db_path)
            self.log.info(f"SignalBridge initialised | db={db_path}")
        except Exception as exc:
            self.log.warning(f"SignalBridge init failed (non-fatal): {exc}")
            self._signal_bridge = None

        # 8. FeaturePipeline for build_from_buffer (Phase 6)
        try:
            from src.features.feature_pipeline import FeaturePipeline
            sym = str(self._instrument_id)
            sym_base = sym.split("-")[0] if "-" in sym else sym.split(".")[0]
            self._pipeline = FeaturePipeline(
                data_store=None,  # type: ignore[arg-type]
                symbol=sym_base,
                interval="4h",
            )
            self.log.info("FeaturePipeline (4H) initialised for live inference")
        except Exception as exc:
            self.log.warning(f"FeaturePipeline init failed (non-fatal): {exc}")
            self._pipeline = None

        # 9. Subscribe to funding rate updates (Phase 6)
        try:
            from nautilus_trader.adapters.binance.futures.types import (
                BinanceFuturesMarkPriceUpdate,
            )
            from nautilus_trader.model.data import DataType
            self.subscribe_data(
                DataType(BinanceFuturesMarkPriceUpdate),
                instrument_id=self._instrument_id,
            )
            self.log.info("Subscribed to funding rate updates")
        except Exception as exc:
            self.log.warning(f"Funding rate subscription unavailable: {exc}")

        # 10. Schedule OI poll every 5 minutes (Phase 6)
        try:
            self.clock.set_timer(
                name="oi_poll",
                interval=__import__("datetime").timedelta(minutes=5),
                callback=self._poll_open_interest,
            )
            self.log.info("OI poll timer scheduled (every 5 min)")
        except Exception as exc:
            self.log.warning(f"OI poll timer failed (non-fatal): {exc}")

        # 11. Preload historical settled funding rates (Phase 6)
        try:
            import requests as _req
            mode = self._config.trading_mode.lower()
            _base = (
                "https://testnet.binancefuture.com"
                if mode == "testnet"
                else "https://fapi.binance.com"
            )
            _sym = str(self._instrument_id)
            _sym_clean = _sym.split("-")[0] if "-" in _sym else _sym.split(".")[0]
            resp = _req.get(
                f"{_base}/fapi/v1/fundingRate",
                params={"symbol": _sym_clean, "limit": 100},
                timeout=5,
            )
            if resp.status_code == 200:
                for fr in resp.json():
                    self._live_state.funding_rate_history.append({
                        "fundingTime": int(fr["fundingTime"]),
                        "fundingRate": float(fr["fundingRate"]),
                    })
                self.log.info(
                    f"Preloaded {len(self._live_state.funding_rate_history)} "
                    f"historical settled funding rates"
                )
        except Exception as exc:
            self.log.warning(f"Funding rate preload failed (non-fatal): {exc}")

        # 12. Dead-man's switch heartbeat — external watchdog reads this
        # key and emergency-closes positions if the bot stops writing.
        self._start_heartbeat()

        # 13. Position reconciler — detects orphan / ghost / mismatched
        # positions between tracker state and the exchange. Runs once at
        # start (catches leftovers from a previous crash) then every 15 min.
        self._schedule_reconciliation()

        # 14. Pending-SL crash-safe store. Restore any entries from a
        # prior run so that fills arriving after restart still trigger
        # SL placement instead of being mis-classified as exits.
        self._init_pending_store()

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
        """Graceful shutdown: stop heartbeat, close all positions, cancel orders."""
        # Stop the heartbeat first so the watchdog sees a clean shutdown
        # (key deleted) rather than mistaking a graceful stop for a crash.
        if self._heartbeat is not None:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(self._heartbeat.stop())
                self.log.info("4H heartbeat stop scheduled")
            except Exception as exc:
                self.log.warning(f"4H heartbeat stop failed: {exc}")

        self.log.info("MLTradingStrategy stopping — closing positions...")
        if not self._config.dry_run:
            self.cancel_all_orders(self._instrument_id)
            self.close_all_positions(self._instrument_id)
        self.log.info(
            f"Strategy stopped | bars_processed={self._bar_count} | "
            f"equity_points={len(self._equity_curve)}"
        )

    # ------------------------------------------------------------------
    # Heartbeat (isolated — atomicortex:heartbeat; fail-soft)
    # ------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        """Create + start the HeartbeatManager on the running event loop.

        Fail-soft on every path: missing Redis, no running loop (e.g.
        unit tests), or any error → log a warning and continue trading.
        ``HeartbeatManager`` itself already retries Redis internally and
        never raises, so a down Redis does not stop the bot.
        """
        try:
            import asyncio
            import os

            from src.execution.heartbeat import HeartbeatManager

            self._heartbeat = HeartbeatManager(
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_password=os.getenv("REDIS_PASSWORD", ""),
                heartbeat_key=self._config.heartbeat_key,
            )
            loop = asyncio.get_running_loop()
            loop.create_task(self._heartbeat.start())
            self.log.info(
                f"4H heartbeat scheduled | key={self._config.heartbeat_key}"
            )
        except RuntimeError as exc:
            # No running loop (unit-test / non-async context).
            self.log.warning(
                f"4H heartbeat not started — no event loop ({exc})"
            )
            self._heartbeat = None
        except Exception as exc:
            self.log.warning(
                f"4H heartbeat init failed (non-fatal): {exc}"
            )
            self._heartbeat = None

    # ------------------------------------------------------------------
    # Position reconciliation (orphan / ghost detection; report-only)
    # ------------------------------------------------------------------

    def _schedule_reconciliation(self) -> None:
        """Create the reconciler + immediate run + 15-min recurring timer.

        Fail-soft: any failure here logs a warning and leaves the bot
        running without reconciliation (a degraded but non-fatal state).
        """
        try:
            from datetime import timedelta as _td

            from src.config import get_settings
            from src.execution.reconciler import PositionReconciler

            settings = get_settings()
            self._reconciler = PositionReconciler(
                binance_api_key=settings.binance_api_key,
                binance_api_secret=settings.binance_api_secret,
                trading_mode=self._config.trading_mode,
                auto_fix=False,  # report-only: operator decides remediation
            )

            # Immediate run — main reason the reconciler exists at startup:
            # catch positions left open on the exchange after a prior crash.
            self._run_reconciliation()

            # Recurring run — every 15 min, matches reconciler_signals cadence.
            try:
                self.clock.set_timer(
                    name="position_reconcile",
                    interval=_td(minutes=15),
                    callback=lambda _ev: self._run_reconciliation(),
                )
                self.log.info("Position reconciler scheduled (every 15 min)")
            except Exception as exc:
                self.log.warning(
                    f"Reconciler timer failed (non-fatal): {exc}"
                )
        except Exception as exc:
            self.log.warning(f"Reconciler init failed (non-fatal): {exc}")
            self._reconciler = None

    def _run_reconciliation(self) -> None:
        """Sync entry point: schedule the async reconcile coroutine."""
        if self._reconciler is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._reconcile_async())
        except Exception as exc:
            self.log.warning(
                f"Reconciler schedule failed (non-fatal): {exc}"
            )

    async def _reconcile_async(self) -> None:
        """Compare tracker state with exchange; log CRITICAL on any drift."""
        if self._reconciler is None:
            return
        try:
            from src.execution.reconciler import InternalPosition

            internal: dict[str, InternalPosition] = {}
            if self._tracker is not None:
                # Tracker is keyed by full instrument id ("BTCUSDT-PERP.BINANCE");
                # Binance positionRisk returns bare "BTCUSDT". Normalise so the
                # symbol sets line up and our positions don't look like ghosts.
                for sym_full, pos in self._tracker._positions.items():
                    norm = (
                        sym_full.split("-")[0]
                        if "-" in sym_full
                        else sym_full.split(".")[0]
                    )
                    internal[norm] = InternalPosition(
                        symbol=norm,
                        direction=pos.direction,
                        quantity=pos.quantity,
                    )

            result = await self._reconciler.reconcile(internal)
            if result.is_clean:
                return

            for orphan in result.orphan_positions:
                self.log.critical(
                    f"ORPHAN POSITION on exchange — manual action required | "
                    f"sym={orphan['symbol']} dir={orphan['direction']} "
                    f"qty={orphan['quantity']} entry={orphan.get('entry_price')}"
                )
            for ghost in result.ghost_positions:
                self.log.critical(
                    f"GHOST POSITION in tracker — closed on exchange | "
                    f"sym={ghost['symbol']} dir={ghost['direction']} "
                    f"qty={ghost['quantity']}"
                )
            for mm in result.mismatched_sizes:
                self.log.critical(
                    f"POSITION SIZE MISMATCH | sym={mm['symbol']} "
                    f"internal=({mm['internal_direction']},"
                    f"{mm['internal_quantity']}) "
                    f"exchange=({mm['exchange_direction']},"
                    f"{mm['exchange_quantity']})"
                )
        except Exception as exc:
            self.log.warning(
                f"Reconciliation failed (non-fatal): {exc}"
            )

    # ------------------------------------------------------------------
    # Pending-SL persistence (crash-safe mirror of _pending_sl_params)
    # ------------------------------------------------------------------

    def _init_pending_store(self) -> None:
        """Create the on-disk store and replay any entries from a prior run.

        Restored entries are NOT used to re-submit stops directly — that
        could double-place SL orders if the fill already happened before
        the crash. They populate ``_pending_sl_params`` so that the next
        fill event for the same client_order_id is recognised as an entry
        and goes through the normal SL placement path. The case where the
        fill happened *during* the crash window (Binance does not replay)
        is covered by the PositionReconciler ORPHAN alert.
        """
        try:
            from src.execution.pending_orders_store import PendingOrdersStore

            project_root = Path(__file__).resolve().parents[3]
            db_path = Path(self._config.signal_db_path)
            if not db_path.is_absolute():
                db_path = project_root / db_path
            store_path = db_path.parent / "pending_sl_4h.json"

            self._pending_store = PendingOrdersStore(store_path)
            restored = self._pending_store.load_all()
            if restored:
                # Replay into the in-memory dict the rest of the strategy uses.
                for oid, entry in restored.items():
                    if oid not in self._pending_sl_params:
                        self._pending_sl_params[oid] = entry
                self.log.warning(
                    f"Restored {len(restored)} pending SL params from "
                    f"{store_path} — awaiting fill events to place stops"
                )
            else:
                self.log.info(
                    f"Pending-SL store initialised | path={store_path}"
                )
        except Exception as exc:
            self.log.warning(
                f"Pending-SL store init failed (non-fatal, in-memory only): {exc}"
            )
            self._pending_store = None

    # ------------------------------------------------------------------
    # Data handler (Phase 6 — live funding feed)
    # ------------------------------------------------------------------

    def on_data(self, data) -> None:
        """Handle incoming data events (funding rate updates).

        Called by Nautilus when subscribed data arrives. Currently handles
        ``BinanceFuturesMarkPriceUpdate`` for live funding rate.

        Point-in-time fix (Phase 6)
        ---------------------------
        ``BinanceFuturesMarkPriceUpdate`` streams the **predicted** funding
        rate every second. Training data uses **settled** rates (final value
        at settlement: 01:00, 09:00, 17:00 UTC).

        We update ``self._live_state.funding_rate`` on every tick (for the
        current ``funding_rate`` feature), but only append to
        ``funding_rate_history`` at settlement times so that rolling features
        (zscore, cum_24h) match the training distribution.
        """
        try:
            from nautilus_trader.adapters.binance.futures.types import (
                BinanceFuturesMarkPriceUpdate,
            )
            if isinstance(data, BinanceFuturesMarkPriceUpdate):
                rate = float(data.funding_rate)
                ts_ms = data.ts_event // 1_000_000

                # Always update current rate (used as point-in-time feature)
                self._live_state.update_funding(
                    rate=rate, timestamp_ms=ts_ms,
                )

                # Append to history ONLY at settlement (every 8h)
                from datetime import datetime as _dt, timezone as _tz
                dt = _dt.fromtimestamp(ts_ms / 1000, tz=_tz.utc)
                if dt.hour in (1, 9, 17) and dt.minute == 0:
                    self._live_state.funding_rate_history.append({
                        "fundingTime": ts_ms,
                        "fundingRate": rate,
                    })
                    self.log.info(
                        f"Funding settled: {rate:.6f} @ {dt.isoformat()}"
                    )
        except Exception as exc:
            self.log.debug(f"on_data error (non-critical): {exc}")

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

            # Phase 6: track bar in live feature state
            self._live_state.add_bar(bar, interval="4h")

            # Timestamp diagnostic (first bar only — verify ts_event conversion)
            if self._bar_count == 1 and self._live_state.bar_buffer_4h:
                _diag = self._live_state.bar_buffer_4h[-1]
                _close_ms = bar.ts_event // 1_000_000
                _open_ms = _diag["open_time"]
                self.log.info(
                    f"TIMESTAMP DIAGNOSTIC: "
                    f"ts_event_ms={_close_ms} "
                    f"open_time_ms={_open_ms} "
                    f"diff_hours={(_close_ms - _open_ms) / 3_600_000:.1f}h "
                    f"(expected 4.0h for 4H bars)"
                )

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

            # 1b. Circuit breaker — block new entries when triggered.
            # Existing positions stay protected by their exchange-side SL/TP;
            # auto-flattening on drawdown would sell into the panic.
            if self._breaker is not None and self._tracker is not None:
                try:
                    funding = float(
                        getattr(self._live_state, "funding_rate", 0.0) or 0.0
                    )
                    # ATR=0 disables only the vol-spike branch (which never
                    # sets is_triggered=True); loss / drawdown / consecutive-
                    # loss guards run from portfolio_state alone.
                    breaker_state = self._breaker.check(
                        portfolio_state=self._tracker.get_state(),
                        current_atr=0.0,
                        avg_atr=0.0,
                        current_funding=funding,
                    )
                    if breaker_state.is_triggered:
                        self.log.warning(
                            f"CIRCUIT BREAKER TRIPPED — skipping bar | "
                            f"reason={breaker_state.trigger_reason}"
                        )
                        return
                except Exception as exc:
                    self.log.warning(
                        f"Circuit breaker check failed (non-fatal): {exc}"
                    )

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

            # 4. Compute features (Phase 6: unified pipeline)
            self.log.info(
                f"on_bar step 4: computing features "
                f"(n={len(features_list)})"
            )
            feature_vector = self._compute_features_unified(features_list)
            if feature_vector is None:
                self.log.info("on_bar: feature_vector is None — skipping")
                return

            # 5. Get ML signal
            self.log.info("on_bar step 5: ML prediction")
            from src.models.lgbm_trainer import LGBMTrainer
            direction, confidence = LGBMTrainer.get_signal(
                model,
                feature_vector,
                confidence_threshold=conf_threshold,
            )
            self.log.info(
                f"on_bar step 6: dir={direction} confidence={confidence:.3f} "
                f"threshold={conf_threshold} regime={regime_label}"
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
            # Mirror the removal to disk so the entry isn't replayed on
            # the next restart (would mis-classify a future exit fill).
            if self._pending_store is not None:
                try:
                    self._pending_store.pop(client_oid)
                except Exception as exc:
                    self.log.warning(
                        f"Pending-SL store pop failed (non-fatal): {exc}"
                    )
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
        client_oid = str(entry_order.client_order_id)
        self._pending_sl_params[client_oid] = {
            "decision": decision,
            "signal": signal,
        }
        # Mirror to disk so a crash between submit_order and on_order_filled
        # cannot lose the SL params (which would leave the position unstopped).
        if self._pending_store is not None:
            try:
                self._pending_store.put(client_oid, decision, signal)
            except Exception as exc:
                self.log.warning(
                    f"Pending-SL store put failed (non-fatal): {exc}"
                )

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

    def _compute_features_unified(
        self, feature_names: list[str],
    ) -> np.ndarray | None:
        """Build a feature vector via FeaturePipeline.build_from_buffer().

        Phase 6 — eliminates train/serve skew by using the same transforms
        as the offline training pipeline. Real funding rate and OI data
        from ``LiveFeatureState`` are injected.

        Falls back gracefully: if the pipeline is unavailable or fails,
        logs an error and returns None (caller skips the bar — never
        trades on bad features).
        """
        if self._pipeline is None:
            self.log.warning(
                "_compute_features_unified: pipeline not available — skipping"
            )
            return None

        try:
            import polars as pl

            df_bars = self._live_state.get_bar_df("4h")
            if df_bars.is_empty() or len(df_bars) < 50:
                self.log.info(
                    f"_compute_features_unified: insufficient bars "
                    f"({len(df_bars)}) — skipping"
                )
                return None

            funding_df = self._live_state.get_funding_df()
            metrics_df = self._live_state.get_metrics_df()

            features_df = self._pipeline.build_from_buffer(
                df=df_bars,
                funding_df=funding_df,
                metrics_df=metrics_df,
                single_row=True,
            )

            if features_df.is_empty():
                self.log.warning("build_from_buffer returned empty — skipping")
                return None

            # Build vector in trained feature order
            rd = {c: features_df[c][0] for c in features_df.columns}

            # Add symbol_encoded (not emitted by pipeline)
            sym_str = str(self._instrument_id)
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            base = sym_str.split("-")[0] if "-" in sym_str else sym_str.split(".")[0]
            rd["symbol_encoded"] = float(sym_map.get(base, -1))

            # Preserve NaN for genuinely-missing features; ±inf collapses
            # to NaN inside _safe_float. LightGBM routes NaN to its
            # optimal branch — keeps train/serve consistent with the
            # nan-preserving fit path in lgbm_trainer (Phase 3 Step 3.3).
            vector = np.array(
                [_safe_float(rd.get(f)) for f in feature_names],
                dtype=np.float64,
            )
            self._consecutive_feature_failures = 0
            return vector

        except Exception as exc:
            self._consecutive_feature_failures += 1
            self.log.error(f"_compute_features_unified failed: {exc}")
            if self._consecutive_feature_failures >= 3:
                self.log.critical(
                    f"ALERT: {self._consecutive_feature_failures} consecutive "
                    f"feature computation failures — bot NOT trading!"
                )
            return None

    def _compute_features(self, feature_names: list[str]) -> np.ndarray | None:
        """Build a feature vector from the current bar buffer.

        .. deprecated::
            DEPRECATED — replaced by ``_compute_features_unified()`` in
            Phase 6.  Kept as reference only; not called from ``on_bar()``.

        Uses the last 50 bars to compute microstructure features,
        then extracts only the features the model was trained on.

        KNOWN ISSUE: Train/serve feature skew
        -------------------------------------
        This method hand-rolls 4H features using heuristics:
          - funding_rate, oi_* = 0.0 (placeholders — no live feed here)
          - CVD computed from the bar buffer (approximation)
          - session features not included (4H does not use them)

        The 4H model was trained with ``FeaturePipeline.build()`` which
        uses real funding / OI from the Binance metrics API. The
        resulting distribution shift may degrade prediction quality
        (it is *not* a sign flip — direction mapping is correct).

        TODO: replace with ``build_from_buffer()`` like
        ``ml_strategy_15m.py`` does. That requires loading
        funding/metrics in ``on_start()`` and maintaining a live
        derivatives feed. Risk: this is live trading code — schedule
        for a maintenance window, not a hot change.
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

            # Build vector in correct order. NaN-preserving (see Phase 3
            # Step 3.3): missing / None / ±inf → NaN so LightGBM handles
            # them as "missing" rather than as a literal zero measurement.
            vector = np.array(
                [_safe_float(feature_dict.get(f)) for f in feature_names],
                dtype=np.float64,
            )
            return vector

        except Exception as exc:
            self.log.error(f"Feature computation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Periodic OI poll (Phase 6)
    # ------------------------------------------------------------------

    def _poll_open_interest(self, event=None) -> None:
        """Non-blocking OI poll — schedules async task on event loop.

        Timer callbacks run on the Nautilus event loop thread.
        ``requests.get()`` is synchronous and would block the loop for
        50-5000 ms.  Instead we offload to a thread via
        ``asyncio.to_thread``.

        Parameters
        ----------
        event:
            Timer event from Nautilus (unused, required by callback API).
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._poll_open_interest_async())
            else:
                # Fallback for non-async context (tests / backtest)
                self._poll_open_interest_sync()
        except Exception as exc:
            self.log.debug(f"OI poll schedule failed (non-critical): {exc}")

    async def _poll_open_interest_async(self) -> None:
        """Async OI fetch — offloads blocking HTTP to thread pool.

        Converts OI from contracts (Binance ``/fapi/v1/openInterest``)
        to USDT-value to match training data (``sum_open_interest_value``).
        """
        try:
            import asyncio
            import requests

            mode = self._config.trading_mode.lower()
            base = (
                "https://testnet.binancefuture.com"
                if mode == "testnet"
                else "https://fapi.binance.com"
            )
            sym = str(self._instrument_id)
            sym_clean = sym.split("-")[0] if "-" in sym else sym.split(".")[0]

            resp = await asyncio.to_thread(
                requests.get,
                f"{base}/fapi/v1/openInterest",
                params={"symbol": sym_clean},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                oi_contracts = float(data.get("openInterest", 0))
                ts = int(data.get("time", 0))

                # Convert contracts → USDT-value (training uses
                # sum_open_interest_value from /futures/data/openInterestHist)
                current_price = (
                    self._bars[-1].close.as_double()
                    if self._bars else 0.0
                )
                oi_usdt = (
                    oi_contracts * current_price
                    if current_price > 0 else oi_contracts
                )

                self._live_state.update_oi(oi=oi_usdt, timestamp_ms=ts)
                self.log.debug(
                    f"OI poll: {oi_contracts:.0f} contracts "
                    f"= ${oi_usdt / 1e9:.2f}B USDT"
                )
            else:
                self.log.debug(f"OI poll: HTTP {resp.status_code}")
        except Exception as exc:
            self.log.debug(f"OI async poll failed (non-critical): {exc}")

    def _poll_open_interest_sync(self) -> None:
        """Synchronous OI fetch — fallback for non-async contexts."""
        try:
            import requests

            mode = self._config.trading_mode.lower()
            base = (
                "https://testnet.binancefuture.com"
                if mode == "testnet"
                else "https://fapi.binance.com"
            )
            sym = str(self._instrument_id)
            sym_clean = sym.split("-")[0] if "-" in sym else sym.split(".")[0]

            resp = requests.get(
                f"{base}/fapi/v1/openInterest",
                params={"symbol": sym_clean},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                oi_contracts = float(data.get("openInterest", 0))
                ts = int(data.get("time", 0))
                current_price = (
                    self._bars[-1].close.as_double()
                    if self._bars else 0.0
                )
                oi_usdt = (
                    oi_contracts * current_price
                    if current_price > 0 else oi_contracts
                )
                self._live_state.update_oi(oi=oi_usdt, timestamp_ms=ts)
                self.log.debug(
                    f"OI sync poll: {oi_contracts:.0f} contracts "
                    f"= ${oi_usdt / 1e9:.2f}B USDT"
                )
        except Exception as exc:
            self.log.debug(f"OI sync poll failed (non-critical): {exc}")

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

    def _bar_period_hours(self) -> float:
        """Bar period in hours from ``config.interval`` ('4h'→4.0,
        '1h'→1.0, '15m'→0.25). Defaults to 4.0 on any parse failure
        (4H is the production timeframe)."""
        iv = str(getattr(self._config, "interval", "4h")).strip().lower()
        try:
            if iv.endswith("h"):
                return float(iv[:-1])
            if iv.endswith("m"):
                return float(iv[:-1]) / 60.0
            if iv.endswith("d"):
                return float(iv[:-1]) * 24.0
        except ValueError:
            pass
        return 4.0

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

        # Freshness guard: the Parquet store can end well before "now"
        # (e.g. data tooling stopped at 2025-12-31 while the bot runs in
        # 2026). If the newest preloaded bar is older than 2× the bar
        # period, the data is stale — discard it so the Binance REST
        # fallback below runs instead of feeding the model a stale /
        # discontinuous window. WARNING only — never stops the bot.
        if bars:
            bar_period_h = self._bar_period_hours()
            latest_dt = datetime.fromtimestamp(
                bars[-1].ts_event / 1e9, tz=timezone.utc
            )
            staleness_h = (
                datetime.now(timezone.utc) - latest_dt
            ).total_seconds() / 3600.0
            if staleness_h > bar_period_h * 2:
                self.log.warning(
                    f"Preloaded Parquet bars are stale: "
                    f"latest={latest_dt.isoformat()} "
                    f"staleness={staleness_h:.1f}h > "
                    f"{bar_period_h * 2:.1f}h — discarding, "
                    f"falling back to Binance REST"
                )
                bars = []
                source = "none"

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

        # Fill bar buffer + live feature state
        if bars:
            for bar in bars[-n_bars:]:
                self._bars.append(bar)
                self._live_state.add_bar(bar, interval="4h")
            self._warmup_complete = True
            self.log.info(
                f"Warmup complete via {source} | "
                f"bars={len(self._bars)} | "
                f"live_state_4h={len(self._live_state.bar_buffer_4h)}"
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
