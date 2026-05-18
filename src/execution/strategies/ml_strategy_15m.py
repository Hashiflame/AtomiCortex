"""
AtomiCortex — 15m ML Trading Strategy (isolated).

Phase 5 — multi-timeframe isolation. This strategy is fully independent
of the 4H bot: its own SQLite (``data/atomicortex_15m.db``), its own
heartbeat key (``bot_15m_heartbeat``, wired in Step 4), its own models
(``trend_model_15m`` + ``orb_model_15m``).

Differences from the 4H ``MLTradingStrategy`` (which it subclasses for
order / risk / SL plumbing):

* 15-MINUTE bar subscription.
* Two models: ``orb_model`` (ORB breakout — the GO model, Sharpe 2.18)
  and ``trend_model`` (trend regime).
* Features are built with ``FeaturePipeline.build_from_buffer`` so the
  live vector is byte-identical to the offline training matrix
  (``scripts/build_15m_dataset.build_feature_matrix``). The ORB model
  needs ``htf_1h_*`` / ``htf_4h_*`` / ``mtf`` features, so 1H and 4H
  frames are resampled from the 15m buffer (15m aligns exactly to the
  hour / 4h boundaries → OHLCV resample is lossless).
* Session-trap filter and ORB-breakout gating are evaluated from the
  produced feature columns (``is_session_trap_zone``,
  ``orb_breakout_bull/bear``) — the same logic ORBDetector used at
  training time, so there is no train/serve skew. (This means the
  trap check runs *after* feature build, not before as the step list
  numbered it — deliberate, to keep one source of truth.)

Known caveats (acceptable for paper trading, flagged for review):

* Nautilus ``Bar`` carries no taker-buy split, so ``build_from_buffer``
  falls back to the ``volume*0.5`` proxy for ``taker_buy_volume`` — the
  same compromise the 4H inline path already makes. CVD/taker features
  are therefore approximations until a taker feed is wired.
* Live funding / OI frames are not supplied (zero-filled, fail-soft —
  identical to the offline build when derivatives are absent).
* 4H HTF context needs a long history (RegimeDetector 4H hurst window);
  with a ~1500-bar 15m preload the htf_4h_* features start partially
  warmed and converge over the first days of paper trading.
"""

from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Price, Quantity

from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
    _bar_to_dict,
)
from src.logger import get_logger
from src.risk.risk_engine import TradeSignal

_log = get_logger(__name__)

_HOUR_MS = 3_600_000
_4H_MS = 4 * _HOUR_MS


# ---------------------------------------------------------------------------
# Config — Nautilus StrategyConfig subclass (keeps parent helpers working)
# ---------------------------------------------------------------------------

class MLStrategy15MConfig(MLStrategyConfig, frozen=True):
    """15m strategy config. Inherits every 4H field (risk_per_trade,
    max_leverage, initial_equity, dry_run, trading_mode, …) so the parent
    helper methods keep working unchanged, and overrides / adds the 15m
    specifics. Defaults mirror ``src.configs.strategy_15m.MLStrategyConfig15M``.
    """

    instrument_id: str = "BTCUSDT-PERP.BINANCE"
    bar_type: str = "BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL"
    interval: str = "15m"
    warmup_bars: int = 200
    confidence_threshold: float = 0.58
    max_open_positions: int = 1
    # Isolation
    signal_db_path: str = "data/atomicortex_15m.db"
    heartbeat_key: str = "bot_15m_heartbeat"
    # Models (two types)
    trend_model_path: str = "data/models/15m/trend_model_15m.pkl"
    orb_model_path: str = "data/models/15m/orb_model_15m.pkl"
    # 15m valid trend regimes (RegimeDetector15M output, lowercase)
    trend_regimes: tuple[str, ...] = ("trend_up", "trend_down")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MLTradingStrategy15M(MLTradingStrategy):
    """15m strategy: ORB-breakout + trend models on isolated infra.

    Reuses the parent's order submission, deferred SL, risk engine,
    portfolio tracker, signal bridge and equity tracking. Overrides
    model loading, feature computation, preload and the bar handler.
    """

    def __init__(self, config: MLStrategy15MConfig) -> None:
        super().__init__(config)
        self._config: MLStrategy15MConfig = config
        self._orb_model: Any = None
        self._orb_features: list[str] = []
        # FeaturePipeline is stateless w.r.t. data; build lazily in on_start.
        self._pipeline: Any = None
        # Dead-man's-switch heartbeat (isolated key). Fail-soft: stays
        # None if Redis / loop unavailable — never blocks trading.
        self._heartbeat: Any = None
        # Bound the buffer: enough 15m history to resample meaningful HTF
        # context while keeping memory finite.
        self._max_bars: int = max(config.warmup_bars, 1600)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Initialise 15m components and subscribe to 15m bars.

        Mirrors the parent's on_start structure but with the 15m regime
        detector, 15m models and 15m preload (parent's on_start would
        load 4H models and preload 4H bars — wrong for this bot — so it
        is intentionally not called).
        """
        self.log.info("MLTradingStrategy15M starting...")

        from src.risk.risk_engine import RiskConfig, RiskEngine
        from src.risk.portfolio_tracker import PortfolioTracker
        from src.features.regime_detector import RegimeDetector15M
        from src.features.feature_pipeline import FeaturePipeline

        risk_cfg = RiskConfig(
            risk_per_trade=self._config.risk_per_trade,
            max_leverage=self._config.max_leverage,
            max_open_positions=self._config.max_open_positions,
            confidence_threshold=self._config.confidence_threshold,
        )
        self._risk_engine = RiskEngine(risk_cfg, equity=self._config.initial_equity)
        self._tracker = PortfolioTracker(self._config.initial_equity)
        self._regime_detector = RegimeDetector15M()
        self._pipeline = FeaturePipeline(
            data_store=None,  # type: ignore[arg-type]
            symbol=self._symbol_base(),
            interval="15m",
        )

        self._load_models()

        self.subscribe_bars(self._bar_type)
        self.log.info(
            f"Subscribed to {self._bar_type} | dry_run={self._config.dry_run} "
            f"| warmup={self._config.warmup_bars} bars"
        )

        if self._config.preload_enabled:
            self._preload_historical_bars()

        try:
            from src.execution.signal_bridge import SignalBridge
            project_root = Path(__file__).resolve().parents[3]
            db_path = str(project_root / self._config.signal_db_path)
            self._signal_bridge = SignalBridge(db_path=db_path)
            self.log.info(f"SignalBridge initialised | db={db_path}")
        except Exception as exc:
            self.log.warning(f"SignalBridge init failed (non-fatal): {exc}")
            self._signal_bridge = None

        self._start_heartbeat()

        state = "COMPLETE" if self._warmup_complete else "IN_PROGRESS"
        self.log.info(
            f"15m strategy started | bars={len(self._bars)} | warmup={state}"
        )

    def on_stop(self) -> None:
        """Stop the heartbeat, then run the parent shutdown (which closes
        positions / cancels orders unless dry_run)."""
        if self._heartbeat is not None:
            try:
                import asyncio

                loop = asyncio.get_running_loop()
                loop.create_task(self._heartbeat.stop())
                self.log.info("15m heartbeat stop scheduled")
            except Exception as exc:
                self.log.warning(f"15m heartbeat stop failed: {exc}")
        super().on_stop()

    # ------------------------------------------------------------------
    # Heartbeat (isolated — bot_15m_heartbeat; fail-soft)
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
                f"15m heartbeat scheduled | key={self._config.heartbeat_key}"
            )
        except RuntimeError as exc:
            # No running loop (unit-test / non-async context).
            self.log.warning(
                f"15m heartbeat not started — no event loop ({exc})"
            )
            self._heartbeat = None
        except Exception as exc:
            self.log.warning(
                f"15m heartbeat init failed (non-fatal): {exc}"
            )
            self._heartbeat = None

    # ------------------------------------------------------------------
    # Model management (two models: trend + orb)
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load the 15m trend + ORB model bundles."""
        for name, path_str in (
            ("trend", self._config.trend_model_path),
            ("orb", self._config.orb_model_path),
        ):
            path = Path(path_str)
            if not path.exists():
                self.log.warning(f"15m {name} model not found: {path}")
                continue
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            booster = bundle["booster"]
            features = bundle.get("feature_columns", [])
            if name == "trend":
                self._trend_model = booster
                self._trend_features = features
            else:
                self._orb_model = booster
                self._orb_features = features
            self.log.info(
                f"Loaded 15m {name} model from {path} ({len(features)} feats)"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _symbol_base(self) -> str:
        """Base symbol (``BTCUSDT``) from the instrument id."""
        sym = str(self._instrument_id)
        return sym.split("-")[0] if "-" in sym else sym.split(".")[0]

    def _bars_to_df(self, bars: list[Bar]) -> pl.DataFrame:
        """Bar list → raw OHLCV DataFrame (build_from_buffer schema)."""
        return pl.DataFrame({
            "open_time": [int(b.ts_event // 1_000_000) for b in bars],  # ns→ms
            "open": [b.open.as_double() for b in bars],
            "high": [b.high.as_double() for b in bars],
            "low": [b.low.as_double() for b in bars],
            "close": [b.close.as_double() for b in bars],
            "volume": [b.volume.as_double() for b in bars],
        })

    @staticmethod
    def _resample(df15: pl.DataFrame, period_ms: int) -> pl.DataFrame:
        """Lossless OHLCV resample of the 15m frame to a higher TF.

        15m bars are aligned to :00/:15/:30/:45, so flooring open_time to
        the period boundary groups exactly 4 bars per hour / 16 per 4h.
        """
        return (
            df15.with_columns(
                ((pl.col("open_time") // period_ms) * period_ms).alias("_bucket")
            )
            .group_by("_bucket")
            .agg([
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ])
            .sort("_bucket")
            .rename({"_bucket": "open_time"})
        )

    def _build_feature_row(self) -> pl.DataFrame | None:
        """Build the current-bar feature row via build_from_buffer.

        Returns a single-row DataFrame with every produced feature plus
        ``symbol_encoded`` (the only model feature the pipeline does not
        emit — added here exactly as the 4H path does).
        """
        try:
            df15 = self._bars_to_df(self._bars)
            df1h = self._resample(df15, _HOUR_MS)
            df4h = self._resample(df15, _4H_MS)
            row = self._pipeline.build_from_buffer(
                df15,
                df_htf_4h=df4h,
                df_htf_1h=df1h,
                single_row=True,
            )
            if row.is_empty():
                return None
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            return row.with_columns(
                pl.lit(float(sym_map.get(self._symbol_base(), -1)))
                .alias("symbol_encoded")
            )
        except Exception as exc:
            self.log.error(f"15m feature build failed: {exc}")
            return None

    @staticmethod
    def _vector(row: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
        """Assemble the model input vector in trained feature order."""
        rd = {c: row[c][0] for c in row.columns}
        vec = np.array(
            [float(rd.get(f, 0.0) or 0.0) for f in feature_names],
            dtype=np.float64,
        )
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Bar handler
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """15m flow: warmup → features → session-trap → model select
        (ORB breakout vs trend) → ML → risk → execute (isolated DB).
        """
        try:
            self._bars.append(bar)
            if len(self._bars) > self._max_bars:
                self._bars = self._bars[-self._max_bars:]
            self._bar_count += 1
            self._record_equity(bar.ts_event)

            # 1. Warmup
            if not self._warmup_complete:
                if len(self._bars) >= self._config.warmup_bars:
                    self._warmup_complete = True
                    self.log.info(
                        f"15m warmup complete | bars={len(self._bars)}"
                    )
                else:
                    return

            # 2. Regime (RegimeDetector15M via parent helper)
            regime_state = self._detect_regime()
            if regime_state is None:
                return
            regime_label = regime_state.regime.value

            # 3. Features (produces ORB + session-trap columns)
            row = self._build_feature_row()
            if row is None:
                return
            rd = {c: row[c][0] for c in row.columns}

            # 4. Session-trap filter — skip first/last 2 bars of a session
            if bool(rd.get("is_session_trap_zone", False)):
                self.log.info(
                    f"15m: session-trap zone (bars_since_open="
                    f"{rd.get('bars_since_session_open')}) — skip"
                )
                return

            # 5. Model selection
            bull = bool(rd.get("orb_breakout_bull", False))
            bear = bool(rd.get("orb_breakout_bear", False))
            allowed_dir: int | None = None
            if bull ^ bear:  # exactly one ORB breakout side
                if self._orb_model is None:
                    self.log.warning("ORB breakout but orb_model unloaded — skip")
                    return
                model = self._orb_model
                feats = self._orb_features
                allowed_dir = 1 if bull else -1
                model_kind = "orb"
            elif regime_label in self._config.trend_regimes:
                if self._trend_model is None:
                    self.log.warning("trend regime but trend_model unloaded — skip")
                    return
                model = self._trend_model
                feats = self._trend_features
                model_kind = "trend"
            else:
                self.log.info(
                    f"15m: no ORB breakout, regime={regime_label} not "
                    f"tradable — skip"
                )
                return

            # 6. ML inference
            from src.models.lgbm_trainer import LGBMTrainer
            vec = self._vector(row, feats)
            direction, confidence = LGBMTrainer.get_signal(
                model, vec, confidence_threshold=self._config.confidence_threshold,
            )
            if direction == 0:
                self.log.info(
                    f"15m: no signal | {model_kind} conf={confidence:.3f}"
                )
                return

            # 7. ORB: only trade WITH the breakout side
            if allowed_dir is not None and direction != allowed_dir:
                self.log.info(
                    f"15m: ORB model dir={direction} vs breakout "
                    f"{allowed_dir} — skip (don't fight breakout)"
                )
                return

            # 8. Risk + execute (parent infra, isolated signal bridge)
            current_price = bar.close.as_double()
            atr_dollar = regime_state.atr_pct * current_price
            now_utc = datetime.fromtimestamp(bar.ts_event / 1e9, tz=timezone.utc)
            funding_rate = self._get_funding_rate(vec, feats)

            signal = TradeSignal(
                symbol=str(self._instrument_id),
                direction=direction,
                confidence=confidence,
                regime=f"{model_kind}:{regime_label}",
                entry_price=current_price,
                atr=atr_dollar,
                atr_pct=regime_state.atr_pct,
                funding_rate=funding_rate,
                timestamp=now_utc,
            )
            decision = self._risk_engine.evaluate(
                signal, self._tracker.get_state()
            )
            if not decision.approved:
                self.log.info(
                    f"15m signal BLOCKED | {model_kind} dir={direction} "
                    f"conf={confidence:.3f} | {decision.reason}"
                )
                return

            self.log.info(
                f"15m signal APPROVED | {model_kind} dir={direction} "
                f"conf={confidence:.3f} | size={decision.position_size:.6f}"
            )
            if not self._config.dry_run:
                self._open_position(decision, signal)
            else:
                self.log.info(
                    f"[DRY RUN 15m] {signal.direction} "
                    f"{decision.position_size:.6f} @ ${current_price:.2f} "
                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
                )

        except Exception as exc:
            import traceback
            self.log.error(f"15m on_bar EXCEPTION: {exc}")
            self.log.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Preload (15m klines — parent's preload is 4H-hardcoded)
    # ------------------------------------------------------------------

    def _preload_historical_bars(self) -> None:
        """Preload recent 15m klines from Binance (fail-soft).

        Fetches up to ``min(max_bars, 1500)`` 15m klines so HTF resamples
        have history. Never raises — on failure the strategy warms up
        from live bars instead.
        """
        try:
            import requests

            mode = self._config.trading_mode.lower()
            base = (
                "https://testnet.binancefuture.com"
                if mode == "testnet"
                else "https://fapi.binance.com"
            )
            limit = min(self._max_bars, 1500)
            resp = requests.get(
                f"{base}/fapi/v1/klines",
                params={
                    "symbol": self._symbol_base(),
                    "interval": "15m",
                    "limit": limit,
                },
                timeout=10,
            )
            resp.raise_for_status()
            klines = resp.json()
            for k in klines:
                ts_ns = int(k[0]) * 1_000_000
                self._bars.append(Bar(
                    bar_type=self._bar_type,
                    open=Price(float(k[1]), precision=1),
                    high=Price(float(k[2]), precision=1),
                    low=Price(float(k[3]), precision=1),
                    close=Price(float(k[4]), precision=1),
                    volume=Quantity(float(k[5]), precision=3),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                ))
            if len(self._bars) >= self._config.warmup_bars:
                self._warmup_complete = True
            self.log.info(
                f"15m preload: {len(self._bars)} bars from {base} "
                f"| warmup_complete={self._warmup_complete}"
            )
        except Exception as exc:
            self.log.warning(
                f"15m preload failed ({exc}) — warming from live bars"
            )
            self._warmup_complete = False
