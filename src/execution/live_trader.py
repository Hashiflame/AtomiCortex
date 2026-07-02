"""
AtomiCortex — Live Trader.

Manages live/testnet connections through Nautilus TradingNode.
Configures Binance USDT-futures data + execution clients and
wires up the MLTradingStrategy.

Phase 4 — Step 4.5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import src.patches.nautilus_enums  # Hotfix for TRADING_HALT

if TYPE_CHECKING:
    from nautilus_trader.trading.strategy import Strategy

from nautilus_trader.adapters.binance import (
    BinanceAccountType,
    BinanceDataClientConfig,
    BinanceExecClientConfig,
    BinanceLiveDataClientFactory,
    BinanceLiveExecClientFactory,
)
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from src.config import get_settings
from src.execution.startup_check import EngineConnectionChecker
from src.execution.strategies.ml_strategy import MLStrategyConfig, MLTradingStrategy
from src.logger import get_logger, setup_logging

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LiveTraderConfig:
    """Top-level config for the live trader."""

    trading_mode: str = "testnet"        # testnet / paper / live
    symbols: list[str] = field(
        default_factory=lambda: ["BTCUSDT-PERP"],
    )
    initial_equity: float = 10_000.0
    dry_run: bool = False
    log_level: str = "INFO"

    # Strategy overrides (forwarded to MLStrategyConfig).
    #
    # H12: confidence_threshold defaults to ``None`` so that build_node()
    # pulls the value from ``Settings.confidence_threshold`` (.env-driven).
    # Pre-H12 the dataclass default 0.65 happened to match the Settings
    # default by coincidence, so changing CONFIDENCE_THRESHOLD in .env
    # silently had no effect. Explicit values (``LiveTraderConfig(
    # confidence_threshold=0.7)``) still win — they short-circuit the
    # Settings lookup.
    #
    # NOTE: the 15m path is wired via ``strategy_factory`` and bypasses
    # this field entirely; per-TF factories must read Settings directly.
    confidence_threshold: float | None = None
    risk_per_trade: float = 0.01
    max_leverage: int = 10
    max_open_positions: int = 3
    models_dir: str = "./data/features/models"
    features_dir: str = "./data/features/ml_features"

    # Fail-fast grace period (seconds).  None → read from Settings
    # (patten H12, same as confidence_threshold).
    startup_grace_sec: float | None = None

    # Strategy injection hook (Phase 5 — multi-timeframe isolation).
    #
    # When None (default) ``build_node`` builds the 4H ``MLStrategyConfig``
    # + ``MLTradingStrategy`` exactly as before — the running 4H bot is
    # byte-for-byte unaffected.
    #
    # When set, it is called once per symbol as
    # ``strategy_factory(cfg, symbol) -> Strategy`` and must return a
    # fully-constructed Nautilus ``Strategy`` (with its own
    # ``StrategyConfig``, bar_type, signal_db_path, heartbeat_key, …).
    # The 15m / 1H launchers supply this so LiveTrader stays generic and
    # no per-timeframe branching leaks into the shared 4H path.
    strategy_factory: Callable[["LiveTraderConfig", str], "Strategy"] | None = None


# ---------------------------------------------------------------------------
# Live Trader
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Builds and runs a Nautilus TradingNode for live/testnet trading
    with the ML strategy.
    """

    def __init__(self, config: LiveTraderConfig) -> None:
        self._config = config
        self._node: TradingNode | None = None
        self._startup_checker: EngineConnectionChecker | None = None
        _log.info(
            f"LiveTrader created | mode={config.trading_mode} | "
            f"symbols={config.symbols} | dry_run={config.dry_run}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def startup_failed(self) -> bool:
        """Whether the post-startup engine check detected a failure."""
        if self._startup_checker is None:
            return False
        return self._startup_checker.engines_failed

    def build_node(self) -> TradingNode:
        """
        Build a fully configured TradingNode with Binance futures
        data + exec clients and the MLTradingStrategy.
        """
        settings = get_settings()
        cfg = self._config

        is_testnet = cfg.trading_mode.lower() == "testnet"

        # Resolve API keys
        if is_testnet:
            api_key = settings.binance_testnet_api_key
            api_secret = settings.binance_testnet_api_secret
        else:
            api_key = settings.binance_mainnet_api_key
            api_secret = settings.binance_mainnet_api_secret

        if not api_key or not api_secret:
            raise ValueError(
                f"Binance API keys not configured for mode '{cfg.trading_mode}'. "
                f"Set BINANCE_{'TESTNET_' if is_testnet else ''}API_KEY "
                f"and BINANCE_{'TESTNET_' if is_testnet else ''}API_SECRET in .env"
            )

        # Instrument provider: load all futures instruments
        instrument_provider = InstrumentProviderConfig(load_all=True)

        # Binance migrated futures WS to /market and /private path prefixes
        # (mainnet deadline 2026-04-23). Nautilus appends "/stream?streams=..."
        # to base_url_ws, so we override only the host+prefix here.
        # Testnet still uses the legacy endpoint without the /market prefix.
        if is_testnet:
            ws_host_market = "wss://stream.binancefuture.com"
            ws_host_private = "wss://stream.binancefuture.com"
        else:
            ws_host_market = "wss://fstream.binance.com/market"
            ws_host_private = "wss://fstream.binance.com/private"

        # Data client
        data_client_config = BinanceDataClientConfig(
            api_key=api_key,
            api_secret=api_secret,
            account_type=BinanceAccountType.USDT_FUTURES,
            testnet=is_testnet,
            instrument_provider=instrument_provider,
            base_url_ws=ws_host_market,
        )

        # Exec client
        exec_client_config = BinanceExecClientConfig(
            api_key=api_key,
            api_secret=api_secret,
            account_type=BinanceAccountType.USDT_FUTURES,
            testnet=is_testnet,
            instrument_provider=instrument_provider,
            use_reduce_only=True,
            base_url_ws=ws_host_private,
        )

        # Build TradingNode config
        node_config = TradingNodeConfig(
            trader_id=TraderId("ATOMICORTEX-001"),
            logging=LoggingConfig(
                log_level=cfg.log_level,
                bypass_logging=False,
            ),
            data_clients={"BINANCE": data_client_config},
            exec_clients={"BINANCE": exec_client_config},
            timeout_connection=30.0,
            timeout_reconciliation=10.0,
            timeout_portfolio=10.0,
            timeout_disconnection=10.0,
            timeout_post_stop=5.0,
        )

        # Build node
        node = TradingNode(config=node_config)

        # Register Binance factories
        node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)

        # Build clients
        node.build()

        # Add strategy instances manually (after build)
        for symbol in cfg.symbols:
            if cfg.strategy_factory is not None:
                # Injected path (15m / 1H launchers). LiveTrader stays
                # generic; the factory owns config + bar_type + isolation.
                strategy = cfg.strategy_factory(cfg, symbol)
                _log.info(
                    f"Strategy via factory | {type(strategy).__name__} "
                    f"| symbol={symbol}"
                )
            else:
                # Default 4H path — unchanged (running bot unaffected).
                instrument_id = f"{symbol}.BINANCE"
                bar_type = f"{instrument_id}-4-HOUR-LAST-EXTERNAL"
                # H12: when LiveTraderConfig didn't pin an explicit
                # confidence_threshold, fall through to the Settings
                # value so CONFIDENCE_THRESHOLD in .env actually flows
                # into the strategy / risk engine.
                conf_threshold = (
                    cfg.confidence_threshold
                    if cfg.confidence_threshold is not None
                    else settings.confidence_threshold
                )
                strat_config = MLStrategyConfig(
                    instrument_id=instrument_id,
                    bar_type=bar_type,
                    confidence_threshold=conf_threshold,
                    models_dir=cfg.models_dir,
                    features_dir=cfg.features_dir,
                    risk_per_trade=cfg.risk_per_trade,
                    max_leverage=cfg.max_leverage,
                    max_open_positions=cfg.max_open_positions,
                    initial_equity=cfg.initial_equity,
                    dry_run=cfg.dry_run,
                    trading_mode=cfg.trading_mode,
                )
                strategy = MLTradingStrategy(config=strat_config)
            node.trader.add_strategy(strategy)

        self._node = node
        _log.info(
            f"TradingNode built | strategies={len(cfg.symbols)} | "
            f"testnet={is_testnet}"
        )
        return node

    def run(self) -> None:
        """Start the TradingNode (blocks until stopped).

        ``TradingNode.run()`` calls ``loop.run_until_complete(run_async())``
        internally and blocks.  When ``stop()`` is invoked (from a signal
        handler or duration timer), ``node.stop()`` schedules
        ``stop_async()`` on the same loop, which cancels the engine tasks
        and lets ``run_async()`` return naturally.
        """
        if self._node is None:
            self.build_node()

        # Fail-fast: daemon thread checks engine connectivity after
        # grace period and sends SIGTERM if engines are disconnected.
        self._start_connection_checker()

        _log.info("Starting TradingNode...")
        try:
            self._node.run()
        except KeyboardInterrupt:
            _log.info("KeyboardInterrupt — stopping...")
        except Exception as exc:
            _log.error(f"TradingNode run error: {exc}")
        finally:
            self._dispose()

    def stop(self) -> None:
        """Request a graceful stop.

        This is safe to call from **any thread** (including signal
        handlers and duration-timer threads).  It calls
        ``TradingNode.stop()`` which internally checks
        ``loop.is_running()`` and schedules ``stop_async()`` as a task
        on the correct event loop — no cross-thread loop access.

        ``dispose()`` is NOT called here; it is called in ``run()``'s
        ``finally`` block after the loop has stopped.
        """
        if self._node is not None:
            _log.info("Requesting TradingNode stop...")
            try:
                self._node.stop()
            except Exception as exc:
                _log.warning(f"Error requesting stop: {exc}")

    def _start_connection_checker(self) -> None:
        """Create and start the engine connection checker (best-effort).

        If checker creation fails for any reason, log a warning and
        continue — degraded monitoring is better than blocking startup.
        """
        try:
            settings = get_settings()
            cfg = self._config

            grace = (
                cfg.startup_grace_sec
                if cfg.startup_grace_sec is not None
                else settings.startup_grace_sec
            )

            reporter = None
            if settings.telegram_bot_token and settings.telegram_admin_id:
                from src.monitoring.telegram_reporter import TelegramReporter
                reporter = TelegramReporter(
                    bot_token=settings.telegram_bot_token,
                    admin_id=settings.telegram_admin_id,
                )
            else:
                _log.debug(
                    "Telegram not configured — startup checker will "
                    "skip alert on failure"
                )

            self._startup_checker = EngineConnectionChecker(
                node=self._node,
                grace_sec=grace,
                reporter=reporter,
            )
            self._startup_checker.start()
            _log.info(
                "Engine connection checker started | grace={g}s",
                g=grace,
            )
        except Exception as exc:
            _log.warning(
                "Failed to start engine connection checker (non-fatal): {err}",
                err=str(exc),
            )

    def _dispose(self) -> None:
        """Dispose of the node after the event loop has stopped.

        Called only from ``run()``'s ``finally`` block, where the loop
        is guaranteed to have stopped.
        """
        if self._node is not None:
            _log.info("Disposing TradingNode...")
            try:
                # Give engines a moment to finish flushing
                time.sleep(1)
                self._node.dispose()
            except Exception as exc:
                _log.warning(f"Error during dispose: {exc}")
            finally:
                self._node = None
            _log.info("TradingNode disposed")
