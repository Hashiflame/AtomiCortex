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
from typing import Any

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

    # Strategy overrides (forwarded to MLStrategyConfig)
    confidence_threshold: float = 0.65
    risk_per_trade: float = 0.01
    max_leverage: int = 10
    max_open_positions: int = 3
    models_dir: str = "./data/features/models"
    features_dir: str = "./data/features/ml_features"


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
        _log.info(
            f"LiveTrader created | mode={config.trading_mode} | "
            f"symbols={config.symbols} | dry_run={config.dry_run}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Data client
        data_client_config = BinanceDataClientConfig(
            api_key=api_key,
            api_secret=api_secret,
            account_type=BinanceAccountType.USDT_FUTURES,
            testnet=is_testnet,
            instrument_provider=instrument_provider,
        )

        # Exec client
        exec_client_config = BinanceExecClientConfig(
            api_key=api_key,
            api_secret=api_secret,
            account_type=BinanceAccountType.USDT_FUTURES,
            testnet=is_testnet,
            instrument_provider=instrument_provider,
            use_reduce_only=True,
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
            instrument_id = f"{symbol}.BINANCE"
            bar_type = f"{instrument_id}-4-HOUR-LAST-EXTERNAL"
            strat_config = MLStrategyConfig(
                instrument_id=instrument_id,
                bar_type=bar_type,
                confidence_threshold=cfg.confidence_threshold,
                models_dir=cfg.models_dir,
                features_dir=cfg.features_dir,
                risk_per_trade=cfg.risk_per_trade,
                max_leverage=cfg.max_leverage,
                max_open_positions=cfg.max_open_positions,
                initial_equity=cfg.initial_equity,
                dry_run=cfg.dry_run,
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
