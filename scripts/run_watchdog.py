#!/usr/bin/env python3
"""
AtomiCortex — External Watchdog launcher.

Runs the Watchdog as a standalone process (separate from the trading bot).
Should ideally be deployed on a different server for true independence.

Usage
-----
    python scripts/run_watchdog.py --redis-host localhost --trading-mode testnet
    python scripts/run_watchdog.py --redis-host 10.0.0.1 --trading-mode live
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.execution.watchdog import Watchdog, WatchdogConfig
from src.logger import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AtomiCortex External Watchdog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis hostname (default: localhost)",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)",
    )
    parser.add_argument(
        "--trading-mode",
        choices=["testnet", "live"],
        default="testnet",
        help="Trading mode (default: testnet)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=15,
        help="Seconds between heartbeat checks (default: 15)",
    )
    parser.add_argument(
        "--max-silence",
        type=int,
        default=60,
        help="Max silence before emergency close (default: 60s)",
    )
    parser.add_argument(
        "--heartbeat-key",
        default="atomicortex:heartbeat",
        help="Redis heartbeat key to monitor "
        "(default: atomicortex:heartbeat — the 4H bot). "
        "Use bot_15m_heartbeat for the isolated 15m watchdog.",
    )
    parser.add_argument(
        "--symbol",
        default="",
        help="Scope emergency close to ONE symbol (e.g. BTCUSDT). "
        "Empty (default) = legacy global close-all (4H watchdog).",
    )
    parser.add_argument(
        "--service-name",
        default="4h",
        help="Label for logs/alerts (e.g. 4h / 15m). Default: 4h.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    log = get_logger("run_watchdog")
    settings = get_settings()

    is_testnet = args.trading_mode == "testnet"

    config = WatchdogConfig(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=settings.redis_password,
        binance_api_key=(
            settings.binance_testnet_api_key if is_testnet
            else settings.binance_mainnet_api_key
        ),
        binance_api_secret=(
            settings.binance_testnet_api_secret if is_testnet
            else settings.binance_mainnet_api_secret
        ),
        trading_mode=args.trading_mode,
        heartbeat_key=args.heartbeat_key,
        symbol=args.symbol,
        service_name=args.service_name,
        check_interval=args.check_interval,
        max_silence_seconds=args.max_silence,
        telegram_token=settings.telegram_bot_token,
        telegram_admin_id=settings.telegram_admin_id,
    )

    watchdog = Watchdog(config)

    log.info("=" * 60)
    log.info("  AtomiCortex External Watchdog")
    log.info("=" * 60)
    log.info(f"  Service:        {args.service_name}")
    log.info(f"  Heartbeat key:  {args.heartbeat_key}")
    log.info(f"  Scope symbol:   {args.symbol or 'ALL (legacy)'}")
    log.info(f"  Redis:          {args.redis_host}:{args.redis_port}")
    log.info(f"  Trading Mode:   {args.trading_mode}")
    log.info(f"  Check Interval: {args.check_interval}s")
    log.info(f"  Max Silence:    {args.max_silence}s")
    log.info("=" * 60)

    # Graceful shutdown
    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        log.info("Signal received — stopping watchdog...")
        asyncio.create_task(watchdog.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    await watchdog.start()

    # Run until stopped
    try:
        while watchdog._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await watchdog.stop()


def main() -> None:
    args = parse_args()
    setup_logging(level_console=args.log_level)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
