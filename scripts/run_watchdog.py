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
from src.execution.watchdog import (
    DEFAULT_HEARTBEAT_KEY,
    STRATEGY_HEARTBEAT_KEYS,
    Watchdog,
    WatchdogConfig,
)
from src.logger import setup_logging, get_logger


def resolve_strategy_args(
    strategy: str | None,
    heartbeat_key: str,
    service_name: str,
    *,
    default_heartbeat_key: str = DEFAULT_HEARTBEAT_KEY,
    default_service_name: str = "4h",
) -> tuple[str, str, str | None]:
    """Resolve --strategy + --heartbeat-key + --service-name into the
    final (heartbeat_key, service_name, warning_or_none) tuple.

    Rules:
      * Explicit ``--heartbeat-key`` (anything other than the launcher
        default) ALWAYS wins — backward compat for ops scripts that
        already pin the key directly.
      * Otherwise, ``--strategy`` derives both the key and (when the
        service name is still default) the service name.
      * Neither flag passed → return the default key plus a warning so
        the operator notices they're implicitly monitoring the 4H bot.
    """
    explicit_key = heartbeat_key != default_heartbeat_key

    if explicit_key and strategy is not None:
        # Both given — explicit key wins. Surface as a warning so the
        # operator sees the conflict.
        warn = (
            f"Both --strategy={strategy} and --heartbeat-key={heartbeat_key} "
            "were supplied — using the explicit key."
        )
        return heartbeat_key, service_name, warn

    if explicit_key:
        return heartbeat_key, service_name, None

    if strategy is not None:
        key = STRATEGY_HEARTBEAT_KEYS[strategy]
        # Only auto-derive service_name if the operator left it at the
        # 4H default, so a custom label stays intact.
        svc = (
            strategy if service_name == default_service_name
            else service_name
        )
        return key, svc, None

    warn = (
        "No --strategy specified and no explicit --heartbeat-key — "
        f"falling back to {default_heartbeat_key!r} (the 4H bot). "
        "Pass --strategy=4h to silence this warning, or "
        "--strategy=15m / --strategy=1h to monitor a different bot."
    )
    return default_heartbeat_key, service_name, warn


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
        "--strategy",
        choices=sorted(STRATEGY_HEARTBEAT_KEYS.keys()),
        default=None,
        help="Strategy to monitor (4h / 1h / 15m). Selects the matching "
        "heartbeat key and labels the service. Recommended over passing "
        "--heartbeat-key directly. Without this flag the launcher "
        "implicitly targets the 4H bot and logs a WARNING.",
    )
    parser.add_argument(
        "--heartbeat-key",
        default=DEFAULT_HEARTBEAT_KEY,
        help=f"Redis heartbeat key to monitor "
        f"(default: {DEFAULT_HEARTBEAT_KEY!r} — the 4H bot). "
        "Use bot_15m_heartbeat for the isolated 15m watchdog. "
        "If supplied, overrides --strategy's auto-derived key.",
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

    # H17: derive the effective heartbeat key / service name from
    # --strategy when present. Surface any conflict / missing-flag
    # condition as a WARNING so the operator notices it in the launch
    # banner.
    heartbeat_key, service_name, warn = resolve_strategy_args(
        strategy=args.strategy,
        heartbeat_key=args.heartbeat_key,
        service_name=args.service_name,
    )
    if warn:
        log.warning(warn)

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
        heartbeat_key=heartbeat_key,
        symbol=args.symbol,
        service_name=service_name,
        check_interval=args.check_interval,
        max_silence_seconds=args.max_silence,
        telegram_token=settings.telegram_bot_token,
        telegram_admin_id=settings.telegram_admin_id,
    )

    watchdog = Watchdog(config)

    log.info("=" * 60)
    log.info("  AtomiCortex External Watchdog")
    log.info("=" * 60)
    log.info(f"  Service:        {service_name}")
    log.info(f"  Heartbeat key:  {heartbeat_key}")
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
