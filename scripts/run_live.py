#!/usr/bin/env python3
"""
AtomiCortex — Live trading launcher.

Usage
-----
    # Testnet dry-run (signal-only, no orders)
    python scripts/run_live.py --mode testnet --symbols BTCUSDT-PERP --capital 10000 --dry-run

    # Testnet real orders
    python scripts/run_live.py --mode testnet --symbols BTCUSDT-PERP --capital 10000

    # With timeout
    python scripts/run_live.py --mode testnet --symbols BTCUSDT-PERP --dry-run --duration 60
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.execution.live_trader import LiveTrader, LiveTraderConfig
from src.logger import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AtomiCortex Live Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["testnet", "paper", "live"],
        default="testnet",
        help="Trading mode (default: testnet)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT-PERP"],
        help="Symbols to trade (default: BTCUSDT-PERP)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial capital in USDT (default: 10000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Signal-only mode — no orders sent",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run duration in seconds (0 = indefinite)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--models-dir",
        default="./data/features/models",
        help="Path to trained model directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level_console=args.log_level)
    log = get_logger("run_live")

    log.info("=" * 60)
    log.info("  AtomiCortex Live Trader")
    log.info("=" * 60)
    log.info(f"  Mode:     {args.mode}")
    log.info(f"  Symbols:  {args.symbols}")
    log.info(f"  Capital:  ${args.capital:,.2f}")
    log.info(f"  Dry Run:  {args.dry_run}")
    log.info(f"  Duration: {args.duration}s" if args.duration else "  Duration: indefinite")
    log.info("=" * 60)

    # Safety: require explicit confirmation for live mode
    if args.mode == "live" and not args.dry_run:
        confirm = input(
            "\n⚠️  LIVE MODE with REAL ORDERS! Type 'YES' to confirm: "
        )
        if confirm.strip() != "YES":
            log.info("Aborted by user")
            return

    # Build config
    config = LiveTraderConfig(
        trading_mode=args.mode,
        symbols=args.symbols,
        initial_equity=args.capital,
        dry_run=args.dry_run,
        log_level=args.log_level,
        models_dir=args.models_dir,
    )

    # Create trader
    trader = LiveTrader(config)

    # Duration-based auto-stop
    if args.duration > 0:
        def _auto_stop() -> None:
            time.sleep(args.duration)
            log.info(f"Duration limit ({args.duration}s) reached — stopping")
            trader.stop()

        t = threading.Thread(target=_auto_stop, daemon=True)
        t.start()

    # Graceful SIGINT / SIGTERM
    def _signal_handler(sig: int, frame: Any) -> None:
        log.info("Signal received — stopping...")
        trader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run
    try:
        trader.run()
    except Exception as exc:
        log.error(f"Fatal error: {exc}", exc_info=True)
        trader.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
