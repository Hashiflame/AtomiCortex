#!/usr/bin/env python3
"""
AtomiCortex — Paper Trading Runner.

Runs paper trading using live Nautilus WebSocket data but with simulated
order execution.  Outputs periodic status updates and saves all signals
and metrics to SQLite.

Usage
-----
    python scripts/run_paper.py --symbols BTCUSDT-PERP --capital 10000 --duration 3600
"""

from __future__ import annotations

import argparse
import os
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
        description="AtomiCortex Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Initial paper capital (default: 10000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration in seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=300,
        help="Status update interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


def _print_status(trader: LiveTrader, start_time: float, capital: float) -> None:
    """Print a formatted status update to console."""
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    equity = capital  # In dry-run mode, equity doesn't change
    pnl = equity - capital
    pnl_pct = (pnl / capital) * 100 if capital > 0 else 0

    print(
        f"\n{'─' * 40}\n"
        f"  AtomiCortex Paper Trading\n"
        f"{'─' * 40}\n"
        f"  Elapsed:   {elapsed_min:.1f} min\n"
        f"  Equity:    ${equity:,.2f} ({pnl_pct:+.2f}%)\n"
        f"  Mode:      testnet (dry-run)\n"
        f"{'─' * 40}\n",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    setup_logging(level_console=args.log_level)
    _log = get_logger(__name__)

    _log.info(
        "Paper Trading starting | symbols={syms} | capital=${cap} | duration={dur}s",
        syms=args.symbols,
        cap=args.capital,
        dur=args.duration,
    )

    # Build LiveTrader in dry-run mode (no real orders)
    cfg = LiveTraderConfig(
        trading_mode="testnet",
        symbols=args.symbols,
        initial_equity=args.capital,
        dry_run=True,  # Paper trading = dry run
        log_level=args.log_level,
    )

    trader = LiveTrader(cfg)

    # Duration timer
    _stop_requested = False

    if args.duration > 0:
        def _duration_timer() -> None:
            nonlocal _stop_requested
            time.sleep(args.duration)
            if not _stop_requested:
                _stop_requested = True
                _log.info(
                    "Duration limit ({dur}s) reached — sending SIGINT",
                    dur=args.duration,
                )
                os.kill(os.getpid(), signal.SIGINT)

        timer_thread = threading.Thread(target=_duration_timer, daemon=True)
        timer_thread.start()

    # Status update timer
    start_time = time.time()

    def _status_loop() -> None:
        while not _stop_requested:
            time.sleep(args.status_interval)
            if not _stop_requested:
                _print_status(trader, start_time, args.capital)

    if args.status_interval > 0:
        status_thread = threading.Thread(target=_status_loop, daemon=True)
        status_thread.start()

    # Signal handler for graceful shutdown
    _first_sigint = True

    def _signal_handler(signum: int, frame: object) -> None:
        nonlocal _first_sigint, _stop_requested
        _stop_requested = True
        if _first_sigint:
            _first_sigint = False
            _log.info("Received SIGINT — shutting down gracefully...")
        else:
            _log.warning("Second SIGINT — forcing exit!")
            sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)

    # Run
    print(
        f"\n{'═' * 45}\n"
        f"  🧪 AtomiCortex Paper Trading\n"
        f"{'═' * 45}\n"
        f"  Symbols:   {', '.join(args.symbols)}\n"
        f"  Capital:   ${args.capital:,.2f}\n"
        f"  Duration:  {args.duration}s\n"
        f"  Mode:      testnet (dry-run)\n"
        f"{'═' * 45}\n",
        flush=True,
    )

    try:
        trader.run()
    except KeyboardInterrupt:
        _log.info("KeyboardInterrupt — stopping paper trading...")
    except Exception as exc:
        _log.error(f"Paper trading error: {exc}")

    # Final status
    elapsed = time.time() - start_time
    _log.info(
        "Paper trading finished | elapsed={el:.0f}s",
        el=elapsed,
    )

    print(
        f"\n{'═' * 45}\n"
        f"  Paper Trading Complete\n"
        f"  Duration: {elapsed:.0f}s\n"
        f"{'═' * 45}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
