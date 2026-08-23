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

import src.patches.nautilus_enums  # Hotfix for TRADING_HALT

from src.config import get_settings
from src.execution.live_trader import LiveTrader, LiveTraderConfig
from src.logger import setup_logging, get_logger
from src.models.model_paths import MODELS_ROOT_4H

# Exit code for a refusal no restart can fix — EX_CONFIG from sysexits.h.
# Mirrored by RestartPreventExitStatus= in deploy/atomicortex-bot.service so
# systemd fails the unit once instead of burning all five restarts allowed by
# StartLimitBurst.  Deliberately NOT 1: the sys.exit(1) at the end of main()
# signals a *transient* connect failure and must stay restartable.
_EXIT_CONFIG_ERROR = 78


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
        default=str(MODELS_ROOT_4H),
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

    # ------------------------------------------------------------------
    # Fail-closed guard
    # ------------------------------------------------------------------
    # Authority is args.mode, NOT settings.trading_mode: build_node()
    # resolves credentials from LiveTraderConfig.trading_mode, which is
    # args.mode.  settings.trading_mode only drives the startup banner and
    # the log filename, so a mismatch is reported as a warning and never as
    # a refusal — an .env edit must not be able to kill a correctly
    # configured unit.
    try:
        env_mode = get_settings().trading_mode
    except Exception as exc:  # diagnostics only — must never be fatal
        env_mode = None
        log.warning(f"Could not read TRADING_MODE for the mode cross-check: {exc}")

    if env_mode is not None and env_mode != args.mode:
        log.warning(
            f"Mode divergence: TRADING_MODE in .env is '{env_mode}' but "
            f"--mode is '{args.mode}'. Orders route as '{args.mode}'; the "
            f"banner and log filename describe '{env_mode}'."
        )

    # This guard sits ABOVE the confirmation prompt on purpose.  Under
    # systemd stdin is /dev/null, where input() raises an unhandled
    # EOFError instead of protecting anything.
    interactive = sys.stdin.isatty()
    refusal: str | None = None
    if args.mode == "paper" and not args.dry_run:
        refusal = (
            "--mode paper resolves to MAINNET credentials and there is no "
            "confirmation prompt on this path — without --dry-run this "
            "sends real orders"
        )
    elif args.mode == "live" and not args.dry_run and not interactive:
        refusal = (
            "--mode live without --dry-run requires interactive "
            "confirmation, but stdin is not a TTY (systemd sets "
            "StandardInput=null)"
        )

    if refusal is not None:
        log.critical(
            f"Refusing to start: {refusal} | mode={args.mode} "
            f"dry_run={args.dry_run} env_trading_mode={env_mode} "
            f"stdin_isatty={interactive} — add --dry-run to ExecStart"
        )
        sys.exit(_EXIT_CONFIG_ERROR)

    # Safety: require explicit confirmation for live mode.
    # Reachable only with a TTY — the guard above returns 78 otherwise.
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

    # Duration-based auto-stop:
    # Send SIGINT to the main process — this is the safest way to
    # stop the Nautilus event loop from a background thread.
    # SIGINT is caught by the signal handler below, which calls
    # trader.stop() → node.stop() → schedules stop_async() on the loop.
    if args.duration > 0:
        def _auto_stop() -> None:
            time.sleep(args.duration)
            log.info(f"Duration limit ({args.duration}s) reached — sending SIGINT")
            os.kill(os.getpid(), signal.SIGINT)

        t = threading.Thread(target=_auto_stop, daemon=True)
        t.start()

    # Graceful SIGINT / SIGTERM
    # Call trader.stop() which schedules stop_async on the Nautilus loop.
    # Do NOT call sys.exit() — that would bypass run()'s finally block
    # where dispose() is called.
    _stop_requested = False

    def _signal_handler(sig: int, frame: Any) -> None:
        nonlocal _stop_requested
        if _stop_requested:
            log.warning("Force exit (second signal)")
            sys.exit(1)
        _stop_requested = True
        log.info("Signal received — requesting graceful stop...")
        trader.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run — blocks until node.stop() causes run_async() to return.
    # The finally block in trader.run() calls _dispose().
    try:
        trader.run()
    except SystemExit:
        pass
    except Exception as exc:
        log.error(f"Fatal error: {exc}", exc_info=True)

    if trader.startup_failed:
        log.critical(
            "Engines failed to connect — exiting for systemd restart"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

