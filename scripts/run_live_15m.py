#!/usr/bin/env python3
"""
AtomiCortex — 15m Live Trader Launcher (ISOLATED).

Same isolation guarantees as run_paper_15m.py (own DB / heartbeat /
models / systemd unit) but can place REAL orders when
``--mode live`` and not ``--dry-run``. Mirrors scripts/run_live.py
semantics (incl. the explicit live-mode confirmation prompt).

Usage
-----
    python scripts/run_live_15m.py --mode testnet
    python scripts/run_live_15m.py --mode testnet --dry-run
    python scripts/run_live_15m.py --mode live           # prompts YES
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.live_trader import LiveTrader, LiveTraderConfig
from src.logger import get_logger, setup_logging


def _make_15m_strategy(cfg: LiveTraderConfig, symbol: str):
    """LiveTrader strategy factory → isolated 15m strategy."""
    from src.execution.strategies.ml_strategy_15m import (
        MLStrategy15MConfig,
        MLTradingStrategy15M,
    )

    instrument_id = f"{symbol}.BINANCE"
    strat_cfg = MLStrategy15MConfig(
        instrument_id=instrument_id,
        bar_type=f"{instrument_id}-15-MINUTE-LAST-EXTERNAL",
        initial_equity=cfg.initial_equity,
        dry_run=cfg.dry_run,
        trading_mode=cfg.trading_mode,
    )
    return MLTradingStrategy15M(config=strat_cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AtomiCortex 15m Live Trader (isolated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["testnet", "paper", "live"], default="testnet"
    )
    p.add_argument("--symbol", default="BTCUSDT-PERP")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument(
        "--dry-run", action="store_true", help="Signal-only — no orders"
    )
    p.add_argument("--duration", type=int, default=0, help="Seconds (0=∞)")
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level_console=args.log_level)
    log = get_logger("run_live_15m")

    log.info("=" * 60)
    log.info("  AtomiCortex 15m Live Trader (ISOLATED)")
    log.info("=" * 60)
    log.info(f"  Mode:     {args.mode}")
    log.info(f"  Symbol:   {args.symbol}")
    log.info(f"  Capital:  ${args.capital:,.2f}")
    log.info(f"  Dry Run:  {args.dry_run}")
    log.info(f"  DB:       data/atomicortex_15m.db")
    log.info(f"  Heartbeat: bot_15m_heartbeat")
    log.info("=" * 60)

    if args.mode == "live" and not args.dry_run:
        confirm = input(
            "\n⚠️  15m LIVE MODE with REAL ORDERS! Type 'YES' to confirm: "
        )
        if confirm.strip() != "YES":
            log.info("Aborted by user")
            return

    cfg = LiveTraderConfig(
        trading_mode=args.mode,
        symbols=[args.symbol],
        initial_equity=args.capital,
        dry_run=args.dry_run,
        log_level=args.log_level,
        strategy_factory=_make_15m_strategy,
    )
    trader = LiveTrader(cfg)

    _stop = False
    if args.duration > 0:
        def _timer() -> None:
            time.sleep(args.duration)
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=_timer, daemon=True).start()

    def _sig(_signum: int, _frame: object) -> None:
        nonlocal _stop
        if not _stop:
            _stop = True
            log.info("SIGINT — stopping 15m live bot...")
        else:
            sys.exit(1)

    signal.signal(signal.SIGINT, _sig)

    try:
        trader.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping 15m live bot")
    except Exception as exc:
        log.error(f"15m live bot error: {exc}")


if __name__ == "__main__":
    main()
