#!/usr/bin/env python3
"""
AtomiCortex — 15m Paper Trading Launcher (ISOLATED).

Runs the 15m strategy fully isolated from the 4H and 1H bots:
  * own SQLite      : data/atomicortex_15m.db
  * own heartbeat   : bot_15m_heartbeat
  * own models      : data/models/15m/{trend,orb}_model_15m.pkl
  * own systemd unit: atomicortex-bot-15m.service

It does NOT share state with, import config from, or otherwise touch the
running 4H bot. Wiring is done through ``LiveTrader.strategy_factory``
(Phase-5 hook) so the shared 4H code path is untouched.

Usage
-----
    python scripts/run_paper_15m.py
    python scripts/run_paper_15m.py --symbol BTCUSDT-PERP --capital 10000
    python scripts/run_paper_15m.py --duration 86400        # 24h then stop
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.patches.nautilus_enums  # Hotfix for TRADING_HALT

from src.execution.live_trader import LiveTrader, LiveTraderConfig
from src.logger import get_logger, setup_logging


def _make_15m_strategy(cfg: LiveTraderConfig, symbol: str):
    """LiveTrader strategy factory → isolated 15m strategy.

    Builds the Nautilus 15m config + ``MLTradingStrategy15M``. Kept here
    (not in LiveTrader) so LiveTrader stays timeframe-agnostic.
    """
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
        description="AtomiCortex 15m Paper Trading (isolated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbol", default="BTCUSDT-PERP", help="Symbol")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument(
        "--trading-mode",
        choices=["testnet", "paper", "live"],
        default="testnet",
        help="Data source mode (orders are always simulated here)",
    )
    p.add_argument("--duration", type=int, default=0, help="Seconds (0=∞)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level_console=args.log_level)
    log = get_logger("run_paper_15m")

    log.info(
        "15m Paper Trading | symbol={s} | capital=${c} | mode={m}",
        s=args.symbol, c=args.capital, m=args.trading_mode,
    )

    cfg = LiveTraderConfig(
        trading_mode=args.trading_mode,
        symbols=[args.symbol],
        initial_equity=args.capital,
        dry_run=True,                       # paper = simulated execution
        log_level=args.log_level,
        strategy_factory=_make_15m_strategy,  # Phase-5 isolated wiring
    )
    trader = LiveTrader(cfg)

    _stop = False

    if args.duration > 0:
        import threading

        def _timer() -> None:
            time.sleep(args.duration)
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=_timer, daemon=True).start()

    def _sig(_signum: int, _frame: object) -> None:
        nonlocal _stop
        if not _stop:
            _stop = True
            log.info("SIGINT — shutting down 15m paper bot...")
        else:
            sys.exit(1)

    signal.signal(signal.SIGINT, _sig)

    print(
        f"\n{'═' * 45}\n"
        f"  🧪 AtomiCortex 15m Paper (ISOLATED)\n"
        f"{'═' * 45}\n"
        f"  Symbol:    {args.symbol}\n"
        f"  Capital:   ${args.capital:,.2f}\n"
        f"  DB:        data/atomicortex_15m.db\n"
        f"  Heartbeat: bot_15m_heartbeat\n"
        f"  Mode:      {args.trading_mode} (dry-run)\n"
        f"{'═' * 45}\n",
        flush=True,
    )

    try:
        trader.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping 15m paper bot")
    except Exception as exc:
        log.error(f"15m paper bot error: {exc}")


if __name__ == "__main__":
    main()
