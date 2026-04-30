#!/usr/bin/env python
"""CLI for running AtomiCortex backtests."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.backtest_runner import BacktestConfig, BacktestRunner
from src.execution.strategies.baseline_strategy import BuyAndHoldStrategy

_STRATEGIES = {
    "buy_and_hold": BuyAndHoldStrategy,
}

_DEFAULT_TRADE_SIZE = {
    "BTCUSDT": 0.001,
    "ETHUSDT": 0.01,
    "SOLUSDT": 1.0,
}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="AtomiCortex Backtest Runner")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT)")
    parser.add_argument("--interval", default="4h", help="Bar interval (1m/5m/15m/1h/4h/1d)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-06-30", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital in USDT")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage multiplier")
    parser.add_argument("--maker-fee", type=float, default=0.0002, help="Maker fee rate")
    parser.add_argument("--taker-fee", type=float, default=0.0005, help="Taker fee rate")
    parser.add_argument(
        "--strategy",
        default="buy_and_hold",
        choices=list(_STRATEGIES),
        help="Strategy name",
    )
    parser.add_argument("--trade-size", type=float, default=None, help="Position size (base)")
    parser.add_argument(
        "--data-dir",
        default="/mnt/hdd/AtomiCortex/data/features",
        help="Path to Parquet feature data",
    )
    args = parser.parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol,
        interval=args.interval,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        initial_capital=args.capital,
        leverage=args.leverage,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        data_dir=Path(args.data_dir),
    )

    trade_size = args.trade_size or _DEFAULT_TRADE_SIZE.get(args.symbol, 0.001)
    strategy_class = _STRATEGIES[args.strategy]
    strategy_config = {"trade_size": trade_size}

    runner = BacktestRunner(cfg)
    result = runner.run(strategy_class, strategy_config)
    runner.print_report(result)


if __name__ == "__main__":
    main()
