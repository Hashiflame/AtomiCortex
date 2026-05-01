#!/usr/bin/env python
"""CLI for AtomiCortex walk-forward validation."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.backtest_runner import BacktestConfig
from src.execution.walk_forward import WalkForwardResult, WalkForwardValidator
from src.execution.strategies.baseline_strategy import BuyAndHoldStrategy
from src.execution.strategies.random_entry_strategy import RandomEntryStrategy

_STRATEGIES = {
    "buy_and_hold": BuyAndHoldStrategy,
    "random_entry": RandomEntryStrategy,
}

_DEFAULT_TRADE_SIZE = {
    "BTCUSDT": 0.001,
    "ETHUSDT": 0.01,
    "SOLUSDT": 1.0,
}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _print_report(result: WalkForwardResult, args: argparse.Namespace) -> None:
    sep = "═" * 74
    thin = "─" * 74
    print(f"\n{sep}")
    print("  AtomiCortex — Walk-Forward Validation Report")
    print(sep)
    print(f"  Symbol   : {args.symbol}  |  Interval : {args.interval}")
    print(f"  Period   : {args.start} → {args.end}")
    print(f"  Train    : {args.train_months}m  |  Test : {args.test_months}m  |  Step : {args.step_months}m")
    print(f"  Strategy : {args.strategy}")
    print(sep)

    if not result.windows:
        print("  No windows completed (no data available for any test period).")
        print(sep)
        print()
        return

    hdr = f"  {'#':<4} {'Train':<22} {'Test':<22} {'Return':>8} {'Sharpe':>7} {'MDD':>6} {'P/L':>4}"
    print(hdr)
    print("  " + thin)
    for i, w in enumerate(result.windows, 1):
        status = "✅" if w.is_profitable else "❌"
        tr = f"{w.train_start.date()}→{w.train_end.date()}"
        te = f"{w.test_start.date()}→{w.test_end.date()}"
        print(
            f"  {i:<4} {tr:<22} {te:<22} "
            f"{w.metrics.total_return_pct:>+7.2f}% "
            f"{w.metrics.sharpe_ratio:>7.3f} "
            f"{w.metrics.max_drawdown_pct:>5.2f}% "
            f"{status}"
        )

    print(sep)
    profitable = sum(1 for w in result.windows if w.is_profitable)
    print(f"  Windows completed    : {len(result.windows)}")
    print(f"  Profitable windows   : {profitable} / {len(result.windows)} "
          f"({result.profitable_windows_pct:.1f}%)")
    print(f"  Average Sharpe ratio : {result.avg_sharpe:.4f}")
    gate = "✅ PASS" if result.passes_walk_forward_test else "❌ FAIL"
    print(f"  Walk-Forward Gate    : {gate}  (≥ 60% profitable windows)")
    print(sep)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AtomiCortex Walk-Forward Validation")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--strategy", default="buy_and_hold", choices=list(_STRATEGIES))
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=None,
                        help="Defaults to test-months")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--trade-size", type=float, default=None)
    parser.add_argument("--data-dir", default="/mnt/hdd/AtomiCortex/data/features")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log results to MLflow (./mlruns)")
    args = parser.parse_args()

    if args.step_months is None:
        args.step_months = args.test_months

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    data_dir = Path(args.data_dir)

    cfg = BacktestConfig(
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
        initial_capital=args.capital,
        leverage=args.leverage,
        data_dir=data_dir,
    )

    trade_size = args.trade_size or _DEFAULT_TRADE_SIZE.get(args.symbol, 0.001)
    strategy_class = _STRATEGIES[args.strategy]
    strategy_config: dict = {"trade_size": trade_size}

    validator = WalkForwardValidator(
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    result = validator.run_validation(
        strategy_class=strategy_class,
        strategy_config=strategy_config,
        backtest_config=cfg,
        data_dir=data_dir,
    )

    _print_report(result, args)

    if args.mlflow and result.windows:
        from src.execution.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        run_id = tracker.log_walk_forward(
            run_name=f"wf_{args.symbol}_{args.strategy}",
            wf_result=result,
            config=cfg,
        )
        print(f"  MLflow run_id: {run_id}")
        print()


if __name__ == "__main__":
    main()
