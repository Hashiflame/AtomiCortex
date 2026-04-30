"""Nautilus BacktestEngine runner for AtomiCortex."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Type

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.trading.strategy import Strategy

from src.execution.data_catalog import AtomiCortexCatalog


# ──────────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    symbol: str
    interval: str
    start: datetime
    end: datetime
    initial_capital: float = 10_000.0
    leverage: int = 5
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    data_dir: Path = field(
        default_factory=lambda: Path("/mnt/hdd/AtomiCortex/data/features")
    )


@dataclass
class BacktestResult:
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    start_equity: float
    end_equity: float
    equity_curve: list[tuple[datetime, float]]


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

class BacktestRunner:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._catalog = AtomiCortexCatalog(config.data_dir)

    def run(
        self,
        strategy_class: Type[Strategy],
        strategy_config: dict[str, Any],
    ) -> BacktestResult:
        cfg = self.config

        # ── Engine ──────────────────────────────────────────────────────────
        engine_config = BacktestEngineConfig(
            trader_id="ATOMICORTEX-001",
            logging=LoggingConfig(log_level="WARNING", bypass_logging=False),
            run_analysis=True,
        )
        engine = BacktestEngine(config=engine_config)

        # ── Venue ───────────────────────────────────────────────────────────
        venue = Venue("BINANCE")
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(cfg.initial_capital, USDT)],
            default_leverage=Decimal(str(cfg.leverage)),
            bar_execution=True,
        )

        # ── Instrument (with fees) ───────────────────────────────────────────
        instrument = self._catalog.get_instrument(
            cfg.symbol,
            maker_fee=Decimal(str(cfg.maker_fee)),
            taker_fee=Decimal(str(cfg.taker_fee)),
        )
        engine.add_instrument(instrument)

        # ── Bar data ─────────────────────────────────────────────────────────
        bars = self._catalog.load_bar_data(cfg.symbol, cfg.interval, cfg.start, cfg.end)
        if not bars:
            raise ValueError(
                f"No bar data found for {cfg.symbol} {cfg.interval} "
                f"{cfg.start} – {cfg.end}"
            )
        engine.add_data(bars)

        # ── Strategy ─────────────────────────────────────────────────────────
        import typing

        bar_type_str = str(bars[0].bar_type)
        full_strategy_config = {
            "instrument_id": str(instrument.id),
            "bar_type": bar_type_str,
            "initial_capital": cfg.initial_capital,
            **strategy_config,
        }
        hints = typing.get_type_hints(strategy_class.__init__)
        config_type = hints["config"]
        strategy = strategy_class(config=config_type(**full_strategy_config))

        engine.add_strategy(strategy)

        # ── Run ─────────────────────────────────────────────────────────────
        engine.run(start=cfg.start, end=cfg.end)

        # ── Collect results ──────────────────────────────────────────────────
        nautilus_result = engine.get_result()
        account = engine.portfolio.account(venue)
        end_equity = (
            account.balance_total(USDT).as_double() if account else cfg.initial_capital
        )

        equity_curve = _build_equity_curve(getattr(strategy, "_equity_curve", []))

        stats_pnls = nautilus_result.stats_pnls.get("USDT", {})
        stats_returns = nautilus_result.stats_returns

        start_equity = cfg.initial_capital
        total_return_pct = (end_equity - start_equity) / start_equity * 100
        sharpe = stats_returns.get("Sharpe Ratio (252 days)", 0.0) or 0.0
        profit_factor = stats_returns.get("Profit Factor", 0.0) or 0.0
        win_rate = stats_pnls.get("Win Rate", 0.0) or 0.0
        max_dd = _max_drawdown([e for _, e in equity_curve]) if equity_curve else 0.0

        engine.dispose()

        return BacktestResult(
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_trades=nautilus_result.total_orders,
            win_rate=win_rate,
            profit_factor=profit_factor,
            start_equity=start_equity,
            end_equity=end_equity,
            equity_curve=equity_curve,
        )

    def print_report(self, result: BacktestResult) -> None:
        cfg = self.config
        sep = "═" * 54

        print(f"\n{sep}")
        print(f"  AtomiCortex Backtest Report")
        print(sep)
        print(f"  Symbol   : {cfg.symbol}  |  Interval: {cfg.interval}")
        print(f"  Period   : {cfg.start.date()} → {cfg.end.date()}")
        print(f"  Leverage : {cfg.leverage}x  |  Capital: ${cfg.initial_capital:,.2f}")
        print(sep)
        _row("Total Return", f"{result.total_return_pct:+.2f}%")
        _row("Start Equity", f"${result.start_equity:,.2f}")
        _row("End Equity", f"${result.end_equity:,.2f}")
        print(sep)
        _row("Sharpe Ratio (252d)", f"{result.sharpe_ratio:.4f}")
        _row("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
        _row("Profit Factor", f"{result.profit_factor:.4f}")
        _row("Win Rate", f"{result.win_rate:.2%}")
        print(sep)
        _row("Total Trades", str(result.total_trades))
        _row("Equity Curve Points", str(len(result.equity_curve)))
        print(sep)
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _row(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def _build_equity_curve(
    raw: list[tuple[int, float]],
) -> list[tuple[datetime, float]]:
    return [
        (datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc), equity)
        for ts_ns, equity in raw
    ]


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return max_dd
