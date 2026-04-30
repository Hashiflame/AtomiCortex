"""Random entry/exit strategy for cost-model validation."""

from __future__ import annotations

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy


class RandomEntryConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    entry_probability: float = 0.1
    hold_bars: int = 6
    trade_size: float = 0.01
    initial_capital: float = 10_000.0
    random_seed: int = 42
    long_only: bool = False  # False → random long/short


class RandomEntryStrategy(Strategy):
    """
    Opens a position with probability `entry_probability` on each bar.
    Closes after `hold_bars` bars.  When long_only=False the direction is
    random, so expected price P&L ≈ 0 and fees dominate over time.
    """

    def __init__(self, config: RandomEntryConfig) -> None:
        super().__init__(config)
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)
        self._trade_size = config.trade_size
        self._initial_capital = config.initial_capital
        self._venue = Venue(self._instrument_id.venue.value)
        self._entry_prob = config.entry_probability
        self._hold_bars = config.hold_bars
        self._long_only = config.long_only
        # Seeded generator: same seed → identical trade sequence every run
        self._rng = np.random.default_rng(config.random_seed)
        self._bars_held = 0
        self._in_position = False
        self._equity_curve: list[tuple[int, float]] = []

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self._in_position:
            self._bars_held += 1
            if self._bars_held >= self._hold_bars:
                self.close_all_positions(self._instrument_id)
                self._in_position = False
                self._bars_held = 0
        else:
            if self._rng.random() < self._entry_prob:
                instrument = self.cache.instrument(self._instrument_id)
                if self._long_only:
                    side = OrderSide.BUY
                else:
                    side = OrderSide.BUY if self._rng.integers(0, 2) == 0 else OrderSide.SELL
                order = self.order_factory.market(
                    instrument_id=self._instrument_id,
                    order_side=side,
                    quantity=instrument.make_qty(self._trade_size),
                )
                self.submit_order(order)
                self._in_position = True
                self._bars_held = 0

        self._record_equity(bar.ts_event)

    def on_stop(self) -> None:
        if self._in_position:
            self.close_all_positions(self._instrument_id)

    def _record_equity(self, ts_ns: int) -> None:
        account = self.portfolio.account(self._venue)
        if account is None:
            return
        try:
            balance = account.balance_total(USDT)
            upnl = self.portfolio.unrealized_pnl(self._instrument_id)
            equity = balance.as_double() + (upnl.as_double() if upnl is not None else 0.0)
            self._equity_curve.append((ts_ns, equity))
        except Exception:
            pass
