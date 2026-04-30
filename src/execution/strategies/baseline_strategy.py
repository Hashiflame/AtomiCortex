"""Buy-and-hold baseline strategy for backtest engine validation."""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy


class BuyAndHoldConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    trade_size: float = 0.001
    initial_capital: float = 10_000.0


class BuyAndHoldStrategy(Strategy):
    """
    Buys one fixed-size position on the first bar and holds until stop.

    Exists solely to verify that the BacktestEngine produces correct P&L,
    fills, and equity-curve data.
    """

    def __init__(self, config: BuyAndHoldConfig) -> None:
        super().__init__(config)
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)
        self._trade_size = config.trade_size
        self._initial_capital = config.initial_capital
        self._venue = Venue(self._instrument_id.venue.value)
        self._ordered = False
        # List of (ts_ns, equity_usdt) snapshots — one per bar after entry
        self._equity_curve: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.subscribe_bars(self._bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self._ordered:
            instrument = self.cache.instrument(self._instrument_id)
            order = self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=OrderSide.BUY,
                quantity=instrument.make_qty(self._trade_size),
            )
            self.submit_order(order)
            self._ordered = True
            self.log.info(f"BUY submitted at ~{bar.close}")

        self._record_equity(bar.ts_event)

    def on_order_filled(self, event) -> None:
        self.log.info(
            f"Filled: {event.order_side} {event.last_qty} @ {event.last_px} "
            f"commission={event.commission}"
        )

    def on_stop(self) -> None:
        self.close_all_positions(self._instrument_id)

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

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
