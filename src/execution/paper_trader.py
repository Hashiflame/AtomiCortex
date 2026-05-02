"""
AtomiCortex — Paper Trading Engine.

Simulates order execution without real capital using live WebSocket
price data.  Models slippage (configurable bps) and exchange fees
(maker/taker) to produce realistic equity curves.

Phase 5 — Paper Trading.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PaperTraderConfig:
    """All paper trading tunables."""

    initial_equity: float = 10_000.0
    maker_fee: float = 0.0002      # 0.02%
    taker_fee: float = 0.0005      # 0.05%
    slippage_bps: float = 2.0      # 2 bps average slippage
    use_live_prices: bool = True


# ---------------------------------------------------------------------------
# Paper Fill
# ---------------------------------------------------------------------------

@dataclass
class PaperFill:
    """Result of a simulated order fill."""

    order_id: str
    symbol: str
    direction: int          # 1=LONG, -1=SHORT
    quantity: float
    fill_price: float       # entry ± slippage
    fee: float
    timestamp: datetime
    slippage_bps: float


# ---------------------------------------------------------------------------
# Paper Position (internal)
# ---------------------------------------------------------------------------

@dataclass
class _PaperPosition:
    """Internal open position record."""

    symbol: str
    direction: int
    quantity: float
    entry_price: float
    entry_fee: float
    open_time: datetime


# ---------------------------------------------------------------------------
# Paper Trader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Simulates order execution with realistic slippage and fees.

    Parameters
    ----------
    config:
        Paper trading configuration.
    """

    def __init__(self, config: PaperTraderConfig | None = None) -> None:
        self._config = config or PaperTraderConfig()
        self._equity = self._config.initial_equity
        self._cash = self._config.initial_equity

        self._positions: dict[str, _PaperPosition] = {}
        self._trade_log: list[dict] = []
        self._pnl_history: list[tuple[datetime, float]] = []
        self._total_trades: int = 0
        self._winning_trades: int = 0
        self._total_pnl: float = 0.0

        _log.info(
            "PaperTrader initialised | equity={eq} | slippage={sl} bps",
            eq=self._config.initial_equity,
            sl=self._config.slippage_bps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_fill(
        self,
        symbol: str,
        direction: int,
        quantity: float,
        current_price: float,
        is_maker: bool = False,
    ) -> PaperFill:
        """Simulate an order fill with slippage and fees.

        Parameters
        ----------
        symbol:
            Instrument symbol.
        direction:
            1 for LONG, -1 for SHORT.
        quantity:
            Number of contracts.
        current_price:
            Current market price.
        is_maker:
            Whether this is a limit order (maker fee) or market (taker fee).

        Returns
        -------
        PaperFill with simulated execution details.
        """
        now = datetime.now(timezone.utc)
        slip_bps = self._config.slippage_bps

        # Apply slippage: LONG pays more, SHORT pays less
        slip_fraction = slip_bps / 10_000
        if direction == 1:
            fill_price = current_price * (1 + slip_fraction)
        else:
            fill_price = current_price * (1 - slip_fraction)

        # Fee calculation
        fee_rate = self._config.maker_fee if is_maker else self._config.taker_fee
        notional = quantity * fill_price
        fee = notional * fee_rate

        # Deduct fee from cash
        self._cash -= fee

        order_id = str(uuid.uuid4())[:12]

        fill = PaperFill(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            fill_price=fill_price,
            fee=fee,
            timestamp=now,
            slippage_bps=slip_bps,
        )

        # Track as open position
        self._positions[f"{symbol}:{order_id}"] = _PaperPosition(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=fill_price,
            entry_fee=fee,
            open_time=now,
        )

        self._trade_log.append({
            "order_id": order_id,
            "type": "ENTRY",
            "symbol": symbol,
            "direction": "LONG" if direction == 1 else "SHORT",
            "quantity": quantity,
            "price": current_price,
            "fill_price": fill_price,
            "fee": fee,
            "slippage_bps": slip_bps,
            "timestamp": now.isoformat(),
        })

        _log.info(
            "Paper FILL | {sym} {dir} {qty:.6f} @ {fp:.2f} "
            "(market={mp:.2f} slip={sl:.1f}bps) fee={fee:.4f}",
            sym=symbol,
            dir="LONG" if direction == 1 else "SHORT",
            qty=quantity,
            fp=fill_price,
            mp=current_price,
            sl=slip_bps,
            fee=fee,
        )

        return fill

    def simulate_close(
        self,
        symbol: str,
        order_id: str,
        current_price: float,
    ) -> float:
        """Close a paper position and return realized PnL.

        Parameters
        ----------
        symbol:
            Instrument symbol.
        order_id:
            The order_id from the original PaperFill.
        current_price:
            Current market price for the close.

        Returns
        -------
        Realized PnL after fees and slippage.
        """
        key = f"{symbol}:{order_id}"
        if key not in self._positions:
            _log.warning("Paper close for unknown position {key}", key=key)
            return 0.0

        pos = self._positions.pop(key)
        now = datetime.now(timezone.utc)

        # Exit slippage (opposite direction)
        slip_fraction = self._config.slippage_bps / 10_000
        if pos.direction == 1:
            exit_price = current_price * (1 - slip_fraction)
        else:
            exit_price = current_price * (1 + slip_fraction)

        # Exit fee
        exit_notional = pos.quantity * exit_price
        exit_fee = exit_notional * self._config.taker_fee

        # PnL
        gross_pnl = pos.direction * (exit_price - pos.entry_price) * pos.quantity
        net_pnl = gross_pnl - pos.entry_fee - exit_fee

        self._cash += net_pnl
        self._total_pnl += net_pnl
        self._total_trades += 1
        if net_pnl > 0:
            self._winning_trades += 1

        self._equity = self._cash
        self._pnl_history.append((now, net_pnl))

        self._trade_log.append({
            "order_id": order_id,
            "type": "EXIT",
            "symbol": symbol,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "exit_fee": exit_fee,
            "timestamp": now.isoformat(),
        })

        _log.info(
            "Paper CLOSE | {sym} {dir} pnl={pnl:.2f} "
            "(entry={ep:.2f} exit={xp:.2f})",
            sym=symbol,
            dir="LONG" if pos.direction == 1 else "SHORT",
            pnl=net_pnl,
            ep=pos.entry_price,
            xp=exit_price,
        )

        return net_pnl

    def get_equity(self) -> float:
        """Return current paper equity (cash minus open position costs)."""
        return self._cash

    def get_pnl_history(self) -> list[tuple[datetime, float]]:
        """Return list of (timestamp, pnl) for each closed trade."""
        return list(self._pnl_history)

    def get_trade_log(self) -> list[dict]:
        """Return full trade log."""
        return list(self._trade_log)

    def get_stats(self) -> dict:
        """Return summary statistics."""
        wr = (
            self._winning_trades / self._total_trades
            if self._total_trades > 0
            else 0.0
        )
        return {
            "equity": self._equity,
            "initial_equity": self._config.initial_equity,
            "total_pnl": self._total_pnl,
            "total_pnl_pct": self._total_pnl / self._config.initial_equity,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._total_trades - self._winning_trades,
            "win_rate": wr,
            "open_positions": len(self._positions),
        }
