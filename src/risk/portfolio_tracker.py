"""
AtomiCortex — Portfolio Tracker.

Real-time portfolio state tracker that maintains equity, positions,
P&L, drawdown, and consecutive-loss counters used by the RiskEngine
and CircuitBreaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.logger import get_logger
from src.risk.risk_engine import PortfolioState

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal position record
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    """Internal representation of a single open position."""

    symbol: str
    direction: int       # 1=LONG, -1=SHORT
    quantity: float
    avg_entry_price: float
    total_fees: float
    open_time: datetime
    current_price: float  # last mark price
    unrealized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio Tracker
# ---------------------------------------------------------------------------

class PortfolioTracker:
    """
    Maintains a live portfolio snapshot used by :class:`RiskEngine` and
    :class:`CircuitBreaker` to make risk decisions.
    """

    def __init__(self, initial_equity: float) -> None:
        self._initial_equity: float = initial_equity
        self._cash: float = initial_equity
        self._peak_equity: float = initial_equity

        # Open positions by symbol
        self._positions: dict[str, _Position] = {}

        # P&L tracking
        self._daily_realized_pnl: float = 0.0
        self._weekly_realized_pnl: float = 0.0
        self._total_realized_pnl: float = 0.0

        # Day/week boundaries
        self._day_start: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        self._week_start: datetime = self._day_start - timedelta(
            days=self._day_start.weekday(),
        )

        # Consecutive losses
        self._consecutive_losses: int = 0
        self._last_loss_time: datetime | None = None

        log.info(
            "PortfolioTracker initialised | equity={eq}",
            eq=initial_equity,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_fill(
        self,
        symbol: str,
        direction: int,
        quantity: float,
        price: float,
        fee: float,
        timestamp: datetime,
    ) -> None:
        """Record an order fill (new position or add to existing)."""
        self._roll_periods(timestamp)
        self._cash -= fee

        if symbol in self._positions:
            pos = self._positions[symbol]
            # Average in
            total_qty = pos.quantity + quantity
            if total_qty > 0:
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.quantity + price * quantity) / total_qty
                )
            pos.quantity = total_qty
            pos.total_fees += fee
        else:
            self._positions[symbol] = _Position(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                avg_entry_price=price,
                total_fees=fee,
                open_time=timestamp,
                current_price=price,
            )

        log.debug(
            "Fill recorded | sym={sym} dir={d} qty={q} price={p} fee={f}",
            sym=symbol,
            d=direction,
            q=quantity,
            p=price,
            f=fee,
        )

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update mark price for an open position."""
        if symbol not in self._positions:
            return
        pos = self._positions[symbol]
        pos.current_price = current_price
        pos.unrealized_pnl = (
            pos.direction * (current_price - pos.avg_entry_price) * pos.quantity
        )

    def close_position(
        self,
        symbol: str,
        close_price: float,
        fee: float,
        timestamp: datetime,
    ) -> float:
        """
        Close a position and return realized P&L (after fees).
        """
        self._roll_periods(timestamp)

        if symbol not in self._positions:
            log.warning("close_position called for unknown symbol {sym}", sym=symbol)
            return 0.0

        pos = self._positions[symbol]
        gross_pnl = pos.direction * (close_price - pos.avg_entry_price) * pos.quantity
        total_fees = pos.total_fees + fee
        realized_pnl = gross_pnl - total_fees

        # Futures accounting: equity = cash + unrealized_pnl (mark-to-market
        # relative to entry). update_fill never debited the notional from
        # cash on open, so close must NOT credit it back here — only the
        # gross PnL and the close-side fee flow into cash now. The open-side
        # fees were already debited at fill time (avoid double-counting).
        self._cash += gross_pnl - fee
        self._daily_realized_pnl += realized_pnl
        self._weekly_realized_pnl += realized_pnl
        self._total_realized_pnl += realized_pnl

        # Track consecutive losses
        if realized_pnl < 0:
            self.record_loss(timestamp)
        else:
            self._consecutive_losses = 0

        # Update peak equity
        current_eq = self._get_equity()
        if current_eq > self._peak_equity:
            self._peak_equity = current_eq

        log.info(
            "Position closed | sym={sym} pnl={pnl:.2f} "
            "fees={fees:.2f} net={net:.2f}",
            sym=symbol,
            pnl=gross_pnl,
            fees=total_fees,
            net=realized_pnl,
        )

        del self._positions[symbol]
        return realized_pnl

    def get_state(self) -> PortfolioState:
        """Return a snapshot for consumption by RiskEngine / CircuitBreaker."""
        equity = self._get_equity()
        return PortfolioState(
            equity=equity,
            open_positions=len(self._positions),
            daily_pnl_pct=self.get_daily_pnl(),
            weekly_pnl_pct=self.get_weekly_pnl(),
            current_drawdown_pct=self.get_drawdown(),
            consecutive_losses=self._consecutive_losses,
            last_loss_time=self._last_loss_time,
            peak_equity=self._peak_equity,
        )

    def get_daily_pnl(self) -> float:
        """Return today's realised P&L as a fraction of initial equity."""
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        daily_total = self._daily_realized_pnl + total_unrealized
        if self._initial_equity <= 0:
            return 0.0
        return daily_total / self._initial_equity

    def get_weekly_pnl(self) -> float:
        """Return this week's realised P&L as a fraction of initial equity."""
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        weekly_total = self._weekly_realized_pnl + total_unrealized
        if self._initial_equity <= 0:
            return 0.0
        return weekly_total / self._initial_equity

    def get_drawdown(self) -> float:
        """Return current drawdown as a positive fraction (0.10 = 10%)."""
        equity = self._get_equity()
        if self._peak_equity <= 0:
            return 0.0
        dd = (self._peak_equity - equity) / self._peak_equity
        return max(dd, 0.0)

    def record_loss(self, timestamp: datetime) -> None:
        """Record a losing trade for the consecutive-loss counter."""
        self._consecutive_losses += 1
        self._last_loss_time = timestamp
        log.debug(
            "Consecutive losses: {n} | last_loss_time={t}",
            n=self._consecutive_losses,
            t=timestamp.isoformat(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_equity(self) -> float:
        """Cash + unrealised P&L of all open positions."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self._cash + unrealized

    def _roll_periods(self, now: datetime) -> None:
        """Reset daily/weekly accumulators if boundaries crossed."""
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today_start > self._day_start:
            self._daily_realized_pnl = 0.0
            self._day_start = today_start
            log.debug("Daily PnL counter rolled over")

        week_start = today_start - timedelta(days=today_start.weekday())
        if week_start > self._week_start:
            self._weekly_realized_pnl = 0.0
            self._week_start = week_start
            log.debug("Weekly PnL counter rolled over")
