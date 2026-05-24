"""
AtomiCortex — Portfolio Tracker.

Real-time portfolio state tracker that maintains equity, positions,
P&L, drawdown, and consecutive-loss counters used by the RiskEngine
and CircuitBreaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.risk.risk_engine import PortfolioState
from src.risk.risk_state_store import RiskStateStore

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

    def __init__(
        self,
        initial_equity: float,
        state_path: Path | str | None = None,
    ) -> None:
        self._initial_equity: float = initial_equity
        self._cash: float = initial_equity
        self._peak_equity: float = initial_equity
        # H5: equity snapshot at the start of the current UTC day. Used as
        # the denominator for daily_pnl_pct so the percent-of-equity stop
        # tracks current portfolio size, not the original deposit.
        self._day_start_equity: float = initial_equity

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

        # Optional crash-safe persistence (None → in-memory only, backward
        # compatible with every existing caller).
        self._store: RiskStateStore | None = (
            RiskStateStore(state_path) if state_path is not None else None
        )
        if self._store is not None:
            self._restore_from_store()

        log.info(
            "PortfolioTracker initialised | equity={eq} | persisted={p}",
            eq=initial_equity,
            p=self._store is not None,
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
        self._persist()

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update mark price for an open position.

        Also bumps ``peak_equity`` when mark-to-market gains push the
        portfolio above the previous high. Without this, drawdown was
        measured from a stale (pre-position) peak and intraday round-trips
        (up 1k → back to flat) registered no drawdown at all.
        """
        if symbol not in self._positions:
            return
        pos = self._positions[symbol]
        pos.current_price = current_price
        pos.unrealized_pnl = (
            pos.direction * (current_price - pos.avg_entry_price) * pos.quantity
        )

        current_eq = self._get_equity()
        if current_eq > self._peak_equity:
            self._peak_equity = current_eq

    def sync_equity(self, nautilus_equity: float) -> None:
        """H6: align tracker equity with the exchange-authoritative value.

        Adjusts ``_cash`` so that ``_get_equity() == nautilus_equity`` —
        ``equity = cash + Σ unrealized_pnl`` is the invariant, so
        ``cash = nautilus_equity - Σ unrealized_pnl``. Funding payments,
        fee-rounding, and timing drifts between the local tracker and the
        Nautilus PortfolioFacade are absorbed here.

        Daily/weekly realised PnL counters, day_start_equity, and
        consecutive-loss state are deliberately untouched — they reflect
        history independent of the cash balance.
        """
        try:
            target = float(nautilus_equity)
        except (TypeError, ValueError):
            return
        if target != target or target in (float("inf"), float("-inf")):
            return  # NaN / inf — drop silently

        unrealised = sum(p.unrealized_pnl for p in self._positions.values())
        self._cash = target - unrealised

        # Bump peak so drawdown gate sees true high-water mark even when
        # equity moves via Nautilus (funding credits, etc.), not via our
        # own update_fill / close_position path.
        if target > self._peak_equity:
            self._peak_equity = target

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
        self._persist()
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
        """Today's realised + unrealised P&L as a fraction of *day-start* equity.

        Denominator is ``_day_start_equity`` (frozen at the day's open),
        not the original deposit — so percent-of-equity daily stops scale
        with the current portfolio after compounding gains/losses.
        """
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        daily_total = self._daily_realized_pnl + total_unrealized
        denom = self._day_start_equity if self._day_start_equity > 0 else self._initial_equity
        if denom <= 0:
            return 0.0
        return daily_total / denom

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
        self._persist()

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
        rolled = False
        if today_start > self._day_start:
            self._daily_realized_pnl = 0.0
            self._day_start = today_start
            # Freeze the new day's denominator at current equity (includes
            # any open unrealised PnL) so daily % stops scale with the
            # portfolio rather than the original deposit.
            self._day_start_equity = self._get_equity()
            rolled = True
            log.debug("Daily PnL counter rolled over")

        week_start = today_start - timedelta(days=today_start.weekday())
        if week_start > self._week_start:
            self._weekly_realized_pnl = 0.0
            self._week_start = week_start
            rolled = True
            log.debug("Weekly PnL counter rolled over")

        if rolled:
            self._persist()

    # ------------------------------------------------------------------
    # Persistence (optional — engaged only when state_path is supplied)
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """Serialise the persistable counters."""
        return {
            "cash": self._cash,
            "peak_equity": self._peak_equity,
            "day_start_equity": self._day_start_equity,
            "daily_realized_pnl": self._daily_realized_pnl,
            "weekly_realized_pnl": self._weekly_realized_pnl,
            "total_realized_pnl": self._total_realized_pnl,
            "consecutive_losses": self._consecutive_losses,
            "last_loss_time": (
                self._last_loss_time.isoformat()
                if self._last_loss_time is not None else None
            ),
            "day_start": self._day_start.isoformat(),
            "week_start": self._week_start.isoformat(),
        }

    def _persist(self) -> None:
        if self._store is None:
            return
        try:
            self._store.save(self._snapshot())
        except Exception as exc:
            log.warning(
                "PortfolioTracker persist failed (non-fatal): {err}",
                err=str(exc),
            )

    def _restore_from_store(self) -> None:
        """Apply persisted scalars on top of the freshly initialised state.

        ``RiskStateStore.load`` has already applied the daily / weekly
        reset semantics, so we can trust the returned dict.
        """
        try:
            state = self._store.load() if self._store is not None else {}
        except Exception as exc:
            log.warning(
                "PortfolioTracker restore failed (non-fatal): {err}",
                err=str(exc),
            )
            return
        if not state:
            return

        try:
            if "cash" in state:
                self._cash = float(state["cash"])
            if "peak_equity" in state:
                self._peak_equity = float(state["peak_equity"])
            if "day_start_equity" in state:
                self._day_start_equity = float(state["day_start_equity"])
            if "daily_realized_pnl" in state:
                self._daily_realized_pnl = float(state["daily_realized_pnl"])
            if "weekly_realized_pnl" in state:
                self._weekly_realized_pnl = float(state["weekly_realized_pnl"])
            if "total_realized_pnl" in state:
                self._total_realized_pnl = float(state["total_realized_pnl"])
            if "consecutive_losses" in state:
                self._consecutive_losses = int(state["consecutive_losses"])
            llt = state.get("last_loss_time")
            if isinstance(llt, str) and llt:
                try:
                    dt = datetime.fromisoformat(llt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    self._last_loss_time = dt
                except ValueError:
                    pass
            ds = state.get("day_start")
            if isinstance(ds, str) and ds:
                try:
                    dt = datetime.fromisoformat(ds)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    self._day_start = dt
                except ValueError:
                    pass
            ws = state.get("week_start")
            if isinstance(ws, str) and ws:
                try:
                    dt = datetime.fromisoformat(ws)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    self._week_start = dt
                except ValueError:
                    pass
            log.info(
                "PortfolioTracker restored | cash={c} peak={p} "
                "daily_pnl={d} weekly_pnl={w} consec_losses={cl}",
                c=self._cash,
                p=self._peak_equity,
                d=self._daily_realized_pnl,
                w=self._weekly_realized_pnl,
                cl=self._consecutive_losses,
            )
        except Exception as exc:
            log.warning(
                "PortfolioTracker restore parse failed (non-fatal): {err}",
                err=str(exc),
            )
