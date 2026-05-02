"""
AtomiCortex — Circuit Breaker.

Multi-level circuit breaker that monitors portfolio health and either
reduces position sizes or halts trading entirely.

Thresholds (from master document)
---------------------------------
- Daily loss  -2%  →  reduce positions 50%
- Daily loss  -3%  →  stop trading today
- Weekly loss -8%  →  stop trading for the week
- Drawdown   -10%  →  alert
- Drawdown   -15%  →  full stop (kill switch)
- Vol spike   2×   →  reduce positions 50%
- Funding >0.1%    →  block
- 5 consecutive losses → pause
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.logger import get_logger

if TYPE_CHECKING:
    from src.risk.risk_engine import PortfolioState

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreakerState:
    """Snapshot returned by :meth:`CircuitBreaker.check`."""

    is_triggered: bool = False
    trigger_reason: str = ""
    trigger_time: datetime | None = None
    resume_time: datetime | None = None


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Evaluates portfolio and market conditions against hard-coded thresholds
    and returns the current breaker state + a position-size multiplier.
    """

    # Class-level thresholds (from master document)
    DAILY_LOSS_SOFT: float = -0.02        # -2%: reduce positions 50%
    DAILY_LOSS_HARD: float = -0.03        # -3%: stop trading today
    WEEKLY_LOSS: float = -0.08            # -8%: stop for the week
    MAX_DRAWDOWN_WARNING: float = -0.10   # -10%: alert
    MAX_DRAWDOWN_KILL: float = -0.15      # -15%: full stop
    VOL_SPIKE: float = 2.0               # ATR > 2× average
    FUNDING_EXTREME: float = 0.001        # |funding| > 0.1%
    CONSECUTIVE_LOSSES: int = 5           # 5 consecutive losses

    def __init__(self) -> None:
        self._daily_triggered: bool = False
        self._daily_trigger_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        portfolio_state: "PortfolioState",
        current_atr: float,
        avg_atr: float,
        current_funding: float,
    ) -> CircuitBreakerState:
        """
        Evaluate all circuit-breaker conditions and return the most
        severe state found.

        Parameters
        ----------
        portfolio_state:
            Current portfolio snapshot.
        current_atr:
            Current ATR in dollars.
        avg_atr:
            Recent average ATR in dollars.
        current_funding:
            Current funding rate (signed).
        """
        now = datetime.now(timezone.utc)

        # --- Kill switch: max drawdown ---
        dd = -portfolio_state.current_drawdown_pct  # convert to negative
        if dd <= self.MAX_DRAWDOWN_KILL:
            log.critical(
                "KILL SWITCH triggered | drawdown={dd:.2%}",
                dd=portfolio_state.current_drawdown_pct,
            )
            return CircuitBreakerState(
                is_triggered=True,
                trigger_reason=(
                    f"KILL SWITCH: drawdown "
                    f"{portfolio_state.current_drawdown_pct:.2%} "
                    f"> {abs(self.MAX_DRAWDOWN_KILL):.2%}"
                ),
                trigger_time=now,
                resume_time=None,  # manual reset required
            )

        # --- Drawdown warning ---
        if dd <= self.MAX_DRAWDOWN_WARNING:
            log.warning(
                "Drawdown WARNING | drawdown={dd:.2%}",
                dd=portfolio_state.current_drawdown_pct,
            )

        # --- Weekly loss ---
        if portfolio_state.weekly_pnl_pct <= self.WEEKLY_LOSS:
            log.error(
                "Weekly loss breaker triggered | weekly_pnl={wp:.2%}",
                wp=portfolio_state.weekly_pnl_pct,
            )
            return CircuitBreakerState(
                is_triggered=True,
                trigger_reason=(
                    f"Weekly loss {portfolio_state.weekly_pnl_pct:.2%} "
                    f"<= {self.WEEKLY_LOSS:.2%}"
                ),
                trigger_time=now,
                resume_time=None,  # resets next week
            )

        # --- Daily loss hard ---
        if portfolio_state.daily_pnl_pct <= self.DAILY_LOSS_HARD:
            self._daily_triggered = True
            self._daily_trigger_reason = (
                f"Daily loss HARD {portfolio_state.daily_pnl_pct:.2%} "
                f"<= {self.DAILY_LOSS_HARD:.2%}"
            )
            log.error(
                "Daily HARD breaker | daily_pnl={dp:.2%}",
                dp=portfolio_state.daily_pnl_pct,
            )
            return CircuitBreakerState(
                is_triggered=True,
                trigger_reason=self._daily_trigger_reason,
                trigger_time=now,
                resume_time=None,  # resets at midnight
            )

        # --- Consecutive losses ---
        if portfolio_state.consecutive_losses >= self.CONSECUTIVE_LOSSES:
            log.warning(
                "Consecutive losses breaker | count={c}",
                c=portfolio_state.consecutive_losses,
            )
            return CircuitBreakerState(
                is_triggered=True,
                trigger_reason=(
                    f"{portfolio_state.consecutive_losses} consecutive losses "
                    f">= {self.CONSECUTIVE_LOSSES}"
                ),
                trigger_time=now,
                resume_time=None,
            )

        # --- Vol spike ---
        if avg_atr > 0 and (current_atr / avg_atr) > self.VOL_SPIKE:
            spike_ratio = current_atr / avg_atr
            log.warning(
                "Vol spike detected | ATR ratio={r:.2f}",
                r=spike_ratio,
            )
            # Not a full trigger — handled via multiplier
            # but return state for visibility
            return CircuitBreakerState(
                is_triggered=False,
                trigger_reason=f"Vol spike: ATR {spike_ratio:.1f}× average",
                trigger_time=now,
            )

        # --- Extreme funding ---
        if abs(current_funding) > self.FUNDING_EXTREME:
            log.warning(
                "Extreme funding rate | rate={r:.4%}",
                r=current_funding,
            )
            return CircuitBreakerState(
                is_triggered=False,
                trigger_reason=f"Extreme funding: {current_funding:.4%}",
                trigger_time=now,
            )

        return CircuitBreakerState()

    def get_position_size_multiplier(
        self,
        portfolio_state: "PortfolioState",
        current_atr: float = 0.0,
        avg_atr: float = 0.0,
    ) -> float:
        """
        Return a multiplier for position size based on portfolio state.

        - ``1.0`` — normal
        - ``0.5`` — soft daily loss (-2%) OR vol spike
        - ``0.0`` — hard daily loss (-3%) or worse
        """
        # Hard stop conditions → 0.0
        if portfolio_state.daily_pnl_pct <= self.DAILY_LOSS_HARD:
            return 0.0
        if portfolio_state.weekly_pnl_pct <= self.WEEKLY_LOSS:
            return 0.0
        dd = -portfolio_state.current_drawdown_pct
        if dd <= self.MAX_DRAWDOWN_KILL:
            return 0.0
        if portfolio_state.consecutive_losses >= self.CONSECUTIVE_LOSSES:
            return 0.0

        # Soft conditions → 0.5
        if portfolio_state.daily_pnl_pct <= self.DAILY_LOSS_SOFT:
            return 0.5
        if avg_atr > 0 and (current_atr / avg_atr) > self.VOL_SPIKE:
            return 0.5

        return 1.0

    def reset_daily(self) -> None:
        """Reset daily counters at midnight."""
        self._daily_triggered = False
        self._daily_trigger_reason = ""
        log.info("Circuit breaker daily counters reset")

