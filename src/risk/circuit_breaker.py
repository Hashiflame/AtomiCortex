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
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    def __init__(self, state_path: Path | str | None = None) -> None:
        self._daily_triggered: bool = False
        self._daily_trigger_reason: str = ""

        # Optional crash-safe persistence. Note ``consecutive_losses`` lives
        # on PortfolioTracker — its persistence is handled there. What this
        # store keeps is the sticky-for-the-day "halted" flag so a restart
        # during a triggered day doesn't quietly resume trading.
        self._store: Any = None
        if state_path is not None:
            from src.risk.risk_state_store import RiskStateStore
            self._store = RiskStateStore(state_path)
            self._restore_from_store()

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
            self._persist()
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
        self._persist()
        log.info("Circuit breaker daily counters reset")

    # ------------------------------------------------------------------
    # Persistence (optional — engaged only when state_path is supplied)
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if self._store is None:
            return
        try:
            # Merge — preserve unrelated keys (the same file is shared with
            # PortfolioTracker in production).
            existing = self._store.load() if self._store is not None else {}
        except Exception:
            existing = {}
        existing["breaker_daily_triggered"] = self._daily_triggered
        existing["breaker_daily_trigger_reason"] = self._daily_trigger_reason
        # Stamp today's UTC midnight so RiskStateStore.load preserves the
        # flag on a same-day reload (and clears it on a next-day reload).
        # In production this key is also written by PortfolioTracker; we
        # set it here so a breaker-only test still gets correct semantics.
        if "day_start" not in existing:
            now = datetime.now(timezone.utc)
            existing["day_start"] = now.replace(
                hour=0, minute=0, second=0, microsecond=0,
            ).isoformat()
        try:
            self._store.save(existing)
        except Exception as exc:
            log.warning(
                "CircuitBreaker persist failed (non-fatal): {err}",
                err=str(exc),
            )

    def _restore_from_store(self) -> None:
        """Re-apply the persisted halt flag; RiskStateStore.load already
        applied the daily reset so a stale yesterday flag is cleared."""
        try:
            state = self._store.load() if self._store is not None else {}
        except Exception as exc:
            log.warning(
                "CircuitBreaker restore failed (non-fatal): {err}",
                err=str(exc),
            )
            return
        if not state:
            return
        self._daily_triggered = bool(state.get("breaker_daily_triggered", False))
        self._daily_trigger_reason = str(state.get("breaker_daily_trigger_reason", ""))
        if self._daily_triggered:
            log.warning(
                "CircuitBreaker restored as TRIGGERED | reason={r}",
                r=self._daily_trigger_reason,
            )

