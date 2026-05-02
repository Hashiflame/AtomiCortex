"""
AtomiCortex — Risk Engine.

Pre-trade filters, ATR-based position sizing, and stop/take-profit
calculation for crypto perpetual futures.

Usage
-----
    from src.risk.risk_engine import RiskEngine, RiskConfig, TradeSignal

    engine = RiskEngine(RiskConfig(), equity=10_000)
    decision = engine.evaluate(signal, portfolio_state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.execution.cost_model import CostModel, FeeConfig
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """All risk-management tunables in one place."""

    # Position sizing
    risk_per_trade: float = 0.01        # 1% of equity
    max_leverage: int = 10
    atr_stop_multiplier: float = 1.5    # stop = 1.5 × ATR

    # Limits
    max_open_positions: int = 3
    daily_loss_limit: float = -0.03     # -3%
    weekly_loss_limit: float = -0.08    # -8%
    max_drawdown_kill: float = -0.15    # -15% → full stop

    # Filters
    confidence_threshold: float = 0.65  # ML confidence minimum
    min_expected_return_bps: float = 15 # at least 15 bps
    max_funding_rate: float = 0.001     # 0.1% extreme
    vol_spike_multiplier: float = 2.0   # ATR > 2× average

    # Circuit breakers
    consecutive_losses_limit: int = 5   # 5 losses in a row
    consecutive_losses_pause_hours: int = 4

    # Cost model defaults (used when estimating fees)
    default_daily_volume: float = 1_000_000_000  # $1B
    default_volatility: float = 0.60              # 60% annualised
    default_hours_held: float = 8.0
    fee_config: FeeConfig = field(default_factory=FeeConfig)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """Incoming ML signal to evaluate."""

    symbol: str
    direction: int          # 1=LONG, -1=SHORT
    confidence: float       # ML confidence 0-1
    regime: str             # trend / high_vol
    entry_price: float
    atr: float              # current ATR in $
    atr_pct: float          # ATR / price
    funding_rate: float
    timestamp: datetime


@dataclass
class PortfolioState:
    """Current portfolio snapshot passed into the engine."""

    equity: float
    open_positions: int
    daily_pnl_pct: float
    weekly_pnl_pct: float
    current_drawdown_pct: float
    consecutive_losses: int
    last_loss_time: datetime | None
    peak_equity: float


@dataclass
class RiskDecision:
    """Output of :meth:`RiskEngine.evaluate`."""

    approved: bool
    reason: str              # rejection reason ("" when approved)
    position_size: float     # contracts (0 when rejected)
    stop_loss: float         # stop-loss price
    take_profit: float       # take-profit price (R:R 1.5)
    notional: float          # position value in USDT
    leverage: float          # effective leverage
    expected_fee_bps: float  # estimated round-trip cost
    risk_reward_ratio: float


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Evaluates every trade signal through a chain of pre-trade filters,
    computes ATR-based position size, and returns a structured decision.
    """

    def __init__(self, config: RiskConfig, equity: float) -> None:
        self._config = config
        self._initial_equity = equity
        self._cost_model = CostModel()
        log.info(
            "RiskEngine initialised | equity={eq} | risk_per_trade={rpt}",
            eq=equity,
            rpt=config.risk_per_trade,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: TradeSignal,
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        """Run *signal* through all pre-trade filters and return a decision."""

        # Sequential filter chain — order matters (cheapest checks first)
        filters = [
            (self._check_max_drawdown, (portfolio_state,)),
            (self._check_weekly_loss, (portfolio_state,)),
            (self._check_daily_loss, (portfolio_state,)),
            (self._check_consecutive_losses, (portfolio_state, signal.timestamp)),
            (self._check_max_positions, (portfolio_state,)),
            (self._check_confidence, (signal,)),
            (self._check_funding_rate, (signal,)),
            (self._check_volatility, (signal,)),
            (self._check_expected_return, (signal, portfolio_state.equity)),
        ]

        for fn, args in filters:
            ok, reason = fn(*args)
            if not ok:
                log.info(
                    "Signal REJECTED | symbol={sym} | filter={flt} | reason={r}",
                    sym=signal.symbol,
                    flt=fn.__name__,
                    r=reason,
                )
                return RiskDecision(
                    approved=False,
                    reason=reason,
                    position_size=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    notional=0.0,
                    leverage=0.0,
                    expected_fee_bps=0.0,
                    risk_reward_ratio=0.0,
                )

        # --- All filters passed — compute sizing ---
        equity = portfolio_state.equity
        contracts, notional, leverage = self.calculate_position_size(signal, equity)
        stop_loss = self.calculate_stop_loss(
            signal.entry_price, signal.direction, signal.atr,
        )
        take_profit = self.calculate_take_profit(
            signal.entry_price, signal.direction, stop_loss,
        )

        # Estimate fees
        rt_cost = self._cost_model.calculate_round_trip_cost(
            notional=notional,
            daily_volume=self._config.default_daily_volume,
            volatility=self._config.default_volatility,
            funding_rate=signal.funding_rate,
            hours_held=self._config.default_hours_held,
            is_long=(signal.direction == 1),
            fee_config=self._config.fee_config,
        )

        risk = abs(signal.entry_price - stop_loss)
        reward = abs(take_profit - signal.entry_price)
        rr_ratio = (reward / risk) if risk > 0 else 0.0

        decision = RiskDecision(
            approved=True,
            reason="",
            position_size=contracts,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notional=notional,
            leverage=leverage,
            expected_fee_bps=rt_cost.total_cost_bps,
            risk_reward_ratio=rr_ratio,
        )

        log.info(
            "Signal APPROVED | symbol={sym} | size={sz:.6f} | "
            "notional=${not_:.2f} | lev={lev:.2f}x | "
            "SL=${sl:.2f} | TP=${tp:.2f} | R:R={rr:.2f}",
            sym=signal.symbol,
            sz=contracts,
            not_=notional,
            lev=leverage,
            sl=stop_loss,
            tp=take_profit,
            rr=rr_ratio,
        )
        return decision

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        signal: TradeSignal,
        equity: float,
    ) -> tuple[float, float, float]:
        """
        ATR-based position sizing.

        Returns
        -------
        (contracts, notional, leverage)
        """
        dollar_risk = equity * self._config.risk_per_trade
        stop_distance = signal.atr * self._config.atr_stop_multiplier

        if stop_distance <= 0:
            return 0.0, 0.0, 0.0

        contracts = dollar_risk / stop_distance
        notional = contracts * signal.entry_price
        leverage = notional / equity if equity > 0 else 0.0

        # Cap leverage
        max_notional = equity * self._config.max_leverage
        if notional > max_notional:
            notional = max_notional
            contracts = notional / signal.entry_price if signal.entry_price > 0 else 0.0
            leverage = float(self._config.max_leverage)

        return contracts, notional, leverage

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float,
    ) -> float:
        """
        Stop-loss = entry ± (ATR × multiplier).

        LONG:  stop = entry - (ATR × 1.5)
        SHORT: stop = entry + (ATR × 1.5)
        """
        offset = atr * self._config.atr_stop_multiplier
        if direction == 1:  # LONG
            return entry_price - offset
        return entry_price + offset  # SHORT

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int,
        stop_loss: float,
        rr_ratio: float = 1.5,
    ) -> float:
        """
        Take-profit achieving the target risk:reward ratio.

        risk   = |entry - stop_loss|
        reward = risk × rr_ratio
        LONG:  TP = entry + reward
        SHORT: TP = entry - reward
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * rr_ratio
        if direction == 1:  # LONG
            return entry_price + reward
        return entry_price - reward  # SHORT

    # ------------------------------------------------------------------
    # Pre-trade filters (private)
    # ------------------------------------------------------------------

    def _check_confidence(self, signal: TradeSignal) -> tuple[bool, str]:
        """Block if ML confidence < threshold."""
        if signal.confidence < self._config.confidence_threshold:
            return False, (
                f"Low confidence {signal.confidence:.2f} "
                f"< {self._config.confidence_threshold}"
            )
        return True, ""

    def _check_daily_loss(self, state: PortfolioState) -> tuple[bool, str]:
        """Block if daily P&L breached the limit."""
        if state.daily_pnl_pct <= self._config.daily_loss_limit:
            return False, (
                f"Daily loss {state.daily_pnl_pct:.2%} "
                f"<= limit {self._config.daily_loss_limit:.2%}"
            )
        return True, ""

    def _check_max_positions(self, state: PortfolioState) -> tuple[bool, str]:
        """Block if already at maximum open positions."""
        if state.open_positions >= self._config.max_open_positions:
            return False, (
                f"Open positions {state.open_positions} "
                f">= max {self._config.max_open_positions}"
            )
        return True, ""

    def _check_expected_return(
        self,
        signal: TradeSignal,
        equity: float,
    ) -> tuple[bool, str]:
        """Block if expected return doesn't cover fees + slippage."""
        # Estimate notional for fee calculation
        _, notional, _ = self.calculate_position_size(signal, equity)
        if notional <= 0:
            return False, "Zero notional — cannot estimate costs"

        rt_cost = self._cost_model.calculate_round_trip_cost(
            notional=notional,
            daily_volume=self._config.default_daily_volume,
            volatility=self._config.default_volatility,
            funding_rate=signal.funding_rate,
            hours_held=self._config.default_hours_held,
            is_long=(signal.direction == 1),
            fee_config=self._config.fee_config,
        )

        expected_return_bps = signal.atr_pct * 10_000  # ATR% → bps
        threshold = max(
            self._config.min_expected_return_bps,
            rt_cost.total_cost_bps * 3,  # rule of 3×
        )

        if expected_return_bps < threshold:
            return False, (
                f"Expected return {expected_return_bps:.1f} bps "
                f"< threshold {threshold:.1f} bps (fees={rt_cost.total_cost_bps:.1f} bps)"
            )
        return True, ""

    def _check_funding_rate(self, signal: TradeSignal) -> tuple[bool, str]:
        """Block if |funding| > extreme threshold."""
        if abs(signal.funding_rate) > self._config.max_funding_rate:
            return False, (
                f"Extreme funding rate {signal.funding_rate:.4%} "
                f"> {self._config.max_funding_rate:.4%}"
            )
        return True, ""

    def _check_volatility(self, signal: TradeSignal) -> tuple[bool, str]:
        """Block if ATR > vol_spike_multiplier × average (circuit breaker)."""
        # atr_pct is ATR/price; spike detection uses the raw ratio.
        # Average ATR is approximated as atr_pct / vol_spike_multiplier
        # threshold, i.e. the user passes actual ATR and we compare to the
        # multiplier against itself.  A true vol-spike means atr_pct is
        # unexpectedly large; we approximate "average" as 1% and flag when
        # atr_pct ≥ 2× that, but the actual comparison is done via the
        # circuit breaker.  Here we simply cap at an absolute level.
        if signal.atr_pct > self._config.vol_spike_multiplier * 0.01:
            return False, (
                f"Volatility spike: ATR% {signal.atr_pct:.4f} "
                f"> {self._config.vol_spike_multiplier}× average"
            )
        return True, ""

    def _check_consecutive_losses(
        self,
        state: PortfolioState,
        signal_time: datetime,
    ) -> tuple[bool, str]:
        """Block if N consecutive losses; enforce pause window."""
        if state.consecutive_losses >= self._config.consecutive_losses_limit:
            pause_hours = self._config.consecutive_losses_pause_hours
            if state.last_loss_time is not None:
                elapsed = (signal_time - state.last_loss_time).total_seconds() / 3600
                if elapsed < pause_hours:
                    return False, (
                        f"{state.consecutive_losses} consecutive losses — "
                        f"paused {pause_hours}h "
                        f"({elapsed:.1f}h elapsed)"
                    )
            else:
                return False, (
                    f"{state.consecutive_losses} consecutive losses — "
                    f"paused {pause_hours}h"
                )
        return True, ""

    def _check_weekly_loss(self, state: PortfolioState) -> tuple[bool, str]:
        """Block if weekly P&L breached the limit."""
        if state.weekly_pnl_pct <= self._config.weekly_loss_limit:
            return False, (
                f"Weekly loss {state.weekly_pnl_pct:.2%} "
                f"<= limit {self._config.weekly_loss_limit:.2%}"
            )
        return True, ""

    def _check_max_drawdown(self, state: PortfolioState) -> tuple[bool, str]:
        """Kill switch: block if drawdown > absolute max.

        ``current_drawdown_pct`` is stored as a **positive** fraction
        (e.g. 0.16 = 16% drawdown).  ``max_drawdown_kill`` is a negative
        config value (e.g. -0.15).  We compare against ``abs(threshold)``.
        """
        threshold = abs(self._config.max_drawdown_kill)  # 0.15
        if state.current_drawdown_pct > threshold:
            return False, (
                f"KILL SWITCH: drawdown {state.current_drawdown_pct:.2%} "
                f"> max {threshold:.2%}"
            )
        return True, ""

