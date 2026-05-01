"""Performance metrics for backtests and walk-forward windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class MetricsResult:
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate: float          # percentage 0-100
    profit_factor: float
    total_return_pct: float
    annualized_return_pct: float
    total_trades: int

    def passes_minimum_thresholds(self) -> bool:
        """Go-live gate from master document."""
        return (
            self.sharpe_ratio >= 1.0
            and self.win_rate >= 52.0
            and self.profit_factor >= 1.3
            and self.max_drawdown_pct <= 20.0
        )

    def to_dict(self) -> dict[str, float]:
        """Flat float dict suitable for MLflow log_metrics."""
        pf = self.profit_factor
        if math.isinf(pf) or math.isnan(pf):
            pf = 999.0
        return {
            "sharpe_ratio": float(self.sharpe_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "win_rate": float(self.win_rate),
            "profit_factor": pf,
            "total_return_pct": float(self.total_return_pct),
            "annualized_return_pct": float(self.annualized_return_pct),
            "total_trades": float(self.total_trades),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Individual metric calculators
# ──────────────────────────────────────────────────────────────────────────────

_CRYPTO_PERIODS_PER_YEAR = 365  # crypto trades every calendar day

def calculate_sharpe_ratio(
    equity_curve: list[tuple[datetime, float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = _CRYPTO_PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio computed from *daily* returns.

    Intra-day points (e.g. 4H bars) are collapsed to end-of-day before
    return computation, so ``periods_per_year`` is always effectively 365
    regardless of input bar frequency.  ``risk_free_rate`` defaults to 0.0
    (no risk-free-rate adjustment) which is the standard convention for
    crypto futures where there is no liquid risk-free instrument.

    Returns 0.0 when fewer than 2 distinct trading days are present or when
    the daily-return std is below the numerical noise floor (~1e-8).
    """
    if len(equity_curve) < 2:
        return 0.0

    # Collapse to end-of-day: intra-day bar frequency does not matter
    daily: dict = {}
    for dt, equity in equity_curve:
        daily[dt.date()] = equity

    sorted_days = sorted(daily)
    if len(sorted_days) < 2:
        return 0.0

    equities = [daily[d] for d in sorted_days]
    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]

    n = len(returns)
    if n < 2:
        return 0.0

    mean_r = sum(returns) / n
    # After daily collapse, rf is always annual / 365 regardless of the
    # caller-supplied periods_per_year (which only affects annualisation).
    rf_per_day = risk_free_rate / _CRYPTO_PERIODS_PER_YEAR
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0

    # Threshold guards against floating-point noise masquerading as volatility
    # (e.g. constant-rate equity where all returns are "equal" in theory).
    if std_r < 1e-8:
        return 0.0

    return (mean_r - rf_per_day) / std_r * math.sqrt(periods_per_year)


def calculate_max_drawdown(
    equity_curve: list[tuple[datetime, float]],
) -> float:
    """Maximum peak-to-trough drawdown in percent (0-100)."""
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return max_dd


def calculate_calmar_ratio(
    equity_curve: list[tuple[datetime, float]],
    periods_per_year: int = 365,
) -> float:
    """Calmar = annualised return / max drawdown.  Returns 0.0 when MDD == 0."""
    if len(equity_curve) < 2:
        return 0.0

    start_eq = equity_curve[0][1]
    end_eq = equity_curve[-1][1]
    if start_eq <= 0:
        return 0.0

    days = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds() / 86400
    if days <= 0:
        return 0.0

    total_ret = end_eq / start_eq - 1
    annual_ret = (1 + total_ret) ** (periods_per_year / days) - 1

    max_dd = calculate_max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0

    return annual_ret / (max_dd / 100)


def calculate_win_rate(trades: list[dict]) -> float:
    """Win rate in percent (0-100).  Returns 0.0 for empty trades list."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades) * 100


def calculate_profit_factor(trades: list[dict]) -> float:
    """PF = gross wins / |gross losses|.  Returns inf if no losing trades."""
    gross_win = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0)
    if gross_loss == 0:
        return float("inf")
    return gross_win / abs(gross_loss)


def calculate_all_metrics(
    equity_curve: list[tuple[datetime, float]],
    trades: list[dict],
    risk_free_rate: float = 0.05,
) -> MetricsResult:
    """Compute all metrics and return a MetricsResult."""
    sharpe = calculate_sharpe_ratio(equity_curve, risk_free_rate)
    calmar = calculate_calmar_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades)
    pf = calculate_profit_factor(trades)

    if len(equity_curve) >= 2:
        s_eq = equity_curve[0][1]
        e_eq = equity_curve[-1][1]
        total_ret = (e_eq - s_eq) / s_eq * 100 if s_eq > 0 else 0.0
        days = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds() / 86400
        if days > 0 and s_eq > 0:
            annual_ret = ((e_eq / s_eq) ** (365 / days) - 1) * 100
        else:
            annual_ret = 0.0
    else:
        total_ret = 0.0
        annual_ret = 0.0

    return MetricsResult(
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        profit_factor=pf,
        total_return_pct=total_ret,
        annualized_return_pct=annual_ret,
        total_trades=len(trades),
    )
