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

# H8: canonical annualisation factor for crypto Sharpe / Sortino / Calmar.
# Crypto trades 24/7/365, so 365 — not the equities-market 252 — is the
# correct number of daily periods per year. Imported by every downstream
# module that annualises (stats_engine, backtest_runner) so the bot
# reports a single Sharpe value everywhere.
CRYPTO_ANNUALIZE: int = 365

# Nautilus-Trader reports Sharpe under the key "Sharpe Ratio (252 days)"
# and we cannot change its internals. Multiply by this factor to convert
# the 252-basis number to a 365-basis one consistent with the rest of
# the project: sharpe_365 = sharpe_252 * sqrt(365 / 252).
NAUTILUS_252_TO_365: float = math.sqrt(CRYPTO_ANNUALIZE / 252)

# Legacy private alias kept so external imports keep working.
_CRYPTO_PERIODS_PER_YEAR = CRYPTO_ANNUALIZE

def calculate_sharpe_ratio(
    equity_curve: list[tuple[datetime, float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = CRYPTO_ANNUALIZE,
    bar_duration_minutes: int | None = None,
) -> float:
    """Annualised Sharpe ratio.

    Default mode (``bar_duration_minutes is None``) collapses the equity
    curve to end-of-day and annualises by ``√CRYPTO_ANNUALIZE`` — the
    pre-H20 behaviour, preserved for backward compatibility.

    H20 mode (``bar_duration_minutes`` set, e.g. 240 for 4H, 15 for 15m):
    skip the daily collapse and compute returns at the native bar
    cadence, then annualise by ``√(365 × 24 × 60 / bar_duration_minutes)``.
    Daily collapse hides intraday volatility — for 15m it shrinks 96
    bars per day into one point, inflating Sharpe 3-5×. Pass the bar
    duration explicitly to get the correct number.

    ``risk_free_rate`` defaults to 0.0 (crypto convention — there is no
    liquid risk-free instrument). H21: every caller in the project
    now defaults to 0.0 so the same equity curve always produces the
    same Sharpe number.

    Returns 0.0 on insufficient data or when the return std is below
    the numerical noise floor.
    """
    if len(equity_curve) < 2:
        return 0.0

    if bar_duration_minutes is not None and bar_duration_minutes > 0:
        # H20: use every bar — no daily collapse.
        sorted_curve = sorted(equity_curve, key=lambda p: p[0])
        equities = [eq for _, eq in sorted_curve]
        bars_per_year = max(
            1, 365 * 24 * 60 // int(bar_duration_minutes),
        )
        rf_per_period = risk_free_rate / bars_per_year
        ann_factor = bars_per_year
    else:
        # Legacy daily-collapse path.
        daily: dict = {}
        for dt, equity in equity_curve:
            daily[dt.date()] = equity
        sorted_days = sorted(daily)
        if len(sorted_days) < 2:
            return 0.0
        equities = [daily[d] for d in sorted_days]
        rf_per_period = risk_free_rate / CRYPTO_ANNUALIZE
        ann_factor = periods_per_year

    returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
        if equities[i - 1] > 0
    ]

    n = len(returns)
    if n < 2:
        return 0.0

    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0

    # Threshold guards against floating-point noise masquerading as volatility
    # (e.g. constant-rate equity where all returns are "equal" in theory).
    if std_r < 1e-8:
        return 0.0

    return (mean_r - rf_per_period) / std_r * math.sqrt(ann_factor)


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
    risk_free_rate: float = 0.0,
    bar_duration_minutes: int | None = None,
) -> MetricsResult:
    """Compute all metrics and return a MetricsResult.

    H21: default risk_free_rate is now 0.0 (was 0.05), matching
    ``calculate_sharpe_ratio`` so the same equity curve produces the
    same Sharpe regardless of which entry point the caller uses.

    H20: ``bar_duration_minutes`` is forwarded to ``calculate_sharpe_ratio``
    so 4H / 15m callers get correctly-annualised Sharpe on bar-level
    returns instead of the silently-inflated daily-collapse number.
    """
    sharpe = calculate_sharpe_ratio(
        equity_curve, risk_free_rate,
        bar_duration_minutes=bar_duration_minutes,
    )
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
