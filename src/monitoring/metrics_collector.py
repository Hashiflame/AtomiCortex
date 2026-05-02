"""
AtomiCortex — Metrics Collector.

Collects trading performance metrics from PortfolioTracker and stores
them in a SQLite database for Grafana dashboards and Telegram reports.

Phase 5 — Paper Trading.
"""

from __future__ import annotations

import math

try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.logger import get_logger
from src.risk.portfolio_tracker import PortfolioTracker

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradingMetrics:
    """Snapshot of trading performance at a given point in time."""

    timestamp: datetime
    equity: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    open_positions: int
    regime: str
    last_signal: str


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects, computes, and persists trading metrics.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    initial_equity:
        Starting equity for PnL percentage calculations.
    """

    def __init__(
        self,
        db_path: str | Path = "data/metrics.db",
        initial_equity: float = 10_000.0,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initial_equity = initial_equity
        self._pnl_returns: list[float] = []
        self._gross_wins: float = 0.0
        self._gross_losses: float = 0.0
        self._total_trades: int = 0
        self._winning_trades: int = 0
        self._max_drawdown: float = 0.0
        self._peak_equity: float = initial_equity
        self._last_regime: str = ""
        self._last_signal: str = ""

        self._init_db()
        _log.info("MetricsCollector initialised | db={db}", db=str(self._db_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        portfolio_tracker: PortfolioTracker,
        regime: str = "",
        last_signal: str = "",
    ) -> TradingMetrics:
        """Collect current metrics from the portfolio tracker.

        Parameters
        ----------
        portfolio_tracker:
            The live PortfolioTracker instance.
        regime:
            Current market regime label.
        last_signal:
            Description of the last signal generated.
        """
        state = portfolio_tracker.get_state()
        self._last_regime = regime or self._last_regime
        self._last_signal = last_signal or self._last_signal

        # Update max drawdown
        if state.current_drawdown_pct > self._max_drawdown:
            self._max_drawdown = state.current_drawdown_pct

        metrics = TradingMetrics(
            timestamp=datetime.now(timezone.utc),
            equity=state.equity,
            daily_pnl=state.daily_pnl_pct * self._initial_equity,
            daily_pnl_pct=state.daily_pnl_pct,
            weekly_pnl=state.weekly_pnl_pct * self._initial_equity,
            weekly_pnl_pct=state.weekly_pnl_pct,
            total_trades=self._total_trades,
            win_rate=self._winning_trades / self._total_trades
            if self._total_trades > 0 else 0.0,
            profit_factor=self._gross_wins / abs(self._gross_losses)
            if self._gross_losses != 0 else 0.0,
            current_drawdown=state.current_drawdown_pct,
            max_drawdown=self._max_drawdown,
            sharpe_ratio=self._calculate_sharpe(),
            open_positions=state.open_positions,
            regime=self._last_regime,
            last_signal=self._last_signal,
        )

        return metrics

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade for win rate / Sharpe calculation."""
        self._total_trades += 1
        self._pnl_returns.append(pnl)
        if pnl > 0:
            self._winning_trades += 1
            self._gross_wins += pnl
        else:
            self._gross_losses += pnl

    def save_to_db(self, metrics: TradingMetrics) -> None:
        """Persist a metrics snapshot to SQLite."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO metrics (
                    timestamp, equity, daily_pnl, daily_pnl_pct,
                    weekly_pnl, weekly_pnl_pct, total_trades, win_rate,
                    profit_factor, current_drawdown, max_drawdown,
                    sharpe_ratio, open_positions, regime, last_signal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    metrics.timestamp.isoformat(),
                    metrics.equity,
                    metrics.daily_pnl,
                    metrics.daily_pnl_pct,
                    metrics.weekly_pnl,
                    metrics.weekly_pnl_pct,
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.current_drawdown,
                    metrics.max_drawdown,
                    metrics.sharpe_ratio,
                    metrics.open_positions,
                    metrics.regime,
                    metrics.last_signal,
                ),
            )
            conn.commit()
            conn.close()
            _log.debug("Metrics saved to DB | equity={eq:.2f}", eq=metrics.equity)
        except Exception as exc:
            _log.error("Failed to save metrics: {err}", err=str(exc))

    def save_signal_to_db(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        regime: str,
        entry_price: float,
        approved: bool,
        reason: str = "",
    ) -> None:
        """Record an ML signal (approved or rejected) in the signals table."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO signals (
                    timestamp, symbol, direction, confidence,
                    regime, entry_price, approved, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    direction,
                    confidence,
                    regime,
                    entry_price,
                    int(approved),
                    reason,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            _log.error("Failed to save signal: {err}", err=str(exc))

    def get_daily_report(self, metrics: TradingMetrics | None = None) -> str:
        """Generate a formatted daily report string."""
        m = metrics
        if m is None:
            return "No metrics available."

        return (
            f"📊 AtomiCortex — Daily Report\n"
            f"{'─' * 35}\n"
            f"Дата:      {m.timestamp.strftime('%Y-%m-%d')}\n"
            f"Equity:    ${m.equity:,.2f}\n"
            f"P&L:       ${m.daily_pnl:+,.2f} ({m.daily_pnl_pct:+.2%})\n"
            f"Сделок:    {m.total_trades}\n"
            f"Win rate:  {m.win_rate:.1%}\n"
            f"PF:        {m.profit_factor:.2f}\n"
            f"Drawdown:  {m.current_drawdown:.2%}\n"
            f"Max DD:    {m.max_drawdown:.2%}\n"
            f"Sharpe:    {m.sharpe_ratio:.2f}\n"
            f"Позиции:   {m.open_positions}\n"
            f"Режим:     {m.regime}\n"
            f"{'─' * 35}"
        )

    def get_weekly_report(self, metrics: TradingMetrics | None = None) -> str:
        """Generate a formatted weekly report string."""
        m = metrics
        if m is None:
            return "No metrics available."

        return (
            f"📈 AtomiCortex — Weekly Report\n"
            f"{'═' * 35}\n"
            f"Неделя:    {m.timestamp.strftime('%Y-W%W')}\n"
            f"Equity:    ${m.equity:,.2f}\n"
            f"P&L:       ${m.weekly_pnl:+,.2f} ({m.weekly_pnl_pct:+.2%})\n"
            f"Сделок:    {m.total_trades}\n"
            f"Win rate:  {m.win_rate:.1%}\n"
            f"PF:        {m.profit_factor:.2f}\n"
            f"Max DD:    {m.max_drawdown:.2%}\n"
            f"Sharpe:    {m.sharpe_ratio:.2f}\n"
            f"{'═' * 35}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _calculate_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualised Sharpe ratio from PnL returns.

        Uses actual PnL amounts normalised by initial equity.
        Annualises using 365 × 6 periods (4H bars).
        """
        if len(self._pnl_returns) < 2:
            return 0.0

        returns = np.array(self._pnl_returns) / self._initial_equity
        mean_ret = float(np.mean(returns)) - risk_free_rate
        std_ret = float(np.std(returns, ddof=1))

        if std_ret < 1e-10:
            return 0.0

        # Annualise: 6 bars/day × 365 days
        periods_per_year = 6 * 365
        return float(mean_ret / std_ret * math.sqrt(periods_per_year))

    def _init_db(self) -> None:
        """Create SQLite tables if they don't exist."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL,
                    daily_pnl REAL,
                    daily_pnl_pct REAL,
                    weekly_pnl REAL,
                    weekly_pnl_pct REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    open_positions INTEGER,
                    regime TEXT,
                    last_signal TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT,
                    direction INTEGER,
                    confidence REAL,
                    regime TEXT,
                    entry_price REAL,
                    approved INTEGER,
                    reason TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as exc:
            _log.error("Failed to init metrics DB: {err}", err=str(exc))
