"""
Trading statistics calculator.

Computes performance metrics from ``signals_log`` across one or more
isolated DBs and caches them in ``performance_cache``. Read path is
cache-first (fresh < 1h); write path recomputes from the closed-signal
ledger maintained by the reconciler.

Metrics: win_rate, profit_factor, expected_value, total_pnl_pct,
max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, avg_rr_ratio,
avg_duration_h, plus the daily equity curve.

Formulas follow the project convention: annualisation factor =
``CRYPTO_ANNUALIZE`` (365), imported from ``src.execution.metrics`` so
backtest reports and Telegram /stats give identical Sharpe numbers
(H8 — was 252, which inflated reported Sharpe by ~20.5% vs. metrics.py).
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.execution.metrics import CRYPTO_ANNUALIZE
from src.logger import get_logger

_log = get_logger(__name__)

# Single source of truth shared with backtest_runner / metrics.py.
_ANNUALIZE = CRYPTO_ANNUALIZE
_CACHE_TTL_SEC = 3600
# Minimum closed signals before risk ratios are statistically meaningful.
_MIN_RATIO_SAMPLE = 10


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_pf(gross_win: float, gross_loss: float) -> float | None:
    """sum(win)/|sum(loss)| — None for the undefined ∞ case (no losses)."""
    if gross_loss != 0:
        return gross_win / abs(gross_loss)
    return None if gross_win > 0 else 0.0


def _std(xs: list[float]) -> float:
    """Population-consistent sample std (ddof=1); 0 for <2 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


class StatsEngine:
    def __init__(
        self, db_paths: list[str], initial_capital: float = 10_000.0,
    ) -> None:
        self.db_paths = [str(p) for p in db_paths]
        self.initial_capital = initial_capital
        # Canonical cache lives in the first (base/4H) DB.
        self._cache_db = self.db_paths[0] if self.db_paths else None

    # ------------------------------------------------------------------
    # Row loading
    # ------------------------------------------------------------------

    def _load_signals(
        self, timeframe: str, period_days: int, symbol: str,
    ) -> list[dict[str, Any]]:
        """Closed+open signals across all DBs for the filter window."""
        out: list[dict[str, Any]] = []
        for db in self.db_paths:
            if not Path(db).exists():
                continue
            conn = _connect(db)
            try:
                cols = {
                    r[1] for r in conn.execute(
                        "PRAGMA table_info(signals_log)"
                    ).fetchall()
                }
                if not cols:
                    continue
                has_tf = "timeframe" in cols
                where = [
                    f"created_at >= datetime('now', '-{int(period_days)} days')"
                ]
                params: list[Any] = []
                if timeframe != "all":
                    if has_tf:
                        where.append("COALESCE(timeframe,'4h') = ?")
                        params.append(timeframe)
                    elif timeframe != "4h":
                        where.append("1 = 0")
                if symbol != "all":
                    where.append("symbol LIKE ?")
                    params.append(f"%{symbol}%")
                sql = (
                    "SELECT * FROM signals_log WHERE "
                    + " AND ".join(where)
                    + " ORDER BY created_at ASC"
                )
                for r in conn.execute(sql, params).fetchall():
                    out.append(dict(r))
            except sqlite3.OperationalError:
                continue
            finally:
                conn.close()
        out.sort(key=lambda s: str(s.get("created_at") or ""))
        return out

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def compute_equity_curve(
        self, timeframe: str = "all", symbol: str = "all",
        period_days: int = 3650,
    ) -> list[dict]:
        """Daily compounded equity + running drawdown from closed signals."""
        sigs = [
            s for s in self._load_signals(timeframe, period_days, symbol)
            if s.get("result") in ("win", "loss", "breakeven")
            and s.get("closed_at")
        ]
        by_day: dict[str, float] = {}
        for s in sigs:
            day = str(s["closed_at"])[:10]
            by_day[day] = by_day.get(day, 0.0) + float(s.get("pnl_pct") or 0.0)

        curve: list[dict] = []
        eq = self.initial_capital
        peak = eq
        for day in sorted(by_day):
            eq *= 1.0 + by_day[day] / 100.0
            peak = max(peak, eq)
            dd = (eq - peak) / peak * 100.0 if peak > 0 else 0.0
            curve.append({
                "date": day,
                "equity": round(eq, 2),
                "drawdown": round(dd, 4),
            })
        return curve

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def _compute(
        self, timeframe: str, period_days: int, symbol: str,
    ) -> dict[str, Any]:
        sigs = self._load_signals(timeframe, period_days, symbol)
        closed = [s for s in sigs if s.get("result") in ("win", "loss", "breakeven")]
        wins = [s for s in closed if s.get("result") == "win"]
        losses = [s for s in closed if s.get("result") == "loss"]
        open_n = sum(1 for s in sigs if s.get("result") == "open")

        pnls = [float(s.get("pnl_pct") or 0.0) for s in closed]
        gross_win = sum(float(s.get("pnl_pct") or 0.0) for s in wins)
        gross_loss = sum(float(s.get("pnl_pct") or 0.0) for s in losses)

        # Daily returns (fraction) from per-day summed pnl_pct.
        by_day: dict[str, float] = {}
        for s in closed:
            if not s.get("closed_at"):
                continue
            day = str(s["closed_at"])[:10]
            by_day[day] = by_day.get(day, 0.0) + float(s.get("pnl_pct") or 0.0)
        daily = [v / 100.0 for _, v in sorted(by_day.items())]

        sharpe = sortino = 0.0
        if len(daily) >= 2:
            mean_d = sum(daily) / len(daily)
            sd = _std(daily)
            if sd > 0:
                sharpe = mean_d / sd * math.sqrt(_ANNUALIZE)
            # Bug 1 fix: no losing days (or zero downside deviation) is
            # *good* — Sortino is undefined there, not zero. Fall back to
            # Sharpe rather than reporting 0.0.
            downside = [r for r in daily if r < 0]
            dsd = _std(downside)
            if len(downside) == 0 or dsd == 0:
                sortino = sharpe
            else:
                sortino = mean_d / dsd * math.sqrt(_ANNUALIZE)

        # Equity / drawdown.
        eq = self.initial_capital
        peak = eq
        max_dd = 0.0
        for _, v in sorted(by_day.items()):
            eq *= 1.0 + v / 100.0
            peak = max(peak, eq)
            dd = (eq - peak) / peak * 100.0 if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        total_pnl = (eq / self.initial_capital - 1.0) * 100.0

        ann_return = (
            (sum(daily) / len(daily)) * _ANNUALIZE * 100.0 if daily else 0.0
        )
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Bug 2 fix: ratios on a tiny sample are statistical noise
        # (a 2-trade "Sharpe 40" is meaningless). Below the minimum
        # sample they are reported as None → JSON null / "мало данных".
        if len(closed) < _MIN_RATIO_SAMPLE:
            sharpe = sortino = calmar = None

        rrs = [
            float(s["rr_ratio"]) for s in closed
            if s.get("rr_ratio") is not None
        ]
        durs = [
            float(s["duration_minutes"]) for s in closed
            if s.get("duration_minutes") is not None
        ]
        confs = [
            float(s["confidence"]) for s in sigs
            if s.get("confidence") is not None
        ]

        created = [str(s["created_at"])[:10] for s in sigs if s.get("created_at")]
        live_since = min(created) if created else None
        days_tracked = 0
        if live_since:
            try:
                ls = datetime.fromisoformat(live_since).replace(
                    tzinfo=timezone.utc
                )
                days_tracked = max(
                    1, (datetime.now(timezone.utc) - ls).days
                )
            except ValueError:
                days_tracked = 0

        decided = len(wins) + len(losses)
        return {
            "timeframe": timeframe,
            "period_days": period_days,
            "symbol": symbol,
            "win_rate": round(len(wins) / decided, 4) if decided else 0.0,
            "profit_factor": _safe_pf(gross_win, gross_loss),
            "expected_value": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
            "total_pnl_pct": round(total_pnl, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": None if sharpe is None else round(sharpe, 4),
            "sortino_ratio": None if sortino is None else round(sortino, 4),
            "calmar_ratio": None if calmar is None else round(calmar, 4),
            "avg_rr_ratio": round(sum(rrs) / len(rrs), 4) if rrs else 0.0,
            "total_signals": len(sigs),
            "open_signals": open_n,
            "closed_signals": len(closed),
            "win_count": len(wins),
            "loss_count": len(losses),
            "avg_duration_h": (
                round(sum(durs) / len(durs) / 60.0, 4) if durs else 0.0
            ),
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "live_since": live_since,
            "days_tracked": days_tracked,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------

    def _read_cache(
        self, timeframe: str, period_days: int, symbol: str,
    ) -> dict | None:
        if not self._cache_db or not Path(self._cache_db).exists():
            return None
        conn = _connect(self._cache_db)
        try:
            row = conn.execute(
                "SELECT * FROM performance_cache "
                "WHERE timeframe=? AND period_days=? AND symbol=?",
                (timeframe, period_days, symbol),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()
        if not row:
            return None
        d = dict(row)
        try:
            updated = datetime.fromisoformat(d["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            if age > _CACHE_TTL_SEC:
                return None
        except (ValueError, TypeError, KeyError):
            return None
        return d

    def _write_cache(self, stats: dict) -> None:
        if not self._cache_db:
            return
        conn = _connect(self._cache_db)
        try:
            pf = stats["profit_factor"]
            conn.execute(
                """INSERT INTO performance_cache
                     (timeframe, period_days, symbol, win_rate,
                      profit_factor, expected_value, total_pnl_pct,
                      max_drawdown, sharpe_ratio, sortino_ratio,
                      calmar_ratio, avg_rr_ratio, total_signals,
                      closed_signals, win_count, loss_count,
                      avg_duration_h, live_since, days_tracked, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(timeframe, period_days, symbol) DO UPDATE SET
                     win_rate=excluded.win_rate,
                     profit_factor=excluded.profit_factor,
                     expected_value=excluded.expected_value,
                     total_pnl_pct=excluded.total_pnl_pct,
                     max_drawdown=excluded.max_drawdown,
                     sharpe_ratio=excluded.sharpe_ratio,
                     sortino_ratio=excluded.sortino_ratio,
                     calmar_ratio=excluded.calmar_ratio,
                     avg_rr_ratio=excluded.avg_rr_ratio,
                     total_signals=excluded.total_signals,
                     closed_signals=excluded.closed_signals,
                     win_count=excluded.win_count,
                     loss_count=excluded.loss_count,
                     avg_duration_h=excluded.avg_duration_h,
                     live_since=excluded.live_since,
                     days_tracked=excluded.days_tracked,
                     updated_at=excluded.updated_at""",
                (
                    stats["timeframe"], stats["period_days"], stats["symbol"],
                    stats["win_rate"], pf, stats["expected_value"],
                    stats["total_pnl_pct"], stats["max_drawdown"],
                    stats["sharpe_ratio"], stats["sortino_ratio"],
                    stats["calmar_ratio"], stats["avg_rr_ratio"],
                    stats["total_signals"], stats["closed_signals"],
                    stats["win_count"], stats["loss_count"],
                    stats["avg_duration_h"], stats["live_since"],
                    stats["days_tracked"], stats["updated_at"],
                ),
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            _log.warning("performance_cache write skipped: {e}", e=exc)
        finally:
            conn.close()

    def compute_performance(
        self, timeframe: str = "all", period_days: int = 30,
        symbol: str = "all", use_cache: bool = True,
    ) -> dict[str, Any]:
        if use_cache:
            cached = self._read_cache(timeframe, period_days, symbol)
            if cached is not None:
                return cached
        stats = self._compute(timeframe, period_days, symbol)
        self._write_cache(stats)
        return stats

    def compute_all(self) -> dict[str, Any]:
        """Recompute + cache every (timeframe, period) combination."""
        timeframes = ["all", "4h", "15m", "1h"]
        periods = [7, 30, 90]
        n = 0
        for tf in timeframes:
            for pd in periods:
                self._write_cache(self._compute(tf, pd, "all"))
                n += 1
        return {"cached": n, "db": self._cache_db}

    def compute_monthly(
        self, timeframe: str = "all", symbol: str = "all",
    ) -> list[dict]:
        """Monthly P&L breakdown from closed signals."""
        sigs = [
            s for s in self._load_signals(timeframe, 3650, symbol)
            if s.get("result") in ("win", "loss", "breakeven")
            and s.get("closed_at")
        ]
        agg: dict[str, dict] = {}
        for s in sigs:
            mo = str(s["closed_at"])[:7]
            a = agg.setdefault(mo, {"pnl": 0.0, "w": 0, "l": 0})
            a["pnl"] += float(s.get("pnl_pct") or 0.0)
            if s["result"] == "win":
                a["w"] += 1
            elif s["result"] == "loss":
                a["l"] += 1
        out = []
        for mo in sorted(agg):
            a = agg[mo]
            dec = a["w"] + a["l"]
            out.append({
                "month": mo,
                "pnl_pct": round(a["pnl"], 4),
                "wins": a["w"], "losses": a["l"],
                "wr": round(a["w"] / dec, 4) if dec else 0.0,
            })
        return out
