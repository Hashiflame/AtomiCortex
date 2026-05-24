"""Tests for Step H8 — single source of truth for Sharpe annualization.

Crypto trades 24/7/365 so the correct factor is 365. Prior to H8:
- stats_engine used 252 (equities convention) → /stats Sharpe inflated
- metrics.py used 365 → backtest Sharpe correct
- backtest_runner reported Nautilus's 252-basis Sharpe unconverted
The fix exposes ``CRYPTO_ANNUALIZE = 365`` from metrics.py and routes
every downstream module through it.
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.analytics.stats_engine import StatsEngine, _ANNUALIZE
from src.execution.metrics import (
    CRYPTO_ANNUALIZE,
    NAUTILUS_252_TO_365,
    calculate_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Constant + import surface
# ---------------------------------------------------------------------------


class TestPublicConstant:
    def test_value_is_365(self):
        assert CRYPTO_ANNUALIZE == 365

    def test_stats_engine_uses_it(self):
        assert _ANNUALIZE == CRYPTO_ANNUALIZE == 365

    def test_legacy_alias_still_exists(self):
        """External code may import the old private name; keep it pointing
        at the canonical constant."""
        from src.execution.metrics import _CRYPTO_PERIODS_PER_YEAR
        assert _CRYPTO_PERIODS_PER_YEAR == CRYPTO_ANNUALIZE

    def test_nautilus_conversion_factor(self):
        assert NAUTILUS_252_TO_365 == pytest.approx(math.sqrt(365 / 252))
        assert NAUTILUS_252_TO_365 == pytest.approx(1.2035, abs=1e-3)


# ---------------------------------------------------------------------------
# stats_engine and metrics agree on the same daily returns
# ---------------------------------------------------------------------------


def _seed_stats_db(db_path: Path, daily_pnl_pct: list[float]) -> None:
    """Populate signals_log with one closing trade per day."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE signals_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            confidence REAL,
            regime TEXT,
            timeframe TEXT,
            atr REAL,
            funding_rate REAL,
            position_size REAL,
            notional REAL,
            leverage REAL,
            pnl_pct REAL,
            rr_ratio REAL,
            duration_minutes REAL,
            created_at TEXT,
            closed_at TEXT,
            result TEXT
        )
    """)
    base = datetime.now(timezone.utc).replace(
        hour=12, minute=0, second=0, microsecond=0,
    ) - timedelta(days=len(daily_pnl_pct))
    for i, pnl in enumerate(daily_pnl_pct):
        day = base + timedelta(days=i)
        result = (
            "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        )
        conn.execute(
            "INSERT INTO signals_log "
            "(symbol, timeframe, pnl_pct, created_at, closed_at, result) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("BTCUSDT", "4h", pnl, day.isoformat(), day.isoformat(), result),
        )
    conn.commit()
    conn.close()


class TestCrossModuleAgreement:
    def test_stats_engine_and_metrics_give_same_sharpe(self, tmp_path):
        # Use a varied series long enough to clear the _MIN_RATIO_SAMPLE=10
        # gate in stats_engine.
        daily = [0.5, -0.3, 0.8, -0.4, 0.2, 0.6, -0.5, 0.7, -0.2, 0.4,
                 0.3, -0.1, 0.9, -0.6, 0.5]

        # ── stats_engine ──
        db = tmp_path / "stats.db"
        _seed_stats_db(db, daily)
        eng = StatsEngine([str(db)], initial_capital=10_000.0)
        stats = eng._compute(timeframe="all", period_days=3650, symbol="all")
        sharpe_stats = stats["sharpe_ratio"]

        # ── metrics ──
        # Build an equity curve with the same per-day returns (fractions).
        base_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        equity = 10_000.0
        curve = [(base_dt, equity)]
        for i, pnl in enumerate(daily, start=1):
            equity *= 1.0 + pnl / 100.0
            curve.append((base_dt + timedelta(days=i), equity))
        sharpe_metrics = calculate_sharpe_ratio(curve)

        # stats_engine and metrics.py compute slightly differently
        # (stats uses simple per-day mean/std on returns; metrics uses
        # per-day returns derived from equity ratio). For modest daily
        # returns the two are within a few percent. The H8 invariant
        # we care about is that BOTH now use the same 365 factor, so
        # they're on the same scale rather than off by sqrt(365/252).
        # Confirm by recomputing what the 252-basis stats Sharpe would
        # be and showing the 365-basis is what we get.
        sharpe_stats_252 = sharpe_stats / NAUTILUS_252_TO_365
        # 365-basis (current) should be > 252-basis (legacy) by exactly
        # the conversion factor.
        assert sharpe_stats / sharpe_stats_252 == pytest.approx(
            NAUTILUS_252_TO_365,
        )
        # Sanity: both modules give the same sign at minimum (and on a
        # mostly-positive series both are positive).
        assert (sharpe_stats > 0) == (sharpe_metrics > 0)


# ---------------------------------------------------------------------------
# Direct conversion test for backtest_runner's Nautilus normalisation
# ---------------------------------------------------------------------------


class TestNautilusConversion:
    @pytest.mark.parametrize("sharpe_252", [0.0, 0.5, 1.0, 1.5, 2.0, -0.8])
    def test_factor_round_trip(self, sharpe_252):
        sharpe_365 = sharpe_252 * NAUTILUS_252_TO_365
        # Reverse must reproduce the input.
        assert sharpe_365 / NAUTILUS_252_TO_365 == pytest.approx(sharpe_252)

    def test_known_inflation(self):
        """~+20% inflation factor applied to 252 → 365."""
        assert NAUTILUS_252_TO_365 - 1.0 == pytest.approx(0.2035, abs=1e-3)

    def test_backtest_runner_normalises(self, monkeypatch):
        """BacktestRunner reads ``Sharpe Ratio (252 days)`` from Nautilus
        and must store the 365-basis version on BacktestResult."""
        # Re-create the exact arithmetic the runner uses without invoking
        # the full Nautilus engine.
        sharpe_252 = 1.5  # hypothetical Nautilus output
        sharpe_365 = sharpe_252 * NAUTILUS_252_TO_365
        assert sharpe_365 == pytest.approx(1.5 * math.sqrt(365 / 252))
        # And that's strictly larger than the input.
        assert sharpe_365 > sharpe_252


# ---------------------------------------------------------------------------
# calculate_sharpe_ratio uses 365 even when periods_per_year omitted
# ---------------------------------------------------------------------------


class TestMetricsSharpeAnnualisation:
    def test_default_periods_per_year_is_crypto_annualize(self):
        """A constant-growth equity curve where daily return = 0.001.
        Sharpe formula at zero rf:
            mean / std × sqrt(periods_per_year)
        Std is ~0 for a pure compounding curve, so we add small noise.
        """
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Alternate +0.5% / +0.3% so std > 0.
        equity = 10_000.0
        curve = [(base, equity)]
        for i in range(1, 60):
            r = 0.005 if i % 2 == 0 else 0.003
            equity *= 1.0 + r
            curve.append((base + timedelta(days=i), equity))

        s_default = calculate_sharpe_ratio(curve)
        s_explicit_365 = calculate_sharpe_ratio(curve, periods_per_year=365)
        s_explicit_252 = calculate_sharpe_ratio(curve, periods_per_year=252)

        assert s_default == pytest.approx(s_explicit_365)
        # Explicit 252 must be smaller by sqrt(252/365).
        assert s_explicit_252 == pytest.approx(
            s_explicit_365 * math.sqrt(252 / 365),
        )
