"""Tests for Steps M6+M7+M8 — Stats / equity display robustness.

M6: pushed the timeframe + wins/losses filters from a Python-side
    post-filter into the SQL WHERE clause; ``_collect_recent`` now
    requests exactly ``limit`` rows per DB instead of a flat 200.
M7: ``min_ratio_sample`` is now part of the StatsEngine payload so the
    Telegram formatter can render "— (нужно 10+, сейчас 3)" instead
    of a bare em-dash with no progress hint.
M8: ``initial_capital`` auto-loads from ``Settings.initial_capital``
    when the caller doesn't pin one explicitly — equity curve no
    longer hard-starts at $10 000 for every premium user.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.execution.signal_bridge import SignalBridge
from src.telegram_bot.database import Database


# ---------------------------------------------------------------------------
# Test rig
# ---------------------------------------------------------------------------


def _seed(db_path: Path, rows: list[dict]) -> None:
    """Populate signals_log via the SignalBridge schema."""
    SignalBridge(str(db_path))  # init schema
    conn = sqlite3.connect(db_path)
    for r in rows:
        conn.execute(
            "INSERT INTO signals_log "
            "(symbol, direction, entry_price, stop_loss, take_profit, "
            "confidence, regime, timeframe, atr, funding_rate, "
            "created_at, closed_at, pnl_pct, result) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r.get("symbol", "BTCUSDT"),
                r.get("direction", "long"),
                50_000.0, 49_000.0, 52_000.0,
                0.7, "trend_up",
                r.get("timeframe", "4h"),
                500.0, 0.0001,
                r["created_at"], r.get("closed_at"),
                r.get("pnl_pct"),
                r.get("result", "open"),
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# M6 — SQL-side filtering in Database.get_recent_signals
# ---------------------------------------------------------------------------


class TestM6SqlFilter:
    def _make_db(self, tmp_path: Path) -> Database:
        path = tmp_path / "bot.db"
        _seed(path, [
            {"created_at": "2026-01-01 10:00:00", "result": "win",
             "timeframe": "4h",  "pnl_pct":  1.0},
            {"created_at": "2026-01-02 10:00:00", "result": "loss",
             "timeframe": "4h",  "pnl_pct": -0.5},
            {"created_at": "2026-01-03 10:00:00", "result": "open",
             "timeframe": "4h"},
            {"created_at": "2026-01-04 10:00:00", "result": "win",
             "timeframe": "15m", "pnl_pct": 0.3},
        ])
        return Database(str(path), init_schema=False)

    def test_result_filter_wins_only(self, tmp_path):
        db = self._make_db(tmp_path)
        rows = db.get_recent_signals(limit=10, result_filter="wins")
        assert all(r["result"] == "win" for r in rows)
        assert len(rows) == 2

    def test_result_filter_losses_only(self, tmp_path):
        db = self._make_db(tmp_path)
        rows = db.get_recent_signals(limit=10, result_filter="losses")
        assert all(r["result"] == "loss" for r in rows)
        assert len(rows) == 1

    def test_result_filter_open_only(self, tmp_path):
        db = self._make_db(tmp_path)
        rows = db.get_recent_signals(limit=10, result_filter="open")
        assert all(r["result"] == "open" for r in rows)
        assert len(rows) == 1

    def test_timeframe_and_result_filter_combined(self, tmp_path):
        db = self._make_db(tmp_path)
        rows = db.get_recent_signals(
            limit=10, timeframe="15m", result_filter="wins",
        )
        assert len(rows) == 1
        assert rows[0]["timeframe"] == "15m"

    def test_limit_applied_at_sql_level(self, tmp_path):
        db = self._make_db(tmp_path)
        rows = db.get_recent_signals(limit=2)
        # No filter — newest 2 of 4.
        assert len(rows) == 2

    def test_unknown_result_filter_ignored(self, tmp_path):
        db = self._make_db(tmp_path)
        rows = db.get_recent_signals(limit=10, result_filter="potato")
        assert len(rows) == 4  # unfiltered


# ---------------------------------------------------------------------------
# M6 — _collect_recent pushes the filter down
# ---------------------------------------------------------------------------


class TestM6CollectRecentPushesDown:
    def test_result_filter_forwarded_to_sql(self, monkeypatch):
        from src.telegram_bot.handlers_premium import _collect_recent
        db = MagicMock()
        db.get_recent_signals.return_value = []
        monkeypatch.setattr(
            "src.telegram_bot.handlers_premium._resolve_stat_dbs",
            lambda c: [("4h", db)],
        )
        ctx = MagicMock(); ctx.bot_data = {}
        _collect_recent(
            ctx, limit=42, timeframe="4h", result_filter="wins",
        )
        kwargs = db.get_recent_signals.call_args.kwargs
        assert kwargs["limit"] == 42  # not the legacy flat 200
        assert kwargs["timeframe"] == "4h"
        assert kwargs["result_filter"] == "wins"

    def test_skips_db_for_mismatched_timeframe(self, monkeypatch):
        from src.telegram_bot.handlers_premium import _collect_recent
        db = MagicMock()
        monkeypatch.setattr(
            "src.telegram_bot.handlers_premium._resolve_stat_dbs",
            lambda c: [("4h", db)],
        )
        ctx = MagicMock(); ctx.bot_data = {}
        _collect_recent(ctx, limit=10, timeframe="15m")
        db.get_recent_signals.assert_not_called()


# ---------------------------------------------------------------------------
# M7 — min_ratio_sample exposed alongside None metrics
# ---------------------------------------------------------------------------


class TestM7MinRatioSampleSurfaced:
    def test_below_threshold_emits_count_and_sample(self, tmp_path):
        from src.analytics.stats_engine import (
            StatsEngine, _MIN_RATIO_SAMPLE,
        )
        path = tmp_path / "stats.db"
        # Three closed signals on three distinct days — below the 10
        # threshold.
        _seed(path, [
            {"created_at": "2026-01-01 10:00:00",
             "closed_at": "2026-01-01 11:00:00",
             "result": "win", "pnl_pct": 1.0},
            {"created_at": "2026-01-02 10:00:00",
             "closed_at": "2026-01-02 11:00:00",
             "result": "loss", "pnl_pct": -0.5},
            {"created_at": "2026-01-03 10:00:00",
             "closed_at": "2026-01-03 11:00:00",
             "result": "win", "pnl_pct": 0.5},
        ])
        eng = StatsEngine([str(path)])
        out = eng._compute(timeframe="all", period_days=3650, symbol="all")
        # Sharpe / Sortino / Calmar suppressed because sample < threshold.
        assert out["sharpe_ratio"] is None
        assert out["sortino_ratio"] is None
        assert out["calmar_ratio"] is None
        # And the formatter now has everything it needs to render the
        # "— (нужно 10+, сейчас 3)" hint.
        assert out["closed_signals"] == 3
        assert out["min_ratio_sample"] == _MIN_RATIO_SAMPLE

    def test_above_threshold_emits_real_metrics(self, tmp_path):
        from src.analytics.stats_engine import StatsEngine
        path = tmp_path / "stats.db"
        rows = []
        for i in range(12):  # ≥ 10 sample
            sign = 1 if i % 2 == 0 else -1
            rows.append({
                "created_at": f"2026-01-{i+1:02d} 10:00:00",
                "closed_at":  f"2026-01-{i+1:02d} 11:00:00",
                "result":     "win" if sign > 0 else "loss",
                "pnl_pct":    sign * 0.5,
            })
        _seed(path, rows)
        eng = StatsEngine([str(path)])
        out = eng._compute(timeframe="all", period_days=3650, symbol="all")
        assert out["sharpe_ratio"] is not None
        assert out["closed_signals"] == 12
        # Threshold still surfaced for the formatter.
        assert "min_ratio_sample" in out


# ---------------------------------------------------------------------------
# M8 — initial_capital auto-loads from Settings when None
# ---------------------------------------------------------------------------


class TestM8InitialCapitalResolution:
    def test_explicit_value_wins(self, tmp_path):
        from src.analytics.stats_engine import StatsEngine
        _seed(tmp_path / "stats.db", [])
        eng = StatsEngine([str(tmp_path / "stats.db")], initial_capital=50_000.0)
        assert eng.initial_capital == 50_000.0

    def test_default_pulls_from_settings(self, tmp_path, monkeypatch):
        from src.analytics.stats_engine import StatsEngine
        # Fake Settings → 25k.
        fake_settings = MagicMock()
        fake_settings.initial_capital = 25_000.0

        monkeypatch.setattr(
            "src.config.get_settings", lambda: fake_settings,
        )
        _seed(tmp_path / "stats.db", [])
        eng = StatsEngine([str(tmp_path / "stats.db")])
        assert eng.initial_capital == 25_000.0

    def test_settings_failure_falls_back_to_10k(self, tmp_path, monkeypatch):
        from src.analytics.stats_engine import StatsEngine

        def _boom():
            raise RuntimeError("synthetic")
        monkeypatch.setattr("src.config.get_settings", _boom)
        _seed(tmp_path / "stats.db", [])
        eng = StatsEngine([str(tmp_path / "stats.db")])
        assert eng.initial_capital == 10_000.0

    def test_equity_curve_starts_from_settings_capital(
        self, tmp_path, monkeypatch,
    ):
        from src.analytics.stats_engine import StatsEngine
        fake_settings = MagicMock()
        fake_settings.initial_capital = 25_000.0
        monkeypatch.setattr(
            "src.config.get_settings", lambda: fake_settings,
        )
        path = tmp_path / "stats.db"
        _seed(path, [
            {"created_at": "2026-01-01 10:00:00",
             "closed_at": "2026-01-01 11:00:00",
             "result": "win", "pnl_pct": 1.0},
            {"created_at": "2026-01-02 10:00:00",
             "closed_at": "2026-01-02 11:00:00",
             "result": "win", "pnl_pct": 0.5},
        ])
        eng = StatsEngine([str(path)])
        curve = eng.compute_equity_curve(period_days=3650)
        # First day: 25k × (1 + 0.01) = 25_250.0.
        assert curve[0]["equity"] == pytest.approx(25_250.0)
        # Curve scales linearly — not the legacy 10k start.
        assert curve[0]["equity"] != pytest.approx(10_100.0)


# ---------------------------------------------------------------------------
# Backward-compat smoke
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_get_recent_signals_no_kwargs(self, tmp_path):
        """Existing callers without the new kwarg keep working."""
        path = tmp_path / "bot.db"
        _seed(path, [
            {"created_at": "2026-01-01 10:00:00", "result": "open"},
        ])
        db = Database(str(path), init_schema=False)
        rows = db.get_recent_signals(limit=5)
        assert len(rows) == 1
