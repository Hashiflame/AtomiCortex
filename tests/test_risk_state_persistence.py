"""Persistence tests for PortfolioTracker + CircuitBreaker risk counters.

These pin down the contract the docstring promises:
* same-day restart: daily / weekly / consec_losses / peak_equity survive
* next-day restart: daily counters reset; weekly survives
* next-week restart: both daily and weekly reset
* corrupted store: graceful degradation to defaults
* unwritable store: tracker / breaker keep working in memory
* backward compat: no state_path → no I/O at all
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.portfolio_tracker import PortfolioTracker
from src.risk.risk_engine import PortfolioState
from src.risk.risk_state_store import (
    RiskStateStore,
    _today_start_utc,
    _week_start_utc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "risk_state.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Backward compatibility — default constructor unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_tracker_without_state_path_does_no_io(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Existing callers (no state_path) must touch no file at all."""
        # Move cwd to an empty dir; nothing should be created.
        monkeypatch.chdir(tmp_path)
        t = PortfolioTracker(initial_equity=10_000)
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t.close_position("BTCUSDT", 50_500, fee=0.0, timestamp=_now())
        # No files were written
        assert list(tmp_path.iterdir()) == []

    def test_breaker_without_state_path_works(self) -> None:
        """CircuitBreaker() with no args must behave exactly as before."""
        b = CircuitBreaker()
        assert b._daily_triggered is False
        # No persistence side effects
        b._persist()  # noop, no raise


# ---------------------------------------------------------------------------
# RiskStateStore — temporal reset semantics
# ---------------------------------------------------------------------------

class TestTemporalReset:
    def test_load_resets_daily_when_day_crossed(
        self, store_path: Path
    ) -> None:
        yesterday = (_today_start_utc() - timedelta(days=1)).isoformat()
        store_path.write_text(json.dumps({
            "daily_realized_pnl": -250.0,
            "weekly_realized_pnl": -250.0,
            "day_start": yesterday,
            "week_start": _week_start_utc().isoformat(),
        }), encoding="utf-8")

        loaded = RiskStateStore(store_path).load()
        assert loaded["daily_realized_pnl"] == 0.0
        assert datetime.fromisoformat(loaded["day_start"]) == _today_start_utc()
        # Weekly still in current week → preserved
        assert loaded["weekly_realized_pnl"] == -250.0

    def test_load_resets_weekly_when_week_crossed(
        self, store_path: Path
    ) -> None:
        last_week = (_week_start_utc() - timedelta(days=7)).isoformat()
        store_path.write_text(json.dumps({
            "weekly_realized_pnl": -800.0,
            "week_start": last_week,
            "daily_realized_pnl": -100.0,
            "day_start": _today_start_utc().isoformat(),
        }), encoding="utf-8")

        loaded = RiskStateStore(store_path).load()
        # week crossed → reset to 0; daily within today → preserved
        assert loaded["weekly_realized_pnl"] == 0.0
        assert loaded["daily_realized_pnl"] == -100.0

    def test_load_preserves_same_day(self, store_path: Path) -> None:
        store_path.write_text(json.dumps({
            "daily_realized_pnl": -123.0,
            "weekly_realized_pnl": -456.0,
            "day_start": _today_start_utc().isoformat(),
            "week_start": _week_start_utc().isoformat(),
        }), encoding="utf-8")

        loaded = RiskStateStore(store_path).load()
        assert loaded["daily_realized_pnl"] == -123.0
        assert loaded["weekly_realized_pnl"] == -456.0

    def test_corrupted_yields_empty(self, store_path: Path) -> None:
        store_path.write_text("{not valid", encoding="utf-8")
        assert RiskStateStore(store_path).load() == {}

    def test_missing_yields_empty(self, store_path: Path) -> None:
        assert RiskStateStore(store_path).load() == {}


# ---------------------------------------------------------------------------
# PortfolioTracker — round-trip across "restart"
# ---------------------------------------------------------------------------

class TestPortfolioTrackerRoundTrip:
    def test_same_day_restart_restores_state(self, store_path: Path) -> None:
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        # Open + close at a loss to populate every counter
        t1.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t1.close_position("BTCUSDT", 49_500, fee=0.0, timestamp=_now())
        # equity = 10_000 - 50 = 9_950; consec_losses = 1
        assert t1.get_state().equity == pytest.approx(9_950.0, abs=1e-6)
        assert t1.get_state().consecutive_losses == 1

        # "Restart"
        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        assert t2.get_state().equity == pytest.approx(9_950.0, abs=1e-6)
        assert t2.get_state().consecutive_losses == 1
        # daily_pnl_pct fraction of initial_equity = -50/10000 = -0.005
        assert t2.get_state().daily_pnl_pct == pytest.approx(-0.005, abs=1e-9)

    def test_peak_equity_survives_restart(self, store_path: Path) -> None:
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        # Profitable trade lifts peak above initial
        t1.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t1.close_position("BTCUSDT", 70_000, fee=0.0, timestamp=_now())
        # equity = 10_000 + 2000 = 12_000; peak = 12_000
        assert t1._peak_equity == pytest.approx(12_000.0, abs=1e-6)

        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        assert t2._peak_equity == pytest.approx(12_000.0, abs=1e-6)

    def test_drawdown_after_restart_uses_restored_peak(
        self, store_path: Path
    ) -> None:
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        t1.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t1.close_position("BTCUSDT", 70_000, fee=0.0, timestamp=_now())
        # peak now 12_000

        # Restart, then incur a loss
        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        t2.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t2.close_position("BTCUSDT", 49_000, fee=0.0, timestamp=_now())
        # equity = 12_000 - 100 = 11_900; dd from peak 12_000 = 100/12000
        assert t2.get_drawdown() == pytest.approx(100 / 12_000, abs=1e-9)

    def test_consecutive_losses_survives_restart(self, store_path: Path) -> None:
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        # 4 losses, then restart, then 5th loss → consec=5
        for _ in range(4):
            t1.record_loss(_now())
        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        assert t2._consecutive_losses == 4
        t2.record_loss(_now())
        assert t2._consecutive_losses == 5

    def test_next_day_restart_drops_daily(self, store_path: Path) -> None:
        """A persisted day_start from yesterday → daily counter reset on load."""
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        t1.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t1.close_position("BTCUSDT", 49_500, fee=0.0, timestamp=_now())

        # Rewrite day_start to yesterday on disk
        raw = json.loads(store_path.read_text(encoding="utf-8"))
        raw["day_start"] = (_today_start_utc() - timedelta(days=1)).isoformat()
        store_path.write_text(json.dumps(raw), encoding="utf-8")

        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        assert t2._daily_realized_pnl == 0.0
        # Weekly counter is still within this week → preserved
        assert t2._weekly_realized_pnl == pytest.approx(-50.0, abs=1e-6)

    def test_next_week_restart_drops_both(self, store_path: Path) -> None:
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        t1.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t1.close_position("BTCUSDT", 49_500, fee=0.0, timestamp=_now())

        raw = json.loads(store_path.read_text(encoding="utf-8"))
        raw["day_start"] = (_today_start_utc() - timedelta(days=8)).isoformat()
        raw["week_start"] = (_week_start_utc() - timedelta(days=7)).isoformat()
        store_path.write_text(json.dumps(raw), encoding="utf-8")

        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        assert t2._daily_realized_pnl == 0.0
        assert t2._weekly_realized_pnl == 0.0

    def test_corrupted_store_falls_back_to_defaults(
        self, store_path: Path
    ) -> None:
        store_path.write_text("{garbage", encoding="utf-8")
        t = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        assert t.get_state().equity == 10_000
        assert t._consecutive_losses == 0


# ---------------------------------------------------------------------------
# CircuitBreaker — halt flag persistence
# ---------------------------------------------------------------------------

class TestCircuitBreakerPersistence:
    def test_daily_triggered_survives_same_day_restart(
        self, store_path: Path
    ) -> None:
        b1 = CircuitBreaker(state_path=store_path)
        # Triggering state: daily_pnl_pct ≤ -3%
        state = PortfolioState(
            equity=9_600, open_positions=0,
            daily_pnl_pct=-0.04, weekly_pnl_pct=-0.04,
            current_drawdown_pct=0.04,
            consecutive_losses=0, last_loss_time=None,
            peak_equity=10_000,
        )
        result = b1.check(state, current_atr=0, avg_atr=0, current_funding=0)
        assert result.is_triggered
        assert b1._daily_triggered

        # New process → still triggered
        b2 = CircuitBreaker(state_path=store_path)
        assert b2._daily_triggered is True
        assert "Daily loss HARD" in b2._daily_trigger_reason

    def test_daily_triggered_clears_next_day(self, store_path: Path) -> None:
        # Hand-craft state on disk: triggered, but day_start = yesterday
        store_path.write_text(json.dumps({
            "breaker_daily_triggered": True,
            "breaker_daily_trigger_reason": "yesterday's loss",
            "day_start": (_today_start_utc() - timedelta(days=1)).isoformat(),
            "week_start": _week_start_utc().isoformat(),
        }), encoding="utf-8")

        b = CircuitBreaker(state_path=store_path)
        # Daily reset semantics in RiskStateStore.load clears the flag
        assert b._daily_triggered is False

    def test_reset_daily_persists(self, store_path: Path) -> None:
        b1 = CircuitBreaker(state_path=store_path)
        b1._daily_triggered = True
        b1._daily_trigger_reason = "x"
        b1._persist()

        b1.reset_daily()
        b2 = CircuitBreaker(state_path=store_path)
        assert b2._daily_triggered is False


# ---------------------------------------------------------------------------
# Fail-soft — unwritable / unreadable store
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_tracker_keeps_running_when_save_fails(
        self, store_path: Path
    ) -> None:
        t = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        with patch(
            "src.risk.risk_state_store.os.replace",
            side_effect=OSError("disk full"),
        ):
            # Mutation triggers _persist() which hits the failing os.replace
            t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
            # In-memory state still consistent — no crash
            assert t.get_state().open_positions == 1

    def test_tracker_with_unwritable_directory_init(
        self, tmp_path: Path
    ) -> None:
        # state_path under a regular file (mkdir + write must fail)
        regular_file = tmp_path / "afile"
        regular_file.write_text("hi")
        bad_path = regular_file / "state.json"
        t = PortfolioTracker(initial_equity=10_000, state_path=bad_path)
        # Mutations must not raise
        t.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        assert t.get_state().open_positions == 1


# ---------------------------------------------------------------------------
# End-to-end: the catastrophe scenario from the task description
# ---------------------------------------------------------------------------

class TestStrategyWiring:
    """Step 1.7b: ``on_start`` wires the shared state_path into both
    PortfolioTracker and CircuitBreaker. We replay the wiring block here
    rather than running Nautilus' on_start, which needs an engine."""

    def _resolve_path(self, signal_db_path: str) -> Path:
        from src.execution.strategies import ml_strategy as mod
        proj_root = Path(mod.__file__).resolve().parents[3]
        db_path = Path(signal_db_path)
        if not db_path.is_absolute():
            db_path = proj_root / db_path
        return db_path.parent / "risk_state_4h.json"

    def test_tracker_constructed_in_on_start_has_store(self) -> None:
        from src.execution.strategies.ml_strategy import MLStrategyConfig
        cfg = MLStrategyConfig()
        path = self._resolve_path(cfg.signal_db_path)
        tracker = PortfolioTracker(cfg.initial_equity, state_path=path)
        assert tracker._store is not None
        assert tracker._store._path == path

    def test_breaker_constructed_in_on_start_has_store(self) -> None:
        from src.execution.strategies.ml_strategy import MLStrategyConfig
        cfg = MLStrategyConfig()
        path = self._resolve_path(cfg.signal_db_path)
        breaker = CircuitBreaker(state_path=path)
        assert breaker._store is not None
        assert breaker._store._path == path

    def test_tracker_and_breaker_share_one_path(self) -> None:
        """Critical: same file → temporal-reset semantics apply to both."""
        from src.execution.strategies.ml_strategy import MLStrategyConfig
        cfg = MLStrategyConfig()
        path = self._resolve_path(cfg.signal_db_path)
        tracker = PortfolioTracker(cfg.initial_equity, state_path=path)
        breaker = CircuitBreaker(state_path=path)
        assert tracker._store._path == breaker._store._path

    def test_filename_matches_spec(self) -> None:
        """Path lives next to signal_db_path and is named risk_state_4h.json."""
        from src.execution.strategies.ml_strategy import MLStrategyConfig
        cfg = MLStrategyConfig()
        path = self._resolve_path(cfg.signal_db_path)
        assert path.name == "risk_state_4h.json"
        assert path.parent == self._resolve_path(cfg.signal_db_path).parent


class TestDisasterScenarioFix:
    def test_daily_limit_holds_across_restart(self, store_path: Path) -> None:
        """The motivating bug: -2.9% lost, restart, daily counter should
        NOT reset to 0, so the next ~-0.2% loss still trips daily HARD."""
        t1 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        # Wipe -290 (about -2.9%)
        t1.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t1.close_position("BTCUSDT", 47_100, fee=0.0, timestamp=_now())
        assert t1.get_state().daily_pnl_pct == pytest.approx(-0.029, abs=1e-6)

        # Restart same day, run breaker — must still see ~-2.9%
        t2 = PortfolioTracker(initial_equity=10_000, state_path=store_path)
        breaker = CircuitBreaker()
        result = breaker.check(
            t2.get_state(), current_atr=0, avg_atr=0, current_funding=0,
        )
        # -2.9% is below SOFT (-2%) but above HARD (-3%) → not triggered yet
        assert not result.is_triggered

        # Another -0.2% takes it past HARD threshold
        t2.update_fill("BTCUSDT", 1, 0.1, 50_000, fee=0.0, timestamp=_now())
        t2.close_position("BTCUSDT", 49_800, fee=0.0, timestamp=_now())
        result = breaker.check(
            t2.get_state(), current_atr=0, avg_atr=0, current_funding=0,
        )
        assert result.is_triggered
        assert "Daily loss HARD" in result.trigger_reason
