"""Tests for S0-2 — seed the drawdown baseline from the exchange balance.

Background
----------
``PortfolioTracker.__init__`` sets ``_peak_equity`` (and the percent
denominators ``_day_start_equity`` / ``_initial_equity``) from the
configured ``initial_equity``. ``sync_equity`` only ever raises the peak,
never lowers it. So a testnet / live account funded below the configured
capital starts every process inside a permanent drawdown: balance 5_000
against a configured 10_000 reads as a 50% drawdown on the very first
bar, and 15% of that is the kill switch.

PR-C closed this for ``dry_run`` only — there the exchange is never read
at all, so the whole path below is dead in that mode.

Contract established here
-------------------------
The first *successful* exchange read seeds the baseline, once per
process, and only when nothing more authoritative is available:

* seeded atomically: ``_peak_equity``, ``_day_start_equity``,
  ``_initial_equity`` — the drawdown denominator and the percent
  denominators must describe the same account;
* a peak restored from the risk-state file wins over the seed;
* a balance inside the zero band does NOT seed — a zero peak makes
  ``get_drawdown()`` return 0.0 forever and disarms the kill switch;
* ``dry_run`` is untouched (PR-C owns that mode);
* ``_cash`` stays with ``sync_equity``, which runs after the seed.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
)
from src.risk.portfolio_tracker import PortfolioTracker


CONFIGURED_EQUITY = 10_000.0
EXCHANGE_BALANCE = 5_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(dry_run: bool = False, trading_mode: str = "testnet") -> MLStrategyConfig:
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=CONFIGURED_EQUITY,
        warmup_bars=2,
        dry_run=dry_run,
        trading_mode=trading_mode,
    )


def _strategy(
    dry_run: bool = False,
    trading_mode: str = "testnet",
    state_path: Path | None = None,
) -> MLTradingStrategy:
    """Strategy with a REAL PortfolioTracker and no network anywhere.

    ``_record_equity`` is called directly, so ``on_bar`` (and with it the
    taker-volume fetch) never runs.
    """
    strat = MLTradingStrategy(config=_cfg(dry_run, trading_mode))
    strat._tracker = PortfolioTracker(
        initial_equity=CONFIGURED_EQUITY, state_path=state_path,
    )
    return strat


def _baseline(tracker: PortfolioTracker) -> tuple[float, float, float]:
    """The three values S0-2 seeds atomically."""
    return (
        tracker._peak_equity,
        tracker._day_start_equity,
        tracker._initial_equity,
    )


def _write_state(path: Path, **extra: object) -> None:
    """Write a risk-state file that RiskStateStore.load will not reset."""
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    week = today - timedelta(days=today.weekday())
    state: dict[str, object] = {
        "day_start": today.isoformat(),
        "week_start": week.isoformat(),
    }
    state.update(extra)
    path.write_text(json.dumps(state), encoding="utf-8")


# ---------------------------------------------------------------------------
# Act 1 — the mechanism of the bug, without the seeding entry point
# ---------------------------------------------------------------------------

class TestMechanism:
    """Green before AND after the fix.

    Pins peak-from-config + sync-only-upwards, so a green Act 2 cannot
    come from ``sync_equity`` quietly changing instead of a seed being
    added.
    """

    def test_peak_from_config_yields_50pct_drawdown_without_seed(self) -> None:
        pt = PortfolioTracker(initial_equity=CONFIGURED_EQUITY)
        pt.sync_equity(EXCHANGE_BALANCE)
        assert pt._peak_equity == pytest.approx(CONFIGURED_EQUITY)
        assert pt.get_drawdown() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Acts 2 and 3 — the same scenario through the real entry point
# ---------------------------------------------------------------------------

class TestSeedOnFirstAuthoritativeRead:
    def test_first_bar_balance_5000_yields_zero_drawdown(self, monkeypatch) -> None:
        strat = _strategy()
        monkeypatch.setattr(
            strat, "_read_nautilus_equity", lambda: EXCHANGE_BALANCE,
        )

        strat._record_equity(ts_ns=1)

        assert strat._tracker.get_drawdown() == pytest.approx(0.0)

    def test_seed_sets_peak_day_start_and_initial_to_5000(self, monkeypatch) -> None:
        strat = _strategy()
        monkeypatch.setattr(
            strat, "_read_nautilus_equity", lambda: EXCHANGE_BALANCE,
        )

        strat._record_equity(ts_ns=1)

        assert _baseline(strat._tracker) == (
            pytest.approx(EXCHANGE_BALANCE),
            pytest.approx(EXCHANGE_BALANCE),
            pytest.approx(EXCHANGE_BALANCE),
        )


# ---------------------------------------------------------------------------
# R2 — a None read defers the seed, it does not lose it
# ---------------------------------------------------------------------------

class TestDeferredSeed:
    def test_none_read_leaves_seed_pending(self, monkeypatch) -> None:
        reads: list[float | None] = [None, 7_000.0]
        monkeypatch.setattr(
            (strat := _strategy()), "_read_nautilus_equity", lambda: reads.pop(0),
        )

        strat._record_equity(ts_ns=1)
        assert _baseline(strat._tracker) == (
            pytest.approx(CONFIGURED_EQUITY),
            pytest.approx(CONFIGURED_EQUITY),
            pytest.approx(CONFIGURED_EQUITY),
        )

        strat._record_equity(ts_ns=2)
        assert _baseline(strat._tracker) == (
            pytest.approx(7_000.0),
            pytest.approx(7_000.0),
            pytest.approx(7_000.0),
        )


# ---------------------------------------------------------------------------
# R5 — the zero band must never become the peak (control: green both sides)
# ---------------------------------------------------------------------------

class TestZeroBandDoesNotSeed:
    @pytest.mark.parametrize("balance", [0.0, 1e-12])
    def test_zero_balance_does_not_seed(self, balance, monkeypatch) -> None:
        strat = _strategy()
        monkeypatch.setattr(strat, "_read_nautilus_equity", lambda: balance)

        strat._record_equity(ts_ns=1)

        # Peak stays at the configured value, so the kill switch still sees
        # a ~100% drawdown instead of a disarmed 0.0.
        assert strat._tracker._peak_equity == pytest.approx(CONFIGURED_EQUITY)
        assert strat._tracker.get_drawdown() == pytest.approx(1.0)
        # PR-C's operator notice must survive untouched.
        assert strat._zero_balance_logged is True


# ---------------------------------------------------------------------------
# R3 — a restored peak outranks the seed (control: green both sides)
# ---------------------------------------------------------------------------

class TestRestoredStateWins:
    def test_restored_peak_wins_over_seed(self, tmp_path, monkeypatch) -> None:
        state_file = tmp_path / "risk_state_4h.json"
        _write_state(state_file, peak_equity=12_000.0)

        strat = _strategy(state_path=state_file)
        assert strat._tracker._peak_equity == pytest.approx(12_000.0)

        monkeypatch.setattr(
            strat, "_read_nautilus_equity", lambda: EXCHANGE_BALANCE,
        )
        strat._record_equity(ts_ns=1)

        assert strat._tracker._peak_equity == pytest.approx(12_000.0)


# ---------------------------------------------------------------------------
# R4 — the seed happens once per process
# ---------------------------------------------------------------------------

class TestSeedIsOneShot:
    def test_seed_runs_once_across_bars(self, monkeypatch) -> None:
        reads = [EXCHANGE_BALANCE, 4_000.0]
        monkeypatch.setattr(
            (strat := _strategy()), "_read_nautilus_equity", lambda: reads.pop(0),
        )

        strat._record_equity(ts_ns=1)
        strat._record_equity(ts_ns=2)

        # A second seed would move the peak to 4_000 and report no drawdown.
        assert strat._tracker._peak_equity == pytest.approx(EXCHANGE_BALANCE)
        assert strat._tracker.get_drawdown() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# R7 — dry-run is PR-C's territory (control: green both sides)
# ---------------------------------------------------------------------------

class TestDryRunUntouched:
    def test_dry_run_does_not_seed(self, monkeypatch) -> None:
        def _boom() -> float:
            raise RuntimeError("exchange must not be consulted in dry-run")

        strat = _strategy(dry_run=True, trading_mode="paper")
        monkeypatch.setattr(strat, "_read_nautilus_equity", _boom)

        strat._record_equity(ts_ns=1)  # must not raise

        assert _baseline(strat._tracker) == (
            pytest.approx(CONFIGURED_EQUITY),
            pytest.approx(CONFIGURED_EQUITY),
            pytest.approx(CONFIGURED_EQUITY),
        )


# ---------------------------------------------------------------------------
# Boundary with sync_equity — cash stays with sync, and sync still runs
# ---------------------------------------------------------------------------

class TestSeedAndSyncBoundary:
    def test_seed_does_not_touch_cash(self, monkeypatch) -> None:
        strat = _strategy()
        monkeypatch.setattr(
            strat, "_read_nautilus_equity", lambda: EXCHANGE_BALANCE,
        )
        synced: list[float] = []
        monkeypatch.setattr(
            strat._tracker, "sync_equity", lambda v: synced.append(float(v)),
        )

        strat._record_equity(ts_ns=1)

        assert strat._tracker._cash == pytest.approx(CONFIGURED_EQUITY)
        assert synced == [EXCHANGE_BALANCE]
        assert _baseline(strat._tracker) == (
            pytest.approx(EXCHANGE_BALANCE),
            pytest.approx(EXCHANGE_BALANCE),
            pytest.approx(EXCHANGE_BALANCE),
        )

    def test_sync_still_runs_after_seed(self, monkeypatch) -> None:
        strat = _strategy()
        monkeypatch.setattr(
            strat, "_read_nautilus_equity", lambda: EXCHANGE_BALANCE,
        )

        strat._record_equity(ts_ns=1)

        st = strat._tracker.get_state()
        assert st.equity == pytest.approx(EXCHANGE_BALANCE)
        assert st.peak_equity == pytest.approx(EXCHANGE_BALANCE)


# ---------------------------------------------------------------------------
# O1 — the seeded baseline survives a restart through the state file
# ---------------------------------------------------------------------------

class TestSeedSurvivesRestart:
    def test_seeded_initial_equity_survives_restart(
        self, tmp_path, monkeypatch,
    ) -> None:
        state_file = tmp_path / "risk_state_4h.json"
        strat = _strategy(state_path=state_file)
        monkeypatch.setattr(
            strat, "_read_nautilus_equity", lambda: EXCHANGE_BALANCE,
        )

        strat._record_equity(ts_ns=1)

        restarted = PortfolioTracker(
            initial_equity=CONFIGURED_EQUITY, state_path=state_file,
        )
        assert _baseline(restarted) == (
            pytest.approx(EXCHANGE_BALANCE),
            pytest.approx(EXCHANGE_BALANCE),
            pytest.approx(EXCHANGE_BALANCE),
        )
