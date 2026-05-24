"""Tests for Step H7 — multi-criteria walk-forward gate.

Old gate: `profitable_windows_pct >= 60`. Vulnerable to one catastrophic
window hidden among many tiny positives. New gate adds avg/worst Sharpe
and aggregate-return checks.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.execution.metrics import MetricsResult
from src.execution.walk_forward import (
    DEFAULT_GATE,
    WalkForwardGateConfig,
    WalkForwardResult,
    WindowResult,
)


_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _window(
    *, sharpe: float = 0.5, return_pct: float = 1.0, profitable: bool | None = None,
) -> WindowResult:
    if profitable is None:
        profitable = return_pct > 0
    m = MetricsResult(
        sharpe_ratio=sharpe,
        calmar_ratio=0.0,
        max_drawdown_pct=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        total_return_pct=return_pct,
        annualized_return_pct=0.0,
        total_trades=10,
    )
    return WindowResult(_DT, _DT, _DT, _DT, m, profitable)


# ---------------------------------------------------------------------------
# Catastrophe scenario from the task
# ---------------------------------------------------------------------------


class TestCatastropheScenario:
    """5 windows at +0.1%, 1 window at -30%, ~5/6 profitable.

    Old gate: 83.3% profitable → PASS.
    New gate: aggregate return ≈ -29.5% and worst Sharpe deeply negative
    → must FAIL.
    """

    def _result(self) -> WalkForwardResult:
        windows = (
            [_window(sharpe=0.4, return_pct=0.1) for _ in range(5)]
            + [_window(sharpe=-3.0, return_pct=-30.0)]
        )
        return WalkForwardResult(windows=windows)

    def test_old_pct_gate_would_pass(self):
        r = self._result()
        assert r.profitable_windows_pct == pytest.approx(83.333, rel=1e-3)

    def test_new_gate_blocks_catastrophe(self):
        r = self._result()
        assert r.passes_walk_forward_test is False
        passed, reasons = r.passes_gate()
        assert passed is False
        # Both Sharpe-based and aggregate-based criteria fire.
        reasons_joined = " | ".join(reasons)
        assert "worst_sharpe" in reasons_joined
        assert "aggregate_return_pct" in reasons_joined

    def test_aggregate_return_pct_value(self):
        r = self._result()
        assert r.aggregate_return_pct == pytest.approx(0.5 - 30.0)

    def test_worst_sharpe_value(self):
        r = self._result()
        assert r.worst_sharpe == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Healthy strategy passes
# ---------------------------------------------------------------------------


class TestHealthyStrategy:
    def test_all_criteria_met(self):
        windows = [
            _window(sharpe=0.8, return_pct=2.5) for _ in range(5)
        ] + [_window(sharpe=0.2, return_pct=0.4)]
        r = WalkForwardResult(windows=windows)
        passed, reasons = r.passes_gate()
        assert passed is True, reasons
        assert reasons == []
        assert r.passes_walk_forward_test is True


# ---------------------------------------------------------------------------
# Individual criteria
# ---------------------------------------------------------------------------


class TestEachCriterion:
    def test_low_pct_fails(self):
        # 2/5 profitable = 40%
        windows = [_window(sharpe=2.0, return_pct=5.0)] * 2 + [
            _window(sharpe=2.0, return_pct=-1.0, profitable=False),
        ] * 3
        r = WalkForwardResult(windows=windows)
        passed, reasons = r.passes_gate()
        assert passed is False
        assert any("profitable_windows_pct" in r for r in reasons)

    def test_low_avg_sharpe_fails(self):
        """All-profitable windows but their Sharpe averages negative."""
        windows = [_window(sharpe=-0.5, return_pct=0.1) for _ in range(5)]
        r = WalkForwardResult(windows=windows)
        passed, reasons = r.passes_gate(
            WalkForwardGateConfig(min_avg_sharpe=0.0),
        )
        assert passed is False
        assert any("avg_sharpe" in r for r in reasons)

    def test_low_worst_sharpe_fails(self):
        """4 good windows + one with Sharpe = -1.5 (below -1 threshold)."""
        windows = [_window(sharpe=1.0, return_pct=2.0) for _ in range(4)] + [
            _window(sharpe=-1.5, return_pct=0.05),
        ]
        r = WalkForwardResult(windows=windows)
        passed, reasons = r.passes_gate()
        assert passed is False
        assert any("worst_sharpe" in r for r in reasons)

    def test_negative_aggregate_fails(self):
        """Mostly tiny wins, one moderate loss outweighs them."""
        windows = [_window(sharpe=0.1, return_pct=0.05) for _ in range(5)] + [
            _window(sharpe=-0.4, return_pct=-1.0),
        ]
        r = WalkForwardResult(windows=windows)
        assert r.aggregate_return_pct < 0
        passed, reasons = r.passes_gate()
        assert passed is False
        assert any("aggregate_return_pct" in r for r in reasons)


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_stricter_avg_sharpe_threshold(self):
        windows = [_window(sharpe=0.3, return_pct=1.0) for _ in range(5)]
        r = WalkForwardResult(windows=windows)
        assert r.passes_gate(WalkForwardGateConfig(min_avg_sharpe=0.0))[0] is True
        assert r.passes_gate(WalkForwardGateConfig(min_avg_sharpe=1.0))[0] is False

    def test_relaxed_worst_sharpe(self):
        windows = [_window(sharpe=1.0, return_pct=2.0) for _ in range(4)] + [
            _window(sharpe=-1.5, return_pct=0.1),
        ]
        r = WalkForwardResult(windows=windows)
        assert r.passes_gate()[0] is False  # default -1.0
        assert r.passes_gate(
            WalkForwardGateConfig(min_worst_sharpe=-2.0),
        )[0] is True


# ---------------------------------------------------------------------------
# Backward compatibility — existing dummy-metrics fixtures still pass
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_zero_metrics_60pct_profitable_still_passes(self):
        """The test_walk_forward.py 3-of-5 fixture uses all-zero MetricsResult
        and asserts passes_walk_forward_test is True. New gate must keep
        that property (>= semantics)."""
        windows = (
            [_window(sharpe=0.0, return_pct=0.0, profitable=True)] * 3
            + [_window(sharpe=0.0, return_pct=0.0, profitable=False)] * 2
        )
        r = WalkForwardResult(windows=windows)
        assert r.passes_walk_forward_test is True

    def test_under_60_pct_still_fails(self):
        windows = (
            [_window(sharpe=2.0, return_pct=5.0, profitable=True)] * 2
            + [_window(sharpe=2.0, return_pct=5.0, profitable=False)] * 3
        )
        r = WalkForwardResult(windows=windows)
        assert r.passes_walk_forward_test is False

    def test_empty_result_does_not_pass(self):
        r = WalkForwardResult(windows=[])
        assert r.passes_walk_forward_test is False

    def test_default_gate_thresholds(self):
        assert DEFAULT_GATE.min_profitable_pct == 60.0
        assert DEFAULT_GATE.min_avg_sharpe == 0.0
        assert DEFAULT_GATE.min_worst_sharpe == -1.0
        assert DEFAULT_GATE.min_aggregate_return == 0.0


# ---------------------------------------------------------------------------
# Aggregate helper smoke
# ---------------------------------------------------------------------------


class TestAggregateProperties:
    def test_aggregate_return_pct_empty(self):
        assert WalkForwardResult(windows=[]).aggregate_return_pct == 0.0

    def test_worst_sharpe_empty(self):
        assert WalkForwardResult(windows=[]).worst_sharpe == 0.0
