"""Tests for Steps H20 + H21 in src/execution/metrics.py.

H20: ``calculate_sharpe_ratio`` collapsed every equity point to one
end-of-day reading, hiding intraday volatility. For 15m, 96 bars/day
shrank to 1 → Sharpe inflated 3-5×. The fix adds an explicit
``bar_duration_minutes`` parameter that skips the daily collapse and
annualises by ``√(bars_per_year)``.

H21: ``calculate_all_metrics`` defaulted ``risk_free_rate=0.05`` while
``calculate_sharpe_ratio`` defaulted to 0.0 — the same equity curve
gave two different Sharpes depending on entry point. Both now default
to 0.0 (crypto convention).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.execution.metrics import (
    CRYPTO_ANNUALIZE,
    calculate_all_metrics,
    calculate_sharpe_ratio,
)


def _bar_curve(
    n_bars: int, bar_minutes: int, *, base: float = 10_000.0,
    pattern: str = "alternating",
) -> list[tuple[datetime, float]]:
    """Synth equity curve with non-trivial intraday volatility.

    ``alternating``: +0.5% / -0.3% bars (mean > 0, std > 0).
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    curve = [(start, base)]
    eq = base
    for i in range(1, n_bars):
        if pattern == "alternating":
            r = 0.005 if i % 2 == 0 else -0.003
        else:
            r = 0.001
        eq *= 1.0 + r
        curve.append((start + timedelta(minutes=bar_minutes * i), eq))
    return curve


# ---------------------------------------------------------------------------
# H20 — bar_duration_minutes flag
# ---------------------------------------------------------------------------


class TestBarFrequencySharpe:
    def test_4h_annualisation_uses_2190_bars_per_year(self):
        """4H: √(365 × 6) = √2190. Compare the bar-frequency Sharpe
        against the same calculation done by hand."""
        curve = _bar_curve(n_bars=200, bar_minutes=240)
        s = calculate_sharpe_ratio(curve, bar_duration_minutes=240)

        equities = [eq for _, eq in curve]
        rets = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
        ]
        n = len(rets)
        mean_r = sum(rets) / n
        var = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
        std = math.sqrt(var)
        bars_per_year = 365 * 24 * 60 // 240  # 2190
        expected = mean_r / std * math.sqrt(bars_per_year)
        assert s == pytest.approx(expected, rel=1e-9)

    def test_15m_annualisation_uses_35040_bars_per_year(self):
        curve = _bar_curve(n_bars=400, bar_minutes=15)
        s = calculate_sharpe_ratio(curve, bar_duration_minutes=15)

        equities = [eq for _, eq in curve]
        rets = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
        ]
        n = len(rets)
        mean_r = sum(rets) / n
        var = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
        std = math.sqrt(var)
        bars_per_year = 365 * 24 * 60 // 15  # 35040
        expected = mean_r / std * math.sqrt(bars_per_year)
        assert s == pytest.approx(expected, rel=1e-9)

    def test_15m_native_differs_from_daily_collapse(self):
        """The whole point of H20: native frequency must NOT equal the
        daily-collapse value when intraday volatility is real."""
        curve = _bar_curve(n_bars=400, bar_minutes=15)
        sharpe_collapsed = calculate_sharpe_ratio(curve)  # legacy path
        sharpe_native = calculate_sharpe_ratio(curve, bar_duration_minutes=15)
        # Different by a meaningful margin — daily collapse averages out
        # most of the +0.5%/-0.3% per-bar oscillation.
        assert abs(sharpe_collapsed - sharpe_native) > 0.5

    @pytest.mark.parametrize("bar_min,expected_bpy", [
        (240,  2190),    # 4H
        (60,   8760),    # 1H
        (15,   35040),   # 15m
        (5,    105120),  # 5m
        (1,    525600),  # 1m
    ])
    def test_bars_per_year_table(self, bar_min, expected_bpy):
        """Annualisation factor scales as 365 × 24 × 60 / bar_minutes."""
        assert 365 * 24 * 60 // bar_min == expected_bpy

    def test_default_mode_unchanged(self):
        """Without bar_duration_minutes, the legacy daily-collapse code
        path runs (one point per UTC day)."""
        # Build a 4H curve spanning many days.
        curve = _bar_curve(n_bars=200, bar_minutes=240)
        s_default = calculate_sharpe_ratio(curve)
        # Manual daily collapse — same as the legacy formula.
        daily = {}
        for dt, eq in curve:
            daily[dt.date()] = eq
        days = sorted(daily)
        eqs = [daily[d] for d in days]
        rets = [
            (eqs[i] - eqs[i - 1]) / eqs[i - 1]
            for i in range(1, len(eqs))
        ]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        expected = mean / math.sqrt(var) * math.sqrt(CRYPTO_ANNUALIZE)
        assert s_default == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# H21 — single risk_free_rate default
# ---------------------------------------------------------------------------


class TestRiskFreeRateConsistency:
    def test_calculate_all_metrics_default_is_zero(self):
        curve = _bar_curve(n_bars=120, bar_minutes=240)
        # No rf passed → both should agree because both default to 0.0.
        sharpe_direct = calculate_sharpe_ratio(curve)
        sharpe_via_all = calculate_all_metrics(
            curve, trades=[],
        ).sharpe_ratio
        assert sharpe_via_all == pytest.approx(sharpe_direct, rel=1e-9)

    def test_explicit_rf_propagates(self):
        curve = _bar_curve(n_bars=120, bar_minutes=240)
        rf = 0.04
        sharpe_direct = calculate_sharpe_ratio(curve, risk_free_rate=rf)
        sharpe_via_all = calculate_all_metrics(
            curve, trades=[], risk_free_rate=rf,
        ).sharpe_ratio
        assert sharpe_via_all == pytest.approx(sharpe_direct, rel=1e-9)

    def test_calculate_all_metrics_forwards_bar_duration(self):
        curve = _bar_curve(n_bars=200, bar_minutes=15)
        direct = calculate_sharpe_ratio(curve, bar_duration_minutes=15)
        via_all = calculate_all_metrics(
            curve, trades=[], bar_duration_minutes=15,
        ).sharpe_ratio
        assert via_all == pytest.approx(direct, rel=1e-9)


# ---------------------------------------------------------------------------
# Edge cases / guards
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_bar_duration_too_few_points(self):
        curve = [
            (datetime(2026, 1, 1, tzinfo=timezone.utc), 10_000.0),
        ]
        assert calculate_sharpe_ratio(curve, bar_duration_minutes=240) == 0.0

    def test_bar_duration_zero_falls_back_to_legacy(self):
        """Defensive: bar_duration_minutes=0 must not divide by zero —
        treated as None and uses the legacy daily-collapse path."""
        curve = _bar_curve(n_bars=200, bar_minutes=240)
        legacy = calculate_sharpe_ratio(curve)
        s0 = calculate_sharpe_ratio(curve, bar_duration_minutes=0)
        assert s0 == pytest.approx(legacy, rel=1e-9)

    def test_constant_equity_returns_zero(self):
        flat = [
            (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i),
             10_000.0)
            for i in range(50)
        ]
        assert calculate_sharpe_ratio(flat, bar_duration_minutes=15) == 0.0
