"""DSR formula correctness regression tests (Phase 3 Step 3.1).

Two bugs in ``calculate_dsr``:
* ``n_obs`` defaulted to ``len(sharpe_ratios)`` (5-15 folds) instead of
  the actual number of return observations T (hundreds-thousands). The
  inflated SE collapsed DSR to ~0.5 regardless of real skill — the
  0.95 go-live gate could not be reached even for an honestly good model.
* The kurtosis correction was ``(γ4 - 1)/4`` instead of the
  Mertens-2002 / Bailey-López-de-Prado ``(γ4 - 3)/4``. For a normal
  distribution (γ4 = 3) the term must vanish; pre-fix it added a
  spurious ``0.5·SR²`` of variance.

These tests pin both fixes via direct closed-form comparisons so a
regression cannot hide behind aggregate behaviour.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sp_stats

from src.models.statistical_tests import calculate_dsr


# ---------------------------------------------------------------------------
# Closed-form SE check for normal returns (γ3=0, γ4=3)
# ---------------------------------------------------------------------------

class TestSENormalReturns:
    """For normal returns with skew=0, kurt=3, the SE simplifies to
    ``1/sqrt(T-1)`` exactly. We back out SE from the DSR formula and
    compare to this analytic target."""

    @pytest.mark.parametrize("T", [100, 1_000, 10_000])
    def test_se_equals_one_over_sqrt_T_minus_1(self, T: int) -> None:
        # Single SR (so best_sr = that one) at SR = 1.0; skew=0, kurt=3.
        best_sr = 1.0
        sharpe_ratios = [best_sr, 0.9]  # two values so the guard passes
        n_trials = 2  # E[SR_max] = sqrt(2 ln 2) - ... ≈ 0.46
        dsr = calculate_dsr(
            sharpe_ratios,
            n_trials=n_trials,
            skewness=0.0,
            kurtosis=3.0,
            n_obs=T,
        )
        # Back out SE from DSR = Φ((best_sr - E[SR_max]) / SE)
        log_n = math.log(n_trials)
        sqrt_2logn = math.sqrt(2 * log_n)
        e_max = sqrt_2logn - (math.log(log_n) + math.log(4 * math.pi)) / (
            2 * sqrt_2logn
        )
        expected_se = 1.0 / math.sqrt(T - 1)
        expected_z = (best_sr - e_max) / expected_se
        expected_dsr = float(sp_stats.norm.cdf(expected_z))
        assert dsr == pytest.approx(expected_dsr, rel=1e-9)


# ---------------------------------------------------------------------------
# Kurtosis term vanishes at γ4 = 3
# ---------------------------------------------------------------------------

class TestKurtosisTermVanishesAtNormal:
    def test_kurt_3_matches_kurt_term_zero(self) -> None:
        """DSR(γ4=3) must equal DSR computed with the kurtosis term
        manually zeroed — proof the (γ4-3)/4 term vanishes at the
        normal distribution."""
        best_sr = 2.0
        n_trials = 10
        T = 500

        dsr_kurt_3 = calculate_dsr(
            [best_sr, 1.5], n_trials=n_trials,
            skewness=0.0, kurtosis=3.0, n_obs=T,
        )

        # Synthesise the "kurtosis term = 0" case by another formula path:
        # pass kurtosis=3 (the only γ4 making the term exactly zero).
        # Compare against analytic SE = 1/sqrt(T-1).
        log_n = math.log(n_trials)
        sqrt_2logn = math.sqrt(2 * log_n)
        e_max = sqrt_2logn - (math.log(log_n) + math.log(4 * math.pi)) / (
            2 * sqrt_2logn
        )
        analytic_se = 1.0 / math.sqrt(T - 1)
        analytic_dsr = float(sp_stats.norm.cdf((best_sr - e_max) / analytic_se))
        assert dsr_kurt_3 == pytest.approx(analytic_dsr, rel=1e-9)

    def test_kurt_above_3_increases_se_lowers_dsr(self) -> None:
        """Excess kurtosis (γ4 > 3) → bigger SE → lower DSR. Verifies the
        sign of the correction is right (positive contribution)."""
        best_sr = 2.0
        T = 500
        dsr_normal = calculate_dsr(
            [best_sr, 1.5], n_trials=5,
            skewness=0.0, kurtosis=3.0, n_obs=T,
        )
        dsr_fat_tail = calculate_dsr(
            [best_sr, 1.5], n_trials=5,
            skewness=0.0, kurtosis=10.0, n_obs=T,
        )
        assert dsr_fat_tail < dsr_normal


# ---------------------------------------------------------------------------
# Monotonicity in T (the bug-1 fix)
# ---------------------------------------------------------------------------

class TestMonotonicityInT:
    def test_larger_T_raises_DSR_for_positive_skill(self) -> None:
        """For a fixed observed SR above the noise floor, bigger T →
        smaller SE → larger z → higher DSR. Pre-fix DSR was nearly flat
        in T because the formula used the fold count instead of T."""
        best_sr = 1.5
        dsr_small = calculate_dsr(
            [best_sr, 1.2], n_trials=5,
            skewness=0.0, kurtosis=3.0, n_obs=50,
        )
        dsr_big = calculate_dsr(
            [best_sr, 1.2], n_trials=5,
            skewness=0.0, kurtosis=3.0, n_obs=5_000,
        )
        assert dsr_big > dsr_small

    def test_T_large_pushes_dsr_above_0_95_for_strong_skill(self) -> None:
        """The whole motivation: a genuinely strong SR with enough data
        must clear the 0.95 go-live gate. Pre-fix it could not."""
        best_sr = 2.5
        dsr = calculate_dsr(
            [best_sr, 2.2, 2.4], n_trials=5,
            skewness=0.0, kurtosis=3.0, n_obs=10_000,
        )
        assert dsr >= 0.95


# ---------------------------------------------------------------------------
# Monotonicity in n_trials (multiple-testing deflation)
# ---------------------------------------------------------------------------

class TestMonotonicityInTrials:
    def test_more_trials_lower_DSR(self) -> None:
        """E[SR_max] grows with n_trials → lower z → lower DSR. Holds
        independently of the T / kurtosis fix."""
        best_sr = 2.0
        dsr_few = calculate_dsr(
            [best_sr, 1.5], n_trials=3,
            skewness=0.0, kurtosis=3.0, n_obs=1_000,
        )
        dsr_many = calculate_dsr(
            [best_sr, 1.5], n_trials=500,
            skewness=0.0, kurtosis=3.0, n_obs=1_000,
        )
        assert dsr_many < dsr_few


# ---------------------------------------------------------------------------
# Backward-compat: n_obs=None falls back to len(sharpe_ratios)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_default_n_obs_uses_fold_count(self) -> None:
        """Without n_obs, the function uses len(sharpe_ratios) as a
        legacy approximation. Result must equal an explicit call with
        n_obs=len(sharpe_ratios)."""
        srs = [1.8, 1.5, 1.3, 1.6, 1.4]
        a = calculate_dsr(srs, n_trials=10)
        b = calculate_dsr(srs, n_trials=10, n_obs=len(srs))
        assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# Bug-1 regression: pre-fix could not pass the 0.95 gate; post-fix can
# ---------------------------------------------------------------------------

class TestBugRegressionPrePostFix:
    def test_pre_fix_formula_would_under_report(self) -> None:
        """Reproduce the pre-fix formula by hand and confirm that for a
        strong-skill / large-T scenario the new (correct) formula gives
        a meaningfully higher DSR. This is the central regression."""
        best_sr = 2.5
        T = 10_000
        n_trials = 5
        skew = 0.0
        kurt = 3.0

        # New (correct) formula via the fixed implementation.
        dsr_new = calculate_dsr(
            [best_sr, 2.2, 2.4], n_trials=n_trials,
            skewness=skew, kurtosis=kurt, n_obs=T,
        )

        # Pre-fix formula reconstructed by hand: kurtosis term (k-1)/4
        # and denominator = number of folds = 3 (NOT T).
        sharpe_ratios = [best_sr, 2.2, 2.4]
        pre_fix_n_obs = len(sharpe_ratios)
        pre_fix_se = math.sqrt(
            (1 - skew * best_sr + ((kurt - 1) / 4) * best_sr ** 2)
            / pre_fix_n_obs
        )
        log_n = math.log(n_trials)
        sqrt_2logn = math.sqrt(2 * log_n)
        e_max = sqrt_2logn - (math.log(log_n) + math.log(4 * math.pi)) / (
            2 * sqrt_2logn
        )
        pre_fix_z = (best_sr - e_max) / pre_fix_se
        pre_fix_dsr = float(sp_stats.norm.cdf(pre_fix_z))

        # The fix must measurably move DSR upward for a genuinely strong
        # model and large T — the whole reason the go-live gate was
        # previously unreachable.
        assert dsr_new > pre_fix_dsr + 0.05
        assert dsr_new >= 0.95
        assert pre_fix_dsr < 0.95  # demonstrates the original bug


# ---------------------------------------------------------------------------
# Caller integration: run_all_tests forwards real T to calculate_dsr
# ---------------------------------------------------------------------------

class TestRunAllTestsIntegration:
    def test_real_returns_path_uses_total_T(self) -> None:
        """Verify ``run_all_tests`` totals T across folds and passes it
        in as ``n_obs`` (so the SE formula sees the right denominator)."""
        from unittest.mock import patch

        from src.models.statistical_tests import run_all_tests

        # Two folds of 250 daily returns each → total T = 500.
        rng = np.random.default_rng(42)
        per_fold = [
            rng.normal(loc=0.001, scale=0.01, size=250),
            rng.normal(loc=0.001, scale=0.01, size=250),
        ]

        # Stub the dependent pieces so we only exercise the DSR path.
        from src.models.lgbm_trainer import EvaluationResult
        from src.models.ml_validator import (
            WalkForwardMLResult,
            WindowMLResult,
        )
        from datetime import datetime

        cv_results = [
            EvaluationResult(
                regime="all", accuracy=0.55, precision=0.55,
                recall=0.55, f1=0.55,
                win_rate=55.0, profit_factor=1.5,
                signal_rate=0.1, avg_confidence=0.6,
                per_symbol={},
            )
            for _ in range(2)
        ]
        wf_result = WalkForwardMLResult(
            regime="all",
            windows=[
                WindowMLResult(
                    train_start=datetime(2024, 1, 1),
                    train_end=datetime(2024, 6, 1),
                    test_start=datetime(2024, 6, 1),
                    test_end=datetime(2024, 9, 1),
                    win_rate=55.0,
                    profit_factor=1.5,
                    signal_rate=0.1,
                    n_signals=50,
                    n_test_bars=300,
                )
            ],
        )

        captured: dict = {}

        def fake_dsr(*args, **kwargs):
            captured["n_obs"] = kwargs.get("n_obs")
            return 0.5

        with patch(
            "src.models.statistical_tests.calculate_dsr",
            side_effect=fake_dsr,
        ):
            run_all_tests(
                cv_results=cv_results,
                wf_result=wf_result,
                n_experiments=10,
                per_fold_daily_returns=per_fold,
            )

        assert captured["n_obs"] == 500


# ---------------------------------------------------------------------------
# Numerical edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_T_less_than_2_returns_zero(self) -> None:
        assert calculate_dsr([1.0, 0.5], n_trials=2, n_obs=1) == 0.0

    def test_negative_skew_raises_se(self) -> None:
        """Negative skew (left-tail) penalises a positive SR — the
        ``-γ3·SR`` term is positive so SE grows, DSR shrinks. Pick
        modest SR and modest T so DSR is well within (0, 1) and the
        difference between configurations is visible."""
        # Setup: best_sr (0.9) is above E[SR_max] for n_trials=3 (~0.6)
        # so z > 0 and "larger SE → smaller DSR" sign convention applies.
        # Modest T (=20) keeps DSR away from the saturated 1.0 plateau
        # so the difference between configurations is observable.
        sr = 0.9
        T = 20
        dsr_neg_skew = calculate_dsr(
            [sr, 0.7], n_trials=3,
            skewness=-1.5, kurtosis=3.0, n_obs=T,
        )
        dsr_zero_skew = calculate_dsr(
            [sr, 0.7], n_trials=3,
            skewness=0.0, kurtosis=3.0, n_obs=T,
        )
        assert dsr_neg_skew < dsr_zero_skew
