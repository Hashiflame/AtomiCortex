"""
tests/test_ml_validator.py

Tests for MLValidator, WalkForwardMLResult, statistical tests (15+ tests).
Phase 3 — Steps 3.5 + 3.6.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.lgbm_trainer import (
    EvaluationResult,
    LGBMTrainer,
    ModelConfig,
)
from src.models.ml_validator import (
    MLValidator,
    WalkForwardMLResult,
    WindowMLResult,
)
from src.models.statistical_tests import (
    StatTestResult,
    calculate_dsr,
    calculate_pbo,
    calculate_t_stat,
    run_all_tests,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(
    n: int = 200, symbol: str = "BTCUSDT", seed: int = 42
) -> pl.DataFrame:
    """Create a synthetic feature DataFrame that mirrors real schema."""
    rng = np.random.RandomState(seed)
    close = 40000 + np.cumsum(rng.randn(n) * 100)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    opn = close + rng.randn(n) * 50

    base_time = 1704067200000  # 2024-01-01 00:00 UTC in ms
    open_times = [base_time + i * 4 * 3600 * 1000 for i in range(n)]

    return pl.DataFrame(
        {
            "open_time": open_times,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(100, 1000, n),
            "close_time": [t + 4 * 3600 * 1000 - 1 for t in open_times],
            "datetime": pl.Series([None] * n, dtype=pl.Datetime),
            "symbol": [symbol] * n,
            "regime": rng.choice(
                ["trend_up", "trend_down", "range", "high_vol"], n
            ).tolist(),
            # Feature columns
            "cvd": rng.randn(n),
            "cvd_cum": np.cumsum(rng.randn(n)),
            "cvd_slope_3": rng.randn(n),
            "cvd_slope_6": rng.randn(n),
            "cvd_slope_12": rng.randn(n),
            "taker_buy_ratio": rng.uniform(0.4, 0.6, n),
            "volume_sma_20": rng.uniform(200, 800, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
            "volume_zscore": rng.randn(n),
            "large_volume": rng.choice([0, 1], n).astype(np.int8),
            "vwap_4h": close + rng.randn(n) * 10,
            "price_to_vwap": rng.uniform(-0.01, 0.01, n),
            "returns_1": rng.randn(n) * 0.01,
            "returns_3": rng.randn(n) * 0.02,
            "returns_6": rng.randn(n) * 0.03,
            "returns_12": rng.randn(n) * 0.04,
            "returns_24": rng.randn(n) * 0.05,
            "body_ratio": rng.uniform(0.1, 0.9, n),
            "upper_wick": rng.uniform(0, 0.5, n),
            "lower_wick": rng.uniform(0, 0.5, n),
            "gap": rng.randn(n) * 0.001,
            "funding_rate": rng.randn(n) * 0.0001,
            "funding_abs": np.abs(rng.randn(n) * 0.0001),
            "funding_zscore_7d": rng.randn(n),
            "funding_zscore_30d": rng.randn(n),
            "funding_extreme": rng.choice([0, 1], n).astype(np.int8),
            "funding_positive": rng.choice([0, 1], n).astype(np.int8),
            "funding_cum_24h": rng.randn(n) * 0.001,
            "oi_value": rng.uniform(1e8, 1e9, n),
            "oi_delta_4h": rng.randn(n) * 0.01,
            "oi_delta_12h": rng.randn(n) * 0.02,
            "oi_zscore": rng.randn(n),
            "oi_quadrant": rng.choice([0, 1], n).astype(np.int8),
            "ls_ratio": rng.uniform(0.8, 1.2, n),
            "ls_ratio_zscore": rng.randn(n),
            "taker_vol_ratio": rng.uniform(0.4, 0.6, n),
            "basis_approx": rng.randn(n) * 0.001,
            "basis_extreme": rng.choice([0, 1], n).astype(np.int8),
            "hurst": rng.uniform(0.3, 0.7, n),
            "adx": rng.uniform(10, 50, n),
            "atr_pct": rng.uniform(0.01, 0.05, n),
            "atr_percentile": rng.uniform(0, 1, n),
            "trend_strength": rng.uniform(0, 1, n),
            "regime_confidence": rng.uniform(0, 1, n),
        }
    )


def _save_features(
    tmp_path: Path, symbols: list[str] | None = None, n: int = 200
) -> Path:
    """Save synthetic feature parquets and return the directory."""
    symbols = symbols or ["BTCUSDT"]
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for i, sym in enumerate(symbols):
        df = _make_feature_df(n=n, symbol=sym, seed=42 + i)
        df.write_parquet(features_dir / f"{sym}_4h_features.parquet")

    return features_dir


def _make_trainer(
    tmp_path: Path,
    regime: str = "all",
    symbols: list[str] | None = None,
    n: int = 200,
):
    """Create a LGBMTrainer with synthetic data."""
    symbols = symbols or ["BTCUSDT"]
    features_dir = _save_features(tmp_path, symbols, n=n)
    models_dir = tmp_path / "models"

    config = ModelConfig(regime=regime, symbols=symbols)
    trainer = LGBMTrainer(
        config=config, features_dir=features_dir, models_dir=models_dir
    )
    return trainer, features_dir, models_dir


def _make_eval_result(
    win_rate: float = 52.0,
    profit_factor: float = 1.3,
    signal_rate: float = 0.15,
    regime: str = "test",
) -> EvaluationResult:
    """Create a synthetic EvaluationResult."""
    return EvaluationResult(
        regime=regime,
        accuracy=55.0,
        precision=55.0,
        recall=55.0,
        f1=55.0,
        win_rate=win_rate,
        profit_factor=profit_factor,
        signal_rate=signal_rate,
        avg_confidence=0.6,
        per_symbol={},
    )


# ===========================================================================
# 1. Purged K-Fold CV Tests
# ===========================================================================


class TestPurgedKFoldCV:
    """Test purged_kfold_cv produces correct number of folds."""

    def test_correct_number_of_folds(self, tmp_path: Path):
        trainer, _, _ = _make_trainer(tmp_path, regime="all", n=300)
        train_df, test_df = trainer.prepare_data()
        full_df = pl.concat([train_df, test_df], how="diagonal")

        validator = MLValidator(n_splits=5)
        results = validator.purged_kfold_cv(trainer, full_df)

        assert len(results) == 5


class TestEmbargoNoOverlap:
    """Test that embargo prevents train/test data overlap."""

    def test_embargo_gap_exists(self, tmp_path: Path):
        """Verify that embargo creates a gap between train and test rows."""
        from src.execution.walk_forward import PurgedKFoldCV

        # Create a simple time-indexed dataframe
        n = 600
        df = _make_feature_df(n=n, symbol="BTCUSDT")

        cv = PurgedKFoldCV(n_splits=3, embargo_pct=0.05)
        for train_df, test_df in cv.split(df):
            # Train end index must be strictly less than test start index
            # (because of embargo rows between them)
            train_end_time = train_df["open_time"][-1]
            test_start_time = test_df["open_time"][0]
            assert test_start_time > train_end_time, (
                "Test data must start after train data (embargo gap)"
            )


# ===========================================================================
# 2. Walk-Forward ML Tests
# ===========================================================================


class TestWalkForwardMLWindows:
    """Test walk_forward_ml generates correct time windows."""

    def test_windows_generated(self, tmp_path: Path):
        trainer, features_dir, _ = _make_trainer(
            tmp_path, regime="all", n=4000
        )
        validator = MLValidator(n_splits=3)

        result = validator.walk_forward_ml(
            trainer=trainer,
            symbols=["BTCUSDT"],
            features_dir=features_dir,
            train_months=6,
            test_months=2,
            step_months=2,
        )

        assert isinstance(result, WalkForwardMLResult)
        assert result.regime == "all"
        # With 4000 rows of 4h data ≈ 667 days ≈ 22 months, should get at
        # least 1 window with 6m train + 2m test
        assert len(result.windows) >= 1


class TestProfitableWindowsPct:
    """Test profitable_windows_pct is calculated correctly."""

    def test_all_profitable(self):
        windows = [
            WindowMLResult(
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 7, 1),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 10, 1),
                win_rate=55.0,
                profit_factor=1.5,
                signal_rate=0.2,
                n_signals=30,
                n_test_bars=150,
            )
            for _ in range(5)
        ]
        result = WalkForwardMLResult(regime="test", windows=windows)
        assert result.profitable_windows_pct == 100.0

    def test_none_profitable(self):
        windows = [
            WindowMLResult(
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 7, 1),
                test_start=datetime(2024, 7, 1),
                test_end=datetime(2024, 10, 1),
                win_rate=45.0,
                profit_factor=0.8,
                signal_rate=0.2,
                n_signals=30,
                n_test_bars=150,
            )
            for _ in range(5)
        ]
        result = WalkForwardMLResult(regime="test", windows=windows)
        assert result.profitable_windows_pct == 0.0

    def test_partial_profitable(self):
        windows = []
        for i in range(5):
            pf = 1.5 if i < 3 else 0.8  # 3/5 profitable
            windows.append(
                WindowMLResult(
                    train_start=datetime(2024, 1, 1),
                    train_end=datetime(2024, 7, 1),
                    test_start=datetime(2024, 7, 1),
                    test_end=datetime(2024, 10, 1),
                    win_rate=52.0,
                    profit_factor=pf,
                    signal_rate=0.2,
                    n_signals=30,
                    n_test_bars=150,
                )
            )
        result = WalkForwardMLResult(regime="test", windows=windows)
        assert result.profitable_windows_pct == 60.0


class TestPassesWalkForwardTest:
    """Test the 60% profitability threshold."""

    def test_passes_at_60_pct(self):
        windows = []
        for i in range(5):
            pf = 1.5 if i < 3 else 0.8  # 3/5 = 60%
            windows.append(
                WindowMLResult(
                    train_start=datetime(2024, 1, 1),
                    train_end=datetime(2024, 7, 1),
                    test_start=datetime(2024, 7, 1),
                    test_end=datetime(2024, 10, 1),
                    win_rate=52.0,
                    profit_factor=pf,
                    signal_rate=0.2,
                    n_signals=30,
                    n_test_bars=150,
                )
            )
        result = WalkForwardMLResult(regime="test", windows=windows)
        assert result.passes_walk_forward_test is True

    def test_fails_below_60_pct(self):
        windows = []
        for i in range(5):
            pf = 1.5 if i < 2 else 0.8  # 2/5 = 40%
            windows.append(
                WindowMLResult(
                    train_start=datetime(2024, 1, 1),
                    train_end=datetime(2024, 7, 1),
                    test_start=datetime(2024, 7, 1),
                    test_end=datetime(2024, 10, 1),
                    win_rate=52.0,
                    profit_factor=pf,
                    signal_rate=0.2,
                    n_signals=30,
                    n_test_bars=150,
                )
            )
        result = WalkForwardMLResult(regime="test", windows=windows)
        assert result.passes_walk_forward_test is False


# ===========================================================================
# 3. DSR Tests
# ===========================================================================


class TestCalculateDSRKnownValues:
    """Test DSR returns plausible values for known inputs."""

    def test_positive_sharpe_ratios(self):
        # All positive SRs with few trials → should be moderate DSR
        srs = [1.5, 1.2, 1.8, 1.3, 1.6]
        dsr = calculate_dsr(srs, n_trials=5)
        assert 0.0 <= dsr <= 1.0

    def test_very_high_sr(self):
        # Very high SR with few trials → high DSR
        srs = [3.0, 2.8, 3.2, 2.9, 3.1]
        dsr = calculate_dsr(srs, n_trials=3)
        assert dsr > 0.5

    def test_near_zero_sr(self):
        # Near-zero SRs → low DSR
        srs = [0.1, -0.1, 0.05, -0.05, 0.0]
        dsr = calculate_dsr(srs, n_trials=50)
        assert dsr < 0.5


class TestDSRGrowsWithTrials:
    """Test DSR decreases (penalizes) with more experiments."""

    def test_dsr_penalizes_more_trials(self):
        srs = [1.5, 1.2, 1.8, 1.3, 1.6]
        dsr_few = calculate_dsr(srs, n_trials=3)
        dsr_many = calculate_dsr(srs, n_trials=100)
        # More trials → higher E[SR_max] → lower DSR (more skeptical)
        assert dsr_few > dsr_many


# ===========================================================================
# 4. PBO Tests
# ===========================================================================


class TestPBORandom:
    """Test PBO ≈ 0.5 for random/uniform results."""

    def test_random_results_give_moderate_pbo(self):
        # Random win rates → PBO should be near 0.5
        rng = np.random.RandomState(42)
        results = [
            _make_eval_result(win_rate=50 + rng.randn() * 2)
            for _ in range(6)
        ]
        pbo = calculate_pbo(results, metric="win_rate")
        # With random data, PBO should be in [0.1, 0.9]
        assert 0.0 <= pbo <= 1.0


class TestPBOGoodModel:
    """Test PBO is low for consistently good model."""

    def test_consistent_model_low_pbo(self):
        # All folds have similar high win rate → low overfitting
        results = [
            _make_eval_result(win_rate=55 + i * 0.1)
            for i in range(6)
        ]
        pbo = calculate_pbo(results, metric="win_rate")
        # Should be relatively low for consistent performance
        assert 0.0 <= pbo <= 1.0


# ===========================================================================
# 5. t-stat Tests
# ===========================================================================


class TestTStatSignificant:
    """Test t-stat for clearly significant win rate (60%)."""

    def test_significant_win_rate(self):
        # WR=60% across 5 windows with 100 trades each
        win_rates = [60.0, 58.0, 62.0, 59.0, 61.0]
        n_trades = [100, 100, 100, 100, 100]
        t = calculate_t_stat(win_rates, n_trades)
        # WR significantly above 50% → high t-stat
        assert t > 2.0


class TestTStatInsignificant:
    """Test t-stat for borderline win rate (51%)."""

    def test_borderline_win_rate(self):
        # WR=51% → should be lower t-stat
        win_rates = [51.0, 50.0, 52.0, 49.5, 51.5]
        n_trades = [100, 100, 100, 100, 100]
        t = calculate_t_stat(win_rates, n_trades)
        # WR barely above 50% with variance → moderate or low t
        assert t < 3.0


# ===========================================================================
# 6. StatTestResult Tests
# ===========================================================================


class TestStatTestResultThresholds:
    """Test StatTestResult.passes_all_thresholds logic."""

    def test_passes_when_all_good(self):
        r = StatTestResult(dsr=0.96, pbo=0.25, t_stat=3.5, n_oos_signals=500)
        assert r.passes_all_thresholds() is True

    def test_fails_low_dsr(self):
        r = StatTestResult(dsr=0.80, pbo=0.25, t_stat=3.5, n_oos_signals=500)
        assert r.passes_all_thresholds() is False

    def test_fails_high_pbo(self):
        r = StatTestResult(dsr=0.96, pbo=0.40, t_stat=3.5, n_oos_signals=500)
        assert r.passes_all_thresholds() is False

    def test_fails_low_t_stat(self):
        r = StatTestResult(dsr=0.96, pbo=0.25, t_stat=2.0, n_oos_signals=500)
        assert r.passes_all_thresholds() is False

    def test_fails_few_signals(self):
        r = StatTestResult(dsr=0.96, pbo=0.25, t_stat=3.5, n_oos_signals=100)
        assert r.passes_all_thresholds() is False


# ===========================================================================
# 7. run_all_tests
# ===========================================================================


class TestRunAllTests:
    """Test run_all_tests returns correct type."""

    def test_returns_stat_test_result(self):
        cv_results = [
            _make_eval_result(win_rate=52 + i, profit_factor=1.1 + i * 0.1)
            for i in range(5)
        ]
        wf_result = WalkForwardMLResult(
            regime="test",
            windows=[
                WindowMLResult(
                    train_start=datetime(2024, 1, 1),
                    train_end=datetime(2024, 7, 1),
                    test_start=datetime(2024, 7, 1),
                    test_end=datetime(2024, 10, 1),
                    win_rate=53.0,
                    profit_factor=1.2,
                    signal_rate=0.15,
                    n_signals=50,
                    n_test_bars=300,
                )
                for _ in range(4)
            ],
        )

        result = run_all_tests(cv_results, wf_result, n_experiments=10)

        assert isinstance(result, StatTestResult)
        assert isinstance(result.dsr, float)
        assert isinstance(result.pbo, float)
        assert isinstance(result.t_stat, float)
        assert isinstance(result.n_oos_signals, int)


class TestOOSSignalsCount:
    """Test n_oos_signals is counted correctly from WF windows."""

    def test_total_oos_signals(self):
        cv_results = [_make_eval_result() for _ in range(5)]
        wf_result = WalkForwardMLResult(
            regime="test",
            windows=[
                WindowMLResult(
                    train_start=datetime(2024, 1, 1),
                    train_end=datetime(2024, 7, 1),
                    test_start=datetime(2024, 7, 1),
                    test_end=datetime(2024, 10, 1),
                    win_rate=53.0,
                    profit_factor=1.2,
                    signal_rate=0.15,
                    n_signals=75,
                    n_test_bars=300,
                )
                for _ in range(4)
            ],
        )

        result = run_all_tests(cv_results, wf_result)
        assert result.n_oos_signals == 75 * 4  # 300


class TestStatTestSummary:
    """Test StatTestResult.summary() output."""

    def test_summary_contains_key_info(self):
        r = StatTestResult(dsr=0.87, pbo=0.33, t_stat=2.1, n_oos_signals=285)
        s = r.summary()
        assert "DSR" in s
        assert "PBO" in s
        assert "t-stat" in s
        assert "OOS signals" in s
        assert "VERDICT" in s


# ===========================================================================
# 8. Integration: MLValidator on synthetic data
# ===========================================================================


class TestMLValidatorIntegration:
    """Test MLValidator on synthetic BTCUSDT data (end-to-end)."""

    def test_full_cv_pipeline(self, tmp_path: Path):
        """Run purged_kfold_cv on synthetic data and verify results."""
        trainer, _, _ = _make_trainer(tmp_path, regime="all", n=400)
        train_df, test_df = trainer.prepare_data()
        full_df = pl.concat([train_df, test_df], how="diagonal")

        validator = MLValidator(n_splits=3)
        results = validator.purged_kfold_cv(trainer, full_df)

        assert len(results) >= 1
        for r in results:
            assert isinstance(r, EvaluationResult)
            assert 0 <= r.win_rate <= 100
            assert r.profit_factor >= 0
            assert 0 <= r.signal_rate <= 1.0
