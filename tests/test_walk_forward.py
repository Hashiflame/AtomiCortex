"""Tests for Phase 2.4-2.6: metrics, walk-forward validation, MLflow tracking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from src.execution.metrics import (
    MetricsResult,
    calculate_all_metrics,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_win_rate,
)
from src.execution.walk_forward import (
    PurgedKFoldCV,
    WalkForwardResult,
    WalkForwardValidator,
    WindowResult,
    _add_months,
)

DATA_DIR = Path("/mnt/hdd/AtomiCortex/data/features")
_data_skip = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="External data drive not mounted"
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

_UTC = timezone.utc
_DT = datetime(2024, 1, 1, tzinfo=_UTC)

_DUMMY_METRICS = MetricsResult(
    sharpe_ratio=0.0,
    calmar_ratio=0.0,
    max_drawdown_pct=0.0,
    win_rate=0.0,
    profit_factor=0.0,
    total_return_pct=0.0,
    annualized_return_pct=0.0,
    total_trades=0,
)


def _make_window(is_profitable: bool) -> WindowResult:
    return WindowResult(
        train_start=_DT,
        train_end=_DT,
        test_start=_DT,
        test_end=_DT,
        metrics=_DUMMY_METRICS,
        is_profitable=is_profitable,
    )


def _daily_equity(start_val: float, returns: list[float]) -> list[tuple[datetime, float]]:
    """Build daily equity curve from a sequence of per-period returns."""
    curve = [(datetime(2024, 1, 1, tzinfo=_UTC), start_val)]
    for i, r in enumerate(returns, 1):
        curve.append((datetime(2024, 1, i + 1, tzinfo=_UTC), curve[-1][1] * (1 + r)))
    return curve


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    # ── Sharpe ───────────────────────────────────────────────────────────────

    def test_sharpe_is_float(self):
        ec = _daily_equity(10_000, [0.01, -0.005, 0.02, -0.008])
        assert isinstance(calculate_sharpe_ratio(ec), float)

    def test_sharpe_positive_for_upward_equity(self):
        # Consistently positive returns → positive Sharpe
        ec = _daily_equity(10_000, [0.01, 0.02, 0.015, 0.008, 0.012])
        assert calculate_sharpe_ratio(ec, risk_free_rate=0.0) > 0

    def test_sharpe_zero_for_constant_equity(self):
        ec = [(datetime(2024, 1, i + 1, tzinfo=_UTC), 10_000.0) for i in range(10)]
        assert calculate_sharpe_ratio(ec) == 0.0

    def test_sharpe_known_data(self):
        """Alternating +2 % / -1 % → mean > 0 → Sharpe > 0 (with rf=0)."""
        ec = _daily_equity(10_000, [0.02, -0.01, 0.02, -0.01, 0.02, -0.01])
        sharpe = calculate_sharpe_ratio(ec, risk_free_rate=0.0)
        assert sharpe > 0.0

    def test_sharpe_negative_for_declining_equity(self):
        ec = _daily_equity(10_000, [-0.01, -0.02, -0.015, -0.005, -0.01])
        assert calculate_sharpe_ratio(ec, risk_free_rate=0.0) < 0

    def test_sharpe_empty_curve_returns_zero(self):
        assert calculate_sharpe_ratio([]) == 0.0

    # ── Bug regression: rf=0.05 default caused negative Sharpe for crypto ────

    def test_sharpe_uniform_growth_10k_to_11k_positive(self):
        """
        Equity grows uniformly from $10 000 to $11 000 over ~365 days with
        realistic daily noise — Sharpe must be positive > 0 using the DEFAULT
        risk_free_rate (now 0.0, not the old 0.05 which gave -12.4 for any
        strategy earning < 5 % / year).

        BEFORE fix (rf=0.05 default): a strategy earning only ~1–2 % annual
        returned a large negative Sharpe because 1 % < 5 % risk-free rate,
        even though the strategy was profitable.

        AFTER fix (rf=0.0 default): any strategy with positive mean return
        and non-zero volatility gives Sharpe > 0.
        """
        import random
        rng = random.Random(42)
        dt = datetime(2024, 1, 1, tzinfo=_UTC)
        equity = 10_000.0
        # 6 × 4H bars per day, equity changes every day with small noise
        curve: list[tuple[datetime, float]] = []
        daily_target = (11_000 / 10_000) ** (1 / 365) - 1  # ~0.0261 % / day
        for day in range(365):
            noise = rng.gauss(0, 0.005)          # 0.5 % daily noise
            equity *= (1 + daily_target + noise)
            # Six 4H bars per day — all with same daily closing equity
            for bar in range(6):
                ts = dt + timedelta(days=day, hours=bar * 4)
                curve.append((ts, equity))

        sharpe = calculate_sharpe_ratio(curve)   # uses default rf=0.0
        assert sharpe > 0, (
            f"Profitable strategy (10k→11k) should have positive Sharpe, got {sharpe:.4f}. "
            f"With old rf=0.05 default this would be negative for annual returns < 5 %."
        )

    def test_sharpe_constant_rate_noise_floor(self):
        """
        Equity growing at EXACTLY constant rate → returns are equal in theory;
        floating-point arithmetic creates std ≈ 1e-18 (not true zero).
        The 1e-8 noise floor must return 0.0 instead of ±1e13 garbage.
        """
        dt = datetime(2024, 1, 1, tzinfo=_UTC)
        equity = 10_000.0
        curve: list[tuple[datetime, float]] = []
        for day in range(365):
            equity *= 1.0001           # exact constant multiplier
            curve.append((dt + timedelta(days=day), equity))
        sharpe = calculate_sharpe_ratio(curve)
        assert sharpe == 0.0, f"Constant-rate equity should give 0.0 (not {sharpe:.3e})"

    def test_sharpe_4h_bars_same_as_daily(self):
        """
        4H equity curve and its daily-equivalent give the same Sharpe —
        the daily-grouping collapse is bar-frequency-agnostic.
        """
        import random
        rng = random.Random(7)
        dt = datetime(2024, 1, 1, tzinfo=_UTC)
        equity = 10_000.0
        daily_curve: list[tuple[datetime, float]] = []
        fourh_curve: list[tuple[datetime, float]] = []
        for day in range(60):
            equity *= (1 + rng.gauss(0.001, 0.01))
            daily_curve.append((dt + timedelta(days=day), equity))
            for bar in range(6):
                fourh_curve.append((dt + timedelta(days=day, hours=bar * 4), equity))

        s_daily = calculate_sharpe_ratio(daily_curve)
        s_4h    = calculate_sharpe_ratio(fourh_curve)
        assert s_daily == pytest.approx(s_4h, rel=1e-9)

    # ── Max Drawdown ─────────────────────────────────────────────────────────

    def test_max_drawdown_known_value(self):
        # Peak 12 000, trough 9 000 → MDD = 25 %
        ec = [
            (datetime(2024, 1, 1, tzinfo=_UTC), 10_000.0),
            (datetime(2024, 1, 2, tzinfo=_UTC), 12_000.0),
            (datetime(2024, 1, 3, tzinfo=_UTC), 9_000.0),
            (datetime(2024, 1, 4, tzinfo=_UTC), 11_000.0),
        ]
        assert calculate_max_drawdown(ec) == pytest.approx(25.0)

    def test_max_drawdown_zero_for_monotone_upward(self):
        ec = _daily_equity(10_000, [0.01, 0.01, 0.01, 0.01])
        assert calculate_max_drawdown(ec) == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_bounded(self):
        ec = _daily_equity(10_000, [0.5, -0.3, 0.2, -0.4, 0.1])
        mdd = calculate_max_drawdown(ec)
        assert 0.0 <= mdd <= 100.0

    # ── Win Rate / Profit Factor ──────────────────────────────────────────────

    def test_win_rate_empty_returns_zero(self):
        assert calculate_win_rate([]) == 0.0

    def test_win_rate_all_wins(self):
        trades = [{"pnl": 100}, {"pnl": 50}, {"pnl": 200}]
        assert calculate_win_rate(trades) == pytest.approx(100.0)

    def test_win_rate_mixed(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}]
        assert calculate_win_rate(trades) == pytest.approx(50.0)

    def test_profit_factor_no_losing_trades(self):
        trades = [{"pnl": 100}, {"pnl": 200}]
        assert calculate_profit_factor(trades) == float("inf")

    def test_profit_factor_known_value(self):
        # wins = 300, losses = 100 → PF = 3.0
        trades = [{"pnl": 100}, {"pnl": 200}, {"pnl": -100}]
        assert calculate_profit_factor(trades) == pytest.approx(3.0)

    # ── MetricsResult ─────────────────────────────────────────────────────────

    def test_metrics_passes_thresholds_true(self):
        m = MetricsResult(
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown_pct=10.0,
            win_rate=55.0,
            profit_factor=1.5,
            total_return_pct=20.0,
            annualized_return_pct=25.0,
            total_trades=50,
        )
        assert m.passes_minimum_thresholds() is True

    def test_metrics_fails_thresholds_low_sharpe(self):
        m = MetricsResult(
            sharpe_ratio=0.5,      # < 1.0
            calmar_ratio=2.0,
            max_drawdown_pct=10.0,
            win_rate=55.0,
            profit_factor=1.5,
            total_return_pct=10.0,
            annualized_return_pct=12.0,
            total_trades=50,
        )
        assert m.passes_minimum_thresholds() is False

    def test_metrics_fails_thresholds_high_drawdown(self):
        m = MetricsResult(
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown_pct=25.0,  # > 20 %
            win_rate=55.0,
            profit_factor=1.5,
            total_return_pct=10.0,
            annualized_return_pct=12.0,
            total_trades=50,
        )
        assert m.passes_minimum_thresholds() is False

    def test_metrics_to_dict_all_keys(self):
        m = _DUMMY_METRICS
        d = m.to_dict()
        expected_keys = {
            "sharpe_ratio", "calmar_ratio", "max_drawdown_pct",
            "win_rate", "profit_factor", "total_return_pct",
            "annualized_return_pct", "total_trades",
        }
        assert expected_keys == set(d.keys())

    def test_metrics_to_dict_inf_profit_factor_capped(self):
        m = MetricsResult(
            sharpe_ratio=0.0, calmar_ratio=0.0, max_drawdown_pct=0.0,
            win_rate=100.0, profit_factor=float("inf"),
            total_return_pct=0.0, annualized_return_pct=0.0, total_trades=1,
        )
        assert m.to_dict()["profit_factor"] == 999.0

    def test_metrics_to_dict_values_are_floats(self):
        m = _DUMMY_METRICS
        for v in m.to_dict().values():
            assert isinstance(v, float)


# ══════════════════════════════════════════════════════════════════════════════
# PurgedKFoldCV
# ══════════════════════════════════════════════════════════════════════════════

def _make_df(n: int = 100) -> pl.DataFrame:
    """Sequential daily DataFrame for CV tests."""
    start = datetime(2024, 1, 1, tzinfo=_UTC)
    return pl.DataFrame(
        {
            "datetime": [start + timedelta(days=i) for i in range(n)],
            "value": list(range(n)),
        }
    )


class TestPurgedKFoldCV:
    def test_correct_number_of_splits(self):
        cv = PurgedKFoldCV(n_splits=5)
        splits = list(cv.split(_make_df(100)))
        assert len(splits) == 5

    def test_no_data_leakage_test_after_train(self):
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=0.01)
        df = _make_df(100)
        for train_df, test_df in cv.split(df):
            last_train_val = train_df["value"][-1]
            first_test_val = test_df["value"][0]
            assert first_test_val > last_train_val, (
                f"Test starts at {first_test_val}, train ends at {last_train_val}"
            )

    def test_embargo_gap_respected(self):
        """First test row must be at least embargo_rows after last train row."""
        n = 100
        embargo_pct = 0.02  # 2 rows
        cv = PurgedKFoldCV(n_splits=5, embargo_pct=embargo_pct)
        df = _make_df(n)
        embargo_rows = max(1, int(n * embargo_pct))  # = 2
        for train_df, test_df in cv.split(df):
            gap = test_df["value"][0] - train_df["value"][-1]
            assert gap >= embargo_rows, f"Gap {gap} < embargo {embargo_rows}"

    def test_train_grows_across_folds(self):
        cv = PurgedKFoldCV(n_splits=4)
        splits = list(cv.split(_make_df(100)))
        train_sizes = [len(tr) for tr, _ in splits]
        assert train_sizes == sorted(train_sizes), "Train sets should grow each fold"

    def test_too_small_dataset_raises(self):
        cv = PurgedKFoldCV(n_splits=5)
        with pytest.raises(ValueError, match="too small"):
            list(cv.split(_make_df(3)))


# ══════════════════════════════════════════════════════════════════════════════
# WalkForwardValidator — period logic (no data needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardValidator:
    def test_correct_number_of_windows(self):
        # train=3m, test=1m, step=1m, start=2024-01-01, end=2024-08-31
        # W1: train=[01,04), test=[04,05)  test_end=2024-05-01 ≤ 2024-08-31 ✓
        # W2: train=[02,05), test=[05,06)  ✓
        # W3: train=[03,06), test=[06,07)  ✓
        # W4: train=[04,07), test=[07,08)  ✓
        # W5 would need test_end=2024-09-01 > 2024-08-31 → excluded
        validator = WalkForwardValidator(train_months=3, test_months=1, step_months=1)
        start = datetime(2024, 1, 1, tzinfo=_UTC)
        end = datetime(2024, 8, 31, tzinfo=_UTC)
        pairs = list(validator.split(start, end))
        assert len(pairs) == 4

    def test_first_window_start(self):
        validator = WalkForwardValidator(train_months=3, test_months=1, step_months=1)
        start = datetime(2024, 1, 1, tzinfo=_UTC)
        pairs = list(validator.split(start, datetime(2024, 12, 31, tzinfo=_UTC)))
        (train_start, train_end), (test_start, test_end) = pairs[0]
        assert train_start == start
        assert train_end == datetime(2024, 4, 1, tzinfo=_UTC)
        assert test_start == train_end
        assert test_end == datetime(2024, 5, 1, tzinfo=_UTC)

    def test_step_advances_by_step_months(self):
        validator = WalkForwardValidator(train_months=3, test_months=1, step_months=2)
        pairs = list(validator.split(
            datetime(2024, 1, 1, tzinfo=_UTC),
            datetime(2024, 12, 31, tzinfo=_UTC),
        ))
        assert len(pairs) >= 2
        (ts1, _), _ = pairs[0]
        (ts2, _), _ = pairs[1]
        assert _add_months(ts1, 2) == ts2

    def test_no_overlapping_test_windows(self):
        validator = WalkForwardValidator(train_months=3, test_months=2, step_months=2)
        pairs = list(validator.split(
            datetime(2024, 1, 1, tzinfo=_UTC),
            datetime(2024, 12, 31, tzinfo=_UTC),
        ))
        test_periods = [tp for _, tp in pairs]
        for i in range(len(test_periods) - 1):
            _, end_i = test_periods[i]
            start_next, _ = test_periods[i + 1]
            assert end_i <= start_next, "Overlapping test windows detected"

    def test_default_step_equals_test_months(self):
        # When step_months not specified it defaults to test_months
        v = WalkForwardValidator(train_months=6, test_months=3)
        assert v.step_months == 3  # default == test_months


# ══════════════════════════════════════════════════════════════════════════════
# WalkForwardResult properties
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardResult:
    def test_profitable_windows_pct_exact(self):
        # 3 / 5 = 60 %
        windows = [
            _make_window(True), _make_window(True), _make_window(True),
            _make_window(False), _make_window(False),
        ]
        result = WalkForwardResult(windows=windows)
        assert result.profitable_windows_pct == pytest.approx(60.0)

    def test_profitable_windows_pct_empty(self):
        assert WalkForwardResult(windows=[]).profitable_windows_pct == 0.0

    def test_passes_walk_forward_test_at_60_pct(self):
        windows = [
            _make_window(True), _make_window(True), _make_window(True),
            _make_window(False), _make_window(False),
        ]
        assert WalkForwardResult(windows=windows).passes_walk_forward_test is True

    def test_fails_walk_forward_test_below_60_pct(self):
        windows = [
            _make_window(True), _make_window(True),
            _make_window(False), _make_window(False), _make_window(False),
        ]
        assert WalkForwardResult(windows=windows).passes_walk_forward_test is False

    def test_avg_sharpe(self):
        from src.execution.metrics import MetricsResult
        make_m = lambda s: MetricsResult(s, 0, 0, 0, 0, 0, 0, 0)
        windows = [
            WindowResult(_DT, _DT, _DT, _DT, make_m(1.0), True),
            WindowResult(_DT, _DT, _DT, _DT, make_m(3.0), True),
        ]
        assert WalkForwardResult(windows=windows).avg_sharpe == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# ExperimentTracker (MLflow — temp SQLite)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def tracker(tmp_path):
    from src.execution.experiment_tracker import ExperimentTracker
    import mlflow

    # Use filesystem store — pyenv Python may lack _sqlite3 module
    uri = f"file:///{tmp_path}/mlruns"
    t = ExperimentTracker("test_atomicortex", tracking_uri=uri)
    yield t
    mlflow.end_run()


def _dummy_backtest_config():
    from src.execution.backtest_runner import BacktestConfig
    return BacktestConfig(
        symbol="BTCUSDT",
        interval="4h",
        start=datetime(2024, 1, 1, tzinfo=_UTC),
        end=datetime(2024, 2, 1, tzinfo=_UTC),
        initial_capital=10_000.0,
    )


def _dummy_backtest_result():
    from src.execution.backtest_runner import BacktestResult
    return BacktestResult(
        total_return_pct=5.0,
        sharpe_ratio=1.2,
        max_drawdown_pct=3.0,
        total_trades=4,
        win_rate=0.75,
        profit_factor=2.0,
        start_equity=10_000.0,
        end_equity=10_500.0,
        equity_curve=[],
    )


def _dummy_metrics(sharpe: float = 1.2) -> MetricsResult:
    return MetricsResult(
        sharpe_ratio=sharpe,
        calmar_ratio=1.0,
        max_drawdown_pct=3.0,
        win_rate=60.0,
        profit_factor=2.0,
        total_return_pct=5.0,
        annualized_return_pct=30.0,
        total_trades=4,
    )


class TestExperimentTracker:
    def test_log_backtest_returns_run_id(self, tracker):
        run_id = tracker.log_backtest(
            run_name="test_run",
            config=_dummy_backtest_config(),
            result=_dummy_backtest_result(),
            metrics=_dummy_metrics(),
        )
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_log_backtest_run_id_is_unique(self, tracker):
        cfg = _dummy_backtest_config()
        r1 = tracker.log_backtest("run_a", cfg, _dummy_backtest_result(), _dummy_metrics(1.0))
        r2 = tracker.log_backtest("run_b", cfg, _dummy_backtest_result(), _dummy_metrics(2.0))
        assert r1 != r2

    def test_get_best_runs_returns_sorted_list(self, tracker):
        cfg = _dummy_backtest_config()
        for sharpe in [0.5, 2.5, 1.5]:
            tracker.log_backtest(f"run_{sharpe}", cfg, _dummy_backtest_result(), _dummy_metrics(sharpe))

        best = tracker.get_best_runs(metric="sharpe_ratio", top_n=2)
        assert len(best) == 2
        assert best[0]["sharpe_ratio"] >= best[1]["sharpe_ratio"]
        assert best[0]["sharpe_ratio"] == pytest.approx(2.5)

    def test_get_best_runs_empty_experiment(self, tmp_path):
        from src.execution.experiment_tracker import ExperimentTracker
        uri = f"file:///{tmp_path}/mlruns_empty"
        t = ExperimentTracker("empty_exp", tracking_uri=uri)
        # No runs logged → should return empty list
        best = t.get_best_runs()
        assert isinstance(best, list)

    def test_get_best_runs_respects_top_n(self, tracker):
        cfg = _dummy_backtest_config()
        for i in range(5):
            tracker.log_backtest(f"run_{i}", cfg, _dummy_backtest_result(), _dummy_metrics(float(i)))
        best = tracker.get_best_runs(top_n=3)
        assert len(best) <= 3

    def test_log_walk_forward_returns_run_id(self, tracker):
        wf = WalkForwardResult(windows=[_make_window(True), _make_window(False)])
        run_id = tracker.log_walk_forward(
            run_name="wf_test",
            wf_result=wf,
            config=_dummy_backtest_config(),
        )
        assert isinstance(run_id, str) and len(run_id) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Integration test — requires data drive
# ══════════════════════════════════════════════════════════════════════════════

@_data_skip
def test_walk_forward_determinism():
    """Same config must produce identical results on two consecutive runs."""
    from src.execution.backtest_runner import BacktestConfig
    from src.execution.strategies.baseline_strategy import BuyAndHoldStrategy

    cfg = BacktestConfig(
        symbol="BTCUSDT",
        interval="4h",
        start=datetime(2024, 1, 1, tzinfo=_UTC),
        end=datetime(2024, 6, 30, tzinfo=_UTC),
        initial_capital=10_000.0,
        data_dir=DATA_DIR,
    )
    validator = WalkForwardValidator(train_months=2, test_months=1, step_months=1)

    r1 = validator.run_validation(BuyAndHoldStrategy, {"trade_size": 0.001}, cfg, DATA_DIR)
    r2 = validator.run_validation(BuyAndHoldStrategy, {"trade_size": 0.001}, cfg, DATA_DIR)

    assert len(r1.windows) == len(r2.windows)
    for w1, w2 in zip(r1.windows, r2.windows):
        assert w1.metrics.total_return_pct == pytest.approx(
            w2.metrics.total_return_pct, abs=1e-6
        )
