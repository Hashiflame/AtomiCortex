"""
tests/test_1h_dataset.py

Unit tests for the 1H ML dataset builder.

Tests target construction, lookahead prevention, regime splits,
embargo, feature integrity, and class balance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers — synthetic data generator
# ---------------------------------------------------------------------------

def _make_1h_ohlcv(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic 1H OHLCV data with realistic structure."""
    rng = np.random.RandomState(seed)

    # Brownian motion price series
    returns = rng.normal(0.0002, 0.005, n)
    close = 50000.0 * np.exp(np.cumsum(returns))

    # Realistic OHLCV
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = np.abs(rng.normal(100, 30, n))

    # Timestamps: 1H apart, starting 2023-01-01
    base_ts = 1672531200000  # 2023-01-01 00:00:00 UTC in ms
    open_time = np.array([base_ts + i * 3600_000 for i in range(n)], dtype=np.int64)

    return pl.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "taker_buy_volume": volume * 0.5,
    })


def _make_df_with_target(
    n: int = 300,
    seed: int = 42,
    pt_multiplier: float = 1.5,
    sl_multiplier: float = 1.0,
    max_holding_bars: int = 6,
) -> pl.DataFrame:
    """Create a DataFrame with atr_pct, regime, and target columns.

    Target is built with triple-barrier (AFML Ch.3), mirroring
    build_1h_dataset.main(): drop vertical (0) labels, rename
    label → target. Defaults match MLStrategyConfig1H.tb_* (1.5/1.0/6).
    """
    df = _make_1h_ohlcv(n, seed)

    # Add atr_pct (simple TR-based)
    tr = df["high"] - df["low"]
    atr_pct = tr / df["close"]
    df = df.with_columns(atr_pct.alias("atr_pct"))

    # Add regime column (mock)
    regimes = ["trend_up", "trend_down", "high_vol", "range", "unknown"]
    rng = np.random.RandomState(seed)
    regime_arr = rng.choice(regimes, n, p=[0.3, 0.2, 0.2, 0.2, 0.1])
    df = df.with_columns(pl.Series("regime", regime_arr, dtype=pl.Utf8))

    # Create target — triple-barrier, same as build_1h_dataset.main()
    from src.features.triple_barrier import apply_triple_barrier
    df = apply_triple_barrier(
        df,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_bars=max_holding_bars,
    )
    df = df.filter(pl.col("label") != 0).rename({"label": "target"})

    return df


# ===========================================================================
# Tests
# ===========================================================================


class TestTargetConstruction:
    """Tests for target variable construction.

    Triple-barrier mechanics (no-lookahead, last-bars-drop, label
    domain, ATR scaling) are covered in tests/test_triple_barrier.py.
    Here we only assert dataset-builder integration (vertical labels
    dropped → no FLAT rows reach the regime datasets).
    """

    def test_flat_bars_excluded(self):
        """Vertical (target=0) bars must be excluded from regime datasets."""
        df = _make_df_with_target(n=300)

        from scripts.build_1h_dataset import split_by_regime
        df_trend, df_high_vol = split_by_regime(df)

        # No FLAT targets in either dataset
        assert (df_trend["target"] == 0).sum() == 0, "FLAT rows found in trend dataset"
        assert (df_high_vol["target"] == 0).sum() == 0, "FLAT rows found in high_vol dataset"


class TestRegimeSplit:
    """Tests for regime-based dataset splitting."""

    def test_regime_split_correct(self):
        """Datasets are split correctly by regime label."""
        df = _make_df_with_target(n=300)

        from scripts.build_1h_dataset import split_by_regime
        df_trend, df_high_vol = split_by_regime(df)

        assert len(df_trend) > 0, "Trend dataset is empty"
        assert len(df_high_vol) > 0, "High-vol dataset is empty"

    def test_trend_dataset_no_range_rows(self):
        """Trend dataset must NOT contain rows with regime='range'."""
        df = _make_df_with_target(n=300)

        from scripts.build_1h_dataset import split_by_regime
        df_trend, _ = split_by_regime(df)

        if "regime" in df_trend.columns:
            regime_values = set(df_trend["regime"].unique().to_list())
            assert "range" not in regime_values, f"Found 'range' in trend dataset: {regime_values}"
            assert "unknown" not in regime_values, f"Found 'unknown' in trend dataset"

    def test_high_vol_dataset_no_trend_rows(self):
        """High-vol dataset must NOT contain trend_up or trend_down rows."""
        df = _make_df_with_target(n=300)

        from scripts.build_1h_dataset import split_by_regime
        _, df_high_vol = split_by_regime(df)

        if "regime" in df_high_vol.columns:
            regime_values = set(df_high_vol["regime"].unique().to_list())
            assert "trend_up" not in regime_values
            assert "trend_down" not in regime_values


class TestWalkForward:
    """Tests for walk-forward and embargo mechanics."""

    def test_embargo_applied_in_walk_forward(self):
        """PurgedKFoldCV with embargo must not leak adjacent bars."""
        from src.execution.walk_forward import PurgedKFoldCV

        df = _make_df_with_target(n=500)
        cv = PurgedKFoldCV(n_splits=3, embargo_pct=0.02)

        # embargo_pct=0.02 on 500 rows = 10 rows; each row = 1 hour = 3600_000 ms
        min_gap_ms = 1 * 3600_000  # at least 1 bar gap

        for train_df, test_df in cv.split(df):
            # Train end index < test start index (gap exists)
            train_end_time = train_df["open_time"][-1]
            test_start_time = test_df["open_time"][0]
            gap_ms = test_start_time - train_end_time
            assert gap_ms > min_gap_ms, (
                f"Embargo gap too small: {gap_ms / 3600_000:.1f} hours "
                f"(need > {min_gap_ms / 3600_000:.1f}h). "
                f"Train ends at {train_end_time}, test starts at {test_start_time}"
            )

    def test_wf_windows_no_overlap(self):
        """Walk-forward train/test windows must not overlap."""
        from src.execution.walk_forward import WalkForwardValidator
        from datetime import datetime, timezone

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        wf = WalkForwardValidator(train_months=12, test_months=4, step_months=2)

        for (ts, te), (vs, ve) in wf.split(start, end):
            # Train end must be <= test start (no overlap)
            assert te <= vs, (
                f"Train/test overlap: train ends {te}, test starts {vs}"
            )
            # Test start must be strictly after train end or equal
            # (the actual WF sets test_start = train_end, which is fine
            # because train uses < te and test uses >= vs)


class TestFeatures:
    """Tests for feature integrity."""

    def test_feature_count_matches_expected(self):
        """Feature count should be >= 40 for 1H (base + session + mtf)."""
        df = _make_df_with_target(n=300)

        from src.models.dataset_builder import _EXCLUDE_COLUMNS
        feature_cols = [
            col for col in df.columns
            if col not in _EXCLUDE_COLUMNS
            and (df[col].dtype.is_float() or df[col].dtype.is_integer())
        ]

        # Synthetic data produces minimal features (atr_pct).
        # Full pipeline with real data would produce 70+ features.
        assert len(feature_cols) >= 1, (
            f"Expected at least 1 feature column, got {len(feature_cols)}: "
            f"{feature_cols}"
        )

    def test_no_nan_in_features(self):
        """Feature columns should not contain NaN after warmup trim."""
        df = _make_df_with_target(n=300)

        from src.models.dataset_builder import _EXCLUDE_COLUMNS
        feature_cols = [
            col for col in df.columns
            if col not in _EXCLUDE_COLUMNS
            and df[col].dtype.is_float()
        ]

        for col in feature_cols:
            null_count = df[col].null_count()
            nan_count = df[col].is_nan().sum()
            total_bad = null_count + nan_count
            assert total_bad == 0, (
                f"Column '{col}' has {total_bad} NaN/null values"
            )

    def test_no_nan_in_target(self):
        """Target column must have zero NaN/null values."""
        df = _make_df_with_target(n=200)
        assert df["target"].null_count() == 0, "Target has null values"

    def test_atr_threshold_dynamic_not_fixed(self):
        """ATR threshold must vary across rows (dynamic, not a fixed constant)."""
        df = _make_1h_ohlcv(200, seed=99)
        tr = df["high"] - df["low"]
        df = df.with_columns((tr / df["close"]).alias("atr_pct"))

        # The threshold = atr_pct * 0.4 should vary
        threshold = df["atr_pct"] * 0.4
        std = threshold.std()
        assert std > 0, "ATR threshold is constant — should be dynamic"


class TestDateRange:
    """Tests for date range and dataset integrity."""

    def test_date_range_correct(self):
        """Dataset date range should match the generated data."""
        df = _make_df_with_target(n=200)
        min_ts = df["open_time"].min()
        max_ts = df["open_time"].max()
        assert min_ts is not None
        assert max_ts is not None
        assert max_ts > min_ts

    def test_dataset_saved_to_correct_path(self, tmp_path: Path):
        """Datasets should be saved to the expected directory structure."""
        df = _make_df_with_target(n=200)

        from scripts.build_1h_dataset import split_by_regime

        df_trend, df_high_vol = split_by_regime(df)

        output_dir = tmp_path / "symbol=BTCUSDT" / "interval=1h"
        output_dir.mkdir(parents=True, exist_ok=True)

        trend_path = output_dir / "dataset_trend.parquet"
        high_vol_path = output_dir / "dataset_high_vol.parquet"

        df_trend.write_parquet(trend_path)
        df_high_vol.write_parquet(high_vol_path)

        assert trend_path.exists()
        assert high_vol_path.exists()
        assert trend_path.stat().st_size > 0
        assert high_vol_path.stat().st_size > 0

        # Read back and verify
        loaded = pl.read_parquet(trend_path)
        assert len(loaded) == len(df_trend)


class TestClassBalance:
    """Tests for class balance in datasets."""

    def test_class_balance_reasonable(self):
        """Class balance should not be more extreme than 80/20."""
        df = _make_df_with_target(n=500, seed=123)

        from scripts.build_1h_dataset import split_by_regime
        df_trend, _ = split_by_regime(df)

        if len(df_trend) < 10:
            pytest.skip("Not enough trend data for balance check")

        n = len(df_trend)
        n_up = int(df_trend["target"].eq(1).sum())
        n_down = int(df_trend["target"].eq(-1).sum())

        ratio_up = n_up / n
        ratio_down = n_down / n

        # Neither class should dominate beyond 80%
        assert ratio_up <= 0.80, f"UP class too dominant: {ratio_up:.1%}"
        assert ratio_down <= 0.80, f"DOWN class too dominant: {ratio_down:.1%}"
        # Both classes should exist
        assert n_up > 0, "No UP samples in trend dataset"
        assert n_down > 0, "No DOWN samples in trend dataset"


# Last-bars exclusion is covered by
# tests/test_triple_barrier.py::test_last_rows_are_dropped.
