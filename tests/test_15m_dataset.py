"""
tests/test_15m_dataset.py

Unit tests for the 15m ML dataset builder.

Tests target construction, lookahead prevention, session trap filter,
regime splits (trend + ORB), embargo, ORB features, MTF context,
feature integrity, and class balance.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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

def _make_15m_ohlcv(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Generate synthetic 15m OHLCV data with realistic structure."""
    rng = np.random.RandomState(seed)

    # Brownian motion price series
    returns = rng.normal(0.0001, 0.003, n)
    close = 50000.0 * np.exp(np.cumsum(returns))

    # Realistic OHLCV
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = np.abs(rng.normal(100, 30, n))

    # Timestamps: 15m apart, starting 2023-01-01 00:00 UTC
    base_ts = 1672531200000  # 2023-01-01 00:00:00 UTC in ms
    open_time = np.array([base_ts + i * 900_000 for i in range(n)], dtype=np.int64)

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
    n: int = 500,
    seed: int = 42,
    pt_multiplier: float = 2.0,
    sl_multiplier: float = 1.0,
    max_holding_bars: int = 8,
    add_orb_features: bool = True,
    add_session_trap: bool = True,
) -> pl.DataFrame:
    """Create a DataFrame with atr_pct, regime, target, and optional ORB columns.

    Target is built with triple-barrier (AFML Ch.3), mirroring
    build_15m_dataset.main(): drop vertical (0) labels, rename
    label → target. Defaults match MLStrategyConfig15M.tb_* (2.0/1.0/8).
    """
    df = _make_15m_ohlcv(n, seed)

    # Add atr_pct (simple TR-based)
    tr = df["high"] - df["low"]
    atr_pct = tr / df["close"]
    df = df.with_columns(atr_pct.alias("atr_pct"))

    # Add regime column (mock)
    regimes = ["trend_up", "trend_down", "high_vol", "range", "unknown"]
    rng = np.random.RandomState(seed)
    regime_arr = rng.choice(regimes, n, p=[0.3, 0.2, 0.15, 0.25, 0.1])
    df = df.with_columns(pl.Series("regime", regime_arr, dtype=pl.Utf8))

    # Add ORB features (mock)
    if add_orb_features:
        rng2 = np.random.RandomState(seed + 1)
        # ORB breakout ~10% of bars
        bull_mask = rng2.random(n) < 0.05
        bear_mask = rng2.random(n) < 0.05
        df = df.with_columns([
            pl.Series("orb_breakout_bull", bull_mask),
            pl.Series("orb_breakout_bear", bear_mask),
            pl.Series("orb_high_asia", np.full(n, 50500.0)),
            pl.Series("orb_low_asia", np.full(n, 49500.0)),
            pl.Series("orb_range_asia", np.full(n, 1000.0)),
            pl.Series("orb_range_asia_atr_pct", np.full(n, 0.5)),
            pl.Series("orb_high_london", np.full(n, 50600.0)),
            pl.Series("orb_low_london", np.full(n, 49400.0)),
            pl.Series("orb_range_london", np.full(n, 1200.0)),
            pl.Series("orb_range_london_atr_pct", np.full(n, 0.6)),
            pl.Series("orb_high_ny", np.full(n, 50700.0)),
            pl.Series("orb_low_ny", np.full(n, 49300.0)),
            pl.Series("orb_range_ny", np.full(n, 1400.0)),
            pl.Series("orb_range_ny_atr_pct", np.full(n, 0.7)),
        ])

    # Add session trap (mock — mirrors ORBDetector._add_session_meta logic)
    if add_session_trap:
        # Derive hour from open_time to create realistic trap zones
        ts = pl.from_epoch(pl.col("open_time"), time_unit="ms")
        df = df.with_columns(ts.dt.hour().alias("_hour"))

        # Session assignment: Asia=1 (0-7), London=2 (8-12), NY=3 (13-21)
        session = (
            pl.when(pl.col("_hour") < 8).then(1)
              .when(pl.col("_hour") < 13).then(2)
              .otherwise(3)
        )
        df = df.with_columns(session.alias("_session"))

        # Session start hour (matching ORBDetector)
        session_start = (
            pl.when(pl.col("_session") == 1).then(0)
              .when(pl.col("_session") == 2).then(8)
              .when(pl.col("_session") == 3).then(13)
              .otherwise(0)
        )
        # Approximate bars since session open (4 bars/hour)
        bars_since = ((pl.col("_hour") - session_start) * 4).cast(pl.Int32).clip(0, 200)
        df = df.with_columns(bars_since.alias("bars_since_session_open"))

        # Session length in bars: Asia=32, London=20, NY=36
        session_len = (
            pl.when(pl.col("_session") == 1).then(32)
              .when(pl.col("_session") == 2).then(20)
              .when(pl.col("_session") == 3).then(36)
              .otherwise(20)
        )
        bars_to_end = session_len - pl.col("bars_since_session_open")

        # Trap zone: first 2 OR last 2 bars of session (same as ORBDetector)
        is_trap = (
            (pl.col("bars_since_session_open") <= 2)
            | (bars_to_end <= 2)
        )
        df = df.with_columns(is_trap.alias("is_session_trap_zone"))
        df = df.drop(["_hour", "_session"])

    # Add MTF context features (mock)
    df = df.with_columns([
        pl.lit(1).cast(pl.Int32).alias("mtf_3tf_alignment"),
        pl.lit(0).cast(pl.Int32).alias("htf_1h_trend_dir"),
        pl.lit(0).cast(pl.Int32).alias("htf_4h_trend_dir"),
    ])

    # Create target — triple-barrier, same as build_15m_dataset.main()
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
    domain, 15m preset 2.0/1.0/8) are covered in
    tests/test_triple_barrier.py. Here we only assert config wiring
    and dataset-builder integration (vertical labels dropped).
    """

    def test_atr_threshold_0_35_not_0_4(self):
        """15m uses ATR threshold multiplier 0.35, not 0.4 like 1H."""
        from src.configs.strategy_15m import MLStrategyConfig15M
        cfg = MLStrategyConfig15M()
        assert cfg.atr_threshold_multiplier == 0.35, (
            f"Expected 0.35, got {cfg.atr_threshold_multiplier}"
        )

    def test_flat_excluded_from_training(self):
        """FLAT (target=0) bars should be excluded from all datasets."""
        df = _make_df_with_target(n=500)

        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        df_trend, df_orb = split_by_regime(df_clean)

        # No FLAT targets in either dataset
        assert (df_trend["target"] == 0).sum() == 0, "FLAT rows found in trend dataset"
        assert (df_orb["target"] == 0).sum() == 0, "FLAT rows found in orb dataset"


class TestSessionTrap:
    """Tests for session trap zone filtering."""

    def test_session_trap_first_bars_excluded(self):
        """Session trap zones must be excluded from training data."""
        df = _make_df_with_target(n=500)

        # Verify some rows are marked as trap
        n_trap = df["is_session_trap_zone"].sum()
        assert n_trap > 0, "No session trap zones found in test data"

        from scripts.build_15m_dataset import filter_session_trap
        df_filtered, n_excluded = filter_session_trap(df)

        # Filtered should have fewer rows
        assert len(df_filtered) < len(df), "Session trap filter didn't remove any rows"
        assert n_excluded == n_trap, (
            f"Expected {n_trap} excluded, got {n_excluded}"
        )

    def test_non_trap_bars_included(self):
        """Bars NOT in session trap zone must be preserved."""
        df = _make_df_with_target(n=500)

        from scripts.build_15m_dataset import filter_session_trap
        df_filtered, _ = filter_session_trap(df)

        # All remaining rows should have is_session_trap_zone == False
        if "is_session_trap_zone" in df_filtered.columns:
            trap_in_filtered = df_filtered["is_session_trap_zone"].sum()
            assert trap_in_filtered == 0, (
                f"Found {trap_in_filtered} trap rows in filtered dataset"
            )

    def test_no_session_trap_in_any_dataset(self):
        """Neither trend nor orb dataset should contain session trap rows."""
        df = _make_df_with_target(n=500)

        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        df_trend, df_orb = split_by_regime(df_clean)

        for name, ds in [("trend", df_trend), ("orb", df_orb)]:
            if "is_session_trap_zone" in ds.columns and len(ds) > 0:
                n_trap = ds["is_session_trap_zone"].sum()
                assert n_trap == 0, (
                    f"Found {n_trap} session trap rows in {name} dataset"
                )


class TestRegimeSplit:
    """Tests for regime-based dataset splitting."""

    def test_regime_split_correct(self):
        """Datasets are split correctly."""
        df = _make_df_with_target(n=500)
        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        df_trend, df_orb = split_by_regime(df_clean)

        assert len(df_trend) > 0, "Trend dataset is empty"
        assert len(df_orb) > 0, "ORB dataset is empty"

    def test_trend_dataset_no_orb_only_rows(self):
        """Trend dataset contains only trend_up/trend_down regime rows."""
        df = _make_df_with_target(n=500)
        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        df_trend, _ = split_by_regime(df_clean)

        if "regime" in df_trend.columns and len(df_trend) > 0:
            regime_values = set(df_trend["regime"].unique().to_list())
            assert regime_values.issubset({"trend_up", "trend_down"}), (
                f"Found unexpected regimes in trend dataset: {regime_values}"
            )

    def test_orb_dataset_only_breakout_rows(self):
        """ORB dataset must contain ONLY rows with breakout signals."""
        df = _make_df_with_target(n=500)
        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        _, df_orb = split_by_regime(df_clean)

        if len(df_orb) > 0:
            has_bull = "orb_breakout_bull" in df_orb.columns
            has_bear = "orb_breakout_bear" in df_orb.columns

            if has_bull and has_bear:
                has_breakout = (
                    df_orb["orb_breakout_bull"] | df_orb["orb_breakout_bear"]
                )
                assert has_breakout.all(), (
                    f"Found {(~has_breakout).sum()} non-breakout rows in ORB dataset"
                )

    def test_orb_dataset_has_breakout_only(self):
        """No non-breakout rows in ORB dataset — redundant check with explicit logic."""
        df = _make_df_with_target(n=1000, seed=123)
        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        _, df_orb = split_by_regime(df_clean)

        if len(df_orb) > 0 and "orb_breakout_bull" in df_orb.columns:
            # Every row must have at least one breakout flag True
            for i in range(len(df_orb)):
                bull = df_orb["orb_breakout_bull"][i]
                bear = df_orb["orb_breakout_bear"][i]
                assert bull or bear, (
                    f"Row {i} in ORB dataset has no breakout signal"
                )


class TestORBFeatures:
    """Tests for ORB feature presence."""

    def test_orb_features_in_dataset(self):
        """ORB features must be present in the dataset."""
        df = _make_df_with_target(n=300)
        expected = [
            "orb_high_asia", "orb_low_asia", "orb_range_asia",
            "orb_high_london", "orb_low_london",
            "orb_high_ny", "orb_low_ny",
            "orb_breakout_bull", "orb_breakout_bear",
        ]
        for col in expected:
            assert col in df.columns, f"Missing ORB feature: {col}"

    def test_mtf_context_3tf_in_dataset(self):
        """MTF 3-TF alignment feature must be present."""
        df = _make_df_with_target(n=300)
        assert "mtf_3tf_alignment" in df.columns, "Missing mtf_3tf_alignment"


class TestWalkForward:
    """Tests for walk-forward and embargo mechanics."""

    def test_embargo_16_bars_applied(self):
        """PurgedKFoldCV with embargo must create a gap between train/test."""
        from src.execution.walk_forward import PurgedKFoldCV

        n = 1000
        df = _make_df_with_target(n=n)
        embargo_pct = 0.02
        cv = PurgedKFoldCV(n_splits=3, embargo_pct=embargo_pct)

        # embargo_pct=0.02 on 1000 rows → ~20 rows gap; each row = 15min
        expected_min_gap_bars = max(int(n * embargo_pct) - 2, 1)  # allow ±2 tolerance
        min_gap_ms = expected_min_gap_bars * 900_000

        for train_df, test_df in cv.split(df):
            train_end_time = train_df["open_time"][-1]
            test_start_time = test_df["open_time"][0]
            gap_ms = test_start_time - train_end_time
            gap_bars = gap_ms / 900_000
            assert gap_ms > min_gap_ms, (
                f"Embargo gap too small: {gap_bars:.1f} bars "
                f"(need > {expected_min_gap_bars} bars for embargo_pct={embargo_pct})"
            )

    def test_wf_windows_no_overlap(self):
        """Walk-forward train/test windows must not overlap."""
        from src.execution.walk_forward import WalkForwardValidator

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        wf = WalkForwardValidator(train_months=10, test_months=3, step_months=2)

        for (ts, te), (vs, ve) in wf.split(start, end):
            assert te <= vs, (
                f"Train/test overlap: train ends {te}, test starts {vs}"
            )


class TestFeatures:
    """Tests for feature integrity."""

    def test_no_nan_in_features(self):
        """Feature columns should not contain NaN after target creation."""
        df = _make_df_with_target(n=500)

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
        df = _make_df_with_target(n=300)
        assert df["target"].null_count() == 0, "Target has null values"

    def test_atr_threshold_dynamic_not_fixed(self):
        """ATR threshold must vary across rows (dynamic, not a fixed constant)."""
        df = _make_15m_ohlcv(300, seed=99)
        tr = df["high"] - df["low"]
        df = df.with_columns((tr / df["close"]).alias("atr_pct"))

        # The threshold = atr_pct * 0.35 should vary
        threshold = df["atr_pct"] * 0.35
        std = threshold.std()
        assert std > 0, "ATR threshold is constant — should be dynamic"


class TestClassBalance:
    """Tests for class balance in datasets."""

    def test_class_balance_reasonable(self):
        """Class balance should not be more extreme than 80/20."""
        df = _make_df_with_target(n=1000, seed=123)

        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        df_trend, _ = split_by_regime(df_clean)

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
        assert n_up > 0, "No UP samples in trend dataset"
        assert n_down > 0, "No DOWN samples in trend dataset"


class TestDatasetIO:
    """Tests for dataset I/O."""

    def test_dataset_saved_correctly(self, tmp_path: Path):
        """Datasets should be saved to the expected directory structure."""
        df = _make_df_with_target(n=500)

        from scripts.build_15m_dataset import filter_session_trap, split_by_regime
        df_clean, _ = filter_session_trap(df)
        df_trend, df_orb = split_by_regime(df_clean)

        output_dir = tmp_path / "symbol=BTCUSDT" / "interval=15m"
        output_dir.mkdir(parents=True, exist_ok=True)

        trend_path = output_dir / "dataset_trend.parquet"
        orb_path = output_dir / "dataset_orb.parquet"

        df_trend.write_parquet(trend_path)
        df_orb.write_parquet(orb_path)

        assert trend_path.exists()
        assert orb_path.exists()
        assert trend_path.stat().st_size > 0
        assert orb_path.stat().st_size > 0

        # Read back and verify
        loaded = pl.read_parquet(trend_path)
        assert len(loaded) == len(df_trend)

    def test_date_range_correct(self):
        """Dataset date range should match the generated data."""
        df = _make_df_with_target(n=300)
        min_ts = df["open_time"].min()
        max_ts = df["open_time"].max()
        assert min_ts is not None
        assert max_ts is not None
        assert max_ts > min_ts


# Last-bars exclusion is covered by
# tests/test_triple_barrier.py::test_last_rows_are_dropped.
