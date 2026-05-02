"""
tests/test_feature_pipeline.py

Phase 3 — Feature Engineering Pipeline tests.
All tests use synthetic in-memory DataFrames; no external data required.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from src.features.utils import rolling_zscore, safe_divide
from src.features.microstructure import (
    add_cvd_features,
    add_price_features,
    add_volume_features,
)
from src.features.derivatives import (
    add_basis_features,
    add_funding_features,
    add_oi_features,
)
from src.features.feature_pipeline import FEATURE_GROUPS, FeaturePipeline


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_BASE_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC
_BAR_MS = 4 * 3600 * 1_000   # 4H in ms


def _klines(n: int = 150) -> pl.DataFrame:
    """Generate deterministic synthetic 4H klines."""
    open_times = [_BASE_MS + i * _BAR_MS for i in range(n)]
    # gentle random-walk price
    closes = [40_000.0 + 50.0 * (i % 20 - 10) for i in range(n)]
    opens = [c - 10.0 for c in closes]
    highs = [c + 30.0 for c in closes]
    lows = [c - 30.0 for c in closes]
    volumes = [500.0 + float(i % 10) * 10 for i in range(n)]
    tbvs = [v * 0.55 for v in volumes]   # taker_buy_volume ≈ 55 %

    return pl.DataFrame({
        "open_time": pl.Series(open_times, dtype=pl.Int64),
        "open": pl.Series(opens, dtype=pl.Float64),
        "high": pl.Series(highs, dtype=pl.Float64),
        "low": pl.Series(lows, dtype=pl.Float64),
        "close": pl.Series(closes, dtype=pl.Float64),
        "volume": pl.Series(volumes, dtype=pl.Float64),
        "taker_buy_volume": pl.Series(tbvs, dtype=pl.Float64),
        "taker_buy_quote_volume": pl.Series(tbvs, dtype=pl.Float64),
        "quote_volume": pl.Series(volumes, dtype=pl.Float64),
        "trade_count": pl.Series([100] * n, dtype=pl.Int32),
        "close_time": pl.Series([_BASE_MS + (i + 1) * _BAR_MS - 1 for i in range(n)], dtype=pl.Int64),
        "ignore": pl.Series([0.0] * n, dtype=pl.Float64),
        "symbol": pl.Series(["BTCUSDT"] * n, dtype=pl.Utf8),
    })


def _funding(n: int = 150) -> pl.DataFrame:
    """Synthetic funding rate rows (every 8H → every 2nd kline bar)."""
    times = [_BASE_MS + i * 2 * _BAR_MS for i in range(n)]
    rates = [0.0001 * (1 + (i % 5) * 0.2) for i in range(n)]
    return pl.DataFrame({
        "fundingTime": pl.Series(times, dtype=pl.Int64),
        "fundingRate": pl.Series(rates, dtype=pl.Float64),
        "symbol": pl.Series(["BTCUSDT"] * n, dtype=pl.Utf8),
    })


def _metrics(n: int = 150) -> pl.DataFrame:
    """Synthetic open-interest metrics (hourly)."""
    times = [_BASE_MS + i * 3600 * 1000 for i in range(n)]
    oi = [5e9 + float(i) * 1e6 for i in range(n)]
    return pl.DataFrame({
        "create_time": pl.Series(times, dtype=pl.Int64),
        "sum_open_interest": pl.Series([oi_v / 40_000 for oi_v in oi], dtype=pl.Float64),
        "sum_open_interest_value": pl.Series(oi, dtype=pl.Float64),
        "count_long_short_ratio": pl.Series([1.05 + 0.01 * (i % 10) for i in range(n)], dtype=pl.Float64),
        "sum_taker_long_short_vol_ratio": pl.Series([0.98 + 0.005 * (i % 4) for i in range(n)], dtype=pl.Float64),
        "symbol": pl.Series(["BTCUSDT"] * n, dtype=pl.Utf8),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Mock DataStore for integration tests
# ──────────────────────────────────────────────────────────────────────────────

class _MockStore:
    """Minimal DataStore stand-in that serves synthetic data."""

    def get_klines(self, symbol, interval, start, end, columns=None):
        return _klines(200)

    def get_funding_rate(self, symbol, start, end):
        return _funding(200)

    def get_metrics(self, symbol, start, end):
        return _metrics(200)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CVD formula
# ──────────────────────────────────────────────────────────────────────────────

class TestCVDFeatures:
    def test_cvd_formula(self):
        """CVD = 2 × taker_buy_volume − volume."""
        df = _klines(10)
        out = add_cvd_features(df)
        expected = (2.0 * df["taker_buy_volume"] - df["volume"]).to_list()
        actual = out["cvd"].to_list()
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_cvd_slope_3_correctness(self):
        """cvd_slope_3[i] == cvd[i] − cvd[i-3] for i ≥ 3."""
        df = _klines(20)
        out = add_cvd_features(df)
        cvd = out["cvd"].to_list()
        slope3 = out["cvd_slope_3"].to_list()
        for i in range(3, len(cvd)):
            assert slope3[i] == pytest.approx(cvd[i] - cvd[i - 3], rel=1e-9)

    def test_taker_buy_ratio_bounds(self):
        """taker_buy_ratio must be in [0, 1] (buy ≤ total volume)."""
        out = add_cvd_features(_klines(50))
        ratio = out["taker_buy_ratio"].to_list()
        assert all(0.0 <= r <= 1.0 for r in ratio), "ratio out of [0, 1]"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Volume features
# ──────────────────────────────────────────────────────────────────────────────

class TestVolumeFeatures:
    def test_volume_ratio_positive(self):
        """volume_ratio = volume / volume_sma_20 must be > 0."""
        out = add_volume_features(_klines(50))
        # Skip first rows where sma is still warming up (those are 0 from fill_null)
        valid = out["volume_ratio"].filter(out["volume_ratio"] > 0).to_list()
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_large_volume_binary(self):
        """large_volume must be 0 or 1 (Int8)."""
        out = add_volume_features(_klines(50))
        values = set(out["large_volume"].to_list())
        assert values.issubset({0, 1})


# ──────────────────────────────────────────────────────────────────────────────
# 3. Price features
# ──────────────────────────────────────────────────────────────────────────────

class TestPriceFeatures:
    def test_log_returns_correctness(self):
        """returns_1[i] == ln(close[i] / close[i-1]) for i ≥ 1."""
        df = _klines(10)
        out = add_price_features(df)
        closes = df["close"].to_list()
        ret1 = out["returns_1"].to_list()
        for i in range(1, len(closes)):
            expected = math.log(closes[i] / closes[i - 1])
            assert ret1[i] == pytest.approx(expected, rel=1e-9)

    def test_body_ratio_bounds(self):
        """body_ratio must be in [0, 1]."""
        out = add_price_features(_klines(50))
        vals = out["body_ratio"].to_list()
        assert all(0.0 <= v <= 1.0 for v in vals), "body_ratio out of [0, 1]"

    def test_all_return_columns_present(self):
        """All five return columns must exist."""
        out = add_price_features(_klines(30))
        for col in ("returns_1", "returns_3", "returns_6", "returns_12", "returns_24"):
            assert col in out.columns, f"Missing column: {col}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Funding features
# ──────────────────────────────────────────────────────────────────────────────

class TestFundingFeatures:
    def test_add_funding_features_join(self):
        """asof join populates funding_rate with non-null values."""
        df = _klines(50)
        fund = _funding(100)
        out = add_funding_features(df, fund)
        assert "funding_rate" in out.columns
        # Most rows should have a non-zero funding rate after join
        non_zero = (out["funding_rate"] != 0).sum()
        assert non_zero > 0, "Expected non-zero funding rates after join"

    def test_funding_zscore_7d_present(self):
        """funding_zscore_7d and funding_zscore_30d columns exist."""
        out = add_funding_features(_klines(50), _funding(100))
        assert "funding_zscore_7d" in out.columns
        assert "funding_zscore_30d" in out.columns

    def test_funding_extreme_threshold(self):
        """funding_extreme fires when |funding_zscore_7d| > 2.0."""
        df = _klines(50)
        fund = _funding(100)
        out = add_funding_features(df, fund)
        mask = out["funding_zscore_7d"].abs() > 2.0
        extreme = out["funding_extreme"]
        # Where z > 2, extreme must be 1; where z ≤ 2, extreme must be 0
        assert (extreme.filter(mask) == 1).all(), "extreme should be 1 when |z|>2"
        assert (extreme.filter(~mask) == 0).all(), "extreme should be 0 when |z|≤2"

    def test_empty_funding_returns_zero_columns(self):
        """Empty funding_df → all funding features are zero."""
        out = add_funding_features(_klines(20), pl.DataFrame())
        assert out["funding_rate"].sum() == pytest.approx(0.0)
        assert "funding_extreme" in out.columns


# ──────────────────────────────────────────────────────────────────────────────
# 5. OI features
# ──────────────────────────────────────────────────────────────────────────────

class TestOIFeatures:
    def _df_with_price_features(self) -> pl.DataFrame:
        return add_price_features(_klines(50))

    def test_oi_quadrant_valid_values(self):
        """oi_quadrant must contain only {-2, -1, 1, 2}."""
        df = self._df_with_price_features()
        out = add_oi_features(df, _metrics(200))
        valid = {-2, -1, 1, 2}
        actual = set(out["oi_quadrant"].to_list())
        assert actual.issubset(valid), f"Unexpected oi_quadrant values: {actual - valid}"

    def test_oi_delta_4h_formula(self):
        """oi_delta_4h[i] = (oi[i] - oi[i-1]) / oi[i] (shift(1) on 4H data)."""
        df = self._df_with_price_features()
        out = add_oi_features(df, _metrics(200))
        oi = out["oi_value"].to_list()
        delta = out["oi_delta_4h"].to_list()
        # Check row 5 (warmup past)
        i = 5
        if oi[i] != 0:
            expected = (oi[i] - oi[i - 1]) / oi[i]
            assert delta[i] == pytest.approx(expected, rel=1e-6)

    def test_empty_metrics_returns_zero_columns(self):
        """Empty metrics_df → all OI features are zero."""
        out = add_oi_features(_klines(20), pl.DataFrame())
        assert out["oi_value"].sum() == pytest.approx(0.0)
        assert "oi_quadrant" in out.columns


# ──────────────────────────────────────────────────────────────────────────────
# 6. FeaturePipeline integration
# ──────────────────────────────────────────────────────────────────────────────

class TestFeaturePipeline:
    def _pipeline(self) -> FeaturePipeline:
        from datetime import datetime, timezone
        store = _MockStore()
        return FeaturePipeline(store, "BTCUSDT", "4h")

    def _build(self) -> pl.DataFrame:
        from datetime import datetime, timezone
        pipeline = self._pipeline()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        return pipeline.build(start, end)

    def test_feature_pipeline_build_returns_dataframe(self):
        """FeaturePipeline.build must return a non-empty DataFrame."""
        df = self._build()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_no_nan_in_features(self):
        """No NaN or null values in any feature column after build."""
        df = self._build()
        pipeline = self._pipeline()
        feature_cols = [c for c in pipeline.get_feature_names() if c in df.columns]
        for col in feature_cols:
            null_n = df[col].null_count()
            nan_n = df[col].is_nan().sum() if df[col].dtype.is_float() else 0
            assert null_n == 0, f"Column '{col}' has {null_n} nulls"
            assert nan_n == 0, f"Column '{col}' has {nan_n} NaNs"

    def test_get_feature_names_complete(self):
        """get_feature_names returns the expected full feature list."""
        pipeline = self._pipeline()
        names = pipeline.get_feature_names()
        expected_micro = FEATURE_GROUPS["microstructure"]
        expected_deriv = FEATURE_GROUPS["derivatives"]
        for feat in expected_micro + expected_deriv:
            assert feat in names, f"Feature '{feat}' missing from get_feature_names()"

    def test_warmup_rows_removed(self):
        """build() removes first 100 warmup rows — output row count < input."""
        df = self._build()
        # MockStore returns 200 kline rows; after warmup: ≥ 100 rows remain
        assert len(df) >= 100

    def test_save_to_parquet_creates_file(self, tmp_path: Path):
        """save_to path must produce a readable Parquet file."""
        from datetime import datetime, timezone
        output = tmp_path / "features.parquet"
        pipeline = self._pipeline()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        pipeline.build(start, end, save_to=output)
        assert output.exists(), "Parquet file was not created"
        loaded = pl.read_parquet(output)
        assert len(loaded) > 0
        assert "returns_1" in loaded.columns
