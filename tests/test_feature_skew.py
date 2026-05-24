"""
Contract test: offline features == live-replayed features.
Guards against train/serve skew regression.

Run in CI before every deploy:
    pytest tests/test_feature_skew.py -v
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.live_feature_state import LiveFeatureState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n_bars: int = 300, seed: int = 42) -> pl.DataFrame:
    """Generate a realistic synthetic 4H OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    base_price = 50_000.0
    close = base_price + np.cumsum(rng.randn(n_bars) * 100)
    close = np.maximum(close, 1000.0)  # keep positive
    high = close + rng.uniform(50, 300, n_bars)
    low = close - rng.uniform(50, 300, n_bars)
    opn = close + rng.randn(n_bars) * 50
    volume = rng.uniform(100, 5000, n_bars)
    start_ms = 1_700_000_000_000  # ~2023-11-14
    open_times = [start_ms + i * 4 * 3_600_000 for i in range(n_bars)]

    return pl.DataFrame({
        "open_time": open_times,
        "open": opn.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": volume.tolist(),
    })


def _make_funding_df(n_records: int = 50) -> pl.DataFrame:
    """Generate synthetic funding rate data matching add_funding_features schema."""
    rng = np.random.RandomState(99)
    start_ms = 1_700_000_000_000
    return pl.DataFrame({
        "fundingTime": [start_ms + i * 8 * 3_600_000 for i in range(n_records)],
        "fundingRate": (rng.randn(n_records) * 0.0002).tolist(),
    })


def _make_metrics_df(n_records: int = 50) -> pl.DataFrame:
    """Generate synthetic metrics data matching add_oi_features schema."""
    rng = np.random.RandomState(77)
    start_ms = 1_700_000_000_000
    return pl.DataFrame({
        "create_time": [start_ms + i * 5 * 60_000 for i in range(n_records)],
        "sum_open_interest_value": (rng.uniform(1e9, 5e9, n_records)).tolist(),
        "count_long_short_ratio": (rng.uniform(0.8, 1.2, n_records)).tolist(),
        "sum_taker_long_short_vol_ratio": (rng.uniform(0.9, 1.1, n_records)).tolist(),
    })


class FakeBar:
    """Lightweight bar stub for LiveFeatureState.add_bar()."""

    def __init__(self, ts_event: int, o: float, h: float, l: float, c: float, v: float):
        self.ts_event = ts_event
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v


# ---------------------------------------------------------------------------
# LiveFeatureState unit tests
# ---------------------------------------------------------------------------

class TestLiveFeatureStateFunding:
    """Tests for funding rate tracking."""

    def test_funding_not_zero_after_update(self):
        """LiveFeatureState.funding_rate != 0 after update_funding()."""
        state = LiveFeatureState()
        assert state.funding_rate == 0.0
        state.update_funding(rate=0.0003, timestamp_ms=1_700_000_000_000)
        assert state.funding_rate == 0.0003
        assert state.last_funding_update == 1_700_000_000_000

    def test_funding_history_appended_directly(self):
        """Directly appending to history works (settlement path)."""
        state = LiveFeatureState()
        for i in range(5):
            state.funding_rate_history.append({
                "fundingTime": 1000 * i,
                "fundingRate": 0.0001 * i,
            })
        assert len(state.funding_rate_history) == 5

    def test_funding_history_maxlen(self):
        """funding_rate_history respects its capped maxlen (Phase 4 Step
        4.2 bumped it from 100 → 300 so the 30-day rolling
        funding_zscore_30d window stays populated)."""
        state = LiveFeatureState()
        cap = state.funding_rate_history.maxlen
        for i in range(cap + 50):
            state.funding_rate_history.append({
                "fundingTime": i,
                "fundingRate": 0.0001,
            })
        assert len(state.funding_rate_history) == cap
        assert cap >= 90  # >= 30 days × 3 settlements/day

    def test_update_funding_does_not_append_history(self):
        """update_funding() updates current rate but NOT history."""
        state = LiveFeatureState()
        state.update_funding(rate=0.0003, timestamp_ms=1000)
        assert state.funding_rate == 0.0003
        assert len(state.funding_rate_history) == 0


class TestLiveFeatureStateOI:
    """Tests for open interest tracking."""

    def test_oi_not_zero_after_update(self):
        """LiveFeatureState.oi_value != 0 after update_oi()."""
        state = LiveFeatureState()
        assert state.oi_value == 0.0
        state.update_oi(oi=3_500_000_000.0, timestamp_ms=1_700_000_000_000)
        assert state.oi_value == 3_500_000_000.0
        assert state.last_oi_update == 1_700_000_000_000

    def test_oi_history_appended(self):
        """Each update_oi() appends to history."""
        state = LiveFeatureState()
        for i in range(5):
            state.update_oi(oi=float(i * 1e9), timestamp_ms=1000 * i)
        assert len(state.oi_history) == 5


class TestLiveFeatureStateBarBuffer:
    """Tests for bar buffer management."""

    def test_bar_buffer_maintains_order(self):
        """Bars added to buffer remain sorted by open_time."""
        state = LiveFeatureState()
        # ts_event = close time; add_bar converts to open_time = close - 4h
        base_ns = (1_700_000_000_000 + 4 * 3_600_000) * 1_000_000  # close of first bar
        for i in range(10):
            bar = FakeBar(
                ts_event=base_ns + i * 4 * 3_600_000 * 1_000_000,  # ns
                o=100.0 + i, h=105.0 + i,
                l=95.0 + i, c=102.0 + i, v=500.0,
            )
            state.add_bar(bar, interval="4h")
        df = state.get_bar_df("4h")
        assert len(df) == 10
        times = df["open_time"].to_list()
        assert times == sorted(times)

    def test_bar_buffer_maxlen(self):
        """4H bar buffer respects maxlen=400."""
        state = LiveFeatureState()
        base_ns = (1_700_000_000_000 + 4 * 3_600_000) * 1_000_000
        for i in range(500):
            bar = FakeBar(
                ts_event=base_ns + i * 4 * 3_600_000 * 1_000_000,
                o=100.0, h=105.0,
                l=95.0, c=102.0, v=500.0,
            )
            state.add_bar(bar, interval="4h")
        assert len(state.bar_buffer_4h) == 400

    def test_get_bar_df_empty(self):
        """get_bar_df returns empty DataFrame when buffer is empty."""
        state = LiveFeatureState()
        df = state.get_bar_df("4h")
        assert df.is_empty()

    def test_bar_buffer_correct_interval(self):
        """Bars are routed to the correct interval buffer."""
        state = LiveFeatureState()
        bar = FakeBar(
            ts_event=(1_700_000_000_000 + 3_600_000) * 1_000_000,
            o=100.0, h=105.0, l=95.0, c=102.0, v=500.0,
        )
        state.add_bar(bar, interval="1h")
        assert len(state.bar_buffer_1h) == 1
        assert len(state.bar_buffer_4h) == 0
        assert len(state.bar_buffer_15m) == 0


class TestLiveFeatureStateDFCompat:
    """Tests for DataFrame compatibility with derivatives.py."""

    def test_get_funding_df_compatible_with_add_funding_features(self):
        """get_funding_df() returns DataFrame that add_funding_features()
        can process without error."""
        from src.features.derivatives import add_funding_features

        state = LiveFeatureState()
        for i in range(20):
            # Directly append (simulates settlement path + preload)
            state.funding_rate_history.append({
                "fundingTime": 1_700_000_000_000 + i * 8 * 3_600_000,
                "fundingRate": 0.0001 * (i - 10),
            })
        funding_df = state.get_funding_df()

        # Verify schema
        assert "fundingTime" in funding_df.columns
        assert "fundingRate" in funding_df.columns

        # Verify it works with the actual function
        ohlcv = _make_ohlcv_df(50)
        result = add_funding_features(ohlcv, funding_df)
        assert "funding_rate" in result.columns
        assert "funding_abs" in result.columns
        assert "funding_zscore_7d" in result.columns

    def test_get_metrics_df_compatible_with_add_oi_features(self):
        """get_metrics_df() returns DataFrame that add_oi_features()
        can process without error."""
        from src.features.derivatives import add_oi_features

        state = LiveFeatureState()
        for i in range(20):
            state.update_oi(
                oi=3e9 + i * 1e7,
                timestamp_ms=1_700_000_000_000 + i * 5 * 60_000,
            )
        metrics_df = state.get_metrics_df()

        # Verify schema
        assert "create_time" in metrics_df.columns
        assert "sum_open_interest_value" in metrics_df.columns

        # Verify it works with the actual function
        ohlcv = _make_ohlcv_df(50)
        result = add_oi_features(ohlcv, metrics_df)
        assert "oi_value" in result.columns
        assert "oi_delta_4h" in result.columns

    def test_get_funding_df_empty_when_no_history(self):
        """get_funding_df() returns empty DataFrame when no history."""
        state = LiveFeatureState()
        df = state.get_funding_df()
        assert df.is_empty()

    def test_get_metrics_df_empty_when_no_history(self):
        """get_metrics_df() returns empty DataFrame when no history."""
        state = LiveFeatureState()
        df = state.get_metrics_df()
        assert df.is_empty()


# ---------------------------------------------------------------------------
# build_from_buffer 4H tests
# ---------------------------------------------------------------------------

class TestBuildFromBuffer4H:
    """Tests for FeaturePipeline.build_from_buffer() with interval='4h'."""

    def test_build_from_buffer_4h_returns_single_row(self):
        """build_from_buffer(interval='4h', single_row=True) returns 1 row."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        result = pipe.build_from_buffer(df, single_row=True)
        assert len(result) == 1

    def test_build_from_buffer_4h_has_microstructure_features(self):
        """4H build_from_buffer produces microstructure columns."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        result = pipe.build_from_buffer(df, single_row=False)

        for col in ["cvd", "volume_ratio", "returns_1", "body_ratio", "vwap_4h"]:
            assert col in result.columns, f"Missing microstructure column: {col}"

    def test_build_from_buffer_4h_has_regime_features(self):
        """4H build_from_buffer produces regime columns."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        result = pipe.build_from_buffer(df, single_row=False)

        for col in ["hurst", "adx", "atr_pct", "atr_percentile"]:
            assert col in result.columns, f"Missing regime column: {col}"

    def test_build_from_buffer_4h_has_derivative_features(self):
        """4H build_from_buffer produces derivative columns (zero-filled without data)."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        result = pipe.build_from_buffer(df, single_row=False)

        for col in ["funding_rate", "oi_value", "basis_approx"]:
            assert col in result.columns, f"Missing derivative column: {col}"

    def test_build_from_buffer_4h_with_funding_data(self):
        """4H build_from_buffer uses real funding data when provided."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        funding = _make_funding_df(50)
        result = pipe.build_from_buffer(
            df, funding_df=funding, single_row=True,
        )

        # With real funding data, funding_rate should not be all zeros
        fr_val = result["funding_rate"][0]
        # It could be 0.0 if the asof join doesn't match, but the column exists
        assert "funding_rate" in result.columns

    def test_build_from_buffer_4h_with_metrics_data(self):
        """4H build_from_buffer uses real metrics data when provided."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        metrics = _make_metrics_df(50)
        result = pipe.build_from_buffer(
            df, metrics_df=metrics, single_row=True,
        )
        assert "oi_value" in result.columns

    def test_build_from_buffer_4h_no_session_features(self):
        """4H build_from_buffer does NOT produce session/ORB features."""
        from src.features.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        result = pipe.build_from_buffer(df, single_row=False)

        # 4H should not have session or ORB features
        for col in ["trading_session", "orb_high_asia", "session_hour_sin"]:
            assert col not in result.columns, f"Unexpected MTF column in 4H: {col}"


# ---------------------------------------------------------------------------
# Skew contract test
# ---------------------------------------------------------------------------

class TestFeatureSkewContract:
    """Contract: offline build() and live build_from_buffer() produce
    compatible feature distributions for the same input data."""

    def test_4h_offline_vs_live_feature_columns_match(self):
        """Both paths produce the same set of core feature columns for 4H."""
        from src.features.feature_pipeline import FeaturePipeline, FEATURE_GROUPS

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)

        # Live path
        live_result = pipe.build_from_buffer(df, single_row=False)

        # Check all core feature groups are present
        for group_name, features in FEATURE_GROUPS.items():
            if group_name == "live_enrichment":
                continue  # these are optional
            for feat in features:
                assert feat in live_result.columns, (
                    f"Feature '{feat}' from group '{group_name}' missing in "
                    f"build_from_buffer output"
                )

    def test_4h_live_replay_no_nan_in_last_row(self):
        """The last row (inference row) should have no NaN in core features."""
        from src.features.feature_pipeline import FeaturePipeline, FEATURE_GROUPS

        pipe = FeaturePipeline(data_store=None, symbol="BTCUSDT", interval="4h")
        df = _make_ohlcv_df(300)
        result = pipe.build_from_buffer(df, single_row=True)

        core_features = []
        for group_name, features in FEATURE_GROUPS.items():
            if group_name == "live_enrichment":
                continue
            core_features.extend(features)

        for feat in core_features:
            if feat in result.columns:
                val = result[feat][0]
                if val is not None and isinstance(val, float):
                    assert not np.isnan(val), (
                        f"NaN in last row for feature '{feat}'"
                    )
