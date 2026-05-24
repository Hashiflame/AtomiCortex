"""Tests for Step H2 — TF-aware rolling windows in derivatives features.

Verifies that ``add_funding_features`` and ``add_oi_features`` correctly
scale their rolling/shift windows from ``bar_duration_minutes`` so the
named horizons (24h / 7d / 30d / 4h / 12h) hold across timeframes.
"""
from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from src.features.derivatives import (
    _bars_for,
    add_funding_features,
    add_oi_features,
)


# ---------------------------------------------------------------------------
# _bars_for: direct unit tests on the scaling helper
# ---------------------------------------------------------------------------


class TestBarsFor:
    @pytest.mark.parametrize("bar_min,expected", [
        (240, 180),  # 4H
        (60,  720),  # 1H
        (15,  2880),  # 15m
        (5,   8640),  # 5m
        (1,   43200),  # 1m
    ])
    def test_30d(self, bar_min, expected):
        assert _bars_for(30 * 24 * 60, bar_min) == expected

    @pytest.mark.parametrize("bar_min,expected", [
        (240, 42),
        (60,  168),
        (15,  672),
    ])
    def test_7d(self, bar_min, expected):
        assert _bars_for(7 * 24 * 60, bar_min) == expected

    @pytest.mark.parametrize("bar_min,expected", [
        (240, 6),
        (60,  24),
        (15,  96),
    ])
    def test_24h(self, bar_min, expected):
        assert _bars_for(24 * 60, bar_min) == expected

    @pytest.mark.parametrize("bar_min,expected", [
        (240, 1),
        (60,  4),
        (15,  16),
    ])
    def test_4h(self, bar_min, expected):
        assert _bars_for(4 * 60, bar_min) == expected

    @pytest.mark.parametrize("bar_min,expected", [
        (240, 3),
        (60,  12),
        (15,  48),
    ])
    def test_12h(self, bar_min, expected):
        assert _bars_for(12 * 60, bar_min) == expected

    def test_floor_at_one(self):
        """Sub-bar horizon never collapses to 0."""
        assert _bars_for(30, 240) == 1  # 30min horizon on 4H bars

    def test_bad_bar_duration(self):
        """Defensive: bar_duration_minutes ≤ 0 clamps to 1."""
        assert _bars_for(60, 0) == 60
        assert _bars_for(60, -5) == 60


# ---------------------------------------------------------------------------
# add_funding_features: bar_duration_minutes wiring
# ---------------------------------------------------------------------------

def _make_klines(n: int, *, bar_min: int) -> pl.DataFrame:
    bar_ms = bar_min * 60_000
    start_ms = 1_700_000_000_000
    return pl.DataFrame({
        "open_time": [start_ms + i * bar_ms for i in range(n)],
        "open":  [100.0 + i * 0.1 for i in range(n)],
        "high":  [101.0 + i * 0.1 for i in range(n)],
        "low":   [ 99.0 + i * 0.1 for i in range(n)],
        "close": [100.5 + i * 0.1 for i in range(n)],
        "volume":[1000.0] * n,
        "taker_buy_volume": [500.0] * n,
    })


def _make_funding(n: int, *, bar_min: int) -> pl.DataFrame:
    bar_ms = bar_min * 60_000
    start_ms = 1_700_000_000_000
    # One funding rate per bar — overshoots reality but is enough for
    # asof-joining; the rolling-window math is what we're testing.
    return pl.DataFrame({
        "fundingTime": [start_ms + i * bar_ms for i in range(n)],
        "fundingRate": [0.0001 * ((i % 17) - 8) for i in range(n)],
    })


class TestAddFundingFeaturesTFAware:
    def test_default_kwarg_matches_4h_behavior(self):
        """No bar_duration_minutes → identical output to explicit 240."""
        kl = _make_klines(300, bar_min=240)
        fd = _make_funding(300, bar_min=240)
        out_default = add_funding_features(kl, fd)
        out_4h = add_funding_features(kl, fd, bar_duration_minutes=240)
        for col in ("funding_zscore_7d", "funding_zscore_30d",
                    "funding_cum_24h"):
            assert out_default[col].to_list() == out_4h[col].to_list(), col

    def test_15m_uses_2880_window_for_30d_zscore(self):
        """rolling_zscore for funding_zscore_30d called with window=2880 on 15m."""
        kl = _make_klines(100, bar_min=15)
        fd = _make_funding(100, bar_min=15)
        with patch(
            "src.features.derivatives.rolling_zscore", wraps=__import__(
                "src.features.utils", fromlist=["rolling_zscore"],
            ).rolling_zscore,
        ) as spy:
            add_funding_features(kl, fd, bar_duration_minutes=15)
        windows = [c.args[1] for c in spy.call_args_list]
        assert 2880 in windows, f"expected 30d-on-15m window 2880, got {windows}"
        assert 672 in windows, f"expected 7d-on-15m window 672, got {windows}"

    def test_1h_uses_720_window_for_30d_zscore(self):
        kl = _make_klines(100, bar_min=60)
        fd = _make_funding(100, bar_min=60)
        with patch(
            "src.features.derivatives.rolling_zscore", wraps=__import__(
                "src.features.utils", fromlist=["rolling_zscore"],
            ).rolling_zscore,
        ) as spy:
            add_funding_features(kl, fd, bar_duration_minutes=60)
        windows = [c.args[1] for c in spy.call_args_list]
        assert 720 in windows
        assert 168 in windows

    def test_funding_cum_24h_window_scales(self):
        """funding_cum_24h rolling_sum window equals bars_24h for each TF."""
        # 15m: 96 bars sum to 24h.
        kl = _make_klines(150, bar_min=15)
        # Make funding_rate non-trivial AND constant so the sum is bars*rate.
        fd = pl.DataFrame({
            "fundingTime": kl["open_time"].to_list(),
            "fundingRate": [1.0] * 150,
        })
        out = add_funding_features(kl, fd, bar_duration_minutes=15)
        # By bar index 95 the rolling_sum(window=96) over a constant 1.0
        # series is exactly 96.0.
        cum = out.sort("open_time")["funding_cum_24h"].to_list()
        assert cum[95] == pytest.approx(96.0)
        assert cum[149] == pytest.approx(96.0)


# ---------------------------------------------------------------------------
# add_oi_features: shift / zscore windows TF-aware
# ---------------------------------------------------------------------------

def _make_metrics(n: int, *, bar_min: int, oi_series) -> pl.DataFrame:
    bar_ms = bar_min * 60_000
    start_ms = 1_700_000_000_000
    return pl.DataFrame({
        "create_time": [start_ms + i * bar_ms for i in range(n)],
        "sum_open_interest_value": oi_series,
        "count_long_short_ratio": [1.0] * n,
        "sum_taker_long_short_vol_ratio": [1.0] * n,
    })


class TestAddOiFeaturesTFAware:
    def test_default_matches_4h(self):
        n = 50
        kl = _make_klines(n, bar_min=240)
        oi = [1_000.0 + i * 10.0 for i in range(n)]
        m = _make_metrics(n, bar_min=240, oi_series=oi)
        out_default = add_oi_features(kl, m)
        out_4h = add_oi_features(kl, m, bar_duration_minutes=240)
        for col in ("oi_delta_4h", "oi_delta_12h", "oi_zscore"):
            assert out_default[col].to_list() == out_4h[col].to_list(), col

    def test_oi_delta_4h_uses_shift_16_on_15m(self):
        """On 15m, oi_delta_4h compares against the bar 16 positions back."""
        n = 50
        kl = _make_klines(n, bar_min=15)
        # Step OI up by 100 at index 20 so we can detect a non-zero delta
        # exactly bars_4h positions later (index 36 on 15m).
        oi = [1_000.0] * 20 + [1_100.0] * (n - 20)
        m = _make_metrics(n, bar_min=15, oi_series=oi)
        out = add_oi_features(kl, m, bar_duration_minutes=15).sort("open_time")
        d4 = out["oi_delta_4h"].to_list()
        # Indices 20..35 see oi=1100 with shift(16) still pointing at 1000 →
        # delta = 100/1000 = 0.1. At index 36, shift(16) = 1100 → delta = 0.
        for i in range(20, 36):
            assert d4[i] == pytest.approx(0.1), f"i={i} d4={d4[i]}"
        assert d4[36] == pytest.approx(0.0)
        # Sanity: with the 4H default this jump would surface at index 21,
        # not 20..35. Recompute with default to confirm divergence.
        out_wrong = add_oi_features(kl, m).sort("open_time")
        d4_wrong = out_wrong["oi_delta_4h"].to_list()
        assert d4_wrong[20] == pytest.approx(0.1)  # shift(1) sees the jump
        assert d4_wrong[21] == pytest.approx(0.0)

    def test_oi_delta_12h_uses_shift_48_on_15m(self):
        n = 80
        kl = _make_klines(n, bar_min=15)
        oi = [1_000.0] * 30 + [1_200.0] * (n - 30)
        m = _make_metrics(n, bar_min=15, oi_series=oi)
        out = add_oi_features(kl, m, bar_duration_minutes=15).sort("open_time")
        d12 = out["oi_delta_12h"].to_list()
        # 12h on 15m = 48 bars. shift(48) is null before i=48; jump at i=30
        # → first observable delta at i=48 (shift sees oi[0]=1000) and
        # delta stays 0.2 through i=77, then 0 at i=78 (shift catches up).
        assert d12[48] == pytest.approx(0.2)
        assert d12[77] == pytest.approx(0.2)
        assert d12[78] == pytest.approx(0.0)

    def test_oi_zscore_window_scales(self):
        """rolling_zscore for oi_zscore uses bars_30d per TF."""
        n = 50
        kl = _make_klines(n, bar_min=60)
        oi = [1_000.0 + i for i in range(n)]
        m = _make_metrics(n, bar_min=60, oi_series=oi)
        with patch(
            "src.features.derivatives.rolling_zscore", wraps=__import__(
                "src.features.utils", fromlist=["rolling_zscore"],
            ).rolling_zscore,
        ) as spy:
            add_oi_features(kl, m, bar_duration_minutes=60)
        windows = [c.args[1] for c in spy.call_args_list]
        assert 720 in windows  # 30d on 1H


# ---------------------------------------------------------------------------
# FeaturePipeline integration: correct bar_duration is forwarded
# ---------------------------------------------------------------------------


class TestFeaturePipelineWiring:
    @pytest.mark.parametrize("interval,expected_min", [
        ("4h", 240),
        ("1h", 60),
        ("15m", 15),
    ])
    def test_build_forwards_bar_duration(self, interval, expected_min):
        """FeaturePipeline.build() passes the correct bar_duration_minutes."""
        from unittest.mock import MagicMock

        from src.features.feature_pipeline import FeaturePipeline

        ds = MagicMock()
        ds.get_klines.return_value = _make_klines(300, bar_min=expected_min)
        ds.get_funding_rate.return_value = pl.DataFrame()
        ds.get_metrics.return_value = pl.DataFrame()

        fp = FeaturePipeline(data_store=ds, symbol="BTCUSDT", interval=interval)

        with patch(
            "src.features.feature_pipeline.add_funding_features",
            wraps=add_funding_features,
        ) as f_spy, patch(
            "src.features.feature_pipeline.add_oi_features",
            wraps=add_oi_features,
        ) as o_spy:
            from datetime import datetime, timezone
            fp.build(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )
        assert f_spy.call_args.kwargs["bar_duration_minutes"] == expected_min
        assert o_spy.call_args.kwargs["bar_duration_minutes"] == expected_min

    @pytest.mark.parametrize("interval,expected_min", [
        ("4h", 240),
        ("1h", 60),
        ("15m", 15),
    ])
    def test_build_from_buffer_forwards_bar_duration(self, interval, expected_min):
        from unittest.mock import MagicMock

        from src.features.feature_pipeline import FeaturePipeline

        fp = FeaturePipeline(
            data_store=MagicMock(), symbol="BTCUSDT", interval=interval,
        )
        df = _make_klines(300, bar_min=expected_min).with_columns(
            (pl.col("volume") * 0.5).alias("taker_buy_volume"),
        )

        with patch(
            "src.features.feature_pipeline.add_funding_features",
            wraps=add_funding_features,
        ) as f_spy, patch(
            "src.features.feature_pipeline.add_oi_features",
            wraps=add_oi_features,
        ) as o_spy:
            try:
                fp.build_from_buffer(df, single_row=False)
            except Exception:
                # build_from_buffer may fail later for non-4h intervals due
                # to missing HTF inputs; we only care that the derivative
                # calls received the correct kwarg.
                pass
        assert f_spy.call_args.kwargs["bar_duration_minutes"] == expected_min
        assert o_spy.call_args.kwargs["bar_duration_minutes"] == expected_min


# ---------------------------------------------------------------------------
# Backward-compatibility smoke: existing tests' column shape unchanged on 4H.
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_4h_columns_match_legacy_set(self):
        kl = _make_klines(300, bar_min=240)
        fd = _make_funding(300, bar_min=240)
        n = 300
        oi = [1_000.0 + i * 5.0 for i in range(n)]
        m = _make_metrics(n, bar_min=240, oi_series=oi)

        out = add_funding_features(kl, fd)
        out = add_oi_features(out, m)
        for col in (
            "funding_rate", "funding_abs", "funding_zscore_7d",
            "funding_zscore_30d", "funding_extreme", "funding_positive",
            "funding_cum_24h", "oi_value", "oi_delta_4h", "oi_delta_12h",
            "oi_zscore", "oi_quadrant", "ls_ratio", "ls_ratio_zscore",
            "taker_vol_ratio",
        ):
            assert col in out.columns, col
