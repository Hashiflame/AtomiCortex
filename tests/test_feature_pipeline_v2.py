"""
tests/test_feature_pipeline_v2.py

Tests for the extended FeaturePipeline (Phase 2).
Verifies backward compatibility with 4H and proper MTF feature addition.

Run:
    pytest tests/test_feature_pipeline_v2.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from src.features.feature_pipeline import (
    FEATURE_GROUPS,
    FEATURE_GROUPS_MTF,
    FeaturePipeline,
)
from src.features.regime_detector import (
    RegimeDetector,
    RegimeDetector1H,
    RegimeDetector15M,
)

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_BASE_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC
_BAR_MS_4H = 4 * 3600 * 1000
_BAR_MS_1H = 3600 * 1000
_BAR_MS_15M = 900 * 1000


def _klines(n: int, bar_ms: int = _BAR_MS_4H) -> pl.DataFrame:
    return pl.DataFrame({
        "open_time": [_BASE_MS + i * bar_ms for i in range(n)],
        "open":  [40000.0 + 50.0 * (i % 20 - 10) - 10.0 for i in range(n)],
        "high":  [40000.0 + 50.0 * (i % 20 - 10) + 30.0 for i in range(n)],
        "low":   [40000.0 + 50.0 * (i % 20 - 10) - 30.0 for i in range(n)],
        "close": [40000.0 + 50.0 * (i % 20 - 10) for i in range(n)],
        "volume": [500.0 + float(i % 10) * 10 for i in range(n)],
        "taker_buy_volume": [275.0 + float(i % 10) * 5 for i in range(n)],
        "quote_volume": [500.0 + float(i % 10) * 10 for i in range(n)],
        "taker_buy_quote_volume": [275.0 + float(i % 10) * 5 for i in range(n)],
        "trade_count": [100] * n,
        "close_time": [_BASE_MS + (i + 1) * bar_ms - 1 for i in range(n)],
        "ignore": [0.0] * n,
        "symbol": ["BTCUSDT"] * n,
    })


def _funding(n: int = 150) -> pl.DataFrame:
    times = [_BASE_MS + i * 2 * _BAR_MS_4H for i in range(n)]
    return pl.DataFrame({
        "fundingTime": pl.Series(times, dtype=pl.Int64),
        "fundingRate": pl.Series([0.0001] * n, dtype=pl.Float64),
        "symbol": ["BTCUSDT"] * n,
    })


def _metrics(n: int = 150) -> pl.DataFrame:
    times = [_BASE_MS + i * 3600 * 1000 for i in range(n)]
    return pl.DataFrame({
        "create_time": pl.Series(times, dtype=pl.Int64),
        "sum_open_interest_value": [5e9] * n,
        "count_long_short_ratio": [1.05] * n,
        "sum_taker_long_short_vol_ratio": [0.98] * n,
        "symbol": ["BTCUSDT"] * n,
    })


class _MockStore:
    def __init__(self, n: int = 500, bar_ms: int = _BAR_MS_4H):
        self._n = n
        self._bar_ms = bar_ms

    def get_klines(self, symbol, interval, start, end, columns=None):
        return _klines(self._n, self._bar_ms)

    def get_funding_rate(self, symbol, start, end):
        return _funding(self._n)

    def get_metrics(self, symbol, start, end):
        return _metrics(self._n)


# ═══════════════════════════════════════════════════════════════
# 1. Backward Compatibility (4H)
# ═══════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    def test_4h_pipeline_unchanged(self) -> None:
        """CRITICAL: 4H pipeline must produce identical output."""
        store = _MockStore(500, _BAR_MS_4H)
        pipeline = FeaturePipeline(store, "BTCUSDT", "4h")
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        df = pipeline.build(start, end)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_4h_feature_names_unchanged(self) -> None:
        """get_feature_names for 4H should match original set."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "4h")
        names = pipeline.get_feature_names()
        original = [feat for group in FEATURE_GROUPS.values() for feat in group]
        assert names == original

    def test_session_features_not_added_for_4h(self) -> None:
        """build_mtf should be a no-op for 4H."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "4h")
        df = _klines(50)
        result = pipeline.build_mtf(df)
        # No session columns should be added.
        session_cols = FEATURE_GROUPS_MTF.get("session", [])
        for col in session_cols:
            assert col not in result.columns, f"4H should not have {col}"

    def test_4h_build_no_nan(self) -> None:
        """4H build should produce no NaN in feature columns."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "4h")
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        df = pipeline.build(start, end)
        feature_cols = [c for c in pipeline.get_feature_names() if c in df.columns]
        for col in feature_cols:
            null_n = df[col].null_count()
            assert null_n == 0, f"Column '{col}' has {null_n} nulls"


# ═══════════════════════════════════════════════════════════════
# 2. MTF Feature Addition
# ═══════════════════════════════════════════════════════════════


class TestMTFFeatures:
    def test_1h_pipeline_adds_session_features(self) -> None:
        """build_mtf for 1H should add session features."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "1h")
        df = _klines(200, _BAR_MS_1H)
        result = pipeline.build_mtf(df)
        assert "trading_session" in result.columns
        assert "session_vwap" in result.columns
        assert "hours_to_funding_mark" in result.columns

    def test_1h_pipeline_adds_mtf_context_when_provided(self) -> None:
        """build_mtf for 1H should add HTF context when 4H data given."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "1h")
        df = _klines(200, _BAR_MS_1H)
        df_4h = _klines(50, _BAR_MS_4H)
        # Add regime columns to 4H.
        det = RegimeDetector()
        df_4h = det.detect_all(df_4h)
        result = pipeline.build_mtf(df, df_htf_4h=df_4h)
        assert "htf_4h_regime" in result.columns
        assert "mtf_1h_4h_aligned" in result.columns

    def test_15m_pipeline_adds_orb_features(self) -> None:
        """build_mtf for 15m should add ORB features."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "15m")
        df = _klines(200, _BAR_MS_15M)
        result = pipeline.build_mtf(df)
        assert "orb_high_asia" in result.columns
        assert "orb_breakout_bull" in result.columns

    def test_15m_pipeline_without_htf_still_works(self) -> None:
        """15m pipeline should work without HTF data."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "15m")
        df = _klines(200, _BAR_MS_15M)
        result = pipeline.build_mtf(df)
        assert len(result) == 200
        assert "trading_session" in result.columns

    def test_1h_feature_names_include_session(self) -> None:
        """get_feature_names for 1H should include session features."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "1h")
        names = pipeline.get_feature_names()
        assert "trading_session" in names
        assert "session_vwap" in names
        assert "hours_to_funding_mark" in names

    def test_15m_feature_names_include_orb(self) -> None:
        """get_feature_names for 15m should include ORB features."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "15m")
        names = pipeline.get_feature_names()
        assert "orb_high_asia" in names
        assert "orb_breakout_bull" in names

    def test_5m_pipeline_adds_session_only(self) -> None:
        """5m should get session features but not ORB or MTF context."""
        pipeline = FeaturePipeline(_MockStore(), "BTCUSDT", "5m")
        df = _klines(200, 300_000)  # 5m = 300000ms
        result = pipeline.build_mtf(df)
        assert "trading_session" in result.columns
        assert "orb_high_asia" not in result.columns  # ORB is 15m only


# ═══════════════════════════════════════════════════════════════
# 3. Regime Detector Subclasses
# ═══════════════════════════════════════════════════════════════


class TestRegimeDetectorSubclasses:
    def test_detector_1h_faster_parameters(self) -> None:
        """RegimeDetector1H should have faster ADX period."""
        det = RegimeDetector1H()
        assert det.adx_period == 10
        assert det.hurst_window == 100
        assert det.atr_lookback == 168

    def test_detector_15m_fastest_parameters(self) -> None:
        """RegimeDetector15M should have fastest ADX period."""
        det = RegimeDetector15M()
        assert det.adx_period == 7
        assert det.hurst_window == 50
        assert det.atr_lookback == 672

    def test_detector_1h_detects_regime(self) -> None:
        """RegimeDetector1H should produce valid regime columns."""
        df = _klines(500, _BAR_MS_1H)
        det = RegimeDetector1H()
        out = det.detect_all(df, min_bars=100)
        assert "regime" in out.columns
        assert "adx" in out.columns

    def test_detector_1h_inherits_classify(self) -> None:
        """RegimeDetector1H inherits _classify from base."""
        det = RegimeDetector1H()
        assert hasattr(det, "_classify")
        assert hasattr(det, "detect")
        assert hasattr(det, "detect_all")


# ═══════════════════════════════════════════════════════════════
# 5. build_from_buffer — live inference parity with offline build
# ═══════════════════════════════════════════════════════════════


class TestBuildFromBuffer:
    """build_from_buffer must reproduce the offline 15m feature matrix
    (scripts/build_15m_dataset.build_feature_matrix) for the current bar.
    """

    def _frames(self):
        # Buffers long enough that offline's .slice(_WARMUP=200) keeps rows
        # and the last row's rolling features are fully warmed.
        df_15m = _klines(500, _BAR_MS_15M)
        df_1h = _klines(300, _BAR_MS_1H)
        df_4h = _klines(150, _BAR_MS_4H)
        return df_15m, df_1h, df_4h, _funding(500), _metrics(500)

    def test_build_from_buffer_matches_offline_for_15m(self) -> None:
        from scripts.build_15m_dataset import build_feature_matrix

        df_15m, df_1h, df_4h, funding, metrics = self._frames()

        offline = build_feature_matrix(
            df_15m.clone(), df_1h.clone(), df_4h.clone(), "BTCUSDT",
            funding_df=funding, metrics_df=metrics,
        )
        pipe = FeaturePipeline(None, "BTCUSDT", "15m")  # type: ignore[arg-type]
        buffered = pipe.build_from_buffer(
            df_15m.clone(),
            df_htf_4h=df_4h.clone(),
            df_htf_1h=df_1h.clone(),
            funding_df=funding,
            metrics_df=metrics,
            single_row=False,
        )

        assert not offline.is_empty(), "offline build returned no rows"
        # Compare the last (current) bar across every model feature column.
        names = [
            c for c in pipe.get_feature_names()
            if c in offline.columns and c in buffered.columns
        ]
        assert len(names) > 100, f"expected MTF+ORB feature set, got {len(names)}"

        off_last = offline.select(names).tail(1)
        buf_last = buffered.select(names).tail(1)
        for col in names:
            o = off_last[col][0]
            b = buf_last[col][0]
            if o is None or b is None:
                assert o == b, f"{col}: null mismatch ({o!r} vs {b!r})"
                continue
            if isinstance(o, float):
                import math
                if math.isnan(o) or math.isnan(b):
                    assert math.isnan(o) and math.isnan(b), f"{col}: {o} vs {b}"
                else:
                    assert abs(o - b) <= 1e-6 + 1e-6 * abs(o), (
                        f"{col}: offline={o} buffer={b}"
                    )
            else:
                assert o == b, f"{col}: offline={o!r} buffer={b!r}"

    def test_build_from_buffer_returns_single_row(self) -> None:
        df_15m, df_1h, df_4h, funding, metrics = self._frames()
        pipe = FeaturePipeline(None, "BTCUSDT", "15m")  # type: ignore[arg-type]
        out = pipe.build_from_buffer(
            df_15m, df_htf_4h=df_4h, df_htf_1h=df_1h,
            funding_df=funding, metrics_df=metrics,
        )  # single_row=True (default)
        assert out.height == 1, f"expected 1 row, got {out.height}"
        # The single row is the most recent bar in the buffer.
        assert out["open_time"][0] == df_15m["open_time"][-1]

    def test_build_from_buffer_accepts_4h(self) -> None:
        """Phase 6: build_from_buffer now supports '4h' interval."""
        pipe = FeaturePipeline(None, "BTCUSDT", "4h")  # type: ignore[arg-type]
        result = pipe.build_from_buffer(_klines(300, _BAR_MS_4H))
        assert result.height == 1  # single_row default
        assert "cvd" in result.columns
        assert "hurst" in result.columns
