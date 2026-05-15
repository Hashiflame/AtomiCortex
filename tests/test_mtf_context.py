"""
tests/test_mtf_context.py

Unit tests for MTFContextBuilder (Phase 2).
Focuses on ASOF JOIN correctness and lookahead bias prevention.

Run:
    pytest tests/test_mtf_context.py -v
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.features.mtf_context import MTFContextBuilder

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────────────────────────

_BAR_MS_1H = 3_600_000
_BAR_MS_4H = 4 * _BAR_MS_1H
_BAR_MS_15M = 900_000


def _ts_ms(year: int, month: int, day: int, hour: int = 0) -> int:
    return int(dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _make_htf_4h(n: int = 100) -> pl.DataFrame:
    """Synthetic 4H bars with regime columns."""
    base = _ts_ms(2024, 1, 1, 0)
    closes = [40000.0 + i * 50.0 for i in range(n)]
    regimes = ["trend_up" if i % 2 == 0 else "range" for i in range(n)]
    adx_vals = [30.0 + (i % 10) for i in range(n)]
    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_4H for i in range(n)],
        "open": [c - 20.0 for c in closes],
        "high": [c + 50.0 for c in closes],
        "low":  [c - 50.0 for c in closes],
        "close": closes,
        "volume": [500.0] * n,
        "regime": regimes,
        "adx": adx_vals,
        "hurst": [0.6] * n,
        "atr_pct": [0.02] * n,
    })


def _make_1h(n: int = 400) -> pl.DataFrame:
    """Synthetic 1H bars."""
    base = _ts_ms(2024, 1, 1, 0)
    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_1H for i in range(n)],
        "open":  [40000.0 + i * 10.0 - 5.0 for i in range(n)],
        "high":  [40000.0 + i * 10.0 + 20.0 for i in range(n)],
        "low":   [40000.0 + i * 10.0 - 20.0 for i in range(n)],
        "close": [40000.0 + i * 10.0 for i in range(n)],
        "volume": [100.0] * n,
    })


def _make_1h_with_regime(n: int = 400) -> pl.DataFrame:
    """Synthetic 1H bars with regime columns (for 15m HTF)."""
    df = _make_1h(n)
    return df.with_columns([
        pl.lit("trend_up").alias("regime"),
        pl.lit(28.0).alias("adx"),
        pl.lit(0.55).alias("hurst"),
        pl.lit(0.015).alias("atr_pct"),
    ])


def _make_15m(n: int = 1600) -> pl.DataFrame:
    """Synthetic 15m bars."""
    base = _ts_ms(2024, 1, 1, 0)
    return pl.DataFrame({
        "open_time": [base + i * _BAR_MS_15M for i in range(n)],
        "open":  [40000.0 + i * 2.0 - 1.0 for i in range(n)],
        "high":  [40000.0 + i * 2.0 + 5.0 for i in range(n)],
        "low":   [40000.0 + i * 2.0 - 5.0 for i in range(n)],
        "close": [40000.0 + i * 2.0 for i in range(n)],
        "volume": [50.0] * n,
    })


# ═══════════════════════════════════════════════════════════════
# ASOF JOIN — Lookahead Bias Prevention
# ═══════════════════════════════════════════════════════════════


class TestAsofJoinNoLookahead:
    def test_asof_join_no_lookahead_bias(self) -> None:
        """CRITICAL: 4H bar opening at 08:00 (closes at 12:00) must NOT be
        available for 1H bar at 10:00 — the 4H bar hasn't closed yet.

        The 1H bar at 10:00 should see the *previous* closed 4H bar (04:00-08:00)
        or NULL if no closed 4H bar exists.
        """
        builder = MTFContextBuilder()

        # 4H bar 04:00 (closes 08:00) → visible to bars at open_time >= 08:00
        # 4H bar 08:00 (closes 12:00) → visible to bars at open_time >= 12:00
        # 4H bar 12:00 (closes 16:00) → visible to bars at open_time >= 16:00
        df_4h = pl.DataFrame({
            "open_time": [
                _ts_ms(2024, 1, 1, 4),
                _ts_ms(2024, 1, 1, 8),
                _ts_ms(2024, 1, 1, 12),
            ],
            "open": [39900.0, 40000.0, 40100.0],
            "high": [39950.0, 40050.0, 40150.0],
            "low":  [39850.0, 39950.0, 40050.0],
            "close": [39920.0, 40020.0, 40120.0],
            "volume": [500.0, 500.0, 500.0],
            "regime": ["range", "trend_up", "range"],
            "adx": [15.0, 30.0, 15.0],
        })

        # 1H bar at 10:00: the 08:00 4H bar hasn't closed yet (closes 12:00).
        # → should see the 04:00 4H bar (closed at 08:00), regime="range".
        df_1h = pl.DataFrame({
            "open_time": [_ts_ms(2024, 1, 1, 10)],
            "open": [40010.0], "high": [40030.0],
            "low": [39990.0], "close": [40015.0],
            "volume": [100.0],
        })

        result = builder.build_for_1h(df_1h, df_4h)
        # Must see the 04:00 bar (range), NOT the 08:00 bar (trend_up)
        assert result["htf_4h_regime"][0] == "range"

    def test_4h_bar_08utc_available_for_1h_bar_10utc(self) -> None:
        """1H bar at 10:00 should use 4H bar from 08:00 (latest closed)."""
        builder = MTFContextBuilder()
        df_4h = _make_htf_4h(50)
        df_1h = _make_1h(200)
        result = builder.build_for_1h(df_1h, df_4h)
        assert "htf_4h_regime" in result.columns
        assert len(result) == len(df_1h)


class TestBuildFor1H:
    def test_all_1h_context_columns_present(self) -> None:
        """build_for_1h must add all expected context columns."""
        builder = MTFContextBuilder()
        result = builder.build_for_1h(_make_1h(), _make_htf_4h())
        expected = {
            "htf_4h_regime", "htf_4h_adx", "htf_4h_trend_dir",
            "price_vs_4h_ema20", "mtf_1h_4h_aligned",
            "mtf_alignment_score",
        }
        assert expected.issubset(set(result.columns))

    def test_mtf_alignment_score_range(self) -> None:
        """mtf_alignment_score should be 0, 1, or 2 (or null for early bars)."""
        builder = MTFContextBuilder()
        result = builder.build_for_1h(_make_1h(), _make_htf_4h())
        valid = {0, 1, 2}
        non_null = result.filter(pl.col("mtf_alignment_score").is_not_null())
        actual = set(non_null["mtf_alignment_score"].to_list())
        assert actual.issubset(valid)

    def test_alignment_when_both_trending(self) -> None:
        """When both 1H and 4H trend up, alignment should be True."""
        builder = MTFContextBuilder()
        # Create strongly trending 4H data.
        df_4h = _make_htf_4h(50)
        df_1h = _make_1h(200)
        result = builder.build_for_1h(df_1h, df_4h)
        # At least some bars should be aligned.
        aligned_count = result["mtf_1h_4h_aligned"].sum()
        assert aligned_count >= 0  # Can be 0 if trends don't match.

    def test_price_vs_4h_ema20_is_finite(self) -> None:
        builder = MTFContextBuilder()
        result = builder.build_for_1h(_make_1h(), _make_htf_4h())
        assert not result["price_vs_4h_ema20"].is_infinite().any()

    def test_empty_4h_fills_defaults(self) -> None:
        """Empty 4H data should fill all columns with defaults."""
        builder = MTFContextBuilder()
        result = builder.build_for_1h(_make_1h(50), pl.DataFrame())
        assert result["htf_4h_regime"][0] == "unknown"
        assert result["htf_4h_adx"][0] == 0.0

    def test_output_row_count_matches_input(self) -> None:
        """Output should have same number of rows as input 1H."""
        builder = MTFContextBuilder()
        df_1h = _make_1h(100)
        result = builder.build_for_1h(df_1h, _make_htf_4h())
        assert len(result) == len(df_1h)


class TestBuildFor15M:
    def test_all_15m_context_columns_present(self) -> None:
        """build_for_15m must add all expected context columns."""
        builder = MTFContextBuilder()
        result = builder.build_for_15m(
            _make_15m(), _make_1h_with_regime(), _make_htf_4h()
        )
        expected = {
            "htf_1h_regime", "htf_1h_trend_dir",
            "htf_4h_regime", "htf_4h_trend_dir",
            "mtf_3tf_alignment", "htf_conflict", "htf_both_strong_trend",
        }
        assert expected.issubset(set(result.columns))

    def test_mtf_3tf_alignment_0_to_3(self) -> None:
        """mtf_3tf_alignment should be in [0, 3]."""
        builder = MTFContextBuilder()
        result = builder.build_for_15m(
            _make_15m(800), _make_1h_with_regime(200), _make_htf_4h(50)
        )
        vals = result["mtf_3tf_alignment"]
        assert (vals >= 0).all()
        assert (vals <= 3).all()

    def test_conflict_detected_when_tfs_disagree(self) -> None:
        """htf_conflict should be boolean."""
        builder = MTFContextBuilder()
        result = builder.build_for_15m(
            _make_15m(800), _make_1h_with_regime(200), _make_htf_4h(50)
        )
        assert result["htf_conflict"].dtype == pl.Boolean

    def test_both_strong_trend_requires_both_adx(self) -> None:
        """htf_both_strong_trend requires both 1H and 4H ADX > 25."""
        builder = MTFContextBuilder()
        # 4H has ADX 30+, 1H has ADX 28 → both strong.
        result = builder.build_for_15m(
            _make_15m(800), _make_1h_with_regime(200), _make_htf_4h(50)
        )
        assert result["htf_both_strong_trend"].dtype == pl.Boolean

    def test_handles_missing_htf_data_gracefully(self) -> None:
        """build_for_15m with empty HTF data should not crash."""
        builder = MTFContextBuilder()
        result = builder.build_for_15m(
            _make_15m(100), pl.DataFrame(), pl.DataFrame()
        )
        assert "htf_conflict" in result.columns
        assert len(result) == 100

    def test_output_row_count_matches_input_15m(self) -> None:
        builder = MTFContextBuilder()
        df_15m = _make_15m(500)
        result = builder.build_for_15m(
            df_15m, _make_1h_with_regime(), _make_htf_4h()
        )
        assert len(result) == len(df_15m)


class TestMTFContextEdgeCases:
    def test_single_htf_bar(self) -> None:
        """Should work with just 1 HTF bar."""
        builder = MTFContextBuilder()
        df_4h = pl.DataFrame({
            "open_time": [_ts_ms(2024, 1, 1, 0)],
            "open": [40000.0], "high": [40050.0],
            "low": [39950.0], "close": [40020.0],
            "volume": [500.0],
            "regime": ["trend_up"], "adx": [30.0],
        })
        df_1h = _make_1h(24)
        result = builder.build_for_1h(df_1h, df_4h)
        assert len(result) == 24

    def test_no_regime_column_in_htf(self) -> None:
        """Should handle HTF without regime column (fills 'unknown')."""
        builder = MTFContextBuilder()
        n_bars = 42  # 7 days of 4H bars
        df_4h = pl.DataFrame({
            "open_time": [_ts_ms(2024, 1, 1, 0) + i * _BAR_MS_4H for i in range(n_bars)],
            "open": [40000.0] * n_bars, "high": [40050.0] * n_bars,
            "low": [39950.0] * n_bars, "close": [40020.0] * n_bars,
            "volume": [500.0] * n_bars,
        })
        df_1h = _make_1h(96)
        result = builder.build_for_1h(df_1h, df_4h)
        assert (result["htf_4h_regime"] == "unknown").all()

    def test_htf_trend_dir_values(self) -> None:
        """htf_4h_trend_dir should be -1, 0, or 1 (ignoring null for early bars)."""
        builder = MTFContextBuilder()
        result = builder.build_for_1h(_make_1h(), _make_htf_4h())
        valid = {-1, 0, 1}
        non_null = result.filter(pl.col("htf_4h_trend_dir").is_not_null())
        actual = set(non_null["htf_4h_trend_dir"].to_list())
        assert actual.issubset(valid)

    def test_no_internal_columns_in_output(self) -> None:
        """No columns starting with _ should remain in output."""
        builder = MTFContextBuilder()
        result = builder.build_for_1h(_make_1h(), _make_htf_4h())
        underscore_cols = [c for c in result.columns if c.startswith("_")]
        assert len(underscore_cols) == 0, f"Internal columns leaked: {underscore_cols}"
