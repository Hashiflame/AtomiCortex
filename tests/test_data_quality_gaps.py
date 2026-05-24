"""Tests for Step H14 — gap detection covers every kline timeframe.

Pre-H14: ``_GAP_THRESHOLD_MS`` only contained ``klines_4h`` and
``klines_1d``; ``klines_1h`` / ``klines_15m`` etc. silently fell through
to ``None`` and ``check_gaps`` reported ``skipped=True`` — so
maintenance-window gaps in 1H/15m training data went unflagged.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.ingestion.data_quality import (
    _GAP_THRESHOLD_MS,
    _KEY_COLS,
    _TIMESTAMP_COL,
    DataQualityChecker,
)


# ---------------------------------------------------------------------------
# Static config: thresholds exist for every supported kline TF
# ---------------------------------------------------------------------------


class TestThresholdsConfig:
    @pytest.mark.parametrize("tf,expected_ms", [
        ("klines_1m",  1 * 60_000),
        ("klines_5m",  5 * 60_000),
        ("klines_15m", 15 * 60_000),
        ("klines_1h",  1 * 3_600_000),
        ("klines_4h",  4 * 3_600_000),  # backward-compat: unchanged
        ("klines_1d",  24 * 3_600_000), # backward-compat: unchanged
    ])
    def test_threshold_for_tf(self, tf, expected_ms):
        assert _GAP_THRESHOLD_MS.get(tf) == expected_ms

    @pytest.mark.parametrize("tf", [
        "klines_1m", "klines_5m", "klines_15m", "klines_1h",
        "klines_4h", "klines_1d",
    ])
    def test_timestamp_col_for_tf(self, tf):
        assert _TIMESTAMP_COL[tf] == "open_time"

    @pytest.mark.parametrize("tf", [
        "klines_1m", "klines_5m", "klines_15m", "klines_1h",
        "klines_4h", "klines_1d",
    ])
    def test_key_cols_for_tf(self, tf):
        cols = _KEY_COLS[tf]
        for c in ("open_time", "open", "high", "low", "close", "volume"):
            assert c in cols


# ---------------------------------------------------------------------------
# End-to-end check_gaps on synthetic parquet fixtures
# ---------------------------------------------------------------------------


def _write_klines(
    data_dir: Path, symbol: str, data_type: str, open_times_ms: list[int],
) -> None:
    """Write a single parquet under data_dir/exchange=…/symbol=…/data_type/."""
    target_dir = (
        data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}"
        / data_type / "date=2026-01-01"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "open_time": open_times_ms,
        "open":   [100.0] * len(open_times_ms),
        "high":   [101.0] * len(open_times_ms),
        "low":    [ 99.0] * len(open_times_ms),
        "close":  [100.5] * len(open_times_ms),
        "volume": [1_000.0] * len(open_times_ms),
    })
    df.write_parquet(target_dir / "part-0.parquet")


def _continuous(bar_ms: int, n: int) -> list[int]:
    start = 1_700_000_000_000
    return [start + i * bar_ms for i in range(n)]


def _with_gap(bar_ms: int, n_before: int, gap_bars: int, n_after: int) -> list[int]:
    """A run of bars, then a gap of (gap_bars + 1) × bar_ms, then more bars."""
    start = 1_700_000_000_000
    before = [start + i * bar_ms for i in range(n_before)]
    # Skip `gap_bars` slots, then continue.
    after_start = start + (n_before + gap_bars) * bar_ms
    after = [after_start + i * bar_ms for i in range(n_after)]
    return before + after


class TestGapDetectionPerTimeframe:
    @pytest.mark.parametrize("tf,bar_ms", [
        ("klines_1h",  1 * 3_600_000),
        ("klines_15m", 15 * 60_000),
        ("klines_5m",  5 * 60_000),
        ("klines_4h",  4 * 3_600_000),
    ])
    def test_continuous_data_no_gaps(self, tmp_path, tf, bar_ms):
        _write_klines(tmp_path, "BTCUSDT", tf, _continuous(bar_ms, 50))
        out = DataQualityChecker(tmp_path).check_gaps("BTCUSDT", tf)
        assert out.get("skipped") is not True
        assert out["gap_count"] == 0
        assert out["largest_gap_ms"] == 0

    def test_one_missing_1h_bar_flagged(self, tmp_path):
        bar_ms = 1 * 3_600_000
        _write_klines(
            tmp_path, "BTCUSDT", "klines_1h",
            _with_gap(bar_ms, n_before=20, gap_bars=1, n_after=20),
        )
        out = DataQualityChecker(tmp_path).check_gaps("BTCUSDT", "klines_1h")
        assert out.get("skipped") is not True
        assert out["gap_count"] == 1
        # 1 missing 1h bar → diff = 2h, must exceed 1h threshold.
        assert out["largest_gap_ms"] == 2 * 3_600_000

    def test_one_missing_15m_bar_flagged(self, tmp_path):
        bar_ms = 15 * 60_000
        _write_klines(
            tmp_path, "BTCUSDT", "klines_15m",
            _with_gap(bar_ms, n_before=20, gap_bars=1, n_after=20),
        )
        out = DataQualityChecker(tmp_path).check_gaps("BTCUSDT", "klines_15m")
        assert out["gap_count"] == 1
        assert out["largest_gap_ms"] == 30 * 60_000

    def test_one_missing_5m_bar_flagged(self, tmp_path):
        bar_ms = 5 * 60_000
        _write_klines(
            tmp_path, "BTCUSDT", "klines_5m",
            _with_gap(bar_ms, n_before=20, gap_bars=1, n_after=20),
        )
        out = DataQualityChecker(tmp_path).check_gaps("BTCUSDT", "klines_5m")
        assert out["gap_count"] == 1
        assert out["largest_gap_ms"] == 10 * 60_000

    def test_multi_bar_gap_size_reported(self, tmp_path):
        """3-bar gap on 1H → diff = 4h."""
        bar_ms = 1 * 3_600_000
        _write_klines(
            tmp_path, "BTCUSDT", "klines_1h",
            _with_gap(bar_ms, n_before=10, gap_bars=3, n_after=10),
        )
        out = DataQualityChecker(tmp_path).check_gaps("BTCUSDT", "klines_1h")
        assert out["gap_count"] == 1
        assert out["largest_gap_ms"] == 4 * 3_600_000


# ---------------------------------------------------------------------------
# Backward compatibility: 4H behaviour preserved
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_4h_threshold_unchanged(self):
        assert _GAP_THRESHOLD_MS["klines_4h"] == 4 * 3_600_000

    def test_4h_one_missing_bar_flagged(self, tmp_path):
        bar_ms = 4 * 3_600_000
        _write_klines(
            tmp_path, "BTCUSDT", "klines_4h",
            _with_gap(bar_ms, n_before=10, gap_bars=1, n_after=10),
        )
        out = DataQualityChecker(tmp_path).check_gaps("BTCUSDT", "klines_4h")
        assert out["gap_count"] == 1
        assert out["largest_gap_ms"] == 8 * 3_600_000

    def test_metrics_still_skipped(self):
        """metrics stays skipped (too high a row count for gap scan)."""
        assert _GAP_THRESHOLD_MS["metrics"] is None

    def test_agg_trades_still_skipped(self):
        assert _GAP_THRESHOLD_MS["agg_trades"] is None
