"""
tests/test_data_quality.py

Unit tests for DataQualityChecker.
All tests use tmp_path and synthetic Parquet fixtures — no live data required.
"""

from __future__ import annotations

from datetime import date, timezone, datetime
from pathlib import Path

import polars as pl
import pytest

from src.ingestion.data_quality import (
    DataQualityChecker,
    _month_range,
    _parse_month,
    row_passes,
)


# ---------------------------------------------------------------------------
# Parquet fixture helpers
# ---------------------------------------------------------------------------

_4H_MS = 4 * 3_600_000   # 14 400 000 ms
_8H_MS = 8 * 3_600_000   # 28 800 000 ms
_BASE_TS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC


def _klines_dir(base: Path, symbol: str, date_str: str) -> Path:
    p = base / "exchange=BINANCE_UM" / f"symbol={symbol}" / "klines_4h" / f"date={date_str}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _funding_dir(base: Path, symbol: str, month_str: str) -> Path:
    p = base / "exchange=BINANCE_UM" / f"symbol={symbol}" / "funding_rate" / f"date={month_str}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _agg_dir(base: Path, symbol: str, date_str: str) -> Path:
    p = base / "exchange=BINANCE_UM" / f"symbol={symbol}" / "agg_trades" / f"date={date_str}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_klines(directory: Path, open_time_start: int, n_bars: int = 6, valid: bool = True) -> None:
    """Write a synthetic klines_4h parquet with n_bars rows."""
    rows = []
    for i in range(n_bars):
        ts = open_time_start + i * _4H_MS
        h = 43_000.0
        l = 41_000.0 if valid else 45_000.0  # high < low when invalid
        rows.append({
            "open_time":              ts,
            "open":                   42_000.0,
            "high":                   h,
            "low":                    l,
            "close":                  42_500.0,
            "volume":                 100.0,
            "close_time":             ts + _4H_MS - 1,
            "quote_volume":           4_250_000.0,
            "trade_count":            1_000,
            "taker_buy_volume":       50.0,
            "taker_buy_quote_volume": 2_125_000.0,
            "ignore":                 0.0,
            "datetime":               datetime.fromtimestamp(ts / 1_000, tz=timezone.utc).replace(tzinfo=None),
            "symbol":                 "BTCUSDT",
        })
    df = pl.DataFrame(rows).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    df.write_parquet(directory / "part-0.parquet")


def _write_funding(directory: Path, start_ts: int, n_records: int = 3, rate: float = 0.0001) -> None:
    rows = [
        {
            "fundingTime": start_ts + i * _8H_MS,
            "fundingRate": rate,
            "markPrice":   42_000.0,
            "symbol":      "BTCUSDT",
            "datetime":    datetime.fromtimestamp((start_ts + i * _8H_MS) / 1_000, tz=timezone.utc).replace(tzinfo=None),
        }
        for i in range(n_records)
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    df.write_parquet(directory / "part-0.parquet")


def _write_agg_trades(directory: Path, start_ts: int, n_rows: int = 1_200, monotonic: bool = True) -> None:
    ts_list = [start_ts + i * 1_000 for i in range(n_rows)]
    if not monotonic:
        # swap two adjacent timestamps near the start to break monotonicity
        ts_list[5], ts_list[6] = ts_list[6], ts_list[5]
    rows = {
        "agg_trade_id":   list(range(n_rows)),
        "price":          [42_000.0] * n_rows,
        "quantity":       [0.1] * n_rows,
        "first_trade_id": list(range(n_rows)),
        "last_trade_id":  list(range(n_rows)),
        "transact_time":  ts_list,
        "is_buyer_maker": [True] * n_rows,
        "datetime":       [
            datetime.fromtimestamp(t / 1_000, tz=timezone.utc).replace(tzinfo=None)
            for t in ts_list
        ],
        "symbol":         ["BTCUSDT"] * n_rows,
    }
    df = pl.DataFrame(rows).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    df.write_parquet(directory / "part-0.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_range_test() -> None:
    result = _month_range(date(2024, 11, 1), date(2025, 2, 1))
    assert result == {"2024-11", "2024-12", "2025-01", "2025-02"}


def test_parse_month() -> None:
    assert _parse_month("2024-01") == date(2024, 1, 1)
    assert _parse_month("2025-12") == date(2025, 12, 1)


def test_month_range_single() -> None:
    assert _month_range(date(2024, 3, 1), date(2024, 3, 1)) == {"2024-03"}


def test_month_range_year_boundary() -> None:
    result = _month_range(date(2024, 11, 1), date(2025, 2, 1))
    assert result == {"2024-11", "2024-12", "2025-01", "2025-02"}


# ---------------------------------------------------------------------------
# Test 1: check_completeness — 100 % (all daily dirs present)
# ---------------------------------------------------------------------------

def test_completeness_full(tmp_path: Path) -> None:
    """Three consecutive days of klines → completeness = 100 %."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    for d in dates:
        d_dir = _klines_dir(tmp_path, "BTCUSDT", d)
        _write_klines(d_dir, _BASE_TS)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_completeness("BTCUSDT", "klines_4h")
    checker.close()

    assert result["expected"] == 3
    assert result["found"] == 3
    assert result["missing_dates"] == []
    assert result["completeness_pct"] == 100.0


# ---------------------------------------------------------------------------
# Test 2: check_completeness — missing date detected
# ---------------------------------------------------------------------------

def test_completeness_missing_day(tmp_path: Path) -> None:
    """Middle day is absent → it appears in missing_dates, pct < 100."""
    for d in ["2024-01-01", "2024-01-03"]:   # gap: 2024-01-02
        d_dir = _klines_dir(tmp_path, "BTCUSDT", d)
        _write_klines(d_dir, _BASE_TS)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_completeness("BTCUSDT", "klines_4h")
    checker.close()

    assert result["expected"] == 3
    assert result["found"] == 2
    assert "2024-01-02" in result["missing_dates"]
    assert result["completeness_pct"] < 100.0


# ---------------------------------------------------------------------------
# Test 3: check_completeness — monthly (funding_rate), 100 %
# ---------------------------------------------------------------------------

def test_completeness_monthly_full(tmp_path: Path) -> None:
    """Two consecutive months of funding_rate → completeness = 100 %."""
    for month in ["2024-01", "2024-02"]:
        m_dir = _funding_dir(tmp_path, "BTCUSDT", month)
        _write_funding(m_dir, _BASE_TS)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_completeness("BTCUSDT", "funding_rate")
    checker.close()

    assert result["expected"] == 2
    assert result["found"] == 2
    assert result["completeness_pct"] == 100.0


# ---------------------------------------------------------------------------
# Test 4: check_gaps — no gaps (consecutive 4h klines)
# ---------------------------------------------------------------------------

def test_gaps_no_gaps(tmp_path: Path) -> None:
    """Two consecutive daily files with perfect 4h cadence → 0 gaps."""
    for i, d in enumerate(["2024-01-01", "2024-01-02"]):
        d_dir = _klines_dir(tmp_path, "BTCUSDT", d)
        _write_klines(d_dir, _BASE_TS + i * 86_400_000)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_gaps("BTCUSDT", "klines_4h")
    checker.close()

    assert result["gap_count"] == 0
    assert result["gap_locations"] == []


# ---------------------------------------------------------------------------
# Test 5: check_gaps — gap detected when a whole day is missing
# ---------------------------------------------------------------------------

def test_gaps_with_missing_day(tmp_path: Path) -> None:
    """Day 2024-01-01 and 2024-01-03 present, day 2024-01-02 absent.
    Gap between last bar of day 1 (20:00) and first bar of day 3 (00:00) = 28h > 4h.
    """
    _write_klines(_klines_dir(tmp_path, "BTCUSDT", "2024-01-01"), _BASE_TS)
    # Day 3 starts at BASE_TS + 2 * 86_400_000
    _write_klines(_klines_dir(tmp_path, "BTCUSDT", "2024-01-03"), _BASE_TS + 2 * 86_400_000)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_gaps("BTCUSDT", "klines_4h")
    checker.close()

    assert result["gap_count"] >= 1
    assert result["largest_gap_ms"] > 4 * 3_600_000


# ---------------------------------------------------------------------------
# Test 6: check_data_integrity — valid klines pass
# ---------------------------------------------------------------------------

def test_integrity_valid_klines(tmp_path: Path) -> None:
    """Clean synthetic klines → null_count=0, anomaly_count=0, is_valid=True."""
    d_dir = _klines_dir(tmp_path, "BTCUSDT", "2024-01-01")
    _write_klines(d_dir, _BASE_TS, valid=True)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_data_integrity("BTCUSDT", "klines_4h")
    checker.close()

    assert result["null_count"] == 0
    assert result["anomaly_count"] == 0
    assert result["is_valid"] is True


# ---------------------------------------------------------------------------
# Test 7: check_data_integrity — anomaly detected (high < low)
# ---------------------------------------------------------------------------

def test_integrity_anomaly_klines(tmp_path: Path) -> None:
    """Klines with high < low → anomaly_count > 0, is_valid=False."""
    d_dir = _klines_dir(tmp_path, "BTCUSDT", "2024-01-01")
    _write_klines(d_dir, _BASE_TS, valid=False)  # high < low

    checker = DataQualityChecker(tmp_path)
    result = checker.check_data_integrity("BTCUSDT", "klines_4h")
    checker.close()

    assert result["anomaly_count"] > 0
    assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# Test 8: check_data_integrity — high funding rate flagged
# ---------------------------------------------------------------------------

def test_integrity_high_funding_rate(tmp_path: Path) -> None:
    """|fundingRate| >= 0.05 (5 %) is flagged as anomaly."""
    m_dir = _funding_dir(tmp_path, "BTCUSDT", "2024-01")
    _write_funding(m_dir, _BASE_TS, rate=0.06)  # 6 % >> 5 % threshold

    checker = DataQualityChecker(tmp_path)
    result = checker.check_data_integrity("BTCUSDT", "funding_rate")
    checker.close()

    assert result["anomaly_count"] > 0
    assert result["is_valid"] is False


# ---------------------------------------------------------------------------
# Test 9: check_clock_drift — monotonic agg_trades
# ---------------------------------------------------------------------------

def test_clock_drift_monotonic(tmp_path: Path) -> None:
    """Monotonically increasing transact_time → is_monotonic=True."""
    d_dir = _agg_dir(tmp_path, "BTCUSDT", "2024-01-01")
    _write_agg_trades(d_dir, _BASE_TS, n_rows=1_200, monotonic=True)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_clock_drift("BTCUSDT")
    checker.close()

    assert result["is_monotonic"] is True
    assert result["rows_sampled"] == 1_000   # capped at AGG_TRADES_SAMPLE


# ---------------------------------------------------------------------------
# Test 10: check_clock_drift — non-monotonic agg_trades detected
# ---------------------------------------------------------------------------

def test_clock_drift_non_monotonic(tmp_path: Path) -> None:
    """Two adjacent transact_time values swapped → is_monotonic=False."""
    d_dir = _agg_dir(tmp_path, "BTCUSDT", "2024-01-01")
    _write_agg_trades(d_dir, _BASE_TS, n_rows=200, monotonic=False)

    checker = DataQualityChecker(tmp_path)
    result = checker.check_clock_drift("BTCUSDT")
    checker.close()

    assert result["is_monotonic"] is False


# ---------------------------------------------------------------------------
# Test 11: check_gaps — skipped for agg_trades
# ---------------------------------------------------------------------------

def test_gaps_skipped_for_agg_trades(tmp_path: Path) -> None:
    """check_gaps for agg_trades returns skipped=True (not checked by design)."""
    checker = DataQualityChecker(tmp_path)
    result = checker.check_gaps("BTCUSDT", "agg_trades")
    checker.close()

    assert result.get("skipped") is True
    assert result["gap_count"] == 0


# ---------------------------------------------------------------------------
# Test 12: row_passes helper
# ---------------------------------------------------------------------------

def test_row_passes_all_good() -> None:
    ok_completeness = {"completeness_pct": 100.0, "missing_dates": []}
    ok_gaps         = {"gap_count": 0, "gap_locations": []}
    ok_integrity    = {"null_count": 0, "anomaly_count": 0, "is_valid": True}
    assert row_passes("BTCUSDT", "klines_4h", ok_completeness, ok_gaps, ok_integrity) is True


def test_row_passes_fails_on_gaps() -> None:
    ok_completeness = {"completeness_pct": 100.0}
    bad_gaps        = {"gap_count": 2, "gap_locations": [...]}
    ok_integrity    = {"null_count": 0, "anomaly_count": 0, "is_valid": True}
    assert row_passes("BTCUSDT", "klines_4h", ok_completeness, bad_gaps, ok_integrity) is False


def test_row_passes_fails_on_low_completeness() -> None:
    low_completeness = {"completeness_pct": 98.0}
    ok_gaps          = {"gap_count": 0}
    ok_integrity     = {"null_count": 0, "anomaly_count": 0, "is_valid": True}
    assert row_passes("BTCUSDT", "klines_4h", low_completeness, ok_gaps, ok_integrity) is False


def test_row_passes_fails_on_non_monotonic() -> None:
    """Non-monotonic timestamps are a hard failure; large drift_ms_max is not."""
    ok_completeness  = {"completeness_pct": 100.0}
    skipped_gaps     = {"gap_count": 0, "skipped": True}
    ok_integrity     = {"null_count": 0, "anomaly_count": 0, "is_valid": True}
    non_monotonic    = {"is_monotonic": False, "drift_ms_max": 5}
    assert row_passes(
        "BTCUSDT", "agg_trades",
        ok_completeness, skipped_gaps, ok_integrity, non_monotonic,
    ) is False


def test_row_passes_large_drift_ms_is_informational() -> None:
    """drift_ms_max for stored data = max gap between trades, not clock skew.
    It does NOT cause a pass/fail failure; only is_monotonic=False does.
    """
    ok_completeness = {"completeness_pct": 100.0}
    skipped_gaps    = {"gap_count": 0, "skipped": True}
    ok_integrity    = {"null_count": 0, "anomaly_count": 0, "is_valid": True}
    large_drift     = {"is_monotonic": True, "drift_ms_max": 6_000}  # 6s gap between trades
    assert row_passes(
        "BTCUSDT", "agg_trades",
        ok_completeness, skipped_gaps, ok_integrity, large_drift,
    ) is True


# ---------------------------------------------------------------------------
# Test 13: empty store returns graceful errors, no exceptions
# ---------------------------------------------------------------------------

def test_empty_store_no_crash(tmp_path: Path) -> None:
    """All checks return error dicts (not exceptions) for a non-existent symbol."""
    checker = DataQualityChecker(tmp_path)

    c = checker.check_completeness("NODATA", "klines_4h")
    g = checker.check_gaps("NODATA", "klines_4h")
    i = checker.check_data_integrity("NODATA", "klines_4h")
    d = checker.check_clock_drift("NODATA")
    checker.close()

    assert c["completeness_pct"] == 0.0
    assert "error" in c
    assert g["gap_count"] == 0
    assert i["null_count"] == 0
    assert "error" in d
