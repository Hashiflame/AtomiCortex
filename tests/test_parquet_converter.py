"""
tests/test_parquet_converter.py

Unit + integration tests for ParquetConverter and DataStore.
All tests use tmp_path and synthetic CSV fixtures — no internet required.
"""

from __future__ import annotations

import textwrap
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from src.ingestion.parquet_converter import (
    ParquetConverter,
    extract_date_from_stem,
)
from src.ingestion.data_store import DataStore


# ---------------------------------------------------------------------------
# Fixtures — sample CSV factories
# ---------------------------------------------------------------------------

def _write_klines_csv(path: Path, rows: int = 6) -> Path:
    """Write a synthetic klines CSV (4h, one day = 6 rows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00 UTC in ms
    interval_ms = 4 * 3_600_000

    lines = [
        "open_time,open,high,low,close,volume,close_time,"
        "quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    ]
    for i in range(rows):
        open_ts = base_ts + i * interval_ms
        close_ts = open_ts + interval_ms - 1
        lines.append(
            f"{open_ts},42000.0,43000.0,41500.0,42500.0,100.0,"
            f"{close_ts},4250000.0,1000,50.0,2125000.0,0.0"
        )
    path.write_text("\n".join(lines))
    return path


def _write_metrics_csv(path: Path, rows: int = 3) -> Path:
    """Write a synthetic metrics CSV with string datetime timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        "count_long_short_ratio,sum_taker_long_short_vol_ratio"
    ]
    for i in range(rows):
        ts = f"2024-01-01 0{i}:05:00"
        lines.append(
            f"{ts},BTCUSDT,500.0,21000000.0,1.5,1.8,1.2,0.7"
        )
    path.write_text("\n".join(lines))
    return path


def _write_funding_csv(path: Path) -> Path:
    """Write a synthetic funding-rate CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent("""\
        fundingTime,fundingRate,markPrice,symbol
        1704067200000,0.0001,42000.0,BTCUSDT
        1704096000000,0.00012,42100.0,BTCUSDT
        1704124800000,0.000095,41900.0,BTCUSDT
    """)
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Test: extract_date_from_stem
# ---------------------------------------------------------------------------

def test_extract_date_from_stem() -> None:
    assert extract_date_from_stem("BTCUSDT-4h-2024-01-01") == "2024-01-01"
    assert extract_date_from_stem("BTCUSDT-fundingRate-2024-12-31") == "2024-12-31"
    assert extract_date_from_stem("ETHUSDT-metrics-2025-06-15") == "2025-06-15"


# ---------------------------------------------------------------------------
# Test: funding CSV conversion
# ---------------------------------------------------------------------------

def test_convert_funding_csv(tmp_path: Path) -> None:
    """Converting a funding CSV produces valid Parquet with correct schema."""
    csv_path = _write_funding_csv(tmp_path / "BTCUSDT-fundingRate-2024-01-01.csv")
    output_dir = tmp_path / "features"

    converter = ParquetConverter()
    result = converter.convert_csv_to_parquet(
        csv_path=csv_path,
        data_type="funding_rate",
        symbol="BTCUSDT",
        output_dir=output_dir,
    )

    assert result is not None, "Expected a Parquet path, got None"
    assert result.exists()
    assert result.suffix == ".parquet"

    df = pl.read_parquet(result, hive_partitioning=False)

    # Row count
    assert len(df) == 3

    # Required columns present
    assert "fundingTime" in df.columns
    assert "fundingRate" in df.columns
    assert "markPrice" in df.columns
    assert "datetime" in df.columns
    assert "symbol" in df.columns

    # Correct dtypes
    assert df["fundingTime"].dtype == pl.Int64
    assert df["fundingRate"].dtype == pl.Float64

    # Sorted by fundingTime
    assert df["fundingTime"].is_sorted()

    # Values are correct
    assert df["fundingTime"][0] == 1_704_067_200_000
    assert df["fundingRate"][0] == pytest.approx(0.0001)
    assert df["symbol"][0] == "BTCUSDT"


# ---------------------------------------------------------------------------
# Test: klines CSV conversion
# ---------------------------------------------------------------------------

def test_convert_klines_csv(tmp_path: Path) -> None:
    """Converting a klines CSV produces a valid Parquet with correct schema."""
    csv_path = _write_klines_csv(tmp_path / "BTCUSDT-4h-2024-01-01.csv")
    output_dir = tmp_path / "features"

    converter = ParquetConverter()
    result = converter.convert_csv_to_parquet(
        csv_path=csv_path,
        data_type="klines_4h",
        symbol="BTCUSDT",
        output_dir=output_dir,
    )

    assert result is not None, "Expected a Parquet path, got None"
    assert result.exists()
    assert result.suffix == ".parquet"

    # hive_partitioning=False: symbol is stored in parquet, not just in the path
    df = pl.read_parquet(result, hive_partitioning=False)

    # Row count
    assert len(df) == 6

    # Column renames applied
    assert "trade_count" in df.columns
    assert "count" not in df.columns

    # datetime column added
    assert "datetime" in df.columns

    # symbol column added
    assert "symbol" in df.columns
    assert df["symbol"][0] == "BTCUSDT"

    # Sorted by open_time
    assert df["open_time"].is_sorted()

    # open_time is Int64 unix ms
    assert df["open_time"].dtype == pl.Int64
    assert df["open_time"][0] == 1_704_067_200_000


# ---------------------------------------------------------------------------
# Test: klines parquet reads back correctly
# ---------------------------------------------------------------------------

def test_klines_parquet_schema(tmp_path: Path) -> None:
    """Parquet file can be round-tripped through Polars with correct dtypes."""
    csv_path = _write_klines_csv(tmp_path / "BTCUSDT-4h-2024-01-01.csv")
    converter = ParquetConverter()
    pq_path = converter.convert_csv_to_parquet(csv_path, "klines_4h", "BTCUSDT", tmp_path / "out")

    df = pl.read_parquet(pq_path, hive_partitioning=False)

    assert df["open"].dtype == pl.Float64
    assert df["trade_count"].dtype == pl.Int32
    assert df["datetime"].dtype in (pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime)
    assert df["close"][0] == pytest.approx(42500.0)


# ---------------------------------------------------------------------------
# Test: metrics CSV conversion (string datetime)
# ---------------------------------------------------------------------------

def test_convert_metrics_csv(tmp_path: Path) -> None:
    """Metrics CSV with string create_time is converted to unix-ms Int64."""
    csv_path = _write_metrics_csv(tmp_path / "BTCUSDT-metrics-2024-01-01.csv")
    converter = ParquetConverter()
    result = converter.convert_csv_to_parquet(
        csv_path=csv_path,
        data_type="metrics",
        symbol="BTCUSDT",
        output_dir=tmp_path / "out",
    )

    assert result is not None
    df = pl.read_parquet(result, hive_partitioning=False)

    assert len(df) == 3
    # create_time must be Int64 unix ms
    assert df["create_time"].dtype == pl.Int64
    # All values should be positive unix timestamps
    assert (df["create_time"] > 0).all()
    # datetime column present
    assert "datetime" in df.columns


# ---------------------------------------------------------------------------
# Test: validate_parquet
# ---------------------------------------------------------------------------

def test_validate_parquet_valid(tmp_path: Path) -> None:
    """validate_parquet returns is_valid=True for a good file."""
    csv_path = _write_klines_csv(tmp_path / "BTCUSDT-4h-2024-01-01.csv")
    converter = ParquetConverter()
    pq_path = converter.convert_csv_to_parquet(csv_path, "klines_4h", "BTCUSDT", tmp_path / "out")

    info = converter.validate_parquet(pq_path)

    assert info["is_valid"] is True
    assert info["row_count"] == 6
    assert info["size_bytes"] > 0
    assert "open_time" in info["columns"]
    assert info["errors"] == []


def test_validate_parquet_missing(tmp_path: Path) -> None:
    """validate_parquet returns is_valid=False when file does not exist."""
    converter = ParquetConverter()
    info = converter.validate_parquet(tmp_path / "nonexistent.parquet")
    assert info["is_valid"] is False
    assert info["errors"]


# ---------------------------------------------------------------------------
# Test: DataStore.get_klines with synthetic parquet data
# ---------------------------------------------------------------------------

def _build_store(tmp_path: Path, symbol: str = "BTCUSDT") -> Path:
    """Create a minimal Parquet store and return its root path."""
    csv_path = _write_klines_csv(tmp_path / "raw" / f"{symbol}-4h-2024-01-01.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    converter = ParquetConverter()
    converter.convert_csv_to_parquet(
        csv_path=csv_path,
        data_type="klines_4h",
        symbol=symbol,
        output_dir=tmp_path / "features",
    )
    return tmp_path / "features"


def test_datastore_get_klines(tmp_path: Path) -> None:
    """DataStore.get_klines returns rows in the requested date range."""
    feat_dir = _build_store(tmp_path)
    store = DataStore(feat_dir)

    df = store.get_klines(
        symbol="BTCUSDT",
        interval="4h",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    )

    assert not df.is_empty()
    assert "open_time" in df.columns
    assert "close" in df.columns
    assert (df["close"] > 0).all()

    store.close()


def test_datastore_get_klines_column_subset(tmp_path: Path) -> None:
    """DataStore.get_klines honours the columns filter."""
    feat_dir = _build_store(tmp_path)
    store = DataStore(feat_dir)

    df = store.get_klines(
        "BTCUSDT", "4h",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
        columns=["open_time", "close"],
    )

    assert set(df.columns) == {"open_time", "close"}
    store.close()


def test_datastore_get_klines_no_data(tmp_path: Path) -> None:
    """DataStore.get_klines returns empty DataFrame when symbol has no files."""
    feat_dir = _build_store(tmp_path, symbol="BTCUSDT")
    store = DataStore(feat_dir)

    # ETHUSDT was never loaded
    df = store.get_klines(
        "ETHUSDT", "4h",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 2),
    )

    assert df.is_empty()
    store.close()


# ---------------------------------------------------------------------------
# Test: DataStore.query (arbitrary SQL)
# ---------------------------------------------------------------------------

def test_datastore_query(tmp_path: Path) -> None:
    """DataStore.query executes arbitrary SQL via DuckDB."""
    feat_dir = _build_store(tmp_path)
    store = DataStore(feat_dir)

    # Find parquet file directly
    pq_files = list(feat_dir.rglob("*.parquet"))
    assert pq_files, "No parquet files to query"

    sql = f"SELECT COUNT(*) AS n FROM read_parquet('{pq_files[0]}')"
    df = store.query(sql)

    assert "n" in df.columns
    assert df["n"][0] == 6   # 6 rows in one day of 4h klines

    store.close()


# ---------------------------------------------------------------------------
# Test: DataStore.get_data_summary
# ---------------------------------------------------------------------------

def test_datastore_get_data_summary(tmp_path: Path) -> None:
    """get_data_summary returns a non-empty dict with expected keys."""
    feat_dir = _build_store(tmp_path)
    store = DataStore(feat_dir)

    summary = store.get_data_summary()
    store.close()

    assert summary, "Summary should not be empty"
    key = "BTCUSDT/klines_4h"
    assert key in summary, f"Expected key '{key}' in summary"

    entry = summary[key]
    assert entry["row_count"] == 6
    assert entry["file_count"] == 1
    # Synthetic parquet is ~2 KB; size_mb rounds to 0.00 at 2 dp — check raw bytes
    pq_file = next(feat_dir.rglob("*.parquet"))
    assert pq_file.stat().st_size > 0
    assert "2024-01-01" in entry["date_range"]
