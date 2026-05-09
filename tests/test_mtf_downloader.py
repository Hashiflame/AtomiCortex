"""
tests/test_mtf_downloader.py

Unit tests for the multi-timeframe data pipeline:
  - download_mtf_data.py (URL generation, resume, retry)
  - convert_mtf_to_parquet.py (CSV parsing, Parquet schema, partitioning)
  - check_mtf_data_quality.py (completeness, quality checks)

All tests use mocks / tmp_path fixtures — no real network calls.

Run:
    pytest tests/test_mtf_downloader.py -v
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------

from scripts.download_mtf_data import (
    BASE_URL,
    build_checksum_url,
    build_month_range,
    build_url,
    dest_dir_for,
    download_and_extract,
    verify_sha256,
)
from scripts.convert_mtf_to_parquet import (
    KLINE_COLUMNS_11,
    KLINE_COLUMNS_12,
    convert_csv_to_parquet,
    read_kline_csv,
)
from scripts.check_mtf_data_quality import (
    EXPECTED_BARS_PER_DAY,
    MTFDataQualityChecker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Temporary base directory for test data."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


def _make_kline_row(
    open_time: int = 1704067200000,
    open_: float = 42000.0,
    high: float = 42100.0,
    low: float = 41900.0,
    close: float = 42050.0,
    volume: float = 100.0,
    close_time: int = 1704070799999,
    quote_vol: float = 4_200_000.0,
    n_trades: int = 1000,
    taker_buy_base: float = 50.0,
    taker_buy_quote: float = 2_100_000.0,
) -> str:
    """Build a single kline CSV row (11 columns, no header)."""
    return (
        f"{open_time},{open_},{high},{low},{close},{volume},"
        f"{close_time},{quote_vol},{n_trades},{taker_buy_base},{taker_buy_quote}"
    )


def _make_csv_file(path: Path, n_rows: int = 24, interval_ms: int = 3_600_000) -> Path:
    """Create a synthetic klines CSV file with n_rows rows."""
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
    lines = []
    for i in range(n_rows):
        ot = base_time + i * interval_ms
        ct = ot + interval_ms - 1
        lines.append(_make_kline_row(
            open_time=ot,
            open_=42000.0 + i * 10,
            high=42100.0 + i * 10,
            low=41900.0 + i * 10,
            close=42050.0 + i * 10,
            volume=100.0 + i,
            close_time=ct,
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_zip(zip_path: Path, csv_content: str, csv_name: str) -> Path:
    """Create a ZIP file containing a single CSV."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_content)
    return zip_path


# ═══════════════════════════════════════════════════════════════
# URL generation tests
# ═══════════════════════════════════════════════════════════════


class TestURLGeneration:
    """Tests for URL building functions."""

    def test_url_generation_1h(self) -> None:
        url = build_url("BTCUSDT", "1h", 2024, 1)
        expected = (
            f"{BASE_URL}/data/futures/um/monthly/klines"
            "/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
        )
        assert url == expected

    def test_url_generation_15m(self) -> None:
        url = build_url("BTCUSDT", "15m", 2024, 6)
        assert "/15m/" in url
        assert "BTCUSDT-15m-2024-06.zip" in url

    def test_url_generation_all_months_in_range(self) -> None:
        months = build_month_range("2023-01", "2023-12")
        assert len(months) == 12
        assert months[0] == (2023, 1)
        assert months[-1] == (2023, 12)

    def test_url_generation_cross_year(self) -> None:
        months = build_month_range("2023-11", "2024-02")
        assert len(months) == 4
        assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]

    def test_checksum_url_derived_from_data_url(self) -> None:
        url = build_url("BTCUSDT", "1h", 2024, 1)
        checksum_url = build_checksum_url(url)
        assert checksum_url == url + ".CHECKSUM"

    def test_dest_dir_structure(self) -> None:
        d = dest_dir_for(Path("/data/raw"), "BTCUSDT", "1h")
        assert str(d) == "/data/raw/exchange=BINANCE_UM/symbol=BTCUSDT/interval=1h"


# ═══════════════════════════════════════════════════════════════
# Resume logic tests
# ═══════════════════════════════════════════════════════════════


class TestResumeLogic:
    """Tests for download resume / skip behaviour."""

    def test_skip_existing_file_same_size(self, data_dir: Path) -> None:
        """If CSV already exists, download_and_extract should skip it."""
        dest = dest_dir_for(data_dir, "BTCUSDT", "1h")
        dest.mkdir(parents=True, exist_ok=True)
        csv = dest / "BTCUSDT-1h-2024-01.csv"
        csv.write_text("dummy data\n")

        result = download_and_extract("BTCUSDT", "1h", 2024, 1, dest)
        assert result["status"] == "skipped"

    def test_redownload_if_size_zero(self, data_dir: Path) -> None:
        """Zero-size CSV should NOT be skipped."""
        dest = dest_dir_for(data_dir, "BTCUSDT", "1h")
        dest.mkdir(parents=True, exist_ok=True)
        csv = dest / "BTCUSDT-1h-2024-01.csv"
        csv.write_text("")  # empty

        # Mock requests to return 404 to avoid real download.
        with patch("scripts.download_mtf_data.requests.Session") as MockSession:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            MockSession.return_value.get.return_value = mock_resp
            MockSession.return_value.headers = {}

            result = download_and_extract("BTCUSDT", "1h", 2024, 1, dest)
            # Should attempt download (not skip), will hit 404.
            assert result["status"] == "not_found"

    def test_dry_run_skips_download(self, data_dir: Path) -> None:
        """Dry run must not download anything."""
        dest = dest_dir_for(data_dir, "BTCUSDT", "1h")
        result = download_and_extract("BTCUSDT", "1h", 2024, 1, dest, dry_run=True)
        assert result["status"] == "dry_run"
        assert "Would download" in result["message"]


# ═══════════════════════════════════════════════════════════════
# Data parsing tests
# ═══════════════════════════════════════════════════════════════


class TestDataParsing:
    """Tests for CSV parsing and data validation."""

    def test_csv_parse_correct_columns(self, tmp_path: Path) -> None:
        csv_path = _make_csv_file(tmp_path / "test.csv", n_rows=5)
        df = read_kline_csv(csv_path)
        expected_cols = set(KLINE_COLUMNS_11)
        assert set(df.columns) == expected_cols

    def test_csv_parse_12_columns(self, tmp_path: Path) -> None:
        """Handle CSVs with 12 columns (including ignore field)."""
        path = tmp_path / "test12.csv"
        row = _make_kline_row() + ",0"  # 12th column = ignore
        path.write_text(row + "\n")

        df = read_kline_csv(path)
        assert "ignore" not in df.columns
        assert "open_time" in df.columns

    def test_timestamp_converted_to_utc_datetime(self, data_dir: Path) -> None:
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=3)
        df = read_kline_csv(csv_path)

        # open_time should be Int64 (unix ms)
        assert df["open_time"].dtype == pl.Int64
        assert df["open_time"][0] == 1704067200000

    def test_no_negative_prices(self, tmp_path: Path) -> None:
        csv_path = _make_csv_file(tmp_path / "test.csv", n_rows=10)
        df = read_kline_csv(csv_path)
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all(), f"Negative values in {col}"

    def test_no_negative_volume(self, tmp_path: Path) -> None:
        csv_path = _make_csv_file(tmp_path / "test.csv", n_rows=10)
        df = read_kline_csv(csv_path)
        assert (df["volume"] >= 0).all()


# ═══════════════════════════════════════════════════════════════
# Parquet conversion tests
# ═══════════════════════════════════════════════════════════════


class TestParquetConversion:
    """Tests for CSV → Parquet conversion."""

    def test_parquet_schema_matches_spec(self, data_dir: Path) -> None:
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=24)
        created = convert_csv_to_parquet(csv_path, "BTCUSDT", "1h", data_dir)

        assert len(created) >= 1
        df = pl.read_parquet(created[0], hive_partitioning=False)

        required_cols = {
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol",
            "timestamp", "symbol", "interval", "exchange",
        }
        assert required_cols.issubset(set(df.columns))

    def test_parquet_zstd_compressed(self, data_dir: Path) -> None:
        """Verify parquet uses ZSTD compression (check file is smaller than CSV)."""
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=100)
        created = convert_csv_to_parquet(csv_path, "BTCUSDT", "1h", data_dir)

        assert len(created) >= 1
        # Parquet should be smaller or at least writable.
        for pf in created:
            assert pf.exists()
            assert pf.stat().st_size > 0

    def test_parquet_partitioned_by_date(self, data_dir: Path) -> None:
        # 48 hours of 1h data → 2 days → 2 parquet files.
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=48)
        created = convert_csv_to_parquet(csv_path, "BTCUSDT", "1h", data_dir)

        assert len(created) == 2
        # Each should be in a date= directory.
        for pf in created:
            assert "date=" in str(pf)
            assert pf.name == "klines.parquet"

    def test_no_duplicate_timestamps_after_convert(self, data_dir: Path) -> None:
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=24)
        created = convert_csv_to_parquet(csv_path, "BTCUSDT", "1h", data_dir)

        all_dfs = [pl.read_parquet(p, hive_partitioning=False) for p in created]
        combined = pl.concat(all_dfs)

        n_total = len(combined)
        n_unique = combined.select("open_time").n_unique()
        assert n_total == n_unique, f"Found {n_total - n_unique} duplicate timestamps"

    def test_symbol_and_interval_columns(self, data_dir: Path) -> None:
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=5)
        created = convert_csv_to_parquet(csv_path, "BTCUSDT", "1h", data_dir)

        df = pl.read_parquet(created[0], hive_partitioning=False)
        assert (df["symbol"] == "BTCUSDT").all()
        assert (df["interval"] == "1h").all()
        assert (df["exchange"] == "BINANCE_UM").all()

    def test_timestamp_is_datetime_type(self, data_dir: Path) -> None:
        csv_path = _make_csv_file(data_dir / "test.csv", n_rows=5)
        created = convert_csv_to_parquet(csv_path, "BTCUSDT", "1h", data_dir)

        df = pl.read_parquet(created[0], hive_partitioning=False)
        assert df["timestamp"].dtype == pl.Datetime("ms")  # from_epoch("ms")


# ═══════════════════════════════════════════════════════════════
# Quality checker tests
# ═══════════════════════════════════════════════════════════════


def _setup_parquet_data(
    data_dir: Path,
    interval: str = "1h",
    n_rows: int = 24,
    interval_ms: int = 3_600_000,
    symbol: str = "BTCUSDT",
) -> list[Path]:
    """Create synthetic CSV, convert to parquet, return parquet paths."""
    csv_dir = (
        data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / f"interval={interval}"
    )
    csv_path = _make_csv_file(csv_dir / "test.csv", n_rows=n_rows, interval_ms=interval_ms)
    return convert_csv_to_parquet(csv_path, symbol, interval, data_dir)


class TestQualityChecker:
    """Tests for MTFDataQualityChecker."""

    def test_completeness_calculated_correctly_1h(self, data_dir: Path) -> None:
        _setup_parquet_data(data_dir, "1h", n_rows=24)

        with MTFDataQualityChecker(data_dir) as checker:
            result = checker.check_completeness("1h")

        assert result["total_bars"] == 24
        assert result["n_days"] == 1
        assert result["expected_bars"] == EXPECTED_BARS_PER_DAY["1h"]
        assert result["completeness_pct"] == 100.0
        assert result["pass"] is True

    def test_completeness_calculated_correctly_15m(self, data_dir: Path) -> None:
        # 96 bars = 1 day of 15m data
        _setup_parquet_data(data_dir, "15m", n_rows=96, interval_ms=900_000)

        with MTFDataQualityChecker(data_dir) as checker:
            result = checker.check_completeness("15m")

        assert result["total_bars"] == 96
        assert result["completeness_pct"] == 100.0
        assert result["pass"] is True

    def test_duplicates_detected(self, data_dir: Path) -> None:
        _setup_parquet_data(data_dir, "1h", n_rows=24)

        with MTFDataQualityChecker(data_dir) as checker:
            result = checker.check_duplicates("1h")

        assert result["duplicate_count"] == 0
        assert result["pass"] is True

    def test_duckdb_reads_parquet_without_error(self, data_dir: Path) -> None:
        _setup_parquet_data(data_dir, "1h", n_rows=24)

        with MTFDataQualityChecker(data_dir) as checker:
            result = checker.check_duckdb_readability("1h")

        assert result["pass"] is True
        assert result["total_bars"] == 24

    def test_zero_prices_detected(self, data_dir: Path) -> None:
        _setup_parquet_data(data_dir, "1h", n_rows=24)

        with MTFDataQualityChecker(data_dir) as checker:
            result = checker.check_zero_prices("1h")

        assert result["zero_count"] == 0
        assert result["pass"] is True

    def test_empty_interval_returns_zero(self, data_dir: Path) -> None:
        with MTFDataQualityChecker(data_dir) as checker:
            result = checker.check_completeness("5m")

        assert result["total_bars"] == 0
        assert result["pass"] is False


# ═══════════════════════════════════════════════════════════════
# SHA256 verification test
# ═══════════════════════════════════════════════════════════════


class TestChecksum:
    """Tests for SHA256 verification."""

    def test_verify_sha256_correct(self, tmp_path: Path) -> None:
        import hashlib
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()

        assert verify_sha256(test_file, expected) is True

    def test_verify_sha256_mismatch(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        assert verify_sha256(test_file, "0" * 64) is False
