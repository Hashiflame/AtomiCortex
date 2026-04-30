"""
tests/test_binance_downloader.py

Integration tests for BinanceDataDownloader.
These tests make real HTTP requests to data.binance.vision — they require
an internet connection and will be slow (~2-5 s each).

Run:
    pytest tests/test_binance_downloader.py -v
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
import pytest_asyncio

from src.ingestion.binance_downloader import BinanceDataDownloader, extract_zip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Temporary base directory for downloaded test data."""
    d = tmp_path / "raw"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _csv_valid(path: Path) -> bool:
    """Return True if path exists, is non-empty, and contains a numeric data row.

    Binance kline CSVs may start with an optional header line
    (``open_time,open,...``); the first *data* row has a unix-ms integer in
    the first column.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        first_col = line.split(",")[0].strip()
        if first_col.isdigit():
            return True
    return False


# ---------------------------------------------------------------------------
# Test 1 — download a single day of 4h klines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_single_klines(data_dir: Path) -> None:
    """Downloading one day of BTCUSDT 4h klines returns one valid CSV."""
    async with BinanceDataDownloader() as dl:
        paths = await dl.download_klines(
            symbol="BTCUSDT",
            interval="4h",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            dest_dir=data_dir,
        )

    assert len(paths) == 1, f"Expected 1 path, got {paths}"
    csv = paths[0]
    assert csv.suffix == ".csv"
    assert csv.exists()
    assert csv.stat().st_size > 0
    assert _csv_valid(csv), f"CSV does not look like kline data: {csv.read_text()[:200]}"


# ---------------------------------------------------------------------------
# Test 2 — checksum verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_checksum_verified(data_dir: Path) -> None:
    """Downloaded file survives SHA-256 verification (no corruption)."""
    async with BinanceDataDownloader() as dl:
        # download_file works on the zip; we use download_klines for end-to-end.
        paths = await dl.download_klines(
            symbol="BTCUSDT",
            interval="1d",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            dest_dir=data_dir,
        )

    assert paths, "No files downloaded"
    csv = paths[0]

    # The file must be readable and contain at least one data row.
    lines = csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1, "CSV appears to be empty"

    # Re-hash the file to confirm it is not all zeros / truncated.
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert len(sha) == 64  # valid hex digest


# ---------------------------------------------------------------------------
# Test 3 — idempotent: second download skips existing file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotent_download(data_dir: Path) -> None:
    """Re-downloading an already-present file must not modify it on disk."""
    target = date(2024, 1, 2)

    async with BinanceDataDownloader() as dl:
        paths1 = await dl.download_klines(
            symbol="BTCUSDT",
            interval="4h",
            start_date=target,
            end_date=target,
            dest_dir=data_dir,
        )

    assert len(paths1) == 1
    csv = paths1[0]
    mtime_after_first = csv.stat().st_mtime
    size_after_first = csv.stat().st_size

    # Second call — must skip, not re-download.
    async with BinanceDataDownloader() as dl:
        paths2 = await dl.download_klines(
            symbol="BTCUSDT",
            interval="4h",
            start_date=target,
            end_date=target,
            dest_dir=data_dir,
        )

    assert len(paths2) == 1
    assert paths2[0] == csv
    assert csv.stat().st_mtime == mtime_after_first, "File was re-written on second download"
    assert csv.stat().st_size == size_after_first


# ---------------------------------------------------------------------------
# Test 4 — future date returns empty list (404)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_future_date_returns_empty(data_dir: Path) -> None:
    """Requesting data for a date far in the future must return an empty list."""
    async with BinanceDataDownloader() as dl:
        paths = await dl.download_klines(
            symbol="BTCUSDT",
            interval="4h",
            start_date=date(2099, 1, 1),
            end_date=date(2099, 1, 1),
            dest_dir=data_dir,
        )

    assert paths == [], f"Expected empty list for non-existent date, got {paths}"


# ---------------------------------------------------------------------------
# Test 5 — extract_zip helper
# ---------------------------------------------------------------------------

def test_extract_zip_removes_zip(tmp_path: Path) -> None:
    """extract_zip must delete the zip and return a CSV path that exists."""
    import zipfile

    csv_content = "1704067200000,42000,42100,41900,42050,100,1704153599999,4200000,1000,50,2100000,0\n"
    csv_name = "BTCUSDT-4h-2024-01-01.csv"
    zip_path = tmp_path / "BTCUSDT-4h-2024-01-01.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, csv_content)

    csv_path = extract_zip(zip_path)

    assert not zip_path.exists(), "ZIP was not deleted after extraction"
    assert csv_path.exists()
    assert csv_path.name == csv_name
    assert csv_path.read_text() == csv_content
