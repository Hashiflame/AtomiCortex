"""
src/ingestion/parquet_converter.py

Converts raw Binance CSV files (from binance_downloader) to Parquet with
ZSTD compression.  Reads with Polars, writes partitioned by date:

    {output_dir}/exchange=BINANCE_UM/symbol={symbol}/{data_type}/date={YYYY-MM-DD}/part-0.parquet

Supported data_type values: klines_4h, klines_1d, funding_rate, metrics, agg_trades
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from src.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public schema constants (final column names after all renames/transforms)
# ---------------------------------------------------------------------------

KLINES_SCHEMA: dict[str, type[pl.DataType]] = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Int64,
    "quote_volume": pl.Float64,
    "trade_count": pl.Int32,
    "taker_buy_volume": pl.Float64,
    "taker_buy_quote_volume": pl.Float64,
    "ignore": pl.Float64,
}

FUNDING_SCHEMA: dict[str, type[pl.DataType]] = {
    "fundingTime": pl.Int64,
    "fundingRate": pl.Float64,
    "markPrice": pl.Float64,
    "symbol": pl.Utf8,
}

METRICS_SCHEMA: dict[str, type[pl.DataType]] = {
    "create_time": pl.Int64,   # stored as unix ms; CSV contains a datetime string
    "symbol": pl.Utf8,
    "sum_open_interest": pl.Float64,
    "sum_open_interest_value": pl.Float64,
    "count_toptrader_long_short_ratio": pl.Float64,
    "sum_toptrader_long_short_ratio": pl.Float64,
    "count_long_short_ratio": pl.Float64,
    "sum_taker_long_short_vol_ratio": pl.Float64,
}

AGG_TRADES_SCHEMA: dict[str, type[pl.DataType]] = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,    # CSV column is "qty"
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "transact_time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
}

# ---------------------------------------------------------------------------
# Internal per-type configuration
# ---------------------------------------------------------------------------

@dataclass
class _TypeConfig:
    """Describes how to read and transform one data type."""
    csv_dtypes: dict[str, type[pl.DataType]]   # dtype overrides using CSV column names
    column_renames: dict[str, str]              # CSV col → target col
    timestamp_col: str                          # column to derive "datetime" from
    sort_col: str
    datetime_is_string: bool = False            # True when timestamp is "YYYY-MM-DD HH:MM:SS"
    datetime_format: str = ""                   # strptime format (when datetime_is_string)


_CONFIGS: dict[str, _TypeConfig] = {
    "klines_4h": _TypeConfig(
        csv_dtypes={
            "open_time": pl.Int64, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
            "close_time": pl.Int64, "quote_volume": pl.Float64,
            "count": pl.Int32, "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64, "ignore": pl.Float64,
        },
        column_renames={"count": "trade_count"},
        timestamp_col="open_time",
        sort_col="open_time",
    ),
    "klines_1d": _TypeConfig(
        csv_dtypes={
            "open_time": pl.Int64, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
            "close_time": pl.Int64, "quote_volume": pl.Float64,
            "count": pl.Int32, "taker_buy_volume": pl.Float64,
            "taker_buy_quote_volume": pl.Float64, "ignore": pl.Float64,
        },
        column_renames={"count": "trade_count"},
        timestamp_col="open_time",
        sort_col="open_time",
    ),
    "funding_rate": _TypeConfig(
        csv_dtypes={
            "fundingTime": pl.Int64,
            "fundingRate": pl.Float64,
            "markPrice": pl.Float64,
            "symbol": pl.Utf8,
        },
        column_renames={},
        timestamp_col="fundingTime",
        sort_col="fundingTime",
    ),
    "metrics": _TypeConfig(
        csv_dtypes={
            "create_time": pl.Utf8,          # string datetime in source
            "symbol": pl.Utf8,
            "sum_open_interest": pl.Float64,
            "sum_open_interest_value": pl.Float64,
            "count_toptrader_long_short_ratio": pl.Float64,
            "sum_toptrader_long_short_ratio": pl.Float64,
            "count_long_short_ratio": pl.Float64,
            "sum_taker_long_short_vol_ratio": pl.Float64,
        },
        column_renames={},
        timestamp_col="create_time",
        sort_col="create_time",
        datetime_is_string=True,
        datetime_format="%Y-%m-%d %H:%M:%S",
    ),
    "agg_trades": _TypeConfig(
        csv_dtypes={
            "agg_trade_id": pl.Int64, "price": pl.Float64,
            "qty": pl.Float64, "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64, "transact_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
        },
        column_renames={"qty": "quantity"},
        timestamp_col="transact_time",
        sort_col="transact_time",
    ),
}

ROW_GROUP_SIZE = 131_072   # 128 K rows per parquet row-group


# ---------------------------------------------------------------------------
# Module-level helper (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _convert_task(args: tuple[str, str, str, str, str, int]) -> dict[str, Any]:
    """Worker function for parallel conversion — must live at module level."""
    csv_path_str, data_type, symbol, output_dir_str, compression, comp_level = args
    # Workers inherit sys.path via fork on Linux; re-insert root for spawn
    _root = str(Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    try:
        converter = ParquetConverter()
        result = converter.convert_csv_to_parquet(
            csv_path=Path(csv_path_str),
            data_type=data_type,
            symbol=symbol,
            output_dir=Path(output_dir_str),
            compression=compression,
            compression_level=comp_level,
        )
        return {"status": "converted" if result else "skipped_empty", "csv": csv_path_str}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "csv": csv_path_str, "error": str(exc)}


import re as _re
_DATE_RE = _re.compile(r"(\d{4}-\d{2}(?:-\d{2})?)$")


def extract_date_from_stem(stem: str) -> str:
    """Extract the date string from a Binance filename stem.

    Handles both daily (``YYYY-MM-DD``) and monthly (``YYYY-MM``) filenames.

    Examples
    --------
    ``"BTCUSDT-4h-2024-01-01"``          →  ``"2024-01-01"``
    ``"BTCUSDT-metrics-2024-01-01"``      →  ``"2024-01-01"``
    ``"BTCUSDT-fundingRate-2024-01"``     →  ``"2024-01"``
    """
    m = _DATE_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot extract date from stem: {stem!r}")
    return m.group(1)


# ---------------------------------------------------------------------------
# ParquetConverter
# ---------------------------------------------------------------------------

class ParquetConverter:
    """Converts Binance raw CSV files to Parquet (ZSTD-3)."""

    # ----------------------------------------------------------------
    # Public: single-file conversion
    # ----------------------------------------------------------------

    def convert_csv_to_parquet(
        self,
        csv_path: Path,
        data_type: str,
        symbol: str,
        output_dir: Path,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> Path | None:
        """Read *csv_path*, apply schema + transforms, write Parquet.

        Returns the output Parquet path, or ``None`` when the CSV is empty.
        """
        if data_type not in _CONFIGS:
            raise ValueError(
                f"Unknown data_type '{data_type}'. "
                f"Valid values: {list(_CONFIGS)}"
            )

        cfg = _CONFIGS[data_type]
        date_str = extract_date_from_stem(csv_path.stem)
        parquet_path = (
            output_dir
            / "exchange=BINANCE_UM"
            / f"symbol={symbol}"
            / data_type
            / f"date={date_str}"
            / "part-0.parquet"
        )

        # ----------------------------------------------------------
        # Read CSV
        # ----------------------------------------------------------
        try:
            df = pl.read_csv(
                csv_path,
                schema_overrides=cfg.csv_dtypes,
                has_header=True,
                try_parse_dates=False,
                null_values=["", "NA", "N/A"],
                ignore_errors=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc

        if df.is_empty():
            _log.debug(f"Empty CSV, skipping: {csv_path.name}")
            return None

        # ----------------------------------------------------------
        # Column renames  (e.g. count → trade_count, qty → quantity)
        # ----------------------------------------------------------
        if cfg.column_renames:
            df = df.rename(cfg.column_renames)

        # ----------------------------------------------------------
        # Datetime column
        # ----------------------------------------------------------
        if cfg.datetime_is_string:
            # metrics: create_time is "YYYY-MM-DD HH:MM:SS" string
            df = df.with_columns(
                pl.col(cfg.timestamp_col)
                  .str.to_datetime(format=cfg.datetime_format, strict=False)
                  .alias("_dt_tmp")
            ).with_columns(
                pl.col("_dt_tmp").alias("datetime"),
                pl.col("_dt_tmp").dt.timestamp("ms").alias(cfg.timestamp_col),
            ).drop("_dt_tmp")
        else:
            # klines / funding_rate / agg_trades: timestamp already unix ms
            df = df.with_columns(
                pl.from_epoch(pl.col(cfg.timestamp_col), time_unit="ms").alias("datetime")
            )

        # ----------------------------------------------------------
        # Add symbol column if missing
        # ----------------------------------------------------------
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

        # ----------------------------------------------------------
        # Sort by primary timestamp
        # ----------------------------------------------------------
        df = df.sort(cfg.sort_col, maintain_order=True)

        # ----------------------------------------------------------
        # Write Parquet
        # ----------------------------------------------------------
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(
            parquet_path,
            compression=compression,
            compression_level=compression_level,
            row_group_size=ROW_GROUP_SIZE,
            statistics=True,
        )

        _log.debug(
            f"Converted {csv_path.name} → {parquet_path.relative_to(output_dir)} "
            f"({len(df)} rows)"
        )
        return parquet_path

    # ----------------------------------------------------------------
    # Public: directory batch conversion
    # ----------------------------------------------------------------

    def convert_directory(
        self,
        raw_dir: Path,
        output_dir: Path,
        symbols: list[str],
        data_types: list[str] | None = None,
        workers: int = 4,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> dict[str, Any]:
        """Discover and convert all CSV files under *raw_dir*.

        Skips files whose Parquet counterpart already exists and is newer
        than the CSV.

        Parameters
        ----------
        raw_dir:    Base raw data directory (exchange= / symbol= tree).
        output_dir: Base Parquet output directory (same Hive tree).
        symbols:    List of Binance symbols to process.
        data_types: Subset of data types to convert; None = all known types.
        workers:    ProcessPoolExecutor worker count.

        Returns
        -------
        dict with ``converted``, ``skipped``, ``skipped_empty``, ``failed``,
        ``errors``, ``elapsed_seconds``.
        """
        import time

        t0 = time.monotonic()
        if data_types is None:
            data_types = list(_CONFIGS)

        stats: dict[str, Any] = {
            "converted": 0,
            "skipped": 0,
            "skipped_empty": 0,
            "failed": 0,
            "errors": [],
            "elapsed_seconds": 0.0,
        }

        # Collect tasks
        tasks: list[tuple[str, str, str, str, str, int]] = []
        skipped_uptodate = 0

        for symbol in symbols:
            for dt in data_types:
                csv_dir = (
                    raw_dir / "exchange=BINANCE_UM" / f"symbol={symbol}" / dt
                )
                if not csv_dir.exists():
                    continue

                for csv_path in sorted(csv_dir.glob("*.csv")):
                    date_str = extract_date_from_stem(csv_path.stem)
                    parquet_path = (
                        output_dir
                        / "exchange=BINANCE_UM"
                        / f"symbol={symbol}"
                        / dt
                        / f"date={date_str}"
                        / "part-0.parquet"
                    )
                    # Skip if parquet is up-to-date
                    if (
                        parquet_path.exists()
                        and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
                    ):
                        skipped_uptodate += 1
                        continue

                    tasks.append((
                        str(csv_path), dt, symbol, str(output_dir),
                        compression, compression_level,
                    ))

        stats["skipped"] = skipped_uptodate
        _log.info(
            f"convert_directory: {len(tasks)} to convert, "
            f"{skipped_uptodate} already up-to-date, workers={workers}"
        )

        if not tasks:
            stats["elapsed_seconds"] = time.monotonic() - t0
            return stats

        # Run in parallel
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_convert_task, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Converting", unit="file") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    status = result["status"]
                    if status == "converted":
                        stats["converted"] += 1
                    elif status == "skipped_empty":
                        stats["skipped_empty"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append(
                            f"{result['csv']}: {result.get('error', '?')}"
                        )
                    pbar.update(1)

        stats["elapsed_seconds"] = time.monotonic() - t0
        _log.info(
            f"Conversion done: converted={stats['converted']} "
            f"skipped={stats['skipped']} failed={stats['failed']}"
        )
        return stats

    # ----------------------------------------------------------------
    # Public: validation
    # ----------------------------------------------------------------

    def validate_parquet(self, parquet_path: Path) -> dict[str, Any]:
        """Read and validate a Parquet file.

        Returns
        -------
        dict with ``row_count``, ``columns``, ``size_bytes``, ``is_valid``,
        and optionally ``errors`` (list of strings).
        """
        result: dict[str, Any] = {
            "path": str(parquet_path),
            "row_count": 0,
            "columns": [],
            "size_bytes": 0,
            "is_valid": False,
            "errors": [],
        }

        if not parquet_path.exists():
            result["errors"].append("File does not exist")
            return result

        result["size_bytes"] = parquet_path.stat().st_size

        try:
            # hive_partitioning=False: symbol is stored in the parquet data
            # itself; auto-detection would create a duplicate column error.
            df = pl.read_parquet(parquet_path, hive_partitioning=False)
        except Exception as exc:
            result["errors"].append(f"Cannot read parquet: {exc}")
            return result

        result["row_count"] = len(df)
        result["columns"] = df.columns

        # Check for all-null key columns (timestamp)
        key_cols = [c for c in ("open_time", "calc_time", "create_time", "transact_time") if c in df.columns]
        for col in key_cols:
            null_count = df[col].null_count()
            if null_count == len(df):
                result["errors"].append(f"Column '{col}' is entirely null")

        result["is_valid"] = len(result["errors"]) == 0 and len(df) > 0
        return result
