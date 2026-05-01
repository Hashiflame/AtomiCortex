"""
src/ingestion/data_store.py

DuckDB-based interface for querying Parquet data produced by ParquetConverter.
The connection is in-memory; data is read directly from Parquet files on disk
via DuckDB's read_parquet() — no data is loaded upfront.

Usage
-----
    store = DataStore(Path("/mnt/hdd/AtomiCortex/data/features"))
    df = store.get_klines("BTCUSDT", "4h", start=datetime(2024,1,1), end=datetime(2024,6,1))
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from src.logger import get_logger

_log = get_logger(__name__)


def _to_ms(dt: datetime) -> int:
    """Convert a datetime to unix milliseconds (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000)


def _glob(base: Path, *parts: str) -> str:
    """Build a glob string for DuckDB read_parquet — list form for robustness."""
    pattern = base.joinpath(*parts)
    return str(pattern)


class DataStore:
    """Query engine over the Parquet feature store.

    Parameters
    ----------
    data_dir:
        Root of the Parquet tree, e.g. ``/mnt/hdd/AtomiCortex/data/features``.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.conn: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")
        # Enable Hive-style partition reading globally
        self.conn.execute("SET enable_progress_bar = false")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parquet_files(self, symbol: str, data_type: str) -> list[str]:
        """Return sorted list of parquet file paths for a symbol/type."""
        base = (
            self.data_dir
            / "exchange=BINANCE_UM"
            / f"symbol={symbol}"
            / data_type
        )
        if not base.exists():
            return []
        return sorted(str(p) for p in base.rglob("*.parquet"))

    def _read_parquet_sql(
        self,
        files: list[str],
        where: str = "",
        cols: str = "*",
        order_by: str = "",
    ) -> str:
        """Build a DuckDB SQL statement for a list of parquet files."""
        if not files:
            return ""
        files_expr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        sql = f"SELECT {cols} FROM read_parquet({files_expr}, union_by_name=true)"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        return sql

    def _run(self, sql: str) -> pl.DataFrame:
        """Execute SQL and return Polars DataFrame. Returns empty DF on error."""
        if not sql:
            return pl.DataFrame()
        try:
            return self.conn.execute(sql).pl()
        except duckdb.IOException as exc:
            _log.warning(f"DuckDB IO error (no data?): {exc}")
            return pl.DataFrame()
        except Exception as exc:
            _log.error(f"DuckDB query failed: {exc}\nSQL: {sql}")
            raise

    # ------------------------------------------------------------------
    # Public: typed accessors
    # ------------------------------------------------------------------

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Return OHLCV klines for *symbol* / *interval* in the date range.

        Parameters
        ----------
        symbol:   Binance symbol, e.g. ``"BTCUSDT"``
        interval: ``"4h"`` or ``"1d"``
        start:    Inclusive start (UTC assumed if tz-naive)
        end:      Inclusive end
        columns:  Optional column subset; ``None`` returns all columns.
        """
        files = self._parquet_files(symbol, f"klines_{interval}")
        if not files:
            _log.warning(f"No klines_{interval} data for {symbol}")
            return pl.DataFrame()

        start_ms, end_ms = _to_ms(start), _to_ms(end)
        cols = "*" if not columns else ", ".join(columns)
        where = f"open_time >= {start_ms} AND open_time <= {end_ms}"
        sql = self._read_parquet_sql(files, where=where, cols=cols, order_by="open_time")
        df = self._run(sql)
        _log.debug(
            f"get_klines {symbol} {interval}: {len(df)} rows "
            f"({start.date()} → {end.date()})"
        )
        return df

    def get_funding_rate(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Return funding-rate rows for *symbol* in the date range."""
        files = self._parquet_files(symbol, "funding_rate")
        if not files:
            _log.warning(f"No funding_rate data for {symbol}")
            return pl.DataFrame()

        start_ms, end_ms = _to_ms(start), _to_ms(end)
        where = f"fundingTime >= {start_ms} AND fundingTime <= {end_ms}"
        sql = self._read_parquet_sql(files, where=where, order_by="fundingTime")
        return self._run(sql)

    def get_metrics(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Return open-interest / long-short metrics for *symbol* in the date range."""
        files = self._parquet_files(symbol, "metrics")
        if not files:
            _log.warning(f"No metrics data for {symbol}")
            return pl.DataFrame()

        start_ms, end_ms = _to_ms(start), _to_ms(end)
        where = f"create_time >= {start_ms} AND create_time <= {end_ms}"
        sql = self._read_parquet_sql(files, where=where, order_by="create_time")
        return self._run(sql)

    # ------------------------------------------------------------------
    # Public: arbitrary SQL
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Execute an arbitrary DuckDB SQL statement and return a Polars DataFrame.

        Useful for ad-hoc exploration in notebooks.  You can reference
        parquet files directly with ``read_parquet('path/**/*.parquet')``.
        """
        return self._run(sql)

    # ------------------------------------------------------------------
    # Public: summary
    # ------------------------------------------------------------------

    def get_data_summary(self) -> dict[str, dict[str, Any]]:
        """Return a summary of all available Parquet data.

        Returns
        -------
        dict keyed by ``"{symbol}/{data_type}"`` with fields:
        ``symbol``, ``data_type``, ``date_range``, ``row_count``,
        ``file_count``, ``size_mb``.
        """
        exchange_dir = self.data_dir / "exchange=BINANCE_UM"
        if not exchange_dir.exists():
            return {}

        summary: dict[str, dict[str, Any]] = {}

        for sym_dir in sorted(exchange_dir.iterdir()):
            if not sym_dir.is_dir() or not sym_dir.name.startswith("symbol="):
                continue
            symbol = sym_dir.name.split("=", 1)[1]

            for dtype_dir in sorted(sym_dir.iterdir()):
                if not dtype_dir.is_dir():
                    continue

                files = sorted(dtype_dir.rglob("*.parquet"))
                if not files:
                    continue

                size_mb = sum(f.stat().st_size for f in files) / 1_048_576

                # Date range from directory names (date=YYYY-MM-DD)
                dates = sorted(
                    d.name.split("=", 1)[1]
                    for d in dtype_dir.iterdir()
                    if d.is_dir() and d.name.startswith("date=")
                )
                date_range = (
                    f"{dates[0]} → {dates[-1]}" if len(dates) >= 2
                    else (dates[0] if dates else "—")
                )

                # Row count via DuckDB (fast — reads only metadata)
                try:
                    files_expr = (
                        "[" + ", ".join(f"'{f}'" for f in files) + "]"
                    )
                    row_count = self.conn.execute(
                        f"SELECT COUNT(*) FROM read_parquet({files_expr}, "
                        f"union_by_name=true)"
                    ).fetchone()[0]
                except Exception as exc:
                    _log.debug(f"Row count failed for {sym_dir.name}/{dtype_dir.name}: {exc}")
                    row_count = -1

                key = f"{symbol}/{dtype_dir.name}"
                summary[key] = {
                    "symbol": symbol,
                    "data_type": dtype_dir.name,
                    "date_range": date_range,
                    "row_count": row_count,
                    "file_count": len(files),
                    "size_mb": round(size_mb, 2),
                }

        return summary

    # ------------------------------------------------------------------
    # Context manager (optional convenience)
    # ------------------------------------------------------------------

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.conn.close()

    def close(self) -> None:
        self.conn.close()
