"""
src/ingestion/data_quality.py

Data quality checks for the AtomiCortex Parquet feature store.

Five checks:
  1. check_completeness  — are all expected date/month partitions present?
  2. check_gaps          — timestamp gaps larger than the expected interval?
  3. check_data_integrity— null values and domain-rule anomalies
  4. check_clock_drift   — agg_trades transact_time monotonicity (sampled)
  5. full_report         — all checks for N symbols × M data types

agg_trades note
---------------
agg_trades files can contain hundreds of millions of rows each.
All agg_trades checks use a sample of the first AGG_TRADES_SAMPLE rows
per file to keep runtime reasonable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from src.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-type configuration
# ---------------------------------------------------------------------------

# Primary timestamp column for each data type
_TIMESTAMP_COL: dict[str, str] = {
    "klines_4h":    "open_time",
    "klines_1d":    "open_time",
    "funding_rate": "fundingTime",
    "metrics":      "create_time",
    "agg_trades":   "transact_time",
}

# Gap threshold in milliseconds — gaps strictly LARGER than this value are flagged.
# None = skip gap check for this type.
_GAP_THRESHOLD_MS: dict[str, int | None] = {
    "klines_4h":    4 * 3_600_000,      # > 4 h  (one missing 4h bar)
    "klines_1d":    24 * 3_600_000,     # > 1 day
    "funding_rate": 9 * 3_600_000,      # > 9 h  (funding every 8 h)
    "metrics":      None,               # skip (every 5 min, too many rows)
    "agg_trades":   None,               # skip (use clock_drift instead)
}

# Key columns checked for NULL values per data type
_KEY_COLS: dict[str, list[str]] = {
    "klines_4h":    ["open_time", "open", "high", "low", "close", "volume"],
    "klines_1d":    ["open_time", "open", "high", "low", "close", "volume"],
    "funding_rate": ["fundingTime", "fundingRate"],
    "metrics":      ["create_time", "sum_open_interest"],
    "agg_trades":   ["transact_time", "price", "quantity"],
}

AGG_TRADES_SAMPLE: int = 1_000   # rows sampled per agg_trades file

# Quality pass thresholds
THRESHOLD_COMPLETENESS_PCT: float = 99.9
THRESHOLD_DRIFT_MS: int = 50


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------

class DataQualityChecker:
    """Run data-quality checks against the Parquet feature store.

    Parameters
    ----------
    data_dir:
        Root of the Parquet tree (``exchange=BINANCE_UM/…``).
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")
        self._conn.execute("SET enable_progress_bar = false")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _type_dir(self, symbol: str, data_type: str) -> Path:
        return (
            self.data_dir
            / "exchange=BINANCE_UM"
            / f"symbol={symbol}"
            / data_type
        )

    def _date_dirs(self, symbol: str, data_type: str) -> list[str]:
        """Return sorted list of ``date=XXXX`` directory names."""
        base = self._type_dir(symbol, data_type)
        if not base.exists():
            return []
        return sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and d.name.startswith("date=")
        )

    def _parquet_files(self, symbol: str, data_type: str) -> list[str]:
        base = self._type_dir(symbol, data_type)
        if not base.exists():
            return []
        return sorted(str(p) for p in base.rglob("*.parquet"))

    @staticmethod
    def _is_monthly(date_dirs: list[str]) -> bool:
        if not date_dirs:
            return False
        return len(date_dirs[0].split("=", 1)[1]) == 7  # "YYYY-MM"

    # ------------------------------------------------------------------
    # 1. check_completeness
    # ------------------------------------------------------------------

    def check_completeness(
        self,
        symbol: str,
        data_type: str,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        """Count present vs expected date/month partitions.

        If *start*/*end* are omitted the range is inferred from the
        earliest and latest partition directories found on disk.

        Returns
        -------
        dict with ``expected``, ``found``, ``missing_dates``,
        ``completeness_pct``.
        """
        date_dirs = self._date_dirs(symbol, data_type)

        if not date_dirs:
            return {
                "expected": 0,
                "found": 0,
                "missing_dates": [],
                "completeness_pct": 0.0,
                "error": "no partition directories found",
            }

        found_set = {d.split("=", 1)[1] for d in date_dirs}
        monthly = self._is_monthly(date_dirs)

        if monthly:
            all_months = sorted(found_set)
            inferred_start = _parse_month(all_months[0])
            inferred_end   = _parse_month(all_months[-1])
            eff_start = start or inferred_start
            eff_end   = end   or inferred_end
            expected_set = _month_range(eff_start, eff_end)
        else:
            all_dates = sorted(found_set)
            eff_start = start or date.fromisoformat(all_dates[0])
            eff_end   = end   or date.fromisoformat(all_dates[-1])
            expected_set = {
                (eff_start + timedelta(days=i)).isoformat()
                for i in range((eff_end - eff_start).days + 1)
            }

        missing = sorted(expected_set - found_set)
        found   = len(found_set & expected_set)
        expected = len(expected_set)
        pct = round(found / expected * 100, 4) if expected > 0 else 0.0

        return {
            "expected":         expected,
            "found":            found,
            "missing_dates":    missing[:50],   # cap output at 50
            "completeness_pct": pct,
        }

    # ------------------------------------------------------------------
    # 2. check_gaps
    # ------------------------------------------------------------------

    def check_gaps(self, symbol: str, data_type: str) -> dict[str, Any]:
        """Find timestamp gaps larger than the expected bar interval.

        Returns
        -------
        dict with ``gap_count``, ``largest_gap_ms``, ``largest_gap_h``,
        ``gap_locations`` (list of up to 10 largest gaps).
        """
        threshold = _GAP_THRESHOLD_MS.get(data_type)
        ts_col = _TIMESTAMP_COL.get(data_type)

        if threshold is None or ts_col is None:
            return {"gap_count": 0, "largest_gap_ms": 0, "gap_locations": [], "skipped": True}

        files = self._parquet_files(symbol, data_type)
        if not files:
            return {"gap_count": 0, "largest_gap_ms": 0, "gap_locations": [], "error": "no files"}

        files_expr = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        sql = f"""
            WITH ordered AS (
                SELECT {ts_col} AS ts,
                       {ts_col} - LAG({ts_col}) OVER (ORDER BY {ts_col}) AS diff_ms
                FROM read_parquet({files_expr}, union_by_name=true)
            )
            SELECT ts, diff_ms
            FROM ordered
            WHERE diff_ms > {threshold}
            ORDER BY diff_ms DESC
        """

        try:
            gaps_df = self._conn.execute(sql).pl()
        except Exception as exc:
            _log.error(f"Gap check failed for {symbol}/{data_type}: {exc}")
            return {"gap_count": 0, "largest_gap_ms": 0, "gap_locations": [], "error": str(exc)}

        gap_count = len(gaps_df)
        largest_gap_ms = int(gaps_df["diff_ms"].max()) if gap_count > 0 else 0

        locations: list[dict[str, Any]] = []
        for row in gaps_df.head(10).iter_rows(named=True):
            ts_ms = int(row["ts"])
            dt_str = datetime.fromtimestamp(ts_ms / 1_000, tz=timezone.utc).isoformat()
            gap_ms = int(row["diff_ms"])
            locations.append({"at": dt_str, "gap_ms": gap_ms, "gap_h": round(gap_ms / 3_600_000, 2)})

        return {
            "gap_count":       gap_count,
            "largest_gap_ms":  largest_gap_ms,
            "largest_gap_h":   round(largest_gap_ms / 3_600_000, 2),
            "gap_locations":   locations,
        }

    # ------------------------------------------------------------------
    # 3. check_data_integrity
    # ------------------------------------------------------------------

    def check_data_integrity(self, symbol: str, data_type: str) -> dict[str, Any]:
        """Check for NULL values and domain-rule anomalies.

        agg_trades is sampled (first ``AGG_TRADES_SAMPLE`` rows per file).

        Returns
        -------
        dict with ``null_count``, ``anomaly_count``, ``is_valid``.
        """
        files = self._parquet_files(symbol, data_type)
        if not files:
            return {"null_count": 0, "anomaly_count": 0, "is_valid": False, "error": "no files"}

        key_cols = _KEY_COLS.get(data_type, [])

        if data_type == "agg_trades":
            return self._integrity_sampled(files, key_cols, data_type)

        return self._integrity_duckdb(files, data_type, key_cols)

    def _integrity_duckdb(
        self,
        files: list[str],
        data_type: str,
        key_cols: list[str],
    ) -> dict[str, Any]:
        files_expr = "[" + ", ".join(f"'{f}'" for f in files) + "]"

        null_expr = (
            " + ".join(f"COUNT(*) FILTER (WHERE {c} IS NULL)" for c in key_cols)
            if key_cols else "0"
        )
        anomaly_expr = _anomaly_sql(data_type)

        sql = f"""
            SELECT
                ({null_expr})  AS null_count,
                {anomaly_expr} AS anomaly_count
            FROM read_parquet({files_expr}, union_by_name=true)
        """

        try:
            row = self._conn.execute(sql).fetchone()
        except Exception as exc:
            _log.error(f"Integrity check failed for {data_type}: {exc}")
            return {"null_count": 0, "anomaly_count": 0, "is_valid": False, "error": str(exc)}

        null_count    = int(row[0]) if row and row[0] is not None else 0
        anomaly_count = int(row[1]) if row and row[1] is not None else 0

        return {
            "null_count":    null_count,
            "anomaly_count": anomaly_count,
            "is_valid":      null_count == 0 and anomaly_count == 0,
        }

    def _integrity_sampled(
        self,
        files: list[str],
        key_cols: list[str],
        data_type: str,
    ) -> dict[str, Any]:
        """Sample-based integrity check for large data types (agg_trades)."""
        null_count    = 0
        anomaly_count = 0

        for f in files:
            try:
                df = (
                    pl.scan_parquet(f, hive_partitioning=False)
                    .head(AGG_TRADES_SAMPLE)
                    .collect()
                )
            except Exception as exc:
                _log.warning(f"Cannot read sample from {f}: {exc}")
                continue

            for col in key_cols:
                if col in df.columns:
                    null_count += int(df[col].null_count())

            if "price" in df.columns:
                anomaly_count += int((df["price"] <= 0).sum())
            if "quantity" in df.columns:
                anomaly_count += int((df["quantity"] <= 0).sum())

        return {
            "null_count":       null_count,
            "anomaly_count":    anomaly_count,
            "is_valid":         null_count == 0 and anomaly_count == 0,
            "sampled":          True,
            "sample_per_file":  AGG_TRADES_SAMPLE,
        }

    # ------------------------------------------------------------------
    # 4. check_clock_drift
    # ------------------------------------------------------------------

    def check_clock_drift(self, symbol: str) -> dict[str, Any]:
        """Check agg_trades transact_time monotonicity and max intra-file tick gap.

        Samples first ``AGG_TRADES_SAMPLE`` rows from every agg_trades file.
        Monotonicity and drift are evaluated **per file** to avoid counting
        expected ~24h gaps between daily partition boundaries as drift.

        Returns
        -------
        dict with ``is_monotonic``, ``drift_ms_max``, ``files_sampled``,
        ``rows_sampled``.
        """
        files = self._parquet_files(symbol, "agg_trades")
        if not files:
            return {"is_monotonic": True, "drift_ms_max": 0, "error": "no agg_trades files"}

        all_monotonic = True
        max_intra_drift = 0
        rows_total = 0

        for f in files:
            try:
                df = (
                    pl.scan_parquet(f, hive_partitioning=False)
                    .select("transact_time")
                    .head(AGG_TRADES_SAMPLE)
                    .collect()
                )
            except Exception as exc:
                _log.warning(f"Cannot sample clock from {f}: {exc}")
                continue

            ts = df["transact_time"]
            rows_total += len(ts)

            if not ts.is_sorted():
                all_monotonic = False

            diffs = ts.diff().drop_nulls().abs()
            if len(diffs) > 0:
                file_max = int(diffs.max())
                if file_max > max_intra_drift:
                    max_intra_drift = file_max

        if rows_total == 0:
            return {"is_monotonic": True, "drift_ms_max": 0, "files_sampled": 0, "rows_sampled": 0}

        return {
            "is_monotonic":  all_monotonic,
            "drift_ms_max":  max_intra_drift,
            "files_sampled": len(files),
            "rows_sampled":  rows_total,
        }

    # ------------------------------------------------------------------
    # 5. full_report
    # ------------------------------------------------------------------

    def full_report(
        self,
        symbols: list[str],
        data_types: list[str],
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        """Run all checks for every symbol × data_type combination.

        Returns
        -------
        Nested dict keyed by ``symbol → data_type → {completeness, gaps, integrity}``.
        Clock-drift result for agg_trades is stored under ``symbol → "_clock_drift"``.
        """
        report: dict[str, Any] = {}

        for symbol in symbols:
            report[symbol] = {}

            for dt in data_types:
                _log.info(f"Checking {symbol}/{dt} …")

                completeness = self.check_completeness(symbol, dt, start, end)

                # Gap check: skip for agg_trades (use clock_drift instead)
                if dt != "agg_trades":
                    gaps = self.check_gaps(symbol, dt)
                else:
                    gaps = {"gap_count": 0, "largest_gap_ms": 0, "gap_locations": [], "skipped": True}

                integrity = self.check_data_integrity(symbol, dt)

                report[symbol][dt] = {
                    "completeness": completeness,
                    "gaps":         gaps,
                    "integrity":    integrity,
                }

            if "agg_trades" in data_types:
                _log.info(f"Checking {symbol}/clock_drift …")
                report[symbol]["_clock_drift"] = self.check_clock_drift(symbol)

        return report

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DataQualityChecker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_month(month_str: str) -> date:
    """``"2024-01"`` → ``date(2024, 1, 1)``."""
    y, m = map(int, month_str.split("-"))
    return date(y, m, 1)


def _month_range(start: date, end: date) -> set[str]:
    """All ``"YYYY-MM"`` strings from *start* month to *end* month (inclusive)."""
    months: set[str] = set()
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y, m) <= (end_y, end_m):
        months.add(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def _anomaly_sql(data_type: str) -> str:
    """SQL expression counting rows that violate domain rules."""
    if data_type in ("klines_4h", "klines_1d"):
        return "COUNT(*) FILTER (WHERE high < low OR close <= 0 OR volume < 0)"
    if data_type == "funding_rate":
        return "COUNT(*) FILTER (WHERE ABS(fundingRate) >= 0.05)"
    if data_type == "metrics":
        return "COUNT(*) FILTER (WHERE sum_open_interest < 0)"
    return "0"


def row_passes(
    symbol: str,
    data_type: str,
    completeness: dict[str, Any],
    gaps: dict[str, Any],
    integrity: dict[str, Any],
    clock_drift: dict[str, Any] | None = None,
) -> bool:
    """Return True when all quality thresholds are met for one symbol/type.

    clock_drift note
    ----------------
    ``drift_ms_max`` in stored agg_trades is the max gap between consecutive
    trades within a file sample — a market liquidity metric, not clock skew.
    Only ``is_monotonic=False`` is treated as a data-quality failure here.
    The 50ms clock-drift threshold applies to the live feed
    (receipt_timestamp vs exchange timestamp), not to historical Parquet.
    """
    if completeness.get("completeness_pct", 0) < THRESHOLD_COMPLETENESS_PCT:
        return False
    if not gaps.get("skipped") and gaps.get("gap_count", 0) > 0:
        return False
    if integrity.get("null_count", 0) > 0:
        return False
    if integrity.get("anomaly_count", 0) > 0:
        return False
    if clock_drift is not None:
        if not clock_drift.get("is_monotonic", True):
            return False
    return True
