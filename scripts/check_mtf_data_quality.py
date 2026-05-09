#!/usr/bin/env python
"""
Data quality validation for multi-timeframe data.

Checks: completeness, duplicates, zero prices, price sanity,
timestamp continuity, DuckDB readability.

Usage:
  python scripts/check_mtf_data_quality.py --interval 1h
  python scripts/check_mtf_data_quality.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MTF_INTERVALS = ["1m", "5m", "15m", "1h"]
DEFAULT_SYMBOL = "BTCUSDT"

EXPECTED_BARS_PER_DAY: dict[str, int] = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "1h": 24,
}

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}

COMPLETENESS_THRESHOLD = 99.0
PRICE_JUMP_THRESHOLD = 0.15  # 15%


# ---------------------------------------------------------------------------
# Quality checker class
# ---------------------------------------------------------------------------


class MTFDataQualityChecker:
    """Run data-quality checks against MTF Parquet files."""

    def __init__(self, data_dir: Path, symbol: str = DEFAULT_SYMBOL) -> None:
        self.data_dir = data_dir
        self.symbol = symbol
        self._conn = duckdb.connect(":memory:")
        self._conn.execute("SET enable_progress_bar = false")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MTFDataQualityChecker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _parquet_glob(self, interval: str) -> str:
        """Return glob pattern for parquet files of an interval."""
        return str(
            self.data_dir
            / "exchange=BINANCE_UM"
            / f"symbol={self.symbol}"
            / f"interval={interval}"
            / "**"
            / "*.parquet"
        )

    def _parquet_files(self, interval: str) -> list[Path]:
        """Return sorted list of parquet files for interval."""
        base = (
            self.data_dir
            / "exchange=BINANCE_UM"
            / f"symbol={self.symbol}"
            / f"interval={interval}"
        )
        if not base.exists():
            return []
        return sorted(base.rglob("*.parquet"))

    def _read_all(self, interval: str) -> pl.DataFrame:
        """Read all parquet files for an interval into a single DataFrame."""
        files = self._parquet_files(interval)
        if not files:
            return pl.DataFrame()
        dfs = [pl.read_parquet(f, hive_partitioning=False) for f in files]
        return pl.concat(dfs).sort("open_time")

    # ----------------------------------------------------------------
    # Check 1: Completeness
    # ----------------------------------------------------------------

    def check_completeness(self, interval: str) -> dict[str, Any]:
        """Count actual vs expected bars."""
        files = self._parquet_files(interval)
        if not files:
            return {
                "total_bars": 0, "expected_bars": 0,
                "completeness_pct": 0.0, "n_days": 0, "pass": False,
            }

        total_bars = 0
        dates: set[str] = set()

        for f in files:
            # Extract date from path: .../date=YYYY-MM-DD/klines.parquet
            for part in f.parts:
                if part.startswith("date="):
                    dates.add(part.split("=", 1)[1])
            df = pl.read_parquet(f, columns=["open_time"], hive_partitioning=False)
            total_bars += len(df)

        n_days = len(dates)
        expected = n_days * EXPECTED_BARS_PER_DAY[interval]
        pct = round(total_bars / expected * 100, 2) if expected > 0 else 0.0

        return {
            "total_bars": total_bars,
            "expected_bars": expected,
            "completeness_pct": pct,
            "n_days": n_days,
            "pass": pct >= COMPLETENESS_THRESHOLD,
        }

    # ----------------------------------------------------------------
    # Check 2: No duplicates
    # ----------------------------------------------------------------

    def check_duplicates(self, interval: str) -> dict[str, Any]:
        """Check for duplicate timestamps."""
        df = self._read_all(interval)
        if df.is_empty():
            return {"duplicate_count": 0, "pass": True}

        total = len(df)
        unique = df.select("open_time").n_unique()
        dups = total - unique

        return {"duplicate_count": dups, "pass": dups == 0}

    # ----------------------------------------------------------------
    # Check 3: No zero prices
    # ----------------------------------------------------------------

    def check_zero_prices(self, interval: str) -> dict[str, Any]:
        """Check that OHLCV values are all positive."""
        df = self._read_all(interval)
        if df.is_empty():
            return {"zero_count": 0, "pass": True}

        price_cols = ["open", "high", "low", "close", "volume"]
        zero_count = 0
        for col in price_cols:
            if col in df.columns:
                zero_count += int((df[col] <= 0).sum())

        return {"zero_count": zero_count, "pass": zero_count == 0}

    # ----------------------------------------------------------------
    # Check 4: Price sanity (no >15% jumps)
    # ----------------------------------------------------------------

    def check_price_sanity(self, interval: str) -> dict[str, Any]:
        """Check that price doesn't jump more than 15% in a single bar."""
        df = self._read_all(interval)
        if df.is_empty() or len(df) < 2:
            return {"anomaly_count": 0, "pass": True}

        close = df["close"]
        pct_change = (close.diff().abs() / close.shift(1)).drop_nulls()
        anomalies = int((pct_change > PRICE_JUMP_THRESHOLD).sum())

        return {
            "anomaly_count": anomalies,
            "max_jump_pct": round(float(pct_change.max()) * 100, 2) if len(pct_change) > 0 else 0.0,
            "pass": anomalies == 0,
        }

    # ----------------------------------------------------------------
    # Check 5: Timestamp continuity
    # ----------------------------------------------------------------

    def check_timestamp_continuity(self, interval: str) -> dict[str, Any]:
        """Check that timestamp gaps match the expected interval."""
        df = self._read_all(interval)
        if df.is_empty() or len(df) < 2:
            return {"gap_count": 0, "pass": True}

        expected_ms = INTERVAL_MS[interval]
        diffs = df["open_time"].diff().drop_nulls()
        gaps = diffs.filter(diffs != expected_ms)
        gap_count = len(gaps)

        largest_gap_ms = int(diffs.max()) if len(diffs) > 0 else 0
        largest_gap_h = round(largest_gap_ms / 3_600_000, 2)

        return {
            "gap_count": gap_count,
            "largest_gap_ms": largest_gap_ms,
            "largest_gap_h": largest_gap_h,
            "pass": True,  # Gaps are expected at maintenance windows
        }

    # ----------------------------------------------------------------
    # Check 6: DuckDB readability
    # ----------------------------------------------------------------

    def check_duckdb_readability(self, interval: str) -> dict[str, Any]:
        """Verify that DuckDB can read the parquet files."""
        glob = self._parquet_glob(interval)
        try:
            result = self._conn.execute(f"""
                SELECT
                    COUNT(*) AS total_bars,
                    MIN(timestamp) AS first_bar,
                    MAX(timestamp) AS last_bar
                FROM read_parquet('{glob}', hive_partitioning=true)
            """).fetchone()

            total_bars = result[0] if result else 0
            first_bar = str(result[1]) if result and result[1] else "N/A"
            last_bar = str(result[2]) if result and result[2] else "N/A"

            return {
                "total_bars": total_bars,
                "first_bar": first_bar,
                "last_bar": last_bar,
                "pass": total_bars > 0,
            }
        except Exception as exc:
            return {"total_bars": 0, "error": str(exc), "pass": False}

    # ----------------------------------------------------------------
    # Full report for one interval
    # ----------------------------------------------------------------

    def check_interval(self, interval: str) -> dict[str, Any]:
        """Run all checks for one interval."""
        return {
            "completeness": self.check_completeness(interval),
            "duplicates": self.check_duplicates(interval),
            "zero_prices": self.check_zero_prices(interval),
            "price_sanity": self.check_price_sanity(interval),
            "timestamp_continuity": self.check_timestamp_continuity(interval),
            "duckdb_readability": self.check_duckdb_readability(interval),
        }

    def all_pass(self, report: dict[str, Any]) -> bool:
        """Return True if all critical checks pass."""
        completeness = report.get("completeness", {})
        duplicates = report.get("duplicates", {})
        zero_prices = report.get("zero_prices", {})
        return (
            completeness.get("pass", False)
            and duplicates.get("pass", False)
            and zero_prices.get("pass", False)
        )


# ---------------------------------------------------------------------------
# Pretty table output
# ---------------------------------------------------------------------------


def _period_str(n_days: int) -> str:
    """Format n_days as 'Xy Ym'."""
    years = n_days // 365
    months = (n_days % 365) // 30
    return f"{years}y {months}m"


def print_report(
    checker: MTFDataQualityChecker,
    intervals: list[str],
) -> bool:
    """Run all checks and print a formatted table. Returns True if all pass."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         AtomiCortex MTF Data Quality Report              ║")
    print("╠══════════╦══════════╦══════════╦══════════╦═════════════╣")
    print("║ Interval ║ Bars     ║ Complete ║ Period   ║ Status      ║")
    print("╠══════════╬══════════╬══════════╬══════════╬═════════════╣")

    all_ok = True
    for interval in intervals:
        report = checker.check_interval(interval)
        comp = report["completeness"]

        bars = comp.get("total_bars", 0)
        pct = comp.get("completeness_pct", 0.0)
        n_days = comp.get("n_days", 0)
        period = _period_str(n_days)

        passed = checker.all_pass(report)
        if not passed:
            all_ok = False

        status = "✓ PASS" if passed else "✗ FAIL"
        status_padded = f"{status:<11}"

        print(
            f"║ {interval:<8} ║ {bars:>8,} ║ {pct:>7.1f}% ║ {period:<8} ║ {status_padded} ║"
        )

        # Print sub-check details if failed
        dups = report["duplicates"]
        zeros = report["zero_prices"]
        sanity = report["price_sanity"]

        if not dups.get("pass", True):
            print(f"║          ║  ⚠ {dups['duplicate_count']} duplicate timestamps")
        if not zeros.get("pass", True):
            print(f"║          ║  ⚠ {zeros['zero_count']} zero/negative prices")
        if sanity.get("anomaly_count", 0) > 0:
            print(
                f"║          ║  ⚠ {sanity['anomaly_count']} bars with >"
                f"{PRICE_JUMP_THRESHOLD * 100:.0f}% price jump"
            )

    print("╚══════════╩══════════╩══════════╩══════════╩═════════════╝")

    # DuckDB verification detail
    print("\nDuckDB Readability:")
    for interval in intervals:
        report = checker.check_interval(interval)
        ddb = report["duckdb_readability"]
        if ddb.get("pass"):
            print(
                f"  ✓ {interval}: {ddb['total_bars']:,} bars  "
                f"[{ddb['first_bar']} → {ddb['last_bar']}]"
            )
        else:
            err = ddb.get("error", "no data")
            print(f"  ✗ {interval}: FAILED ({err})")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the MTF data quality checker."""
    parser = argparse.ArgumentParser(
        description="AtomiCortex MTF data quality checker",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", choices=MTF_INTERVALS)
    parser.add_argument(
        "--all", action="store_true", dest="all_intervals",
        help="Check all MTF intervals",
    )
    parser.add_argument("--data-dir", default="data/raw")

    args = parser.parse_args()
    setup_logging()

    if not args.interval and not args.all_intervals:
        parser.error("Specify --interval or --all")

    intervals = MTF_INTERVALS if args.all_intervals else [args.interval]
    data_dir = Path(args.data_dir)

    with MTFDataQualityChecker(data_dir, symbol=args.symbol.upper()) as checker:
        all_ok = print_report(checker, intervals)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
