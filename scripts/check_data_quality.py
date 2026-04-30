#!/usr/bin/env python
"""
scripts/check_data_quality.py

Data-quality report for the AtomiCortex Parquet feature store.

Usage
-----
    python scripts/check_data_quality.py \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --data-dir /mnt/hdd/AtomiCortex/data/features

    # Specific data types only
    python scripts/check_data_quality.py \\
        --symbols BTCUSDT \\
        --data-types klines_4h,funding_rate \\
        --data-dir /mnt/hdd/AtomiCortex/data/features

Pass criteria (from master spec)
---------------------------------
  Completeness  >= 99.9 %
  Gaps          == 0
  Nulls         == 0
  Anomalies     == 0
  Clock drift   < 50 ms  (agg_trades only)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import click

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.data_quality import (
    AGG_TRADES_SAMPLE,
    THRESHOLD_COMPLETENESS_PCT,
    THRESHOLD_DRIFT_MS,
    DataQualityChecker,
    row_passes,
)
from src.ingestion.parquet_converter import _CONFIGS as _ALL_TYPES
from src.logger import get_logger, setup_logging

_COLUMN_WIDTHS = {
    "symbol":   10,
    "dtype":    13,
    "complete": 10,
    "gaps":      6,
    "nulls":     7,
    "anomalies": 9,
    "drift":     8,
    "valid":     5,
}

_HDR = (
    f"  {'Symbol':<{_COLUMN_WIDTHS['symbol']}} "
    f"{'Type':<{_COLUMN_WIDTHS['dtype']}} "
    f"{'Complete%':>{_COLUMN_WIDTHS['complete']}} "
    f"{'Gaps':>{_COLUMN_WIDTHS['gaps']}} "
    f"{'Nulls':>{_COLUMN_WIDTHS['nulls']}} "
    f"{'Anomalies':>{_COLUMN_WIDTHS['anomalies']}} "
    f"{'Drift ms':>{_COLUMN_WIDTHS['drift']}} "
    f"{'OK':^{_COLUMN_WIDTHS['valid']}}"
)
_SEP = "  " + "─" * (len(_HDR) - 2)


def _fmt_gaps(gaps: dict[str, Any]) -> str:
    if gaps.get("skipped"):
        return "N/A"
    return str(gaps.get("gap_count", 0))


def _fmt_drift(drift: dict[str, Any] | None) -> str:
    if drift is None:
        return "N/A"
    return str(drift.get("drift_ms_max", 0))


def _print_report(report: dict[str, Any], data_types: list[str]) -> None:
    click.echo(f"\n{_SEP}")
    click.echo(_HDR)
    click.echo(_SEP)

    total_checks = 0
    total_pass   = 0
    issues: list[str] = []

    for symbol, sym_data in sorted(report.items()):
        clock_drift = sym_data.get("_clock_drift")

        for dt in data_types:
            if dt not in sym_data:
                continue

            entry        = sym_data[dt]
            completeness = entry["completeness"]
            gaps         = entry["gaps"]
            integrity    = entry["integrity"]

            pct          = completeness.get("completeness_pct", 0.0)
            gap_count    = gaps.get("gap_count", 0) if not gaps.get("skipped") else 0
            null_count   = integrity.get("null_count", 0)
            anomaly_count= integrity.get("anomaly_count", 0)
            drift_str    = _fmt_drift(clock_drift) if dt == "agg_trades" else "N/A"
            drift_obj    = clock_drift if dt == "agg_trades" else None

            passed = row_passes(symbol, dt, completeness, gaps, integrity, drift_obj)
            total_checks += 1
            if passed:
                total_pass += 1
            else:
                reasons = []
                if pct < THRESHOLD_COMPLETENESS_PCT:
                    reasons.append(f"completeness={pct:.2f}%")
                if gap_count > 0:
                    reasons.append(f"gaps={gap_count}")
                if null_count > 0:
                    reasons.append(f"nulls={null_count}")
                if anomaly_count > 0:
                    reasons.append(f"anomalies={anomaly_count}")
                if drift_obj and not drift_obj.get("is_monotonic", True):
                    reasons.append("non-monotonic")
                if drift_obj and drift_obj.get("drift_ms_max", 0) >= THRESHOLD_DRIFT_MS:
                    reasons.append(f"drift={drift_obj['drift_ms_max']}ms")
                issues.append(f"  ❌ {symbol}/{dt}: {', '.join(reasons)}")

            valid_mark = "✅" if passed else "❌"

            click.echo(
                f"  {symbol:<{_COLUMN_WIDTHS['symbol']}} "
                f"{dt:<{_COLUMN_WIDTHS['dtype']}} "
                f"{pct:>{_COLUMN_WIDTHS['complete']}.2f}% "
                f"{_fmt_gaps(gaps):>{_COLUMN_WIDTHS['gaps']}} "
                f"{null_count:>{_COLUMN_WIDTHS['nulls']}} "
                f"{anomaly_count:>{_COLUMN_WIDTHS['anomalies']}} "
                f"{drift_str:>{_COLUMN_WIDTHS['drift']}} "
                f"{valid_mark:^{_COLUMN_WIDTHS['valid']}}"
            )

    click.echo(_SEP)
    click.echo(f"\n  Pass: {total_pass}/{total_checks} checks")

    if issues:
        click.echo(f"\n  Issues found:")
        for issue in issues:
            click.echo(issue)
    else:
        click.echo("  All checks passed ✅")

    click.echo()


def _print_gaps_detail(report: dict[str, Any], data_types: list[str]) -> None:
    """Print detailed gap locations if any gaps were found."""
    any_gap = False
    for symbol, sym_data in sorted(report.items()):
        for dt in data_types:
            if dt not in sym_data:
                continue
            gaps = sym_data[dt]["gaps"]
            if gaps.get("skipped") or gaps.get("gap_count", 0) == 0:
                continue
            if not any_gap:
                click.echo("  Gap details:")
                any_gap = True
            click.echo(f"    {symbol}/{dt}: {gaps['gap_count']} gap(s), "
                       f"largest={gaps.get('largest_gap_h', 0):.1f}h")
            for loc in gaps.get("gap_locations", [])[:5]:
                click.echo(f"      → {loc['at']}  ({loc['gap_h']:.1f}h)")


def _print_completeness_detail(report: dict[str, Any], data_types: list[str]) -> None:
    """Print missing dates for incomplete symbols."""
    for symbol, sym_data in sorted(report.items()):
        for dt in data_types:
            if dt not in sym_data:
                continue
            c = sym_data[dt]["completeness"]
            missing = c.get("missing_dates", [])
            if missing:
                click.echo(f"  Missing partitions for {symbol}/{dt} "
                           f"(showing first 10 of {len(missing)}):")
                for d in missing[:10]:
                    click.echo(f"    date={d}")


@click.command()
@click.option(
    "--symbols",
    default="BTCUSDT,ETHUSDT,SOLUSDT",
    show_default=True,
    help="Comma-separated Binance symbols.",
)
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(),
    help="Path to the Parquet feature store root.",
)
@click.option(
    "--data-types",
    default=None,
    help=(
        f"Comma-separated data types to check. "
        f"Default: all ({', '.join(_ALL_TYPES)})."
    ),
)
@click.option(
    "--detail",
    is_flag=True,
    default=False,
    help="Print gap locations and missing dates for failed checks.",
)
def main(symbols: str, data_dir: str, data_types: str | None, detail: bool) -> None:
    """Run data-quality checks on the Parquet feature store."""
    setup_logging()
    log = get_logger(__name__)

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    dt_list: list[str] = (
        [d.strip() for d in data_types.split(",") if d.strip()]
        if data_types
        else list(_ALL_TYPES)
    )
    feat_dir = Path(data_dir)

    if not feat_dir.exists():
        click.echo(f"\n⚠️  Directory not found: {feat_dir}\n")
        sys.exit(1)

    click.echo(
        f"\n{'─'*60}\n"
        f"  AtomiCortex — Data Quality Report\n"
        f"  Store     : {feat_dir}\n"
        f"  Symbols   : {', '.join(symbol_list)}\n"
        f"  Types     : {', '.join(dt_list)}\n"
        f"  Thresholds: completeness>={THRESHOLD_COMPLETENESS_PCT}%  "
        f"drift<{THRESHOLD_DRIFT_MS}ms\n"
        f"  Note      : agg_trades sampled ({AGG_TRADES_SAMPLE} rows/file)\n"
        f"{'─'*60}"
    )

    t0 = time.monotonic()

    with DataQualityChecker(feat_dir) as checker:
        report = checker.full_report(symbol_list, dt_list)

    elapsed = time.monotonic() - t0
    _print_report(report, dt_list)

    if detail:
        _print_gaps_detail(report, dt_list)
        _print_completeness_detail(report, dt_list)

    click.echo(f"  Elapsed: {elapsed:.1f}s\n")

    # Exit 1 if any check failed
    any_fail = any(
        not row_passes(
            sym, dt,
            report[sym][dt]["completeness"],
            report[sym][dt]["gaps"],
            report[sym][dt]["integrity"],
            report[sym].get("_clock_drift") if dt == "agg_trades" else None,
        )
        for sym in report
        for dt in dt_list
        if dt in report[sym]
    )
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
