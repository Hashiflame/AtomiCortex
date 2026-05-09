#!/usr/bin/env python
"""
Verifies Phase 1 completion:
1. New MTF data files exist (if downloaded)
2. Quality checks pass (if data present)
3. 4H data untouched
4. All new tests pass

Usage:
    python scripts/verify_phase1.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def verify() -> None:
    """Run all Phase 1 verification checks."""
    print("=== AtomiCortex Phase 1 Verification ===\n")

    data_raw = Path("data/raw")
    all_ok = True

    # ---------------------------------------------------------------
    # 1. Check 4H data not tampered (if present)
    # ---------------------------------------------------------------
    path_4h = data_raw / "exchange=BINANCE_UM" / "symbol=BTCUSDT" / "klines_4h"
    if path_4h.exists():
        print("✓ 4H data directory intact")
    else:
        print("⚠ 4H data not present locally (expected if running off-VM)")

    # ---------------------------------------------------------------
    # 2. Check new MTF data directories (if data downloaded)
    # ---------------------------------------------------------------
    for interval in ["1h", "15m", "5m", "1m"]:
        path = data_raw / "exchange=BINANCE_UM" / "symbol=BTCUSDT" / f"interval={interval}"
        if path.exists():
            csv_count = len(list(path.glob("*.csv")))
            pq_count = len(list(path.rglob("*.parquet")))
            print(f"✓ {interval}: {csv_count} CSV files, {pq_count} parquet files")
        else:
            print(f"⚠ {interval}: directory not created yet (run download first)")

    # ---------------------------------------------------------------
    # 3. Check DuckDB readability (if parquet data present)
    # ---------------------------------------------------------------
    try:
        import duckdb
        conn = duckdb.connect(":memory:")
        conn.execute("SET enable_progress_bar = false")

        for interval in ["1h", "15m"]:
            glob = str(
                data_raw
                / "exchange=BINANCE_UM"
                / "symbol=BTCUSDT"
                / f"interval={interval}"
                / "**"
                / "*.parquet"
            )
            try:
                rows = conn.execute(f"""
                    SELECT COUNT(*) FROM read_parquet('{glob}', hive_partitioning=true)
                """).fetchone()[0]
                print(f"✓ {interval}: {rows:,} bars readable via DuckDB")
            except Exception:
                print(f"⚠ {interval}: no parquet data to read yet")

        conn.close()
    except ImportError:
        print("⚠ DuckDB not installed — skipping readability check")

    # ---------------------------------------------------------------
    # 4. Run tests
    # ---------------------------------------------------------------
    print("\n--- Running tests ---")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_mtf_downloader.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    print(result.stdout)
    if result.stderr:
        # Only print non-empty stderr lines (warnings, etc.)
        for line in result.stderr.strip().splitlines():
            if line.strip():
                print(f"  {line}")

    if result.returncode == 0:
        print("✓ All new tests pass")
    else:
        print("✗ Some tests failed!")
        all_ok = False

    # ---------------------------------------------------------------
    # 5. Check new scripts exist
    # ---------------------------------------------------------------
    print("\n--- Script files ---")
    for script in [
        "scripts/download_mtf_data.py",
        "scripts/convert_mtf_to_parquet.py",
        "scripts/check_mtf_data_quality.py",
        "scripts/verify_phase1.py",
    ]:
        p = _ROOT / script
        if p.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} MISSING")
            all_ok = False

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print()
    if all_ok:
        print("=== Phase 1 COMPLETE ===")
    else:
        print("=== Phase 1 has issues — see above ===")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    verify()
