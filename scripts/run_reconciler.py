#!/usr/bin/env python3
"""
Signal reconciliation runner.

Usage:
  python scripts/run_reconciler.py --dry-run          # preview only
  python scripts/run_reconciler.py                    # close + update stats
  python scripts/run_reconciler.py --all              # every atomicortex*.db
  python scripts/run_reconciler.py --db data/atomicortex_15m.db
  python scripts/run_reconciler.py --offline          # DataStore only (no net)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.reconciler_signals import (
    DataStorePriceSource,
    SignalReconciler,
)
from src.logger import get_logger, setup_logging

# Per-DB default bar size (used when a signal lacks a timeframe column).
_BAR_HOURS = {"atomicortex.db": 4.0, "atomicortex_1h.db": 1.0,
              "atomicortex_15m.db": 0.25}


def _discover() -> list[str]:
    return sorted(str(p) for p in (_ROOT / "data").glob("atomicortex*.db"))


def main() -> None:
    ap = argparse.ArgumentParser(description="AtomiCortex signal reconciler")
    ap.add_argument("--db", default="data/atomicortex.db")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="Use only local DataStore (no Binance calls)")
    ap.add_argument("--trading-mode", default="live",
                    choices=["live", "testnet"])
    ap.add_argument("--data-dir", default="data/features")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(level_console=args.log_level)
    log = get_logger("run_reconciler")

    targets = _discover() if args.all else [args.db]
    if not targets:
        print("No atomicortex*.db found")
        return

    grand = {"checked": 0, "closed_win": 0, "closed_loss": 0,
             "still_open": 0, "skipped_recent": 0, "errors": 0}

    for db in targets:
        if not Path(db).exists():
            print(f"⚠️  {db}: not found — skipped")
            continue
        bar_h = _BAR_HOURS.get(Path(db).name, 4.0)
        src = DataStorePriceSource(args.data_dir) if args.offline else None
        rec = SignalReconciler(
            db_path=db,
            data_dir=args.data_dir,
            bar_hours=bar_h,
            dry_run=args.dry_run,
            price_source=src,
            trading_mode=args.trading_mode,
        )
        res = rec.reconcile()
        for k in grand:
            grand[k] += res.get(k, 0)

        mode = "DRY-RUN" if args.dry_run else "APPLIED"
        print(f"\n=== {db} [{mode}] ===")
        print(f"  checked={res['checked']} win={res['closed_win']} "
              f"loss={res['closed_loss']} still_open={res['still_open']} "
              f"skipped_recent={res['skipped_recent']} errors={res['errors']}")
        for d in res["details"]:
            print("  " + json.dumps(d, default=str))

    print(f"\n=== TOTAL === {json.dumps(grand)}")


if __name__ == "__main__":
    main()
