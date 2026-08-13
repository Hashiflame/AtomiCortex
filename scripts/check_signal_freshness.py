#!/usr/bin/env python3
"""
AtomiCortex Signal Freshness Checker.

Reads MAX(created_at) from SQLite DBs in readonly mode and sends a
Telegram alert if any DB has not recorded a signal for a prolonged
period (starvation).
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.logger import get_logger, setup_logging
from src.monitoring.telegram_reporter import TelegramReporter

_log = get_logger("signal_freshness")


def check_freshness(
    db_paths: list[str],
    thresholds: dict[str, float],
    reporter: Any,
    now_fn: Callable[[], datetime]
) -> int:
    """Core logic to check freshness of signals in given DBs.

    Returns the number of alerts generated.
    """
    alerts = 0
    now = now_fn()

    for db_path_str in db_paths:
        path = Path(db_path_str)
        if not path.exists():
            _log.warning(f"DB not found, skipping: {path.name}")
            continue

        threshold_hours = thresholds.get(path.name, thresholds.get("default", 48.0))
        uri = f"file:{path.absolute()}?mode=ro"

        try:
            # Must use uri=True for readonly access
            conn = sqlite3.connect(uri, uri=True)
            try:
                # Check if signals_log exists
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals_log'")
                if not cur.fetchone():
                    msg = f"⚠️ Starvation Alert: DB '{path.name}' has no signals_log table!"
                    _log.error(msg)
                    asyncio.run(reporter.send_alert(msg))
                    alerts += 1
                    continue
                
                cur.execute("SELECT MAX(created_at) FROM signals_log")
                row = cur.fetchone()
                if not row or not row[0]:
                    msg = f"⚠️ Starvation Alert: DB '{path.name}' has no signals ever recorded!"
                    _log.error(msg)
                    asyncio.run(reporter.send_alert(msg))
                    alerts += 1
                    continue
                
                last_time_str = row[0]
                # Parse datetime with timezone correctly
                last_time = datetime.fromisoformat(last_time_str)
                if not last_time.tzinfo:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                
                diff = now - last_time
                diff_hours = diff.total_seconds() / 3600.0
                
                if diff_hours > threshold_hours:
                    msg = (
                        f"⚠️ Starvation Alert: DB '{path.name}' has not had a signal "
                        f"in {diff_hours:.1f} hours! (Threshold: {threshold_hours}h)\n"
                        f"Last signal: {last_time_str}"
                    )
                    _log.error(msg)
                    asyncio.run(reporter.send_alert(msg))
                    alerts += 1
                else:
                    _log.info(f"DB '{path.name}' is fresh. Last signal: {diff_hours:.1f} hours ago.")

            finally:
                conn.close()

        except Exception as e:
            _log.error(f"Error checking DB '{path.name}': {e}")
            # Fail-soft
            pass

    return alerts


class _NullReporter:
    async def send_alert(self, msg: str) -> bool:
        _log.error(f"ALERT (telegram off): {msg}")
        return False


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Signal freshness checker")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--threshold-4h", type=float, default=None, help="Override threshold for atomicortex.db")
    ap.add_argument("--threshold-15m", type=float, default=None, help="Override threshold for atomicortex_15m.db")
    ap.add_argument("--threshold-default", type=float, default=None, help="Override default threshold")
    return ap


def main() -> None:
    ap = get_parser()
    args = ap.parse_args()
    setup_logging(level_console=args.log_level)

    settings = get_settings()
    
    if not settings.telegram_bot_token or not settings.telegram_admin_id:
        _log.warning("Telegram not configured, alerts will be skipped.")
        reporter = _NullReporter()
    else:
        reporter = TelegramReporter(
            bot_token=settings.telegram_bot_token,
            admin_id=settings.telegram_admin_id,
        )

    db_paths = sorted(str(p) for p in (_ROOT / "data").glob("atomicortex*.db"))
    if not db_paths:
        _log.warning("No atomicortex*.db found in data/")
        return

    thresholds = {
        "atomicortex.db": args.threshold_4h if args.threshold_4h is not None else settings.signal_stale_hours_4h,
        "atomicortex_15m.db": args.threshold_15m if args.threshold_15m is not None else settings.signal_stale_hours_15m,
        "default": args.threshold_default if args.threshold_default is not None else settings.signal_stale_hours_default
    }

    def _now() -> datetime:
        return datetime.now(timezone.utc)

    alerts = check_freshness(db_paths, thresholds, reporter, _now)
    _log.info(f"Freshness check completed. Alerts generated: {alerts}")


if __name__ == "__main__":
    main()
