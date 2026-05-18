#!/usr/bin/env python3
"""
AtomiCortex REST API launcher.

Usage:
  python scripts/run_api.py
  python scripts/run_api.py --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="AtomiCortex REST API")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
