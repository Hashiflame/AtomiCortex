#!/usr/bin/env python3
"""
scripts/train_models.py

Train LightGBM models for each market regime.

Usage
-----
    python scripts/train_models.py \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \
        --features-dir /mnt/hdd/AtomiCortex/data/features/ml_features \
        --models-dir /mnt/hdd/AtomiCortex/data/features/models \
        --regimes trend,range,high_vol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging
from src.models.training_pipeline import TrainingPipeline

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM regime models")
    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,SOLUSDT",
    )
    p.add_argument(
        "--features-dir",
        required=True,
        type=Path,
        help="Directory with {SYMBOL}_4h_features.parquet files",
    )
    p.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="Directory to save trained models",
    )
    p.add_argument(
        "--regimes",
        default="trend,range,high_vol",
        help="Comma-separated regimes to train (default: trend,range,high_vol)",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    regimes = [r.strip() for r in args.regimes.split(",")]

    print(f"\n{'='*60}")
    print(f"  AtomiCortex — LightGBM Model Training")
    print(f"{'='*60}")
    print(f"  Symbols     : {', '.join(symbols)}")
    print(f"  Regimes     : {', '.join(regimes)}")
    print(f"  Features dir: {args.features_dir}")
    print(f"  Models dir  : {args.models_dir}")
    print(f"{'='*60}\n")

    pipeline = TrainingPipeline()
    results = pipeline.run(
        symbols=symbols,
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        regimes=regimes,
    )

    pipeline.print_report(results)


if __name__ == "__main__":
    main()
