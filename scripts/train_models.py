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
        --regimes trend,range,high_vol \
        --model-suffix _v4

``--model-suffix`` is required and has no default (PR-Э1.4, A2-036).
This script trains on the legacy 1-bar ``sign_return`` target, and the
unsuffixed filename is the one the live 4H strategy loads; ``save_bundle``
refuses to write it for anything but a triple-barrier model.
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
    # Required, and deliberately without a default (PR-Э1.4, A2-036).
    # This script trains on the legacy sign_return target, and an empty
    # suffix names its output exactly like the bundle the live 4H
    # strategy loads. A default -- even "_vN" -- would put that filename
    # one forgotten argument away again, so the name is made a decision
    # the operator has to type.
    p.add_argument(
        "--model-suffix",
        required=True,
        help="Filename suffix written between regime and .pkl (e.g. '_v4'). "
             "Required: the unsuffixed name is reserved for the production "
             "triple-barrier bundles and save_bundle refuses it.",
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
    print(f"  Suffix      : {args.model_suffix}")
    print(f"{'='*60}\n")

    pipeline = TrainingPipeline()
    results = pipeline.run(
        symbols=symbols,
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        regimes=regimes,
        model_suffix=args.model_suffix,
    )

    pipeline.print_report(results)


if __name__ == "__main__":
    main()
