#!/usr/bin/env python3
"""
scripts/analyze_regimes.py

Analyse market regimes on historical data and print summary statistics.

Usage
-----
    python scripts/analyze_regimes.py \\
        --symbol BTCUSDT \\
        --start 2024-01-01 \\
        --end 2025-12-31 \\
        --data-dir /mnt/hdd/AtomiCortex/data/features
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import polars as pl

from src.ingestion.data_store import DataStore
from src.features.feature_pipeline import FeaturePipeline
from src.features.regime_detector import MarketRegime, RegimeDetector
from src.logger import get_logger

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse market regimes")
    p.add_argument("--symbol", required=True, help="Binance symbol, e.g. BTCUSDT")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--data-dir", required=True, type=Path, help="Root Parquet directory")
    p.add_argument("--interval", default="4h", help="Kline interval (default: 4h)")
    return p.parse_args()


def _ascii_histogram(values: np.ndarray, bins: int = 20, width: int = 40) -> str:
    """Produce a simple ASCII histogram string."""
    counts, edges = np.histogram(values, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    lines: list[str] = []
    for i, c in enumerate(counts):
        bar_len = int(c / max_count * width)
        bar = "█" * bar_len
        lo, hi = edges[i], edges[i + 1]
        lines.append(f"  {lo:6.3f}–{hi:6.3f} | {bar} ({c})")
    return "\n".join(lines)


def _regime_transitions(regimes: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """Count regime → regime transitions."""
    transitions: Counter[str] = Counter()
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            key = f"{regimes[i - 1]} → {regimes[i]}"
            transitions[key] += 1
    return transitions.most_common(top_n)


def main() -> None:
    args = _parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"\n{'='*60}")
    print(f"  AtomiCortex Regime Analyser — Phase 3.3")
    print(f"{'='*60}")
    print(f"  Symbol   : {args.symbol}")
    print(f"  Interval : {args.interval}")
    print(f"  Range    : {args.start} → {args.end}")
    print(f"  Data dir : {args.data_dir}")
    print(f"{'='*60}\n")

    # --- Load klines ---
    with DataStore(args.data_dir) as store:
        pipeline = FeaturePipeline(store, args.symbol, args.interval)
        df = pipeline.build(start, end)

    if df.is_empty():
        print("ERROR: No data produced. Check data-dir and date range.")
        sys.exit(1)

    # --- Detect regimes ---
    detector = RegimeDetector()
    df = detector.detect_all(df)
    stats = detector.get_regime_statistics(df)

    # ========== Output ==========

    # 1. Regime distribution
    print(f"{'─'*60}")
    print(f"  REGIME DISTRIBUTION ({stats['total_bars']} bars)")
    print(f"{'─'*60}")
    for regime in MarketRegime:
        pct = stats["regime_pct"].get(regime.value, 0)
        bar = "█" * int(pct / 2)
        print(f"  {regime.value:12s} : {pct:6.2f}%  {bar}")

    # 2. Average metrics per regime
    print(f"\n{'─'*60}")
    print(f"  AVERAGE METRICS PER REGIME")
    print(f"{'─'*60}")
    print(f"  {'Regime':12s} {'Hurst':>8s} {'ATR %':>10s}")
    for regime in MarketRegime:
        h = stats["mean_hurst"].get(regime.value, 0)
        a = stats["mean_atr"].get(regime.value, 0)
        print(f"  {regime.value:12s} {h:8.4f} {a:10.6f}")

    # 3. Hurst histogram
    hurst_vals = df["hurst"].to_numpy()
    # Only show non-default (non-warmup) values
    active_hurst = hurst_vals[hurst_vals != 0.5]
    if len(active_hurst) > 10:
        print(f"\n{'─'*60}")
        print(f"  HURST EXPONENT DISTRIBUTION")
        print(f"{'─'*60}")
        print(_ascii_histogram(active_hurst, bins=15, width=35))
        print(f"  mean={active_hurst.mean():.4f}  std={active_hurst.std():.4f}  "
              f"min={active_hurst.min():.4f}  max={active_hurst.max():.4f}")

    # 4. Top transitions
    regimes_list = df["regime"].to_list()
    transitions = _regime_transitions(regimes_list, top_n=10)
    if transitions:
        print(f"\n{'─'*60}")
        print(f"  TOP-10 REGIME TRANSITIONS")
        print(f"{'─'*60}")
        for label, count in transitions:
            print(f"  {label:35s} : {count:4d}")

    # 5. Current regime (last bar)
    last_state = detector.detect(df)
    print(f"\n{'─'*60}")
    print(f"  CURRENT REGIME (last bar)")
    print(f"{'─'*60}")
    print(f"  Regime       : {last_state.regime.value}")
    print(f"  Hurst        : {last_state.hurst}")
    print(f"  ADX          : {last_state.adx}")
    print(f"  ATR %        : {last_state.atr_pct:.6f}")
    print(f"  ATR pctile   : {last_state.atr_percentile}")
    print(f"  Trend str    : {last_state.trend_strength}")
    print(f"  Confidence   : {last_state.confidence}")
    print(f"  Tradeable    : {'YES' if last_state.is_tradeable() else 'NO'}")
    print(f"  Pos mult     : {last_state.position_size_multiplier()}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
