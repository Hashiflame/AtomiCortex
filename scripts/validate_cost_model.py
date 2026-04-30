#!/usr/bin/env python
"""Validate CostModel: print round-trip cost breakdown for various position sizes."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.execution.cost_model import CostModel, FeeConfig

DAILY_VOLUME = 30_000_000_000  # $30B (typical BTC futures)
VOLATILITY = 0.60              # 60% annualised
FUNDING_RATE = 0.0001          # 0.01% per 8h (typical)
HOLD_HOURS = 24                # 24h holding period
FEE_CONFIG = FeeConfig()       # BNB discount enabled


def _bar(char: str = "─", width: int = 80) -> None:
    print(char * width)


def main() -> None:
    cm = CostModel()
    sizes = [1_000, 5_000, 10_000, 50_000]

    print()
    _bar("═")
    print("  AtomiCortex — Cost Model Validation")
    print(f"  Volume: ${DAILY_VOLUME/1e9:.0f}B  |  Vol: {VOLATILITY:.0%}  |  "
          f"Funding: {FUNDING_RATE*100:.3f}%/8h  |  Hold: {HOLD_HOURS}h")
    _bar("═")

    header = (
        f"  {'Size':>8}  │  {'Fee(entry)':>10}  │  {'Slippage':>9}  │  "
        f"{'Funding24h':>10}  │  {'Round-trip':>10}  │  {'Min return':>10}"
    )
    print(header)
    _bar()

    for notional in sizes:
        rt = cm.calculate_round_trip_cost(
            notional=notional,
            daily_volume=DAILY_VOLUME,
            volatility=VOLATILITY,
            funding_rate=FUNDING_RATE,
            hours_held=HOLD_HOURS,
            is_long=True,
            fee_config=FEE_CONFIG,
        )
        entry_fee = cm.calculate_fee(notional, is_maker=True, fee_config=FEE_CONFIG)
        funding_24h = cm.calculate_funding_cost(notional, FUNDING_RATE, HOLD_HOURS, is_long=True)
        slippage_1w = cm.calculate_slippage(notional, DAILY_VOLUME, VOLATILITY)

        print(
            f"  ${notional:>7,}  │  ${entry_fee:>9.2f}  │  ${slippage_1w:>7.4f}  │  "
            f"${funding_24h:>8.4f}   │  ${rt.total_cost:>8.4f}   │  "
            f"{rt.min_required_return_bps:>8.2f} bps"
        )

    _bar()
    print()
    print("  Column notes:")
    print("    Fee(entry)  — single-side maker fee with BNB discount (0.018%)")
    print("    Slippage    — one-way square-root impact model")
    print("    Round-trip  — entry+exit fees + both slippages + funding")
    print("    Min return  — round-trip total × 3  (rule of 3×)")
    print()

    # Criterion (spec): $1,000 BTC round-trip with maker fees (limit orders) < 10 bps
    rt_1k_maker = cm.calculate_round_trip_cost(
        notional=1_000,
        daily_volume=DAILY_VOLUME,
        volatility=VOLATILITY,
        funding_rate=FUNDING_RATE,
        hours_held=HOLD_HOURS,
        is_long=True,
        fee_config=FEE_CONFIG,
        is_maker=True,  # limit-order fees (0.018% with BNB discount)
    )
    rt_1k_taker = cm.calculate_round_trip_cost(
        notional=1_000,
        daily_volume=DAILY_VOLUME,
        volatility=VOLATILITY,
        funding_rate=FUNDING_RATE,
        hours_held=HOLD_HOURS,
        is_long=True,
        fee_config=FEE_CONFIG,
        is_maker=False,  # market-order fees (0.045% with BNB discount)
    )
    _bar("═")
    check = "✅ PASS" if rt_1k_maker.total_cost_bps < 10 else "❌ FAIL"
    print(f"  $1,000 BTC round-trip (maker) = {rt_1k_maker.total_cost_bps:.2f} bps  →  {check} (< 10 bps)")
    print(f"  $1,000 BTC round-trip (taker) = {rt_1k_taker.total_cost_bps:.2f} bps  (market orders)")
    _bar("═")
    print()


if __name__ == "__main__":
    main()
