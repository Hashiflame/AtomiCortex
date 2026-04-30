"""Realistic transaction cost model: fees, slippage, and funding."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeeConfig:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    use_bnb_discount: bool = True

    @property
    def effective_maker(self) -> float:
        return self.maker_fee * (0.9 if self.use_bnb_discount else 1.0)

    @property
    def effective_taker(self) -> float:
        return self.taker_fee * (0.9 if self.use_bnb_discount else 1.0)


@dataclass
class RoundTripCost:
    entry_fee: float
    exit_fee: float
    entry_slippage: float
    exit_slippage: float
    funding_cost: float
    total_cost: float
    total_cost_bps: float
    min_required_return_bps: float  # total_cost_bps × 3 (rule of 3×)

    def is_tradeable(self, expected_return_bps: float) -> bool:
        return expected_return_bps >= self.min_required_return_bps


class CostModel:
    """Analytical transaction cost model for crypto perpetual futures."""

    def calculate_fee(
        self,
        notional: float,
        is_maker: bool,
        fee_config: FeeConfig,
    ) -> float:
        """Return commission in USDT for a single order side."""
        rate = fee_config.effective_maker if is_maker else fee_config.effective_taker
        fee = notional * rate
        if fee > notional * 0.001:
            logger.warning(
                "Anomalous fee %.4f USDT on %.2f notional (%.4f%%)",
                fee, notional, fee / notional * 100,
            )
        return fee

    def calculate_slippage(
        self,
        notional: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """
        Return one-way slippage in USDT using the square-root market impact model.

        slippage = notional × 0.5 × σ_annual × √(Q / V)
        where Q = notional, V = daily_volume_usdt, σ = annualised fractional vol.
        """
        if daily_volume <= 0:
            return 0.0
        slippage_fraction = 0.5 * volatility * math.sqrt(notional / daily_volume)
        return notional * slippage_fraction

    def calculate_funding_cost(
        self,
        position_size: float,
        funding_rate: float,
        hours_held: float,
        is_long: bool,
    ) -> float:
        """
        Return funding cost in USDT over the holding period.

        Positive → net cost for the caller (long pays when funding > 0).
        Negative → net gain (short receives when funding > 0).
        """
        num_payments = hours_held / 8.0
        gross = position_size * funding_rate * num_payments
        return gross if is_long else -gross

    def calculate_round_trip_cost(
        self,
        notional: float,
        daily_volume: float,
        volatility: float,
        funding_rate: float,
        hours_held: float,
        is_long: bool,
        fee_config: FeeConfig,
        is_maker: bool = False,
    ) -> RoundTripCost:
        """Return full round-trip cost breakdown for one position.

        is_maker=True uses maker (limit-order) fees; False uses taker (market-order) fees.
        """
        entry_fee = self.calculate_fee(notional, is_maker=is_maker, fee_config=fee_config)
        exit_fee = self.calculate_fee(notional, is_maker=is_maker, fee_config=fee_config)
        entry_slippage = self.calculate_slippage(notional, daily_volume, volatility)
        exit_slippage = self.calculate_slippage(notional, daily_volume, volatility)
        funding_cost = self.calculate_funding_cost(
            position_size=notional,
            funding_rate=funding_rate,
            hours_held=hours_held,
            is_long=is_long,
        )
        total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage + funding_cost
        total_cost_bps = (total_cost / notional * 10_000) if notional > 0 else 0.0
        min_required_return_bps = total_cost_bps * 3
        return RoundTripCost(
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            funding_cost=funding_cost,
            total_cost=total_cost,
            total_cost_bps=total_cost_bps,
            min_required_return_bps=min_required_return_bps,
        )
