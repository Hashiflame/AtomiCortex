"""Tests for Phase 2.3: CostModel, FeeConfig, RoundTripCost, RandomEntryStrategy."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.execution.cost_model import CostModel, FeeConfig, RoundTripCost
from src.execution.backtest_runner import BacktestConfig, BacktestRunner
from src.execution.strategies.random_entry_strategy import RandomEntryStrategy

DATA_DIR = Path("/mnt/hdd/AtomiCortex/data/features")
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_3M = datetime(2024, 3, 31, tzinfo=timezone.utc)

_data_skip = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="External data drive not mounted",
)

DAILY_VOLUME = 30_000_000_000
VOLATILITY = 0.60
FUNDING_RATE = 0.0001  # 0.01% per 8h


# ──────────────────────────────────────────────────────────────────────────────
# FeeConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestFeeConfig:
    def test_maker_with_bnb_discount(self):
        cfg = FeeConfig(maker_fee=0.0002, use_bnb_discount=True)
        assert cfg.effective_maker == pytest.approx(0.0002 * 0.9)

    def test_taker_with_bnb_discount(self):
        cfg = FeeConfig(taker_fee=0.0005, use_bnb_discount=True)
        assert cfg.effective_taker == pytest.approx(0.0005 * 0.9)

    def test_maker_without_bnb_discount(self):
        cfg = FeeConfig(maker_fee=0.0002, use_bnb_discount=False)
        assert cfg.effective_maker == pytest.approx(0.0002)

    def test_taker_without_bnb_discount(self):
        cfg = FeeConfig(taker_fee=0.0005, use_bnb_discount=False)
        assert cfg.effective_taker == pytest.approx(0.0005)

    def test_bnb_discount_reduces_fee(self):
        cfg_discount = FeeConfig(taker_fee=0.0005, use_bnb_discount=True)
        cfg_no_discount = FeeConfig(taker_fee=0.0005, use_bnb_discount=False)
        assert cfg_discount.effective_taker < cfg_no_discount.effective_taker


# ──────────────────────────────────────────────────────────────────────────────
# CostModel.calculate_fee
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateFee:
    def test_maker_fee_value(self):
        cm = CostModel()
        fee_cfg = FeeConfig(maker_fee=0.0002, use_bnb_discount=True)
        fee = cm.calculate_fee(notional=1_000, is_maker=True, fee_config=fee_cfg)
        assert fee == pytest.approx(1_000 * 0.0002 * 0.9)

    def test_taker_fee_value(self):
        cm = CostModel()
        fee_cfg = FeeConfig(taker_fee=0.0005, use_bnb_discount=False)
        fee = cm.calculate_fee(notional=1_000, is_maker=False, fee_config=fee_cfg)
        assert fee == pytest.approx(1_000 * 0.0005)

    def test_fee_scales_linearly(self):
        cm = CostModel()
        fee_cfg = FeeConfig()
        fee_1k = cm.calculate_fee(1_000, is_maker=False, fee_config=fee_cfg)
        fee_10k = cm.calculate_fee(10_000, is_maker=False, fee_config=fee_cfg)
        assert fee_10k == pytest.approx(fee_1k * 10)


# ──────────────────────────────────────────────────────────────────────────────
# CostModel.calculate_slippage
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateSlippage:
    def test_larger_order_has_more_slippage(self):
        cm = CostModel()
        s_small = cm.calculate_slippage(1_000, DAILY_VOLUME, VOLATILITY)
        s_large = cm.calculate_slippage(10_000, DAILY_VOLUME, VOLATILITY)
        assert s_large > s_small

    def test_slippage_fraction_scales_sublinearly(self):
        """Square-root model: slippage per unit notional grows as √Q (sub-linear).

        10× order → √10 ≈ 3.16× larger slippage fraction (not 10× linear).
        """
        cm = CostModel()
        s_small = cm.calculate_slippage(1_000, DAILY_VOLUME, VOLATILITY)
        s_large = cm.calculate_slippage(10_000, DAILY_VOLUME, VOLATILITY)
        frac_small = s_small / 1_000
        frac_large = s_large / 10_000
        ratio = frac_large / frac_small  # should be ≈ √10 ≈ 3.16
        assert 2.0 < ratio < 5.0

    def test_1k_btc_slippage_below_1_usdt(self):
        """$1,000 BTC order at $30B volume: slippage should be < $1."""
        cm = CostModel()
        s = cm.calculate_slippage(1_000, DAILY_VOLUME, VOLATILITY)
        assert s < 1.0

    def test_zero_volume_returns_zero(self):
        cm = CostModel()
        assert cm.calculate_slippage(1_000, 0, VOLATILITY) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CostModel.calculate_funding_cost
# ──────────────────────────────────────────────────────────────────────────────

class TestCalculateFundingCost:
    def test_long_pays_when_positive_funding(self):
        cm = CostModel()
        cost = cm.calculate_funding_cost(10_000, FUNDING_RATE, 24, is_long=True)
        assert cost > 0  # long pays

    def test_short_receives_when_positive_funding(self):
        cm = CostModel()
        cost = cm.calculate_funding_cost(10_000, FUNDING_RATE, 24, is_long=False)
        assert cost < 0  # short receives (negative cost = gain)

    def test_funding_scales_with_holding_period(self):
        cm = CostModel()
        cost_24h = cm.calculate_funding_cost(1_000, FUNDING_RATE, 24, is_long=True)
        cost_48h = cm.calculate_funding_cost(1_000, FUNDING_RATE, 48, is_long=True)
        assert cost_48h == pytest.approx(cost_24h * 2)

    def test_funding_zero_when_rate_zero(self):
        cm = CostModel()
        cost = cm.calculate_funding_cost(10_000, 0.0, 24, is_long=True)
        assert cost == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# RoundTripCost
# ──────────────────────────────────────────────────────────────────────────────

class TestRoundTripCost:
    def _rt(self, notional: float = 1_000) -> RoundTripCost:
        return CostModel().calculate_round_trip_cost(
            notional=notional,
            daily_volume=DAILY_VOLUME,
            volatility=VOLATILITY,
            funding_rate=FUNDING_RATE,
            hours_held=24,
            is_long=True,
            fee_config=FeeConfig(),
        )

    def test_total_cost_is_sum_of_components(self):
        rt = self._rt()
        expected = rt.entry_fee + rt.exit_fee + rt.entry_slippage + rt.exit_slippage + rt.funding_cost
        assert rt.total_cost == pytest.approx(expected)

    def test_total_cost_bps(self):
        rt = self._rt(1_000)
        assert rt.total_cost_bps == pytest.approx(rt.total_cost / 1_000 * 10_000)

    def test_min_required_return_is_3x(self):
        rt = self._rt()
        assert rt.min_required_return_bps == pytest.approx(rt.total_cost_bps * 3)

    def test_is_tradeable_true(self):
        rt = self._rt()
        assert rt.is_tradeable(rt.min_required_return_bps + 1) is True

    def test_is_tradeable_false(self):
        rt = self._rt()
        assert rt.is_tradeable(rt.min_required_return_bps - 1) is False

    def test_1k_btc_round_trip_below_10_bps(self):
        """$1,000 BTC round-trip with maker (limit-order) fees must be < 10 bps.

        Maker fees give: 2×0.018% + slippage + funding = ~7-8 bps.
        (Taker-only round-trip would be ~13 bps due to higher execution fee.)
        """
        rt = CostModel().calculate_round_trip_cost(
            notional=1_000,
            daily_volume=DAILY_VOLUME,
            volatility=VOLATILITY,
            funding_rate=FUNDING_RATE,
            hours_held=24,
            is_long=True,
            fee_config=FeeConfig(),
            is_maker=True,
        )
        assert rt.total_cost_bps < 10.0, (
            f"Round-trip {rt.total_cost_bps:.2f} bps exceeds 10 bps limit"
        )


# ──────────────────────────────────────────────────────────────────────────────
# RandomEntryStrategy — requires data drive
# ──────────────────────────────────────────────────────────────────────────────

def _run_random(seed: int = 42, **kwargs) -> object:
    cfg = BacktestConfig(
        symbol="BTCUSDT",
        interval="4h",
        start=START,
        end=END_3M,
        initial_capital=10_000.0,
        leverage=5,
        maker_fee=0.0002,
        taker_fee=0.0005,
        data_dir=DATA_DIR,
    )
    default_params = {
        "entry_probability": 0.5,
        "hold_bars": 1,
        "trade_size": 0.01,
        "random_seed": seed,
        "long_only": False,
    }
    default_params.update(kwargs)
    return BacktestRunner(cfg).run(RandomEntryStrategy, default_params)


@_data_skip
class TestRandomEntryStrategy:
    def test_strategy_runs_without_error(self):
        result = _run_random()
        assert result is not None

    def test_deterministic_with_same_seed(self):
        r1 = _run_random(seed=42)
        r2 = _run_random(seed=42)
        assert r1.total_return_pct == pytest.approx(r2.total_return_pct, abs=1e-6)
        assert r1.end_equity == pytest.approx(r2.end_equity, abs=1e-6)

    def test_different_seeds_differ(self):
        r1 = _run_random(seed=42)
        r2 = _run_random(seed=99)
        assert r1.end_equity != pytest.approx(r2.end_equity, abs=1e-2)

    def test_places_orders(self):
        result = _run_random()
        assert result.total_trades >= 1

    def test_equity_curve_populated(self):
        result = _run_random()
        assert len(result.equity_curve) > 0

    @pytest.mark.slow
    def test_random_entry_loses_money(self):
        """With fees and random long/short, expected return ≈ 0 from price,
        accumulated fees dominate → end_equity < start_equity."""
        result = _run_random(
            seed=42,
            entry_probability=0.5,
            hold_bars=1,
            trade_size=0.01,
            long_only=False,
        )
        assert result.end_equity < result.start_equity, (
            f"Expected fee-dominated loss; got end={result.end_equity:.2f} "
            f"start={result.start_equity:.2f} return={result.total_return_pct:.4f}%"
        )
