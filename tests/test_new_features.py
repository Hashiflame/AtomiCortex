"""Tests for the 5 new live-enrichment features."""

from __future__ import annotations

import math
import random

import pytest

from src.features.derivatives import (
    compute_basis_annualized,
    compute_liquidation_proximity,
    compute_oi_velocity,
    compute_sentiment_features,
)
from src.features.microstructure import compute_vpin


def _is_finite(x: float) -> bool:
    return isinstance(x, float) and not math.isnan(x) and not math.isinf(x)


# -------------------------------- liquidation proximity ----------------------

def test_liq_proximity_empty_returns_neutral():
    out = compute_liquidation_proximity(50_000.0, [])
    assert out == {
        "liq_cluster_long_pct": 0.0,
        "liq_cluster_short_pct": 0.0,
        "liq_imbalance": 0.0,
        "liq_cascade_risk": 0.0,
        "liq_volume_1h": 0.0,
    }


def test_liq_proximity_invalid_price():
    out = compute_liquidation_proximity(0.0, [{"price": 1, "origQty": 1, "side": "BUY"}])
    assert out["liq_imbalance"] == 0.0
    assert out["liq_cluster_short_pct"] == 0.0


def test_liq_proximity_imbalance_long_heavy():
    cp = 50_000.0
    # 3 long liquidations (SELL) above + 1 short below → imbalance > 0.
    liqs = [
        {"price": 50_500.0, "origQty": 10.0, "side": "SELL", "time": 1_700_000_000_000},
        {"price": 50_520.0, "origQty": 12.0, "side": "SELL", "time": 1_700_000_000_000},
        {"price": 50_400.0, "origQty": 8.0,  "side": "SELL", "time": 1_700_000_000_000},
        {"price": 49_500.0, "origQty": 2.0,  "side": "BUY",  "time": 1_700_000_000_000},
    ]
    out = compute_liquidation_proximity(cp, liqs, atr=200.0)
    assert out["liq_imbalance"] > 0.5
    assert 0.0 < out["liq_cluster_long_pct"] <= 10.0
    assert _is_finite(out["liq_volume_1h"])
    # imbalance bounded
    assert -1.0 <= out["liq_imbalance"] <= 1.0


def test_liq_proximity_cascade_risk_bounded():
    now = 1_700_000_000_000
    liqs = [
        {"price": 50_500.0, "origQty": 5.0, "side": "SELL", "time": now},
        {"price": 50_510.0, "origQty": 5.0, "side": "SELL", "time": now},
        {"price": 49_400.0, "origQty": 5.0, "side": "BUY",  "time": now - 23 * 3_600_000},
    ]
    out = compute_liquidation_proximity(50_000.0, liqs, atr=300.0)
    assert 0.0 <= out["liq_cascade_risk"] <= 1.0


# -------------------------------- VPIN ---------------------------------------

def test_vpin_empty_returns_neutral():
    assert compute_vpin([]) == 0.5


def test_vpin_constant_price_returns_neutral():
    # Zero return variance → can't classify → neutral.
    trades = [{"p": 100.0, "q": 1.0} for _ in range(500)]
    assert compute_vpin(trades, bucket_size=10, num_buckets=10) == 0.5


def test_vpin_strong_uptrend_high_imbalance():
    # Monotonic uptrend → BVC should classify ~all volume as buys.
    trades = [{"p": 100.0 + i * 0.1, "q": 1.0} for i in range(1, 600)]
    v = compute_vpin(trades, bucket_size=10, num_buckets=20)
    assert 0.0 <= v <= 1.0
    assert v > 0.5  # imbalanced


def test_vpin_random_walk_moderate():
    random.seed(42)
    price = 100.0
    trades = []
    for _ in range(2000):
        price *= math.exp(random.gauss(0, 0.001))
        trades.append({"p": price, "q": 1.0})
    v = compute_vpin(trades, bucket_size=20, num_buckets=30)
    assert 0.0 <= v <= 1.0
    assert _is_finite(v)


# -------------------------------- basis annualized ---------------------------

def test_basis_annualized_basic():
    out = compute_basis_annualized(50_050.0, 50_000.0, 0.0001)
    # basis_bps = 10, annualized = 0.0001 * 3 * 365 * 100 = 10.95% p.a.
    assert out["basis_bps"] == pytest.approx(10.0, rel=1e-6)
    assert out["basis_annualized"] == pytest.approx(10.95, rel=1e-3)
    assert out["basis_extreme"] == 0


def test_basis_annualized_overheated_longs():
    # funding 0.0005 every 8h → ~54.75% p.a. → extreme = -1 (bearish)
    out = compute_basis_annualized(50_000.0, 50_000.0, 0.0005)
    assert out["basis_annualized"] > 30.0
    assert out["basis_extreme"] == -1


def test_basis_annualized_zero_inputs_safe():
    out = compute_basis_annualized(0.0, 0.0, 0.0)
    assert out["basis_bps"] == 0.0
    assert out["basis_annualized"] == 0.0
    assert out["basis_extreme"] == 0


def test_basis_annualized_zscore():
    hist = [5.0, 6.0, 4.0, 5.5, 4.5]
    out = compute_basis_annualized(50_500.0, 50_000.0, 0.0001, basis_history_bps=hist)
    # basis_bps = 100, far above ~5 → big positive z.
    assert out["basis_zscore_30d"] > 3.0


# -------------------------------- OI velocity --------------------------------

def test_oi_velocity_empty_returns_neutral():
    out = compute_oi_velocity([], [])
    assert out["oi_velocity"] == 0.0
    assert out["oi_acceleration"] == 0.0
    assert out["oi_exhaustion"] == 0


def test_oi_velocity_acceleration_sign():
    oi = [100.0, 101.0, 103.0, 106.0]  # accelerating
    px = [50_000.0, 50_100.0, 50_200.0, 50_300.0]
    out = compute_oi_velocity(oi, px)
    assert out["oi_velocity"] > 0
    assert out["oi_acceleration"] > 0
    assert -1.0 <= out["oi_price_divergence"] <= 1.0


def test_oi_velocity_exhaustion_flag():
    # Sustained rise then a drop.
    oi = [100.0, 102.0, 105.0, 108.0, 110.0, 108.0]
    px = [50_000.0] * 6
    out = compute_oi_velocity(oi, px)
    assert out["oi_exhaustion"] == 1


# -------------------------------- sentiment ----------------------------------

def test_sentiment_neutral_inputs():
    out = compute_sentiment_features(50, 1.0, 1.0)
    assert out["fear_greed_norm"] == pytest.approx(0.5)
    assert out["fear_greed_extreme"] == 0
    assert abs(out["ls_divergence"]) < 1e-9
    assert abs(out["sentiment_score"]) < 1e-9


def test_sentiment_extreme_fear():
    out = compute_sentiment_features(10, 1.0, 1.0)
    assert out["fear_greed_extreme"] == 1
    assert out["fear_greed_norm"] == pytest.approx(0.1)


def test_sentiment_top_traders_diverge():
    # Top traders long-heavy (2.0) vs retail short-heavy (0.5).
    out = compute_sentiment_features(50, 0.5, 2.0)
    assert out["ls_divergence"] > 0
    assert -1.0 <= out["ls_divergence"] <= 1.0


def test_sentiment_clamps_out_of_range_fg():
    out = compute_sentiment_features(150, 1.0, 1.0)
    assert out["fear_greed_norm"] == 1.0
    assert out["fear_greed_extreme"] == -1


# -------------------------------- generic guards -----------------------------

@pytest.mark.parametrize(
    "result",
    [
        compute_liquidation_proximity(50_000.0, []),
        compute_basis_annualized(50_000.0, 50_000.0, 0.0001),
        compute_oi_velocity([100.0, 101.0, 102.0], [1.0, 1.0, 1.0]),
        compute_sentiment_features(50, 1.0, 1.0),
    ],
)
def test_no_nan_in_dict_outputs(result):
    for k, v in result.items():
        assert v is not None, f"{k} is None"
        if isinstance(v, float):
            assert not math.isnan(v), f"{k} is NaN"
            assert not math.isinf(v), f"{k} is inf"


def test_vpin_returns_float_in_range():
    v = compute_vpin([{"p": 100.0 + i * 0.01, "q": 1.0} for i in range(300)])
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0
