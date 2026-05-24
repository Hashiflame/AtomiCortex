"""
tests/test_cvd_stationarity.py

Verifies that the cvd_cum→cvd_rolling_N migration (Step H1) actually
delivers stationary CVD features and remains train/serve consistent.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.microstructure import add_cvd_features


def _synthetic_ohlcv(n: int, seed: int = 0) -> pl.DataFrame:
    """OHLCV-like frame with mean-zero CVD increments (no global drift).

    taker_buy_volume / volume are positive; their relationship is random
    around 0.5 so cvd = 2*tbv - volume has mean ≈ 0 and bounded variance.
    """
    rng = np.random.default_rng(seed)
    volume = rng.uniform(100.0, 1000.0, size=n)
    buy_frac = rng.uniform(0.3, 0.7, size=n)
    tbv = volume * buy_frac
    return pl.DataFrame({
        "open_time": np.arange(n, dtype=np.int64) * 14_400_000,  # 4h step (ms)
        "volume": volume,
        "taker_buy_volume": tbv,
    })


# ---------------------------------------------------------------------------
# 1. cvd_cum no longer appears in the output schema.
# ---------------------------------------------------------------------------


def test_cvd_cum_removed_from_output_columns():
    df = _synthetic_ohlcv(200)
    out = add_cvd_features(df)
    assert "cvd_cum" not in out.columns
    assert "cvd_rolling_24" in out.columns
    assert "cvd_rolling_96" in out.columns


# ---------------------------------------------------------------------------
# 2. rolling CVD is bounded — never grows with dataset epoch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col,window", [
    ("cvd_rolling_24", 24),
    ("cvd_rolling_96", 96),
])
def test_rolling_cvd_is_bounded(col, window):
    n = 5000
    df = _synthetic_ohlcv(n)
    out = add_cvd_features(df)

    cvd_max = float(out["cvd"].abs().max())
    rolling_max = float(out[col].abs().max())

    # Trivial sanity bound: a sum of `window` terms cannot exceed
    # window * max(|term|). This holds independent of dataset length —
    # the very property cvd_cum violated.
    assert rolling_max <= window * cvd_max + 1e-6


# ---------------------------------------------------------------------------
# 3. Stationarity — std of rolling CVD is stable across dataset halves,
#    whereas cumsum's std would grow ~sqrt(n) with the epoch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col", ["cvd_rolling_24", "cvd_rolling_96"])
def test_rolling_cvd_is_stationary_across_dataset_halves(col):
    n = 10_000
    df = _synthetic_ohlcv(n, seed=42)
    out = add_cvd_features(df)

    series = out[col].to_numpy()
    # Skip the warmup region (first `window` rows) for a fair comparison.
    warmup = 96
    body = series[warmup:]
    half = len(body) // 2
    std_first = float(np.std(body[:half]))
    std_second = float(np.std(body[half:]))

    # On stationary data the two halves should have comparable spread.
    # We allow a 2× ratio either way — this fails dramatically for
    # cumsum (would be ~sqrt(2) ratio AND both stds explode with n).
    assert std_first > 0
    assert std_second > 0
    ratio = max(std_first, std_second) / min(std_first, std_second)
    assert ratio < 2.0, f"{col}: std ratio across halves = {ratio:.2f}"


def test_cumsum_would_have_failed_the_stationarity_check():
    """Regression guard: confirms the test above actually discriminates
    between stationary (rolling) and non-stationary (cumulative) signals.
    A cumsum's std is dominated by epoch and grows with dataset length —
    the ratio across halves should be much larger than the 2.0 bound.
    """
    n = 10_000
    df = _synthetic_ohlcv(n, seed=42)
    out = add_cvd_features(df)
    cumsum = out["cvd"].cum_sum().to_numpy()
    half = len(cumsum) // 2
    std_first = float(np.std(cumsum[:half]))
    std_second = float(np.std(cumsum[half:]))
    ratio = max(std_first, std_second) / min(std_first, std_second)
    assert ratio > 2.0, (
        f"cumsum-CVD std ratio = {ratio:.2f} — expected >2 to confirm "
        "the stationarity test discriminates correctly"
    )


# ---------------------------------------------------------------------------
# 4. Train/serve consistency — offline (full history) and live (tail-only
#    buffer) yield identical rolling values on the overlapping rows,
#    provided the live buffer is at least `window` bars long.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col,window", [
    ("cvd_rolling_24", 24),
    ("cvd_rolling_96", 96),
])
def test_offline_vs_live_buffer_consistency(col, window):
    n = 500
    df_full = _synthetic_ohlcv(n, seed=7)
    out_full = add_cvd_features(df_full)

    # Live "buffer": just the last `window + 10` rows. add_cvd_features is
    # re-applied on this slice (mirrors build_from_buffer's pattern).
    buf_len = window + 10
    df_buf = df_full.tail(buf_len)
    out_buf = add_cvd_features(df_buf)

    # The last row should match exactly — the rolling window over the
    # tail of the live buffer covers identical bars to the full-history run.
    last_full = float(out_full[col][-1])
    last_buf = float(out_buf[col][-1])
    assert last_full == pytest.approx(last_buf, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. The pre-existing cvd_slope_* family is still stationary and intact.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col", ["cvd_slope_3", "cvd_slope_6", "cvd_slope_12"])
def test_cvd_slope_features_still_present_and_stationary(col):
    df = _synthetic_ohlcv(5000, seed=3)
    out = add_cvd_features(df)
    assert col in out.columns
    series = out[col].to_numpy()
    half = len(series) // 2
    std_first = float(np.std(series[10:half]))
    std_second = float(np.std(series[half:]))
    ratio = max(std_first, std_second) / min(std_first, std_second)
    assert ratio < 2.0


# ---------------------------------------------------------------------------
# 6. FEATURE_GROUPS index reflects the new feature names (no orphan refs).
# ---------------------------------------------------------------------------


def test_feature_pipeline_group_lists_rolling_cvd():
    from src.features.feature_pipeline import FEATURE_GROUPS
    micro = FEATURE_GROUPS["microstructure"]
    assert "cvd_cum" not in micro
    assert "cvd_rolling_24" in micro
    assert "cvd_rolling_96" in micro
