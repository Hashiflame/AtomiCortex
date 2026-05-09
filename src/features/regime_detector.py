"""
src/features/regime_detector.py

Market regime detection: Hurst exponent (R/S analysis), ADX, ATR percentile,
and a composite classifier that labels each bar as TREND_UP, TREND_DOWN,
RANGE, HIGH_VOL, or UNKNOWN.

Phase 3 — Step 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

from src.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Part 1 — Hurst Exponent (R/S analysis)
# ---------------------------------------------------------------------------

def calculate_hurst_exponent(
    prices: np.ndarray,
    min_lag: int = 2,
    max_lag: int = 20,
) -> float:
    """Rescaled-range (R/S) analysis for the Hurst exponent.

    Algorithm
    ---------
    1. For each *lag* from *min_lag* to *max_lag*:
       a. Split the price series into non-overlapping chunks of size *lag*.
       b. For each chunk compute R/S = (max − min of cumulative deviations)
          divided by the standard deviation of the chunk.
       c. Average R/S across all chunks for that lag.
    2. Hurst = slope of the OLS regression  log(lag) vs log(mean R/S).

    Interpretation
    --------------
    * H > 0.55  →  trending (momentum)
    * H ≈ 0.50  →  random walk
    * H < 0.45  →  mean-reverting

    Returns a float in [0, 1].  Returns 0.5 (neutral) on any error.
    """
    try:
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)
        if n < max(min_lag, 10):
            return 0.5

        # R/S analysis operates on log-returns, not raw price levels.
        # Raw prices have a non-stationary trend that dominates the Hurst
        # estimate.  Log-returns capture the serial-dependence structure.
        returns = np.diff(np.log(prices))
        if len(returns) < max(min_lag, 10):
            return 0.5

        m = len(returns)
        lags: list[int] = []
        rs_values: list[float] = []

        for lag in range(min_lag, min(max_lag + 1, m + 1)):
            n_chunks = m // lag
            if n_chunks < 1:
                continue

            rs_list: list[float] = []
            for i in range(n_chunks):
                chunk = returns[i * lag : (i + 1) * lag]
                mean = chunk.mean()
                deviations = np.cumsum(chunk - mean)
                r = deviations.max() - deviations.min()
                s = chunk.std(ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                lags.append(lag)
                rs_values.append(np.mean(rs_list))

        if len(lags) < 2:
            return 0.5

        log_lags = np.log(np.array(lags, dtype=np.float64))
        log_rs = np.log(np.array(rs_values, dtype=np.float64))

        # OLS slope:  H = cov(x,y) / var(x)
        slope = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(slope, 0.0, 1.0))

    except Exception:  # noqa: BLE001
        _log.debug("calculate_hurst_exponent: error — returning 0.5")
        return 0.5


# ---------------------------------------------------------------------------
# Part 2 — Technical Indicators
# ---------------------------------------------------------------------------

def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index — trend strength (0–100).

    Uses the ``ta`` library.  ADX > 25 → strong trend, ADX < 20 → ranging.

    Returns an ndarray the same length as the inputs (NaN for warmup).
    """
    adx = ADXIndicator(
        high=pd.Series(high, dtype=np.float64),
        low=pd.Series(low, dtype=np.float64),
        close=pd.Series(close, dtype=np.float64),
        window=period,
    )
    return adx.adx().values


def calculate_atr_percentile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    lookback: int = 90 * 6,   # 90 days × 6 bars/day on 4H
) -> tuple[float, float]:
    """Current ATR relative to its historical distribution.

    Returns ``(current_atr, atr_percentile)`` where *atr_percentile* ∈ [0, 1].

    * atr_percentile > 0.8  →  high-volatility regime
    * atr_percentile < 0.3  →  low-volatility regime
    """
    atr_ind = AverageTrueRange(
        high=pd.Series(high, dtype=np.float64),
        low=pd.Series(low, dtype=np.float64),
        close=pd.Series(close, dtype=np.float64),
        window=period,
    )
    atr_series = atr_ind.average_true_range().values

    # Remove NaN from the warmup period
    valid = atr_series[~np.isnan(atr_series)]
    if len(valid) == 0:
        return 0.0, 0.5

    current_atr = float(valid[-1])

    # Historical window for percentile comparison
    history = valid[-lookback:] if len(valid) > lookback else valid
    pct = float(np.searchsorted(np.sort(history), current_atr) / len(history))
    return current_atr, np.clip(pct, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Part 3 — Regime Classifier
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    """Discrete market regime labels."""
    TREND_UP   = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE      = "range"
    HIGH_VOL   = "high_vol"
    UNKNOWN    = "unknown"


# Position-size multiplier per regime (from master document)
_REGIME_SIZE_MULT: dict[MarketRegime, float] = {
    MarketRegime.TREND_UP:   1.0,
    MarketRegime.TREND_DOWN: 1.0,
    MarketRegime.RANGE:      0.7,
    MarketRegime.HIGH_VOL:   0.5,
    MarketRegime.UNKNOWN:    0.0,
}


@dataclass
class RegimeState:
    """Snapshot of the detected market regime for a single bar."""
    regime: MarketRegime
    hurst: float
    adx: float
    atr_pct: float            # current ATR as % of price
    atr_percentile: float     # ATR in historical context (0–1)
    trend_strength: float     # composite score 0–1
    confidence: float         # confidence in regime classification 0–1

    def is_tradeable(self) -> bool:
        """Return *True* unless the regime is UNKNOWN."""
        return self.regime != MarketRegime.UNKNOWN

    def position_size_multiplier(self) -> float:
        """Risk-scaled position multiplier per regime.

        HIGH_VOL  → 0.5  (reduce size)
        RANGE     → 0.7
        TREND_*   → 1.0  (full size)
        UNKNOWN   → 0.0  (don't trade)
        """
        return _REGIME_SIZE_MULT.get(self.regime, 0.0)


class RegimeDetector:
    """Classify the current market regime from OHLCV data.

    Parameters
    ----------
    hurst_window:
        Number of bars fed into Hurst calculation (300 ≈ 50 days on 4H).
        Hurst is computed as a numeric feature for ML; it is **not** used
        in the classification rules (ADX-first design).
    adx_period:
        ADX indicator period.
    atr_period:
        ATR indicator period.
    atr_lookback:
        Bars of ATR history used for percentile ranking (540 ≈ 90 days).
    adx_trend_threshold:
        ADX above this → TREND regime.
    adx_range_threshold:
        ADX below this → RANGE regime.
    high_vol_percentile:
        ATR percentile above this → HIGH_VOL regime.
    """

    def __init__(
        self,
        hurst_window: int = 300,
        adx_period: int = 14,
        atr_period: int = 14,
        atr_lookback: int = 540,       # 90 days × 6 bars
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        high_vol_percentile: float = 0.8,
    ) -> None:
        self.hurst_window = hurst_window
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.high_vol_percentile = high_vol_percentile

    # ------------------------------------------------------------------
    # Single-bar detection
    # ------------------------------------------------------------------

    def detect(
        self,
        df: pl.DataFrame,
        idx: int = -1,
    ) -> RegimeState:
        """Detect the market regime for one bar (default: last bar).

        ADX-first classification rules
        ------------------------------
        1. ATR percentile > high_vol_percentile       → HIGH_VOL
        2. ADX > adx_trend_threshold (25)             → TREND_UP / TREND_DOWN
        3. ADX < adx_range_threshold (20)             → RANGE
        4. adx_range_threshold ≤ ADX ≤ adx_trend_threshold → UNKNOWN

        Hurst is computed but used only as a numeric ML feature,
        not as a classification rule (R/S estimator bias on crypto).

        Confidence = 0.4 × |hurst − 0.5| × 2  +  0.6 × min(adx / 50, 1.0)
        """
        n = len(df)
        if n == 0:
            return self._unknown()

        # Resolve negative index
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            return self._unknown()

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        # Hurst — over the preceding window ending at idx+1
        start = max(0, idx + 1 - self.hurst_window)
        hurst = calculate_hurst_exponent(close[start : idx + 1])

        # ADX — compute over full series, then pick idx
        adx_arr = calculate_adx(high, low, close, self.adx_period)
        adx_val = float(adx_arr[idx]) if not np.isnan(adx_arr[idx]) else 0.0

        # ATR percentile — up to idx+1
        current_atr, atr_percentile = calculate_atr_percentile(
            high[: idx + 1],
            low[: idx + 1],
            close[: idx + 1],
            period=self.atr_period,
            lookback=self.atr_lookback,
        )

        # ATR as a percentage of the current price
        price = close[idx]
        atr_pct = (current_atr / price) if price > 0 else 0.0

        # --- Classify ---
        regime = self._classify(hurst, adx_val, atr_percentile, close, idx)

        # --- Composite scores ---
        hurst_distance = abs(hurst - 0.5) * 2.0  # 0–1
        adx_normalized = min(adx_val / 50.0, 1.0)
        # ADX-first: ADX carries more weight than Hurst in confidence
        trend_strength = 0.4 * hurst_distance + 0.6 * adx_normalized
        confidence = float(np.clip(trend_strength, 0.0, 1.0))

        return RegimeState(
            regime=regime,
            hurst=round(hurst, 4),
            adx=round(adx_val, 2),
            atr_pct=round(atr_pct, 6),
            atr_percentile=round(atr_percentile, 4),
            trend_strength=round(trend_strength, 4),
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Vectorised (whole-DataFrame) detection
    # ------------------------------------------------------------------

    def detect_all(
        self,
        df: pl.DataFrame,
        min_bars: int = 300,
    ) -> pl.DataFrame:
        """Add regime columns for every row in *df*.

        Rows before *min_bars* receive neutral defaults.

        Optimisation
        ------------
        Hurst is O(n) per call ⇒ computing it every bar would be O(n²).
        Instead, it is recomputed every 6 bars (once per day on 4H);
        intermediate bars reuse the previous value.

        Added columns
        -------------
        regime (str), hurst (f64), adx (f64), atr_pct (f64),
        atr_percentile (f64), trend_strength (f64), regime_confidence (f64).
        """
        n = len(df)
        _log.info(f"RegimeDetector.detect_all: {n} bars, min_bars={min_bars}")

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        # Pre-compute full ADX array once — O(n)
        adx_arr = calculate_adx(high, low, close, self.adx_period)
        adx_arr = np.where(np.isnan(adx_arr), 0.0, adx_arr)

        # Pre-compute full ATR array once — O(n)
        atr_ind = AverageTrueRange(
            high=pd.Series(high, dtype=np.float64),
            low=pd.Series(low, dtype=np.float64),
            close=pd.Series(close, dtype=np.float64),
            window=self.atr_period,
        )
        atr_arr = atr_ind.average_true_range().values
        atr_arr = np.where(np.isnan(atr_arr), 0.0, atr_arr)

        # Result arrays
        regimes = ["unknown"] * n
        hursts = np.full(n, 0.5)
        adxs = np.zeros(n)
        atr_pcts = np.zeros(n)
        atr_percentiles = np.full(n, 0.5)
        trend_strengths = np.zeros(n)
        confidences = np.zeros(n)

        last_hurst = 0.5
        _RECOMPUTE_EVERY = 6  # recompute Hurst every 6 bars (1 day on 4H)

        for i in range(n):
            if i < min_bars:
                adxs[i] = float(adx_arr[i])
                continue

            # --- Hurst (amortised) ---
            if (i - min_bars) % _RECOMPUTE_EVERY == 0:
                start = max(0, i + 1 - self.hurst_window)
                last_hurst = calculate_hurst_exponent(close[start : i + 1])
            hursts[i] = last_hurst

            # --- ADX ---
            adx_val = float(adx_arr[i])
            adxs[i] = adx_val

            # --- ATR percentile ---
            current_atr = float(atr_arr[i])
            lb_start = max(0, i + 1 - self.atr_lookback)
            history = atr_arr[lb_start : i + 1]
            history_valid = history[history > 0]
            if len(history_valid) > 0:
                pct = float(
                    np.searchsorted(np.sort(history_valid), current_atr)
                    / len(history_valid)
                )
            else:
                pct = 0.5
            atr_percentiles[i] = np.clip(pct, 0.0, 1.0)

            price = close[i]
            atr_pcts[i] = (current_atr / price) if price > 0 else 0.0

            # --- Classify ---
            regime = self._classify(last_hurst, adx_val, pct, close, i)
            regimes[i] = regime.value

            # --- Composite scores ---
            hurst_distance = abs(last_hurst - 0.5) * 2.0
            adx_normalized = min(adx_val / 50.0, 1.0)
            ts = 0.4 * hurst_distance + 0.6 * adx_normalized
            trend_strengths[i] = np.clip(ts, 0.0, 1.0)
            confidences[i] = np.clip(ts, 0.0, 1.0)

        # --- Add columns to DataFrame ---
        df = df.with_columns([
            pl.Series("regime", regimes, dtype=pl.Utf8),
            pl.Series("hurst", hursts, dtype=pl.Float64),
            pl.Series("adx", adxs, dtype=pl.Float64),
            pl.Series("atr_pct", atr_pcts, dtype=pl.Float64),
            pl.Series("atr_percentile", atr_percentiles, dtype=pl.Float64),
            pl.Series("trend_strength", trend_strengths, dtype=pl.Float64),
            pl.Series("regime_confidence", confidences, dtype=pl.Float64),
        ])

        _log.debug("RegimeDetector.detect_all: done")
        return df

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_regime_statistics(self, df: pl.DataFrame) -> dict[str, Any]:
        """Aggregate regime statistics from a DataFrame enriched by detect_all.

        Returns
        -------
        dict with:
        * ``regime_pct``   — % of bars in each regime
        * ``mean_hurst``   — average Hurst per regime
        * ``mean_atr``     — average ATR % per regime
        * ``total_bars``   — total bar count
        """
        if "regime" not in df.columns:
            return {}

        total = len(df)
        stats: dict[str, Any] = {"total_bars": total, "regime_pct": {}, "mean_hurst": {}, "mean_atr": {}}

        for regime in MarketRegime:
            mask = df["regime"] == regime.value
            count = mask.sum()
            stats["regime_pct"][regime.value] = round(100.0 * count / total, 2) if total > 0 else 0.0

            subset = df.filter(mask)
            if len(subset) > 0:
                stats["mean_hurst"][regime.value] = round(float(subset["hurst"].mean()), 4)
                stats["mean_atr"][regime.value] = round(float(subset["atr_pct"].mean()), 6)
            else:
                stats["mean_hurst"][regime.value] = 0.0
                stats["mean_atr"][regime.value] = 0.0

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        hurst: float,
        adx: float,
        atr_percentile: float,
        close: np.ndarray,
        idx: int,
    ) -> MarketRegime:
        """ADX-first classification rules (deterministic, no ML).

        1. atr_percentile > 0.8   → HIGH_VOL
        2. adx > 25               → TREND_UP / TREND_DOWN
        3. adx < 20               → RANGE
        4. 20 ≤ adx ≤ 25          → UNKNOWN (transitional)

        Hurst is deliberately excluded from the rules — it serves
        as a numeric feature for downstream ML models.
        """
        if atr_percentile > self.high_vol_percentile:
            return MarketRegime.HIGH_VOL

        if adx > self.adx_trend_threshold:
            # Direction from recent return
            if idx >= 1 and close[idx] >= close[idx - 1]:
                return MarketRegime.TREND_UP
            return MarketRegime.TREND_DOWN

        if adx < self.adx_range_threshold:
            return MarketRegime.RANGE

        return MarketRegime.UNKNOWN

    @staticmethod
    def _unknown() -> RegimeState:
        """Return a neutral UNKNOWN state."""
        return RegimeState(
            regime=MarketRegime.UNKNOWN,
            hurst=0.5,
            adx=0.0,
            atr_pct=0.0,
            atr_percentile=0.5,
            trend_strength=0.0,
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Extended Regime Detectors for MTF (Phase 2)
# ---------------------------------------------------------------------------


class RegimeDetector1H(RegimeDetector):
    """Extended regime detection for 1H timeframe.

    Inherits from RegimeDetector but uses faster parameters suited
    to the higher bar frequency of 1H data.

    Key differences from 4H RegimeDetector:
    - Faster ADX period (10 vs 14)
    - Shorter Hurst window (100 vs 300)
    - Shorter ATR lookback (168 = 1 week of 1H bars)
    """

    def __init__(self, **kwargs) -> None:
        defaults = dict(
            hurst_window=100,
            adx_period=10,
            atr_period=10,
            atr_lookback=168,        # 1 week * 24 bars/day
            adx_trend_threshold=25.0,
            adx_range_threshold=20.0,
            high_vol_percentile=0.8,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


class RegimeDetector15M(RegimeDetector):
    """Micro-regime detection for 15m timeframe.

    Uses very fast parameters for the highest bar frequency.

    Key differences:
    - Very fast ADX/ATR period (7)
    - Shortest Hurst window (50)
    - ATR lookback of 672 (1 week of 15m bars)
    """

    def __init__(self, **kwargs) -> None:
        defaults = dict(
            hurst_window=50,
            adx_period=7,
            atr_period=7,
            atr_lookback=672,        # 1 week * 96 bars/day
            adx_trend_threshold=25.0,
            adx_range_threshold=20.0,
            high_vol_percentile=0.8,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)

