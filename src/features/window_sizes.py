"""Feature-window sizes — the single source of truth for lookback lengths.

Leaf module: it imports nothing from ``src`` so that every consumer
(``regime_detector``, ``feature_pipeline``, ``live_feature_state``, the live
strategies and the tests) can depend on it without creating an import cycle
and without dragging pandas / ta into modules that do not need them.

Why the numbers matter
----------------------
``RegimeDetector.detect_all()`` gives every row before ``min_bars``
(= ``hurst_window``) neutral constants, and ``atr_percentile`` is measured over
the trailing ``atr_lookback`` rows. ``FeaturePipeline.build_from_buffer()``
returns only the *last* row of the buffer, so that row is a valid feature
vector only when the buffer is at least ``WARMUP_ROWS + atr_lookback`` long —
the contract documented in ``build_from_buffer``'s docstring. A shorter buffer
does not fail loudly; it silently emits neutral constants or a percentile
measured over a partial window (train/serve skew).
"""
from __future__ import annotations

# Rows the offline pipeline slices off the head of every feature matrix
# (``build()`` step 11) because their rolling features have not converged yet;
# the longest of those rollings is funding_zscore_30d at 180 4H bars.
WARMUP_ROWS = 200

# ATR-percentile lookbacks per interval — the longest lookback in each chain.
ATR_LOOKBACK_4H = 540    # 90 days x 6 bars/day
ATR_LOOKBACK_1H = 168    # 1 week x 24 bars/day
ATR_LOOKBACK_15M = 672   # 1 week x 96 bars/day

# Hurst window for the 4H detector — also its ``min_bars`` threshold.
HURST_WINDOW_4H = 300


def required_history_bars(atr_lookback: int) -> int:
    """Bars needed for the last row of a buffer to be a valid feature vector.

    ``WARMUP_ROWS`` covers the pipeline's rolling warm-up, ``atr_lookback``
    covers the longest regime window, and the last row must sit past both.
    """
    return WARMUP_ROWS + atr_lookback


# 740 bars = 123.3 days of 4H history; one Binance REST call (ceiling 1500).
HISTORY_BARS_4H = required_history_bars(ATR_LOOKBACK_4H)
