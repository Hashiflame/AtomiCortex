"""
src/features/utils.py

Polars expression utilities shared across feature modules.
"""

from __future__ import annotations

import polars as pl


def safe_divide(a: pl.Expr, b: pl.Expr, fill: float = 0.0) -> pl.Expr:
    """Divide a by b, replacing zero-division, inf, and NaN with *fill*."""
    return (
        pl.when(b == 0)
        .then(fill)
        .otherwise(a / b)
        .fill_nan(fill)
        .fill_null(fill)
    )


def rolling_zscore(series: pl.Expr, window: int) -> pl.Expr:
    """Z-score = (x - rolling_mean) / rolling_std over *window* bars.

    Returns 0.0 for any NaN or undefined values.
    """
    mean = series.rolling_mean(window_size=window)
    std = series.rolling_std(window_size=window)
    return safe_divide(series - mean, std)  # safe_divide already fills NaN/null


def rolling_correlation(x: pl.Expr, y: pl.Expr, window: int) -> pl.Expr:
    """Pearson correlation of x and y over a rolling *window*.

    Implements: corr = cov(X,Y) / sqrt(var(X) * var(Y))
    using the identity:  cov = E[XY] - E[X]E[Y]

    Returns 0.0 where undefined (constant window or insufficient data).
    """
    mean_x = x.rolling_mean(window_size=window)
    mean_y = y.rolling_mean(window_size=window)
    mean_xy = (x * y).rolling_mean(window_size=window)
    mean_x2 = (x * x).rolling_mean(window_size=window)
    mean_y2 = (y * y).rolling_mean(window_size=window)

    cov_xy = mean_xy - mean_x * mean_y
    var_x_raw = mean_x2 - mean_x * mean_x
    var_y_raw = mean_y2 - mean_y * mean_y

    # Clamp to zero to guard against tiny negatives from floating-point arithmetic
    var_x = pl.when(var_x_raw < 0).then(0.0).otherwise(var_x_raw)
    var_y = pl.when(var_y_raw < 0).then(0.0).otherwise(var_y_raw)

    denom = (var_x * var_y).sqrt()
    return safe_divide(cov_xy, denom).fill_nan(0.0).fill_null(0.0)
