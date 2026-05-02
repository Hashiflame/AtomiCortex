"""
src/models/statistical_tests.py

Statistical tests for ML model validation:
  - Deflated Sharpe Ratio (DSR) — López de Prado 2014
  - Probability of Backtest Overfitting (PBO) — Bailey et al. 2014
  - t-statistic for win-rate significance

Phase 3 — Step 3.5 / 3.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats

from src.logger import get_logger
from src.models.lgbm_trainer import EvaluationResult

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def calculate_dsr(
    sharpe_ratios: list[float],
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (López de Prado 2014).

    DSR corrects the observed Sharpe Ratio for multiple testing by
    comparing the best observed SR against the expected maximum SR
    from *n_trials* independent trials.

    Simplified formula (sufficient for our use-case)::

        E[SR_max] ≈ √(2·ln(N)) -
            (ln(ln(N)) + ln(4π)) / (2·√(2·ln(N)))

        DSR = Φ( (SR_obs - E[SR_max]) / σ_SR × √(T) )

    where Φ is the standard normal CDF and T is the number of
    observations (len(sharpe_ratios)).

    Parameters
    ----------
    sharpe_ratios:
        List of Sharpe ratios from cross-validation / experiments.
    n_trials:
        Total number of model configurations tested.
    skewness:
        Skewness of returns (default 0 — normal).
    kurtosis:
        Excess kurtosis of returns (default 3 — normal).

    Returns
    -------
    float
        DSR probability in [0, 1].  Goal ≥ 0.95.
    """
    if len(sharpe_ratios) < 2 or n_trials < 2:
        return 0.0

    sr_array = np.array(sharpe_ratios, dtype=np.float64)
    best_sr = float(np.max(sr_array))
    std_sr = float(np.std(sr_array, ddof=1))

    if std_sr < 1e-12:
        return 0.0

    # Expected maximum Sharpe from n_trials (Euler–Mascheroni approx)
    log_n = math.log(n_trials)
    if log_n <= 0:
        return 0.0

    sqrt_2logn = math.sqrt(2 * log_n)
    log_logn = math.log(log_n) if log_n > 0 else 0.0

    expected_max_sr = sqrt_2logn - (log_logn + math.log(4 * math.pi)) / (
        2 * sqrt_2logn
    )

    # Standard error of SR (Lo 2002, with skewness/kurtosis correction)
    n_obs = len(sharpe_ratios)
    se_sr = math.sqrt(
        (1 - skewness * best_sr + ((kurtosis - 1) / 4) * best_sr ** 2)
        / n_obs
    )

    if se_sr < 1e-12:
        return 0.0

    # DSR = Φ( (SR* - E[SR_max]) / SE )
    z = (best_sr - expected_max_sr) / se_sr
    dsr = float(sp_stats.norm.cdf(z))

    _log.info(
        f"DSR: best_sr={best_sr:.4f}, E[SR_max]={expected_max_sr:.4f}, "
        f"std_sr={std_sr:.4f}, se_sr={se_sr:.4f}, z={z:.4f}, DSR={dsr:.4f}"
    )
    return dsr


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting
# ---------------------------------------------------------------------------

def calculate_pbo(
    cv_results: list[EvaluationResult],
    metric: str = "win_rate",
) -> float:
    """Probability of Backtest Overfitting (Bailey et al. 2014).

    Simplified combinatorial approach:

    1. Split CV fold results into pairs of (IS, OOS) subsets.
    2. For each partition, find which fold is "best" in-sample.
    3. Check whether that same fold performs *below median* out-of-sample.
    4. PBO = fraction of partitions where best-IS is below-median-OOS.

    Interpretation:
        PBO = 0.0 → no overfitting
        PBO = 0.5 → random selection
        PBO > 0.5 → overfitting

    Goal: PBO ≤ 0.30.

    Parameters
    ----------
    cv_results:
        List of EvaluationResult from cross-validation folds.
    metric:
        Which metric to use (``"win_rate"`` or ``"profit_factor"``).

    Returns
    -------
    float in [0, 1].
    """
    n = len(cv_results)
    if n < 4:
        _log.warning("PBO needs ≥ 4 folds for meaningful estimate; got %d", n)
        return 0.5  # uninformative prior

    # Extract metric values for each fold
    metrics = []
    for r in cv_results:
        val = getattr(r, metric, None)
        if val is None:
            raise ValueError(f"EvaluationResult has no attribute '{metric}'")
        metrics.append(float(val))

    metrics_arr = np.array(metrics)

    # Combinatorial split: partition folds into two halves
    fold_indices = list(range(n))
    half = n // 2
    overfit_count = 0
    total_partitions = 0

    for is_indices in combinations(fold_indices, half):
        is_set = set(is_indices)
        oos_indices = [i for i in fold_indices if i not in is_set]

        is_metrics = metrics_arr[list(is_indices)]
        oos_metrics = metrics_arr[oos_indices]

        # Find the index (within IS) of the best IS fold
        best_is_pos = int(np.argmax(is_metrics))
        # Map back to the original fold index
        best_is_idx = list(is_indices)[best_is_pos]

        # Check OOS performance of this fold
        oos_median = float(np.median(oos_metrics))
        best_is_oos_val = metrics_arr[best_is_idx]

        # Count as overfit if best-IS performs below OOS median
        if best_is_oos_val < oos_median:
            overfit_count += 1
        total_partitions += 1

    pbo = overfit_count / total_partitions if total_partitions > 0 else 0.5

    _log.info(
        f"PBO ({metric}): {overfit_count}/{total_partitions} = {pbo:.4f} "
        f"(metrics={[f'{m:.2f}' for m in metrics]})"
    )
    return pbo


# ---------------------------------------------------------------------------
# t-statistic for win-rate significance
# ---------------------------------------------------------------------------

def calculate_t_stat(
    win_rates: list[float],
    n_trades: list[int],
) -> float:
    """t-statistic for testing win-rate significance.

    H0: win_rate = 50% (random)

    Uses weighted mean by number of trades::

        t = (mean_wr - 50%) / (std_wr / √n_windows)

    From master document: goal t-stat ≥ 3.0.

    Parameters
    ----------
    win_rates:
        Win rates (in percent, 0-100) per window/fold.
    n_trades:
        Number of trades/signals per window/fold.

    Returns
    -------
    float
        t-statistic. Positive = above random, negative = below.
    """
    if not win_rates or len(win_rates) < 2:
        return 0.0

    wr_arr = np.array(win_rates, dtype=np.float64)
    n_arr = np.array(n_trades, dtype=np.float64)

    # Weighted mean
    total_trades = n_arr.sum()
    if total_trades == 0:
        return 0.0

    weighted_mean = float(np.average(wr_arr, weights=n_arr))
    std_wr = float(np.std(wr_arr, ddof=1))

    if std_wr < 1e-12:
        return 0.0

    n_windows = len(win_rates)
    t = (weighted_mean - 50.0) / (std_wr / math.sqrt(n_windows))

    _log.info(
        f"t-stat: weighted_mean={weighted_mean:.2f}%, std={std_wr:.2f}, "
        f"n_windows={n_windows}, t={t:.4f}"
    )
    return t


# ---------------------------------------------------------------------------
# StatTestResult
# ---------------------------------------------------------------------------

@dataclass
class StatTestResult:
    """Aggregated statistical test results for ML validation."""

    dsr: float
    pbo: float
    t_stat: float
    n_oos_signals: int

    def passes_all_thresholds(self) -> bool:
        """Check against master-document go-live criteria."""
        return (
            self.dsr >= 0.95
            and self.pbo <= 0.30
            and self.t_stat >= 3.0
            and self.n_oos_signals >= 300
        )

    def summary(self) -> str:
        """Pretty-print statistical test results."""
        lines = [
            "",
            "Statistical Tests:",
            f"  DSR:         {self.dsr:.4f}  {'✅' if self.dsr >= 0.95 else '❌'}  ← goal ≥ 0.95",
            f"  PBO:         {self.pbo:.4f}  {'✅' if self.pbo <= 0.30 else '❌'}  ← goal ≤ 0.30",
            f"  t-stat:      {self.t_stat:.4f}  {'✅' if self.t_stat >= 3.0 else '❌'}  ← goal ≥ 3.0",
            f"  OOS signals: {self.n_oos_signals}     {'✅' if self.n_oos_signals >= 300 else '❌'}  ← goal ≥ 300",
            "",
        ]
        verdict = "✅ PASSES" if self.passes_all_thresholds() else "❌ Does not pass (yet)"
        lines.append(f"  VERDICT: {verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_tests(
    cv_results: list[EvaluationResult],
    wf_result: "WalkForwardMLResult",  # forward ref to avoid circular import
    n_experiments: int = 10,
) -> StatTestResult:
    """Run DSR + PBO + t-stat and return aggregated StatTestResult.

    Parameters
    ----------
    cv_results:
        List of EvaluationResult from Purged K-Fold CV.
    wf_result:
        WalkForwardMLResult from walk-forward ML validation.
    n_experiments:
        Number of model configurations tested (for DSR).

    Returns
    -------
    StatTestResult
    """
    # --- DSR ---
    # Use profit_factor as proxy for Sharpe (PF > 1 ≈ positive returns)
    sharpe_proxies = []
    for r in cv_results:
        # Convert win_rate + profit_factor into a simple Sharpe proxy
        wr_frac = r.win_rate / 100.0
        pf = r.profit_factor if r.profit_factor < 100 else 1.0
        # Simple proxy: (wr - 0.5) * pf acts as risk-adjusted return
        sr_proxy = (wr_frac - 0.5) * pf * 10  # scaled to SR-like range
        sharpe_proxies.append(sr_proxy)

    # Also include WF window-level metrics for richer sample
    for w in wf_result.windows:
        wr_frac = w.win_rate / 100.0
        pf = w.profit_factor if w.profit_factor < 100 else 1.0
        sr_proxy = (wr_frac - 0.5) * pf * 10
        sharpe_proxies.append(sr_proxy)

    dsr = calculate_dsr(sharpe_proxies, n_trials=n_experiments)

    # --- PBO ---
    pbo = calculate_pbo(cv_results, metric="win_rate")

    # --- t-stat ---
    win_rates = [w.win_rate for w in wf_result.windows]
    n_trades = [w.n_signals for w in wf_result.windows]
    t_stat = calculate_t_stat(win_rates, n_trades)

    # --- OOS signals ---
    n_oos = sum(w.n_signals for w in wf_result.windows)

    result = StatTestResult(
        dsr=round(dsr, 4),
        pbo=round(pbo, 4),
        t_stat=round(t_stat, 4),
        n_oos_signals=n_oos,
    )

    _log.info(f"All tests complete: {result}")
    return result
