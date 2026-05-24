"""Walk-forward validation and purged K-fold cross-validation."""

from __future__ import annotations

import calendar
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import polars as pl

from src.execution.backtest_runner import BacktestConfig, BacktestRunner
from src.execution.metrics import (
    MetricsResult,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from src.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Month arithmetic helper
# ──────────────────────────────────────────────────────────────────────────────

def _add_months(dt: datetime, n: int) -> datetime:
    """Add *n* calendar months to *dt*, clamping the day to the last valid day."""
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


# ──────────────────────────────────────────────────────────────────────────────
# Purged K-Fold CV
# ──────────────────────────────────────────────────────────────────────────────

class PurgedKFoldCV:
    """Time-series cross-validation with an embargo gap between train and test.

    Fold layout (expanding train, fixed-size test block):

        Fold 1: [==TRAIN==][GAP][TEST]
        Fold 2: [=====TRAIN=====][GAP][TEST]
        Fold 3: [========TRAIN========][GAP][TEST]

    The embargo removes the first ``embargo_pct × N`` rows immediately after
    the training set to prevent look-ahead leakage from overlapping features.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        data: pl.DataFrame,
        timestamp_col: str = "datetime",
        symbol_col: str | None = None,
    ) -> Generator[tuple[pl.DataFrame, pl.DataFrame], None, None]:
        """Yield ``(train_df, test_df)`` for each fold.

        Boundaries are computed over the **time axis**, not row indices.
        With multi-symbol concatenated frames (``[BTC...][ETH...][SOL...]``)
        the row index is not monotone in time — slicing by row would put
        late BTC and early ETH in the same fold's "test". Pass
        ``symbol_col`` to filter every symbol independently by time so no
        symbol's future leaks into another symbol's training set
        (same fix pattern as ``temporal_split_multi``).

        ``embargo_pct`` is the fraction of the **total time range** (not
        of rows) to discard immediately after train_end. For a uniformly
        spaced single-symbol frame this is numerically equivalent to the
        old row-based embargo, so existing single-symbol tests still hold.
        """
        n = len(data)
        if n < self.n_splits + 1:
            raise ValueError(
                f"Dataset too small ({n} rows) for {self.n_splits} splits"
            )
        # Auto-fallback: synthetic / test frames frequently have a real
        # ``open_time`` (epoch ms) but a null / missing ``datetime`` column.
        # When the caller did not pick an alternative explicitly, try
        # ``open_time`` before failing — this preserves the old behaviour
        # of split() working on these frames without an explicit kwarg.
        candidates = [timestamp_col]
        if timestamp_col == "datetime" and "open_time" in data.columns:
            candidates.append("open_time")

        ts = None
        chosen_col = None
        for col in candidates:
            if col not in data.columns:
                continue
            s = data[col]
            if s.null_count() >= len(s):
                continue
            if s.min() is None or s.max() is None:
                continue
            ts = s
            chosen_col = col
            break

        if ts is None or chosen_col is None:
            raise ValueError(
                f"PurgedKFoldCV.split: no usable timestamp column among "
                f"{candidates} in data (columns: {data.columns})"
            )
        timestamp_col = chosen_col
        t_min = ts.min()
        t_max = ts.max()

        # ``t_max - t_min`` works for both Datetime (→ timedelta) and
        # integer epoch (→ int). Subsequent multiplication / addition
        # follow the same generic arithmetic.
        total = t_max - t_min
        block = total / (self.n_splits + 1)
        embargo = self.embargo_pct * total

        use_per_symbol = (
            symbol_col is not None and symbol_col in data.columns
        )
        if use_per_symbol:
            symbols = sorted(data[symbol_col].unique().to_list())

        for i in range(self.n_splits):
            train_end_t = t_min + (i + 1) * block
            test_start_t = train_end_t + embargo
            test_end_t = t_min + (i + 2) * block
            if test_end_t > t_max:
                test_end_t = t_max
            if test_start_t >= test_end_t:
                log.warning(
                    "Fold %d: test window is empty after embargo — skipping", i,
                )
                continue

            if use_per_symbol:
                train_parts: list[pl.DataFrame] = []
                test_parts: list[pl.DataFrame] = []
                for sym in symbols:
                    sub = (
                        data.filter(pl.col(symbol_col) == sym)
                        .sort(timestamp_col)
                    )
                    train_parts.append(
                        sub.filter(pl.col(timestamp_col) < train_end_t)
                    )
                    test_parts.append(
                        sub.filter(
                            (pl.col(timestamp_col) >= test_start_t)
                            & (pl.col(timestamp_col) < test_end_t)
                        )
                    )
                train_df = pl.concat(train_parts, how="diagonal").sort(
                    timestamp_col
                )
                test_df = pl.concat(test_parts, how="diagonal").sort(
                    timestamp_col
                )
            else:
                s = data.sort(timestamp_col)
                train_df = s.filter(pl.col(timestamp_col) < train_end_t)
                test_df = s.filter(
                    (pl.col(timestamp_col) >= test_start_t)
                    & (pl.col(timestamp_col) < test_end_t)
                )
            yield train_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metrics: MetricsResult
    is_profitable: bool


@dataclass
class WalkForwardGateConfig:
    """Acceptance thresholds for ``WalkForwardResult.passes_walk_forward_test``.

    H7 — single binary "% profitable" gate was vulnerable to one
    catastrophic window hidden among many tiny positives (5×+0.1% then
    1×-30% scored 5/6 profitable but lost ~30%). Multi-criteria gate
    requires every threshold to pass.

    Defaults are tuned to catch true catastrophes without rejecting
    healthy strategies:
    * ``min_profitable_pct=60`` — legacy threshold.
    * ``min_avg_sharpe=0.0`` — mean Sharpe across windows must be ≥ 0;
       a strategy that loses more than it wins fails here.
    * ``min_worst_sharpe=-1.0`` — single worst window's Sharpe must be
       ≥ -1.0; rejects "one big blowup" patterns.
    * ``min_aggregate_return=0.0`` — sum of per-window total_return_pct
       must be non-negative; the headline arithmetic rejects the
       catastrophe scenario above (aggregate ≈ -29.5%).

    Use ``>=`` semantics throughout so all-zero / dummy metrics still
    pass (backward-compat for existing fixtures).
    """
    min_profitable_pct: float = 60.0
    min_avg_sharpe: float = 0.0
    min_worst_sharpe: float = -1.0
    min_aggregate_return: float = 0.0


# Module-level default — `passes_walk_forward_test` (a zero-arg property)
# reads thresholds from here so callers who want different gates can
# either pass their own config to `passes_gate()` or mutate this object.
DEFAULT_GATE = WalkForwardGateConfig()


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]

    @property
    def profitable_windows_pct(self) -> float:
        """Percentage of windows where the test period was profitable."""
        if not self.windows:
            return 0.0
        n_prof = sum(1 for w in self.windows if w.is_profitable)
        return n_prof / len(self.windows) * 100

    @property
    def avg_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.metrics.sharpe_ratio for w in self.windows) / len(self.windows)

    @property
    def worst_sharpe(self) -> float:
        """Minimum Sharpe across all windows (catastrophe detector)."""
        if not self.windows:
            return 0.0
        return min(w.metrics.sharpe_ratio for w in self.windows)

    @property
    def aggregate_return_pct(self) -> float:
        """Sum of per-window ``total_return_pct``.

        Sum rather than compound average — the gate cares about whether
        the headline P&L is positive, and summing per-window % returns is
        the same shape ``WindowResult.is_profitable`` uses internally.
        """
        if not self.windows:
            return 0.0
        return sum(w.metrics.total_return_pct for w in self.windows)

    def passes_gate(
        self,
        config: WalkForwardGateConfig | None = None,
    ) -> tuple[bool, list[str]]:
        """Multi-criteria walk-forward acceptance check.

        Returns ``(passed, reasons)``. ``reasons`` lists every criterion
        that failed (empty when ``passed`` is True). Useful for surfacing
        *why* a strategy was rejected instead of a bare bool.
        """
        cfg = config or DEFAULT_GATE
        reasons: list[str] = []
        if self.profitable_windows_pct < cfg.min_profitable_pct:
            reasons.append(
                f"profitable_windows_pct {self.profitable_windows_pct:.1f} "
                f"< {cfg.min_profitable_pct:.1f}"
            )
        if self.avg_sharpe < cfg.min_avg_sharpe:
            reasons.append(
                f"avg_sharpe {self.avg_sharpe:.3f} < {cfg.min_avg_sharpe:.3f}"
            )
        if self.worst_sharpe < cfg.min_worst_sharpe:
            reasons.append(
                f"worst_sharpe {self.worst_sharpe:.3f} "
                f"< {cfg.min_worst_sharpe:.3f}"
            )
        if self.aggregate_return_pct < cfg.min_aggregate_return:
            reasons.append(
                f"aggregate_return_pct {self.aggregate_return_pct:.3f} "
                f"< {cfg.min_aggregate_return:.3f}"
            )
        return (not reasons, reasons)

    @property
    def passes_walk_forward_test(self) -> bool:
        """Multi-criteria gate using ``DEFAULT_GATE`` thresholds.

        Backward-compatible: the legacy "≥ 60% profitable" rule is one
        of the criteria (``min_profitable_pct=60``); the additional
        Sharpe and aggregate-return checks only fire on catastrophic
        windows, leaving normal strategies unaffected.
        """
        passed, _ = self.passes_gate()
        return passed


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validator
# ──────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """Sliding-window walk-forward validation.

    Window layout (step = step_months each iteration):

        Window 1: [train_months TRAIN][test_months TEST]
        Window 2:          [train_months TRAIN][test_months TEST]
        Window 3:                   [train_months TRAIN][test_months TEST]
    """

    def __init__(
        self,
        train_months: int = 18,
        test_months: int = 6,
        step_months: int | None = None,
        embargo: timedelta = timedelta(0),
    ) -> None:
        self.train_months = train_months
        self.test_months = test_months
        # Default: step equals test window so windows don't overlap
        self.step_months = step_months if step_months is not None else test_months
        # AFML Ch.7 embargo: gap between train_end and test_start to prevent
        # triple-barrier labels from peeking into the test window. Express
        # as a duration (caller computes max_holding_bars × bar_duration).
        # Default timedelta(0) preserves the legacy zero-gap behaviour.
        self.embargo = embargo

    def split(
        self,
        start: datetime,
        end: datetime,
    ) -> Generator[
        tuple[tuple[datetime, datetime], tuple[datetime, datetime]], None, None
    ]:
        """Yield ``((train_start, train_end), (test_start, test_end))`` pairs."""
        cursor = start
        while True:
            train_start = cursor
            train_end = _add_months(cursor, self.train_months)
            # Embargo shifts the test window forward so triple-barrier
            # labels generated in the last bars of train cannot reach
            # into the test window. With embargo=timedelta(0) this is
            # a no-op (legacy behaviour).
            test_start = train_end + self.embargo
            test_end = _add_months(test_start, self.test_months)

            if test_end > end:
                break

            yield (train_start, train_end), (test_start, test_end)
            cursor = _add_months(cursor, self.step_months)

    def run_validation(
        self,
        strategy_class: type,
        strategy_config: dict,
        backtest_config: BacktestConfig,
        data_dir: Path,
    ) -> WalkForwardResult:
        """Run the strategy on every test window and collect metrics."""
        pairs = list(self.split(backtest_config.start, backtest_config.end))
        log.info(
            "Walk-forward: %d windows | train=%dm test=%dm step=%dm",
            len(pairs),
            self.train_months,
            self.test_months,
            self.step_months,
        )

        windows: list[WindowResult] = []
        for i, ((train_start, train_end), (test_start, test_end)) in enumerate(pairs):
            log.info(
                "  Window %d/%d  test: %s → %s",
                i + 1,
                len(pairs),
                test_start.date(),
                test_end.date(),
            )
            test_cfg = replace(
                backtest_config,
                start=test_start,
                end=test_end,
                data_dir=data_dir,
            )
            try:
                runner = BacktestRunner(test_cfg)
                result = runner.run(strategy_class, strategy_config)
            except ValueError as exc:
                log.warning("  Skipping window %s–%s: %s", test_start.date(), test_end.date(), exc)
                continue

            metrics = _metrics_from_result(result)
            windows.append(
                WindowResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    metrics=metrics,
                    is_profitable=result.total_return_pct > 0,
                )
            )

        return WalkForwardResult(windows=windows)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _metrics_from_result(result) -> MetricsResult:  # type: ignore[return]
    """Build MetricsResult from a BacktestResult."""
    ec = result.equity_curve
    days = (
        (ec[-1][0] - ec[0][0]).total_seconds() / 86400
        if len(ec) >= 2 else 0.0
    )
    s_eq = ec[0][1] if ec else 1.0
    e_eq = ec[-1][1] if ec else 1.0
    annual_ret = (
        ((e_eq / s_eq) ** (365 / days) - 1) * 100
        if (days > 0 and s_eq > 0)
        else 0.0
    )
    return MetricsResult(
        sharpe_ratio=calculate_sharpe_ratio(ec),
        calmar_ratio=calculate_calmar_ratio(ec),
        max_drawdown_pct=calculate_max_drawdown(ec),
        win_rate=result.win_rate * 100,      # BacktestResult stores fraction 0-1
        profit_factor=result.profit_factor,
        total_return_pct=result.total_return_pct,
        annualized_return_pct=annual_ret,
        total_trades=result.total_trades,
    )
