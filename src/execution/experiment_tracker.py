"""MLflow experiment tracking for AtomiCortex backtests and walk-forward runs."""

from __future__ import annotations

import mlflow
import mlflow.tracking

from src.execution.backtest_runner import BacktestConfig, BacktestResult
from src.execution.metrics import MetricsResult
from src.execution.walk_forward import WalkForwardResult
from src.logger import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """Thin MLflow wrapper for logging backtests and walk-forward results.

    Tracking URI defaults to ``./mlruns`` (local filesystem).  For tests,
    pass a temp SQLite URI::

        ExperimentTracker("test_exp", tracking_uri="sqlite:///tmp/test.db")
    """

    def __init__(
        self,
        experiment_name: str = "AtomiCortex",
        tracking_uri: str = "./mlruns",
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        log.info("ExperimentTracker: experiment='%s' uri='%s'", experiment_name, tracking_uri)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_backtest(
        self,
        run_name: str,
        config: BacktestConfig,
        result: BacktestResult,
        metrics: MetricsResult,
    ) -> str:
        """Log one backtest run.  Returns the MLflow run_id."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "symbol": config.symbol,
                    "interval": config.interval,
                    "start": str(config.start.date()),
                    "end": str(config.end.date()),
                    "initial_capital": config.initial_capital,
                    "leverage": config.leverage,
                    "maker_fee": config.maker_fee,
                    "taker_fee": config.taker_fee,
                    "strategy": run_name,
                }
            )
            mlflow.log_metrics(metrics.to_dict())
            run_id = run.info.run_id
        log.info("Logged backtest run '%s' → run_id=%s", run_name, run_id)
        return run_id

    def log_walk_forward(
        self,
        run_name: str,
        wf_result: WalkForwardResult,
        config: BacktestConfig,
    ) -> str:
        """Log aggregate walk-forward statistics.  Returns the MLflow run_id."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "symbol": config.symbol,
                    "interval": config.interval,
                    "start": str(config.start.date()),
                    "end": str(config.end.date()),
                    "n_windows": len(wf_result.windows),
                }
            )
            mlflow.log_metrics(
                {
                    "profitable_windows_pct": wf_result.profitable_windows_pct,
                    "avg_sharpe": wf_result.avg_sharpe,
                    "passes_walk_forward_test": float(wf_result.passes_walk_forward_test),
                    "total_windows": float(len(wf_result.windows)),
                }
            )
            # Per-window metrics as nested runs
            for i, w in enumerate(wf_result.windows):
                with mlflow.start_run(
                    run_name=f"{run_name}_w{i + 1}", nested=True
                ) as child:
                    mlflow.log_params(
                        {
                            "test_start": str(w.test_start.date()),
                            "test_end": str(w.test_end.date()),
                        }
                    )
                    mlflow.log_metrics(w.metrics.to_dict())

            run_id = run.info.run_id
        log.info("Logged walk-forward run '%s' → run_id=%s", run_name, run_id)
        return run_id

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_best_runs(
        self,
        metric: str = "sharpe_ratio",
        top_n: int = 5,
    ) -> list[dict]:
        """Return the top *top_n* runs sorted by *metric* (descending)."""
        client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
        experiment = client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            log.warning("Experiment '%s' not found", self._experiment_name)
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n,
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name or "",
                metric: r.data.metrics.get(metric, 0.0),
                "params": dict(r.data.params),
            }
            for r in runs
        ]
