"""
src/models/training_pipeline.py

End-to-end training pipeline: iterate over regimes, train a LightGBM model
for each, evaluate, and produce a summary report.

Phase 3 — Step 3.4.
"""

from __future__ import annotations

from pathlib import Path

from src.logger import get_logger
from src.models.lgbm_trainer import EvaluationResult, LGBMTrainer, ModelConfig

_log = get_logger(__name__)


class TrainingPipeline:
    """Train and evaluate LightGBM models for multiple market regimes."""

    def run(
        self,
        symbols: list[str],
        features_dir: Path,
        models_dir: Path,
        regimes: list[str] | None = None,
    ) -> dict[str, EvaluationResult]:
        """Train one model per regime and evaluate each.

        Parameters
        ----------
        symbols:
            Binance symbols, e.g. ``["BTCUSDT", "ETHUSDT", "SOLUSDT"]``.
        features_dir:
            Path to directory with ``{SYMBOL}_4h_features.parquet`` files.
        models_dir:
            Path to save trained models (``{regime}_model.pkl``).
        regimes:
            List of regimes to train, e.g. ``["trend", "range", "high_vol"]``.

        Returns
        -------
        dict mapping regime name → EvaluationResult.
        """
        if regimes is None:
            regimes = ["trend", "range", "high_vol"]

        features_dir = Path(features_dir)
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, EvaluationResult] = {}

        for regime in regimes:
            _log.info(f"{'='*60}")
            _log.info(f"Training regime: {regime}")
            _log.info(f"{'='*60}")

            config = ModelConfig(
                regime=regime,
                symbols=symbols,
            )

            trainer = LGBMTrainer(
                config=config,
                features_dir=features_dir,
                models_dir=models_dir,
            )

            try:
                train_df, test_df = trainer.prepare_data()
                model = trainer.train(train_df)
                result = trainer.evaluate(model, test_df)
                results[regime] = result
            except Exception as exc:
                _log.error(f"Training failed for regime '{regime}': {exc}")
                continue

        return results

    def print_report(self, results: dict[str, EvaluationResult]) -> None:
        """Print a formatted summary table of evaluation results."""
        print(f"\n{'═'*80}")
        print(f"  AtomiCortex — LightGBM Training Report")
        print(f"{'═'*80}")
        print()

        # Header
        header = (
            f"  {'Regime':<12} | {'Win Rate':>9} | {'PF':>7} | "
            f"{'Signal%':>8} | {'Accuracy':>9} | {'F1':>7} | {'Passes?':>8}"
        )
        print(header)
        print(f"  {'─'*12}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*8}")

        for regime, result in results.items():
            passes = "✅" if result.passes_minimum_thresholds() else "❌"
            print(
                f"  {regime:<12} | {result.win_rate:>8.1f}% | "
                f"{result.profit_factor:>7.2f} | "
                f"{result.signal_rate * 100:>7.1f}% | "
                f"{result.accuracy:>8.1f}% | "
                f"{result.f1:>6.1f}% | "
                f"  {passes}"
            )

        print()

        # Per-symbol breakdown
        print(f"  {'─'*78}")
        print(f"  Per-symbol breakdown:")
        print()

        for regime, result in results.items():
            print(f"  [{regime}]")
            if result.per_symbol:
                for sym, metrics in result.per_symbol.items():
                    print(
                        f"    {sym:<10}: WR={metrics['win_rate']:.1f}%, "
                        f"PF={metrics['profit_factor']:.2f}, "
                        f"sig={metrics['signal_rate']*100:.1f}%, "
                        f"bars={metrics['n_bars']}, signals={metrics['n_signals']}"
                    )
            else:
                print(f"    (no per-symbol data)")
            print()

        # Summary
        any_pass = any(r.passes_minimum_thresholds() for r in results.values())
        print(f"  {'═'*78}")
        if any_pass:
            print(f"  ✅  At least one model passes minimum thresholds!")
        else:
            print(f"  ❌  No models pass minimum thresholds yet")
        print(f"  {'═'*78}\n")
