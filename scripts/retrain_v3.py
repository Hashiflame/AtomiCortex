#!/usr/bin/env python3
"""
scripts/retrain_v3.py

Block 2 — vol-scaled symmetric triple-barrier labels + AFML sample
uniqueness weights + LightGBM retrain for the v3 model line.

Grid-searches barrier multipliers (pt, sl, max_holding) for each regime,
ranks configurations by an OOS Sharpe proxy and a Deflated Sharpe Ratio
(López de Prado 2014) computed across the grid, then refits the winning
config and saves to ``<models_dir>/v3/<regime>_model_v3.pkl``.

Production models in ``<models_dir>/*.pkl`` are never touched (suffix
"_v3" on the v3 path; v3 lives in a sibling subfolder).

Usage
-----
    # Dry run — show class balance and uniqueness-weight stats for every
    # grid cell, no training:
    python3 scripts/retrain_v3.py --dry-run

    # Full retrain (default: trend + high_vol on BTC/ETH/SOL 4H):
    python3 scripts/retrain_v3.py \\
        --features-dir data/features/ml_features \\
        --models-dir   data/features/models
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging
from src.models.dataset_builder import DatasetBuilder
from src.models.lgbm_trainer import EvaluationResult, LGBMTrainer, ModelConfig
from src.models.statistical_tests import calculate_dsr

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Grid (matches Block-2 spec). Skip cells whose class balance falls outside
# [35%, 65%] — they re-create the same imbalance v3 is trying to fix.
# ---------------------------------------------------------------------------
BARRIER_GRID: list[dict[str, Any]] = [
    {"pt": 1.0,  "sl": 1.0, "hold": 6},
    {"pt": 1.0,  "sl": 0.8, "hold": 6},
    {"pt": 1.25, "sl": 1.0, "hold": 6},
    {"pt": 1.25, "sl": 1.0, "hold": 4},
    {"pt": 1.0,  "sl": 1.0, "hold": 4},
    {"pt": 1.5,  "sl": 1.0, "hold": 8},
]

DEFAULT_REGIMES = ["trend", "high_vol"]
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

CLASS_BALANCE_MIN = 0.35
CLASS_BALANCE_MAX = 0.65

# Winning cells from the Block-2 baseline grid. --best-only refits at
# these so the only delta vs the 45-feature baseline is the whitelist.
BEST_CELLS: dict[str, dict[str, Any]] = {
    "trend":    {"pt": 1.25, "sl": 1.0, "hold": 4},
    "high_vol": {"pt": 1.25, "sl": 1.0, "hold": 6},
}


def _sharpe_proxy(result: EvaluationResult) -> float:
    """OOS Sharpe proxy from WR & PF — same shape used in statistical_tests
    so DSR rankings line up with the rest of the validation stack."""
    wr_frac = result.win_rate / 100.0
    pf = result.profit_factor if result.profit_factor < 100 else 1.0
    return (wr_frac - 0.5) * pf * 10.0


def _build_config(
    regime: str,
    symbols: list[str],
    pt: float,
    sl: float,
    hold: int,
    feature_whitelist: list[str] | None = None,
    model_suffix: str = "_v3",
) -> ModelConfig:
    return ModelConfig(
        regime=regime,
        symbols=symbols,
        use_triple_barrier=True,
        use_uniqueness_weights=True,
        barrier_pt_multiplier=pt,
        barrier_sl_multiplier=sl,
        barrier_max_holding=hold,
        model_suffix=model_suffix,
        feature_whitelist=feature_whitelist,
    )


def run_best_only(
    features_dir: Path,
    models_dir: Path,
    symbols: list[str],
    regimes: list[str],
    feature_whitelist: list[str] | None,
    model_suffix: str,
) -> dict[str, dict[str, Any]]:
    """Refit at the winning Block-2 cells only — fast A/B vs the baseline.

    Used to isolate the effect of the clustered-MDA feature selection
    (or any other single change) without rerunning the full grid.
    """
    v3_dir = models_dir / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}

    for regime in regimes:
        cell = BEST_CELLS.get(regime)
        if cell is None:
            _log.warning(f"No BEST_CELLS entry for {regime} — skipping")
            continue
        print(f"\n{'='*78}\n  BEST-ONLY refit | regime={regime} | "
              f"cell={cell} | suffix={model_suffix}\n{'='*78}")
        if feature_whitelist:
            print(f"  Whitelist active: {len(feature_whitelist)} features")
        trainer = LGBMTrainer(
            config=_build_config(
                regime, symbols,
                pt=cell["pt"], sl=cell["sl"], hold=cell["hold"],
                feature_whitelist=feature_whitelist,
                model_suffix=model_suffix,
            ),
            features_dir=features_dir,
            models_dir=v3_dir,
            use_mtf_params=True,
        )
        train_df, test_df = trainer.prepare_data()
        n_pos = int((train_df["target"] == 1).sum())
        pos_frac = n_pos / len(train_df) if len(train_df) else 0.0
        model = trainer.train(train_df)
        result = trainer.evaluate(model, test_df)
        summary[regime] = {
            "cell": cell,
            "pos_frac": pos_frac,
            "n_features": len(trainer._feature_columns),
            "final": asdict(result),
        }
        print(f"  ✓ WR={result.win_rate}% PF={result.profit_factor} "
              f"sig={result.signal_rate*100:.1f}% acc={result.accuracy}% "
              f"avg_conf={result.avg_confidence}  "
              f"(features={len(trainer._feature_columns)})")
    return summary


def _dry_run_cell(
    builder: DatasetBuilder,
    features_dir: Path,
    symbol: str,
    pt: float,
    sl: float,
    hold: int,
) -> dict[str, Any]:
    """Per-symbol class balance + uniqueness weight stats (no training)."""
    df = builder.load_and_combine(features_dir, symbols=[symbol])
    if df.is_empty():
        return {"symbol": symbol, "skipped": True}

    labeled = builder.create_target_triple_barrier(
        df, pt_multiplier=pt, sl_multiplier=sl, max_holding=hold,
    )
    n = len(labeled)
    n_up = int((labeled["target"] == 1).sum())
    pos_frac = n_up / n if n else 0.0
    weights = builder.compute_uniqueness_weights(n_samples=n, max_holding=hold)
    return {
        "symbol": symbol,
        "rows": n,
        "pos_frac": round(pos_frac, 4),
        "w_min": round(float(weights.min()), 3) if n else 0.0,
        "w_max": round(float(weights.max()), 3) if n else 0.0,
        "w_mean": round(float(weights.mean()), 3) if n else 0.0,
    }


def run_dry_run(
    features_dir: Path,
    symbols: list[str],
    regimes: list[str],
) -> None:
    print(f"\n{'='*78}")
    print("  AtomiCortex v3 — DRY RUN (no training)")
    print(f"{'='*78}")

    for regime in regimes:
        builder = DatasetBuilder(features_dir.parent, symbols)
        print(f"\n[regime={regime}]")
        print(f"  {'pt':>4} {'sl':>4} {'h':>3} | {'symbol':<10} "
              f"{'rows':>6} {'pos%':>6} {'w_min':>6} {'w_max':>6} {'w_mean':>6}")
        for cell in BARRIER_GRID:
            for sym in symbols:
                stats = _dry_run_cell(
                    builder, features_dir, sym,
                    cell["pt"], cell["sl"], cell["hold"],
                )
                if stats.get("skipped"):
                    continue
                print(
                    f"  {cell['pt']:>4} {cell['sl']:>4} {cell['hold']:>3} | "
                    f"{stats['symbol']:<10} {stats['rows']:>6} "
                    f"{stats['pos_frac']*100:>5.1f}% "
                    f"{stats['w_min']:>6.3f} {stats['w_max']:>6.3f} "
                    f"{stats['w_mean']:>6.3f}"
                )
    print(f"\n{'='*78}\n")


def run_grid(
    features_dir: Path,
    models_dir: Path,
    symbols: list[str],
    regimes: list[str],
) -> dict[str, dict[str, Any]]:
    """Grid search → refit best config → save to <models_dir>/v3/."""
    v3_dir = models_dir / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}

    for regime in regimes:
        print(f"\n{'='*78}")
        print(f"  GRID — regime={regime}")
        print(f"{'='*78}")
        cells: list[dict[str, Any]] = []
        proxies: list[float] = []

        for cell in BARRIER_GRID:
            pt, sl, hold = cell["pt"], cell["sl"], cell["hold"]
            tag = f"pt{pt}_sl{sl}_h{hold}"
            print(f"\n--- {regime} | {tag} ---")

            try:
                config = _build_config(regime, symbols, pt, sl, hold)
                trainer = LGBMTrainer(
                    config=config,
                    features_dir=features_dir,
                    # Train into a scratch subdir so grid pickles don't
                    # collide with the winning model written below.
                    models_dir=v3_dir / "_grid",
                    use_mtf_params=True,
                )
                train_df, test_df = trainer.prepare_data()

                # Class balance gate — skip imbalanced cells.
                n_pos = int((train_df["target"] == 1).sum())
                pos_frac = n_pos / len(train_df) if len(train_df) else 0.0
                if not (CLASS_BALANCE_MIN <= pos_frac <= CLASS_BALANCE_MAX):
                    print(f"  ❌ skip — pos_frac={pos_frac:.3f} outside "
                          f"[{CLASS_BALANCE_MIN}, {CLASS_BALANCE_MAX}]")
                    continue

                model = trainer.train(train_df)
                result = trainer.evaluate(model, test_df)
                proxy = _sharpe_proxy(result)
                proxies.append(proxy)
                cells.append({
                    "cell": cell, "tag": tag, "result": result,
                    "proxy": proxy, "pos_frac": pos_frac,
                })
                print(f"  ✓ WR={result.win_rate:.2f}% PF={result.profit_factor:.3f} "
                      f"sig={result.signal_rate*100:.1f}% acc={result.accuracy:.2f}% "
                      f"proxy={proxy:.3f}")
            except Exception as exc:
                _log.exception(f"Grid cell {tag} failed: {exc}")
                continue

        if not cells:
            print(f"\n  ⚠ No valid grid cells for {regime}")
            continue

        # Rank by OOS Sharpe proxy; report DSR across the grid for
        # multiple-testing context.
        cells.sort(key=lambda c: c["proxy"], reverse=True)
        best = cells[0]
        dsr = calculate_dsr(proxies, n_trials=len(proxies))
        print(f"\n  ► BEST {regime}: {best['tag']} | proxy={best['proxy']:.3f} | "
              f"DSR={dsr:.3f} (n_trials={len(proxies)})")

        # Refit best config and save to v3 dir under the canonical name.
        print(f"  ► Refit + save → {v3_dir}/{regime}_model_v3.pkl")
        winner = LGBMTrainer(
            config=_build_config(regime, symbols, **{
                "pt": best["cell"]["pt"],
                "sl": best["cell"]["sl"],
                "hold": best["cell"]["hold"],
            }),
            features_dir=features_dir,
            models_dir=v3_dir,
            use_mtf_params=True,
        )
        train_df, test_df = winner.prepare_data()
        model = winner.train(train_df)
        final_result = winner.evaluate(model, test_df)

        summary[regime] = {
            "best_cell": best["cell"],
            "best_proxy": best["proxy"],
            "dsr": dsr,
            "n_valid_cells": len(cells),
            "final": asdict(final_result),
        }

    return summary


def _print_summary(
    summary: dict[str, dict[str, Any]],
    models_dir: Path,
) -> None:
    print(f"\n{'='*78}")
    print("  v3 RETRAIN SUMMARY")
    print(f"{'='*78}")
    for regime, s in summary.items():
        f = s["final"]
        cell = s.get("best_cell") or s.get("cell")
        extras = []
        if "dsr" in s:
            extras.append(f"DSR={s['dsr']:.3f}")
        if "n_valid_cells" in s:
            extras.append(f"n_valid={s['n_valid_cells']}")
        if "n_features" in s:
            extras.append(f"feats={s['n_features']}")
        suffix = f"  ({', '.join(extras)})" if extras else ""
        print(f"\n[{regime}] cell={cell}{suffix}")
        print(f"  WR={f['win_rate']}%  PF={f['profit_factor']}  "
              f"sig={f['signal_rate']*100:.1f}%  acc={f['accuracy']}%  "
              f"avg_conf={f['avg_confidence']}")
    print(f"\n  Models saved to: {models_dir / 'v3'}")
    print(f"{'='*78}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 retrain — triple-barrier + uniqueness weights")
    p.add_argument("--features-dir", type=Path,
                   default=Path("data/features/ml_features"))
    p.add_argument("--models-dir", type=Path,
                   default=Path("data/features/models"))
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    p.add_argument("--dry-run", action="store_true",
                   help="Print class balance + uniqueness stats per grid cell; no training.")
    p.add_argument("--best-only", action="store_true",
                   help="Skip grid; refit only at BEST_CELLS (Block-2 winners). "
                        "Pair with --selected-features for fast A/B.")
    p.add_argument("--selected-features", type=Path, default=None,
                   help="Path to selected_features_v3.json (whitelist). "
                        "Restricts X to these columns (+ symbol_encoded).")
    p.add_argument("--model-suffix", default="_v3",
                   help="Filename suffix written between regime and .pkl "
                        "(e.g. '_v3_sel' for the selected-features variant).")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    if args.dry_run:
        run_dry_run(args.features_dir, symbols, regimes)
        return

    whitelist: list[str] | None = None
    if args.selected_features:
        payload = json.loads(args.selected_features.read_text())
        whitelist = payload["selected_features"]
        _log.info(f"Loaded whitelist ({len(whitelist)}): {whitelist}")

    if args.best_only:
        summary = run_best_only(
            args.features_dir, args.models_dir, symbols, regimes,
            feature_whitelist=whitelist, model_suffix=args.model_suffix,
        )
    else:
        summary = run_grid(args.features_dir, args.models_dir, symbols, regimes)
    _print_summary(summary, args.models_dir)


if __name__ == "__main__":
    main()
