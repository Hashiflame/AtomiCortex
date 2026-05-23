#!/usr/bin/env python3
"""
scripts/feature_selection_v3.py

Block 3 / Step 1 — Clustered-MDA feature selection for v3 models.

Pipeline
--------
1. Pre-drop manually-vetoed features (zero MDI + obvious duplicates).
2. Build per-symbol triple-barrier targets matching v3's winning grid
   cells (trend pt=1.25/sl=1.0/h=4, high_vol pt=1.25/sl=1.0/h=6).
3. Cluster the surviving features hierarchically by |correlation|
   (Ward linkage on 1-|ρ| distance) — train portion only, no leakage.
4. For each regime, run Clustered-MDA on the OOS slice against the
   already-trained v3 booster: permute *all* features of a cluster
   simultaneously, measure accuracy drop (López de Prado MLAM 2020 §6.5).
5. Pick top-K clusters by cluster-MDA, then 1-2 representatives per
   cluster by MDI gain. Union across regimes → final list.
6. Write to data/features/models/v3/selected_features_v3.json.

No model retraining here — selection only.

Usage
-----
    python3 scripts/feature_selection_v3.py             # full run
    python3 scripts/feature_selection_v3.py --dry-run   # clusters + MDA, no JSON
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import accuracy_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logger import get_logger, setup_logging
from src.models.dataset_builder import DatasetBuilder
from src.models.lgbm_trainer import LABEL_TO_CLASS, SYMBOL_ENCODING

_log = get_logger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_FEATURES_DIR = Path("data/features/ml_features")
DEFAULT_MODELS_DIR = Path("data/features/models/v3")

# Pre-vetoed drops (from baseline trend_model_v3 inspection):
#   - 6 zero-MDI features
#   - 3 duplicate-pair losers (keep the higher-ranked sibling)
# symbol_encoded stays out of the selection list (it is auto-appended
# by LGBMTrainer._prepare_xy and irrelevant here).
PREDROP_FEATURES: set[str] = {
    "funding_extreme", "large_volume", "funding_positive",
    "basis_extreme", "gap",
    "ls_ratio",        # keep ls_ratio_zscore
    "atr_percentile",  # keep atr_pct
    "oi_value",        # keep oi_zscore
    "symbol_encoded",  # auto-appended, never a feature col here
}

# Per-regime barrier params that won the Block-2 grid.
REGIME_BARRIERS: dict[str, dict[str, Any]] = {
    "trend":    {"pt": 1.25, "sl": 1.0, "hold": 4},
    "high_vol": {"pt": 1.25, "sl": 1.0, "hold": 6},
}


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_features(
    X: np.ndarray,
    feature_cols: list[str],
    n_clusters: int = 10,
    method: str = "ward",
) -> tuple[dict[int, list[str]], np.ndarray]:
    """Hierarchical cluster by |correlation|; returns (clusters, corr_matrix)."""
    # Robust correlation: drop constant cols by adding tiny noise to the
    # diagonal so corrcoef doesn't NaN out on zero-variance columns.
    X = np.where(np.isfinite(X), X, 0.0)
    var = X.var(axis=0)
    keep_mask = var > 1e-12
    if not keep_mask.all():
        dropped = [f for f, k in zip(feature_cols, keep_mask) if not k]
        _log.warning(f"Dropping constant cols for clustering: {dropped}")

    X_kept = X[:, keep_mask]
    kept_cols = [f for f, k in zip(feature_cols, keep_mask) if k]
    corr = np.corrcoef(X_kept.T)
    corr = np.nan_to_num(corr, nan=0.0)

    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters: dict[int, list[str]] = {}
    for feat, cid in zip(kept_cols, labels):
        clusters.setdefault(int(cid), []).append(feat)
    return clusters, corr


# ---------------------------------------------------------------------------
# Clustered MDA
# ---------------------------------------------------------------------------

def clustered_mda(
    booster: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    clusters: dict[int, list[str]],
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[int, float]:
    """Permute each cluster as a block; return mean accuracy drop per cluster.

    The booster is the v3 model trained on `feature_names` (in that order).
    X must therefore be shaped (n, len(feature_names)) and match the
    training feature order exactly, including the trailing symbol_encoded
    column if the booster expects it.
    """
    baseline_pred = (booster.predict(X) >= 0.5).astype(int)
    baseline_acc = accuracy_score(y, baseline_pred)
    rng = np.random.RandomState(random_state)

    importance: dict[int, float] = {}
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    for cid, feats in clusters.items():
        idx = [name_to_idx[f] for f in feats if f in name_to_idx]
        if not idx:
            continue
        drops = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            X_perm = X.copy()
            perm = rng.permutation(len(X))
            X_perm[:, idx] = X_perm[np.ix_(perm, idx)]
            pred = (booster.predict(X_perm) >= 0.5).astype(int)
            drops[r] = baseline_acc - accuracy_score(y, pred)
        importance[cid] = float(drops.mean())

    return importance


# ---------------------------------------------------------------------------
# Per-regime selection
# ---------------------------------------------------------------------------

def _load_regime_data(
    builder: DatasetBuilder,
    features_dir: Path,
    symbols: list[str],
    regime: str,
    pt: float,
    sl: float,
    hold: int,
    test_pct: float = 0.20,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replicate LGBMTrainer.prepare_data: per-symbol triple-barrier
    target → regime filter → 80/20 temporal split → concat."""
    train_parts, test_parts = [], []
    for sym in symbols:
        df = builder.load_and_combine(features_dir, symbols=[sym])
        if df.is_empty():
            continue
        df = builder.create_target_triple_barrier(
            df, pt_multiplier=pt, sl_multiplier=sl, max_holding=hold,
        )
        if regime == "trend":
            df = df.filter(pl.col("regime").is_in(["trend_up", "trend_down"]))
        elif regime == "high_vol":
            df = df.filter(pl.col("regime") == "high_vol")
        elif regime == "range":
            df = df.filter(pl.col("regime") == "range")
        if df.is_empty():
            continue
        n = len(df)
        n_tr = int(n * (1.0 - test_pct))
        train_parts.append(df.head(n_tr))
        test_parts.append(df.tail(n - n_tr))
    return pl.concat(train_parts, how="diagonal"), pl.concat(test_parts, how="diagonal")


def _prepare_xy_for_booster(
    df: pl.DataFrame,
    booster_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build X (in booster's feature order, incl. symbol_encoded) + y∈{0,1}."""
    cols = [c for c in booster_features if c != "symbol_encoded"]
    X = df.select(cols).to_numpy().astype(np.float64)
    if "symbol_encoded" in booster_features and "symbol" in df.columns:
        sym = (
            df["symbol"]
            .replace(SYMBOL_ENCODING, default=-1)
            .cast(pl.Float64)
            .to_numpy()
            .reshape(-1, 1)
        )
        X = np.hstack([X, sym])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([LABEL_TO_CLASS[int(v)] for v in df["target"].to_numpy()],
                 dtype=np.int32)
    return X, y


def select_for_regime(
    regime: str,
    features_dir: Path,
    models_dir: Path,
    symbols: list[str],
    n_clusters: int,
    top_k_clusters: int,
    min_cluster_importance: float,
    max_per_cluster: int,
    n_repeats: int,
) -> dict[str, Any]:
    """Cluster + MDA + per-cluster pick for a single regime."""
    barrier = REGIME_BARRIERS[regime]
    builder = DatasetBuilder(features_dir.parent, symbols)
    train_df, test_df = _load_regime_data(
        builder, features_dir, symbols, regime,
        pt=barrier["pt"], sl=barrier["sl"], hold=barrier["hold"],
    )
    _log.info(
        f"[{regime}] train={len(train_df)} test={len(test_df)} | "
        f"barriers pt={barrier['pt']} sl={barrier['sl']} h={barrier['hold']}"
    )

    # Load v3 booster — defines the feature order for X.
    bundle_path = models_dir / f"{regime}_model_v3.pkl"
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    booster = bundle["booster"]
    booster_features: list[str] = bundle["feature_columns"]

    # MDI from booster gain — used as tiebreaker inside clusters.
    mdi_arr = booster.feature_importance(importance_type="gain")
    mdi = {f: float(g) for f, g in zip(booster_features, mdi_arr)}

    # Cluster on TRAIN slice only — no leakage.
    cluster_feats = [f for f in booster_features if f != "symbol_encoded"]
    X_tr, _ = _prepare_xy_for_booster(train_df, cluster_feats)
    clusters, _ = cluster_features(X_tr, cluster_feats, n_clusters=n_clusters)

    # MDA on OOS slice with the booster's full feature order.
    X_te, y_te = _prepare_xy_for_booster(test_df, booster_features)
    cluster_imp = clustered_mda(
        booster, X_te, y_te, booster_features, clusters,
        n_repeats=n_repeats,
    )

    # Selection: top-K clusters above importance floor, drop PREDROP
    # candidates, take up to `max_per_cluster` by MDI gain.
    ranked = sorted(cluster_imp.items(), key=lambda kv: kv[1], reverse=True)
    selected: list[str] = []
    chosen_clusters: list[dict[str, Any]] = []
    for cid, imp in ranked[:top_k_clusters]:
        candidates = [f for f in clusters[cid] if f not in PREDROP_FEATURES]
        if not candidates:
            chosen_clusters.append({"cluster": cid, "mda": imp,
                                    "members": clusters[cid], "picked": []})
            continue
        candidates.sort(key=lambda f: mdi.get(f, 0.0), reverse=True)
        picks = candidates[:max_per_cluster]
        if imp < min_cluster_importance:
            picks = picks[:1]   # below floor → at most one representative
        selected.extend(picks)
        chosen_clusters.append({
            "cluster": cid, "mda": round(imp, 6),
            "members": clusters[cid], "picked": picks,
        })

    return {
        "regime": regime,
        "barrier": barrier,
        "n_train": len(train_df), "n_test": len(test_df),
        "clusters": chosen_clusters,
        "all_clusters_ranked": [
            {"cluster": cid, "mda": round(imp, 6),
             "members": clusters[cid]}
            for cid, imp in ranked
        ],
        "selected": selected,
        "mdi": mdi,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_regime_report(report: dict[str, Any]) -> None:
    print(f"\n{'='*78}")
    print(f"  REGIME = {report['regime']}  (train={report['n_train']}, "
          f"test={report['n_test']})")
    print(f"{'='*78}")
    print(f"  Clusters by MDA (showing top-{len(report['clusters'])}):")
    for c in report["clusters"]:
        members_str = ", ".join(c["members"][:6])
        if len(c["members"]) > 6:
            members_str += f", ... (+{len(c['members'])-6})"
        picks = ", ".join(c["picked"]) if c["picked"] else "— (all pre-dropped)"
        print(f"   cluster #{c['cluster']:>2} mda={c['mda']:+.4f}  "
              f"size={len(c['members'])}  ➜ pick: {picks}")
        print(f"        members: {members_str}")
    print(f"\n  Selected for this regime ({len(report['selected'])}): "
          f"{report['selected']}")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--regimes", default="trend,high_vol")
    p.add_argument("--n-clusters", type=int, default=10)
    p.add_argument("--top-k-clusters", type=int, default=10)
    p.add_argument("--max-per-cluster", type=int, default=2)
    p.add_argument("--min-cluster-importance", type=float, default=0.0005)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="Print clusters + MDA without writing JSON.")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    per_regime: dict[str, dict[str, Any]] = {}
    union: list[str] = []
    for regime in regimes:
        report = select_for_regime(
            regime=regime,
            features_dir=args.features_dir,
            models_dir=args.models_dir,
            symbols=symbols,
            n_clusters=args.n_clusters,
            top_k_clusters=args.top_k_clusters,
            min_cluster_importance=args.min_cluster_importance,
            max_per_cluster=args.max_per_cluster,
            n_repeats=args.n_repeats,
        )
        _print_regime_report(report)
        per_regime[regime] = report
        for f in report["selected"]:
            if f not in union:
                union.append(f)

    print(f"\n{'='*78}\n  UNION across regimes ({len(union)}): {union}\n"
          f"  Pre-dropped (always): {sorted(PREDROP_FEATURES)}\n{'='*78}\n")

    if args.dry_run:
        print("  --dry-run: not writing selected_features_v3.json")
        return

    out_path = args.models_dir / "selected_features_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_features": union,
        "predropped": sorted(PREDROP_FEATURES),
        "n_clusters": args.n_clusters,
        "top_k_clusters": args.top_k_clusters,
        "max_per_cluster": args.max_per_cluster,
        "min_cluster_importance": args.min_cluster_importance,
        "n_repeats": args.n_repeats,
        "regimes": {
            r: {
                "barrier": rep["barrier"],
                "selected": rep["selected"],
                "n_train": rep["n_train"],
                "n_test": rep["n_test"],
                "clusters_picked": [
                    {"cluster": c["cluster"], "mda": c["mda"],
                     "members": c["members"], "picked": c["picked"]}
                    for c in rep["clusters"]
                ],
            }
            for r, rep in per_regime.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  ✓ Written: {out_path}")


if __name__ == "__main__":
    main()
