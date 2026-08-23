#!/usr/bin/env python3
"""Promote a trained candidate bundle into the tracked production root.

The trainer writes candidates into an untracked directory; the bot loads
from a directory that git tracks.  This script is the only supported way
across, and it exists so that the crossing is gated rather than being a
``cp`` somebody ran once.

The gate is modelled on ``LGBMTrainer.save_bundle``: a broken artifact is
refused before a weak one, every check runs before anything touches the
filesystem, and a refusal leaves whatever is already in production
exactly as it was.  ``--force`` permits an overwrite; it waives no check.
No backup copy is ever written beside the artifact -- git is the history,
and a stray ``.pkl.bak`` would be one more untracked file to explain.

With ``--regime all`` the two bundles are gated together and written only
if both pass.  They are one traded configuration, not two independent
files: they share a feature list, and the strategy routes a third regime
onto the first of them.  A production where one is new and the other is
months old is a state nobody designed.

Run:
    python scripts/promote_model.py --regime all --dry-run
    python scripts/promote_model.py --regime all
    python scripts/promote_model.py --regime trend --force

Exit code is 0 on success -- including a dry run that passed the gate --
and 1 on any refusal or on an unreadable registry.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.model_paths import (  # noqa: E402  (path shim must run first)
    CANDIDATES_ROOT_4H,
    MODELS_ROOT_4H,
    PROD_STEMS_4H,
    REGISTRY_PATH,
    candidate_path,
    prod_path,
)
from src.models.model_registry import (  # noqa: E402
    RegistryError,
    build_entry,
    load_registry,
    registry_entry_for,
    save_registry,
    sha256_file,
)

# The value of --regime that means "every production stem".
ALL_REGIMES = "all"

# Refusal codes.  Named constants rather than bare strings so a test and
# the message a human reads cannot drift apart.
SOURCE_MISSING = "source_missing"
UNREADABLE = "unreadable"
NO_MANIFEST = "no_manifest"
NO_FEATURE_COLUMNS = "no_feature_columns"
REGIME_MISMATCH = "regime_mismatch"
NOT_PASSING = "not_passing"
RESEARCH_ARTIFACT = "research_artifact"
DEST_EXISTS = "dest_exists"


@dataclass(frozen=True)
class PromotionPlan:
    """A candidate that passed the gate, and what promoting it would do."""

    stem: str
    source: Path
    dest: Path
    manifest: dict[str, Any]
    sha256: str
    size_bytes: int
    dest_exists: bool
    # What is being replaced, sampled while the gate runs -- i.e. before
    # the copy. Reading it at report time would describe the new file and
    # make --force claim it had overwritten the thing it just wrote.
    prev_sha256: str | None = None
    prev_size_bytes: int | None = None


@dataclass(frozen=True)
class Refusal:
    """A candidate that did not pass, and the check that stopped it."""

    stem: str
    code: str
    detail: str


@dataclass(frozen=True)
class PromotionResult:
    """What a run did, or would have done."""

    planned: list[PromotionPlan]
    promoted: list[PromotionPlan]
    refusals: list[Refusal]
    dry_run: bool

    @property
    def ok(self) -> bool:
        return not self.refusals


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def gate_candidate(
    stem: str,
    *,
    candidates_dir: Path,
    prod_dir: Path,
    force: bool = False,
) -> PromotionPlan | Refusal:
    """Decide whether one candidate may be promoted.  Writes nothing.

    The order of the checks is the argument.  A bundle that could not be
    used for inference is refused before one whose numbers are merely bad
    -- the same precedence ``save_bundle`` uses, and for the same reason:
    "broken" outranks "weak" as an explanation.  ``force`` reaches only
    the last check.
    """
    source = candidate_path(candidates_dir, stem)
    dest = prod_path(prod_dir, stem)

    # 1. Nothing to promote.
    if not source.exists():
        return Refusal(stem, SOURCE_MISSING, f"no candidate at {source}")

    # 2. Present but not a bundle.
    try:
        with open(source, "rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001 -- any failure means unusable
        return Refusal(stem, UNREADABLE, f"{source} could not be unpickled: {exc}")
    if not isinstance(bundle, dict):
        return Refusal(
            stem, UNREADABLE,
            f"{source} unpickled to {type(bundle).__name__}, not a bundle dict",
        )

    # 3. No provenance.  Promoting one would put a file in production
    #    whose training data, target and metrics cannot be recovered.
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict):
        return Refusal(
            stem, NO_MANIFEST,
            f"{source} carries no manifest -- provenance unknown",
        )

    # 4. Broken artifact.  Nothing records which columns to feed the
    #    booster, so no consumer could build an input vector.  Checked
    #    ahead of the metrics for the same reason save_bundle does it,
    #    and --force does not waive it.
    if not bundle.get("feature_columns"):
        return Refusal(
            stem, NO_FEATURE_COLUMNS,
            f"{source} has an empty feature_columns -- unusable for inference",
        )

    # 5. Wrong file for this stem.  Manifest and bundle carry the regime
    #    independently, so both are checked: swapped arguments would
    #    otherwise install a high-vol model as the trend one.
    manifest_regime = manifest.get("regime")
    bundle_regime = bundle.get("regime")
    if manifest_regime != stem or bundle_regime != stem:
        return Refusal(
            stem, REGIME_MISMATCH,
            f"{source} is regime {manifest_regime!r} in its manifest and "
            f"{bundle_regime!r} in the bundle, promoted as {stem!r}",
        )

    # 6. Failed the go-live gate.  Strictly "is not True": a bundle old
    #    enough to have no verdict must refuse, not default to one.
    if manifest.get("passes") is not True:
        evaluation = manifest.get("eval") or {}
        return Refusal(
            stem, NOT_PASSING,
            f'manifest["passes"] is {manifest.get("passes")!r} '
            f'(WR={evaluation.get("win_rate")}, '
            f'PF={evaluation.get("profit_factor")})',
        )

    # 7. Research artifact.  Redundant with 6 for a bundle written by
    #    allow_failing, which stamps both -- but not for one that failed
    #    the thresholds at write time and would pass them now.
    if manifest.get("written_despite_failing") is True:
        return Refusal(
            stem, RESEARCH_ARTIFACT,
            'manifest["written_despite_failing"] is True -- written by the '
            "escape hatch, not for production",
        )

    # 8. Occupied.  The only check --force reaches.
    dest_exists = dest.exists()
    if dest_exists and not force:
        return Refusal(
            stem, DEST_EXISTS,
            f"{dest} already exists; pass --force to replace it",
        )

    prev_sha256: str | None = None
    prev_size_bytes: int | None = None
    if dest_exists:
        try:
            prev_sha256 = sha256_file(dest)
            prev_size_bytes = dest.stat().st_size
        except OSError as exc:
            prev_sha256 = f"<unreadable: {exc}>"

    return PromotionPlan(
        stem=stem,
        source=source,
        dest=dest,
        manifest=manifest,
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        dest_exists=dest_exists,
        prev_sha256=prev_sha256,
        prev_size_bytes=prev_size_bytes,
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _atomic_copy(source: Path, dest: Path) -> None:
    """Copy bytes through a sibling temp file and one rename.

    A half-written bundle in production is worse than none: the loader
    would find the file, fail to unpickle it, and the failure would look
    like a code bug rather than an interrupted copy.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent),
    )
    try:
        with os.fdopen(fd, "wb") as out:
            with open(source, "rb") as src:
                for chunk in iter(lambda: src.read(65_536), b""):
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def promote(
    *,
    stems: list[str],
    candidates_dir: Path,
    prod_dir: Path,
    registry_path: Path,
    force: bool = False,
    dry_run: bool = False,
) -> PromotionResult:
    """Gate every stem, then write all of them or none.

    The registry is read first, before the gate: an unreadable one has to
    stop the run while nothing has been written yet.  It is written last,
    after every bundle is in place, so it never describes a file that is
    not there.

    Raises
    ------
    RegistryError
        The registry file exists and could not be understood.  Raised
        before any write, so a refusal here costs nothing.
    """
    registry = load_registry(registry_path)

    planned: list[PromotionPlan] = []
    refusals: list[Refusal] = []
    for stem in stems:
        outcome = gate_candidate(
            stem, candidates_dir=candidates_dir, prod_dir=prod_dir, force=force,
        )
        if isinstance(outcome, Refusal):
            refusals.append(outcome)
        else:
            planned.append(outcome)

    if refusals or dry_run:
        return PromotionResult(
            planned=planned, promoted=[], refusals=refusals, dry_run=dry_run,
        )

    for plan in planned:
        _atomic_copy(plan.source, plan.dest)

    promoted_at = datetime.now(timezone.utc).isoformat()
    for plan in planned:
        registry["models"][plan.stem] = build_entry(
            bundle_path=plan.dest,
            manifest=plan.manifest,
            promoted_at_utc=promoted_at,
            source_path=plan.source,
            prod_root=prod_dir,
        )
    save_registry(registry, registry_path)

    return PromotionResult(
        planned=planned, promoted=planned, refusals=refusals, dry_run=False,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _describe_overwrite(plan: PromotionPlan, registry: dict[str, Any]) -> list[str]:
    """What --force is about to destroy, in the words of the old file.

    Printed unconditionally on an overwrite: a flag that silently
    replaces a production model is a flag nobody can review.
    """
    old_entry = registry_entry_for(registry, plan.stem)
    old_sha = plan.prev_sha256
    old_size = plan.prev_size_bytes
    trained = old_entry.get("trained_at_utc") if old_entry else None
    return [
        f"WARNING  {plan.stem}: overwriting {plan.dest}",
        f"    old: sha256={old_sha} size={old_size} trained={trained}"
        + ("" if old_entry else "  (old registry entry: none)"),
        f"    new: sha256={plan.sha256} size={plan.size_bytes} "
        f'trained={plan.manifest.get("created_at")}',
    ]


def _report(result: PromotionResult, registry: dict[str, Any], registry_path: Path) -> None:
    """Print the outcome.  Refusals go to stderr, successes to stdout."""
    prefix = "DRY-RUN " if result.dry_run else ""

    for refusal in result.refusals:
        print(
            f"REFUSED  {refusal.stem}: {refusal.code} -- {refusal.detail}",
            file=sys.stderr,
        )

    for plan in result.planned:
        if plan.dest_exists:
            for line in _describe_overwrite(plan, registry):
                print(prefix + line)
        verb = "would write" if result.dry_run else "promoted"
        print(f"{prefix}OK  {plan.stem}: {plan.source} -> {plan.dest} ({verb})")
        print(
            f"{prefix}    sha256={plan.sha256} size={plan.size_bytes} "
            f'passes={plan.manifest.get("passes")} '
            f'target_kind={plan.manifest.get("target_kind")} '
            f'trained={plan.manifest.get("created_at")}'
        )

    if result.refusals:
        print(
            f"Refused: {len(result.refusals)} of "
            f"{len(result.refusals) + len(result.planned)}. "
            "No files written, registry untouched.",
            file=sys.stderr,
        )
    elif result.dry_run:
        print(
            f"DRY-RUN Registry would gain or update {len(result.planned)} "
            f"entries in {registry_path}. Nothing written."
        )
    else:
        print(f"Registry updated: {registry_path} ({len(result.promoted)} promoted)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promote a trained candidate bundle into production",
    )
    ap.add_argument(
        "--regime",
        required=True,
        choices=list(PROD_STEMS_4H) + [ALL_REGIMES],
        help="Which model to promote, or every production one",
    )
    ap.add_argument(
        "--candidates-dir",
        type=Path,
        default=CANDIDATES_ROOT_4H,
        help="Root the trainer writes into (default: %(default)s)",
    )
    ap.add_argument(
        "--prod-dir",
        type=Path,
        default=MODELS_ROOT_4H,
        help="Root the bot loads from (default: %(default)s)",
    )
    ap.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Registry file to update (default: %(default)s)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing artifact. Waives no other check.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full gate and report, but write nothing",
    )
    args = ap.parse_args()

    stems = list(PROD_STEMS_4H) if args.regime == ALL_REGIMES else [args.regime]

    try:
        registry_before = load_registry(args.registry)
        result = promote(
            stems=stems,
            candidates_dir=args.candidates_dir,
            prod_dir=args.prod_dir,
            registry_path=args.registry,
            force=args.force,
            dry_run=args.dry_run,
        )
    except RegistryError as exc:
        print(f"REFUSED  registry unusable: {exc}", file=sys.stderr)
        print("Nothing written.", file=sys.stderr)
        sys.exit(1)

    _report(result, registry_before, args.registry)
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
