"""The single point of definition for model artifact names and roots.

Before this module the same six filenames were spelled out by hand in
seventeen places and the four roots in fifteen more, so a trainer and the
strategy that loads its output could disagree without anything failing
(A2-031).  Everything here exists to make that disagreement impossible to
express.

All six naming schemes in the tree are one scheme with a different suffix::

    trend_model.pkl         trend   + _model +          + .pkl
    trend_model_v3.pkl      trend   + _model + _v3      + .pkl
    trend_model_v3_sel.pkl  trend   + _model + _v3_sel  + .pkl
    trend_model_1h.pkl      trend   + _model + _1h      + .pkl
    orb_model_15m.pkl       orb     + _model + _15m     + .pkl
    meta_model_v3.pkl       meta    + _model + _v3      + .pkl

so there is one primitive — :func:`bundle_filename` — and the rest is
named constants and thin path constructors over it.

Roots are always passed in, never read from here by the constructors: every
one of them is configurable through a CLI flag or a strategy config, and a
root baked into the constructor would just move the duplication one level
up.  Depends on ``pathlib`` alone: ``lgbm_trainer`` imports this module, so
any richer dependency risks an import cycle.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------

# Kept as a template rather than an f-string so the shape of a bundle name
# is one readable literal instead of being spread across an expression.
_FILENAME_TEMPLATE = "{stem}_model{suffix}.pkl"


def bundle_filename(stem: str, suffix: str = "") -> str:
    """Filename of a model bundle.

    ``stem`` is not always a regime: the 15m line trains an ``orb`` model
    and the meta gate an entire model named ``meta``, so the parameter is
    named for what it is rather than for the common case.
    """
    return _FILENAME_TEMPLATE.format(stem=stem, suffix=suffix)


# ---------------------------------------------------------------------------
# Suffixes — the only place these strings live
# ---------------------------------------------------------------------------

# Named rather than defaulted so that a production name reads as a decision
# and not as a forgotten argument.
SUFFIX_PROD = ""
SUFFIX_V3 = "_v3"
SUFFIX_V3_SELECTED = "_v3_sel"
SUFFIX_1H = "_1h"
SUFFIX_15M = "_15m"


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

# Tracked in git, unlike everything under data/: this is the artifact the VM
# pulls, so the bundle the bot loads is pinned by the same commit as the code
# that loads it. CANDIDATES_ROOT_4H below stays under data/, which is why
# these are two names and not one.
MODELS_ROOT_4H = Path("models/prod")

# Where retrain_v3.py writes and where the meta artifacts live. Untracked.
CANDIDATES_ROOT_4H = Path("data/features/models")

MODELS_ROOT_1H = Path("data/models/1h")
MODELS_ROOT_15M = Path("data/models/15m")

# "candidate", not "v3": the subdirectory means "trained but not promoted",
# which outlives the name of any particular model line.
CANDIDATE_SUBDIR = "v3"


# What describes the promoted bundles: one entry per stem, carrying the
# hash and the provenance of the file actually on disk. Tracked next to the
# systemd units rather than beside the bundles, so a diff of the deployment
# shows the model change as text.
REGISTRY_PATH = Path("deploy/model_registry.json")


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------

# Exactly what the 4H strategy loads. ``range`` is absent on purpose: it is
# a regime label routed onto the trend model, never a bundle of its own.
PROD_STEMS_4H = ("trend", "high_vol")
STEMS_1H = ("trend", "high_vol")
STEMS_15M = ("trend", "orb")
META_STEM = "meta"


# ---------------------------------------------------------------------------
# Path constructors
# ---------------------------------------------------------------------------

def prod_path(models_dir: Path | str, stem: str) -> Path:
    """Bundle the production loader reads, directly under the root."""
    return Path(models_dir) / bundle_filename(stem)


def candidate_dir(candidates_dir: Path | str) -> Path:
    """Subdirectory holding trained-but-not-promoted bundles."""
    return Path(candidates_dir) / CANDIDATE_SUBDIR


def candidate_path(
    candidates_dir: Path | str, stem: str, suffix: str = SUFFIX_V3,
) -> Path:
    """Bundle a retrain writes, inside :func:`candidate_dir`."""
    return candidate_dir(candidates_dir) / bundle_filename(stem, suffix)


def meta_path(candidates_dir: Path | str) -> Path:
    """Bundle of the meta gate — a candidate artifact like any other."""
    return candidate_path(candidates_dir, META_STEM, SUFFIX_V3)


def path_1h(models_dir: Path | str, stem: str) -> Path:
    """Bundle of the 1H line."""
    return Path(models_dir) / bundle_filename(stem, SUFFIX_1H)


def path_15m(models_dir: Path | str, stem: str) -> Path:
    """Bundle of the 15m line."""
    return Path(models_dir) / bundle_filename(stem, SUFFIX_15M)
