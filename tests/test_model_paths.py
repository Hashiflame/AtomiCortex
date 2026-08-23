"""Tests for ``src.models.model_paths`` — PR-Э1.1, A2-031 criterion 3.

Two kinds of test live here.

*Unit tests* pin the shape of every name and path the module builds.  Their
expected values are spelled out as **literal strings**, never derived from
the production configs: after the migration those configs compute their
defaults through this very module, so reading the expectation back from
them would make the test compare the module with itself.

*Linters* assert that no other module under ``src/`` or ``scripts/`` spells
an artifact name or a models root itself.  They parse each file with ``ast``
rather than grepping its text: a regex cannot tell a docstring from code,
and reformatting an f-string would slip past it.  ``tests/`` is deliberately
out of scope — a test that asserted ``"trend_model.pkl"`` via
``bundle_filename()`` would be a tautology, so tests keep their literals.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.models import model_paths as mp
from src.models.model_paths import (
    CANDIDATE_SUBDIR,
    CANDIDATES_ROOT_4H,
    META_STEM,
    MODELS_ROOT_1H,
    MODELS_ROOT_4H,
    MODELS_ROOT_15M,
    PROD_STEMS_4H,
    STEMS_1H,
    STEMS_15M,
    SUFFIX_1H,
    SUFFIX_15M,
    SUFFIX_PROD,
    SUFFIX_V3,
    SUFFIX_V3_SELECTED,
    bundle_filename,
    candidate_dir,
    candidate_path,
    meta_path,
    path_1h,
    path_15m,
    prod_path,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCAN_ROOTS = ("src", "scripts")
_MODEL_PATHS_MODULE = _REPO_ROOT / "src" / "models" / "model_paths.py"
_ML_STRATEGY_MODULE = _REPO_ROOT / "src" / "execution" / "strategies" / "ml_strategy.py"

# An artifact filename spelled out by hand.  Matches "trend_model.pkl",
# "orb_model_15m.pkl", "meta_model_v3.pkl" and the constant fragments that
# f-strings such as f"{regime}_model_v3.pkl" leave in the AST.
_FILENAME_RE = re.compile(r"_model[A-Za-z0-9_]*\.pkl")

# A models root spelled out by hand.  "models/prod" is listed ahead of time
# so that the PR which introduces it cannot smuggle in a fresh literal.
_ROOT_RE = re.compile(r"(?:data/(?:features/)?models|models/prod)(?:/|\b)")


# ---------------------------------------------------------------------------
# Linter machinery
# ---------------------------------------------------------------------------

def _scanned_files() -> list[Path]:
    """Every ``.py`` under src/ and scripts/, minus caches and the module
    that is allowed to hold the literals."""
    out: list[Path] = []
    for root in _SCAN_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path == _MODEL_PATHS_MODULE:
                continue
            out.append(path)
    return out


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Ids of the Constant nodes that are docstrings, not code."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _string_constants(path: Path) -> list[tuple[int, str]]:
    """(lineno, value) for every string constant in the file that is code.

    Constant fragments inside f-strings are ordinary Constant nodes, so
    ``f"{regime}_model_v3.pkl"`` is caught through its ``_model_v3.pkl``
    piece.  Comments never reach the AST at all.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_constant_ids(tree)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            out.append((node.lineno, node.value))
    return out


def _violations(pattern: re.Pattern[str]) -> list[str]:
    """``path:lineno: 'literal'`` for every match, so a failure is fixable
    without re-running the search by hand."""
    found: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(_REPO_ROOT)
        for lineno, value in _string_constants(path):
            if pattern.search(value):
                found.append(f"{rel}:{lineno}: {value!r}")
    return found


def _load_models_function() -> ast.FunctionDef:
    """The 4H strategy's ``_load_models`` node."""
    tree = ast.parse(_ML_STRATEGY_MODULE.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_models"
    ]
    assert len(matches) == 1, (
        f"expected exactly one _load_models in {_ML_STRATEGY_MODULE.name}, "
        f"found {len(matches)}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# 1. bundle_filename — the single primitive
# ---------------------------------------------------------------------------

def test_bundle_filename_prod_has_no_suffix() -> None:
    assert bundle_filename("trend") == "trend_model.pkl"
    assert bundle_filename("trend", SUFFIX_PROD) == "trend_model.pkl"
    assert bundle_filename("high_vol") == "high_vol_model.pkl"


def test_bundle_filename_applies_suffix() -> None:
    """One test with an internal loop, not a parametrize: parametrising over
    the suffixes would inflate the collected count for no extra signal."""
    cases = [
        ("trend", SUFFIX_V3, "trend_model_v3.pkl"),
        ("trend", SUFFIX_V3_SELECTED, "trend_model_v3_sel.pkl"),
        ("trend", SUFFIX_1H, "trend_model_1h.pkl"),
        ("trend", SUFFIX_15M, "trend_model_15m.pkl"),
        ("high_vol", SUFFIX_V3, "high_vol_model_v3.pkl"),
    ]
    for stem, suffix, expected in cases:
        assert bundle_filename(stem, suffix) == expected, (stem, suffix)


def test_bundle_filename_accepts_non_regime_stem() -> None:
    """``orb`` and ``meta`` are not regimes, and ``all`` is the whole-dataset
    trainer's stem — the primitive must not assume a regime."""
    assert bundle_filename("orb", SUFFIX_15M) == "orb_model_15m.pkl"
    assert bundle_filename(META_STEM, SUFFIX_V3) == "meta_model_v3.pkl"
    assert bundle_filename("all") == "all_model.pkl"


# ---------------------------------------------------------------------------
# 2. Path constructors
# ---------------------------------------------------------------------------

def test_prod_path_sits_directly_under_root() -> None:
    path = prod_path("some/root", "trend")
    assert path == Path("some/root/trend_model.pkl")
    assert path.parent == Path("some/root")


def test_candidate_path_sits_under_candidate_subdir() -> None:
    path = candidate_path("some/root", "trend")
    assert path == Path("some/root") / CANDIDATE_SUBDIR / "trend_model_v3.pkl"
    assert candidate_path("some/root", "trend", SUFFIX_V3_SELECTED).name == (
        "trend_model_v3_sel.pkl"
    )


def test_candidate_dir_is_parent_of_candidate_path() -> None:
    assert candidate_path("some/root", "trend").parent == candidate_dir("some/root")
    assert meta_path("some/root").parent == candidate_dir("some/root")


def test_meta_path_matches_meta_strategy_default() -> None:
    """Literal on purpose — ``MetaMLStrategyConfig.meta_model_path`` is
    computed through this module after the migration."""
    assert str(meta_path(CANDIDATES_ROOT_4H)) == (
        "data/features/models/v3/meta_model_v3.pkl"
    )


def test_path_1h_and_path_15m_shapes() -> None:
    """Literals on purpose, for the same reason as the meta path above."""
    assert str(path_1h(MODELS_ROOT_1H, "trend")) == "data/models/1h/trend_model_1h.pkl"
    assert str(path_1h(MODELS_ROOT_1H, "high_vol")) == (
        "data/models/1h/high_vol_model_1h.pkl"
    )
    assert str(path_15m(MODELS_ROOT_15M, "trend")) == (
        "data/models/15m/trend_model_15m.pkl"
    )
    assert str(path_15m(MODELS_ROOT_15M, "orb")) == "data/models/15m/orb_model_15m.pkl"


# ---------------------------------------------------------------------------
# 3. Contracts the loader depends on
# ---------------------------------------------------------------------------

def test_prod_stems_4h_is_exactly_trend_and_high_vol() -> None:
    """Two-regime production is a contract, not an accident: ``range`` is
    routed onto the trend model and no third bundle is ever loaded."""
    assert PROD_STEMS_4H == ("trend", "high_vol")
    assert STEMS_1H == ("trend", "high_vol")
    assert STEMS_15M == ("trend", "orb")


def test_loader_loop_has_no_literal_regime_list() -> None:
    """``_load_models`` must take its regimes from ``PROD_STEMS_4H``.

    A literal list inside the loader is how the loader and the constant
    drift apart, so neither a string sequence nor a bare regime name may
    appear in the function body.
    """
    fn = _load_models_function()
    for node in ast.walk(fn):
        if isinstance(node, (ast.List, ast.Tuple)):
            literals = [
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
            assert not literals, (
                f"_load_models carries a literal string sequence {literals} "
                f"at line {node.lineno} — it must read PROD_STEMS_4H"
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in PROD_STEMS_4H, (
                f"_load_models spells the regime {node.value!r} at line "
                f"{node.lineno} — it must come from PROD_STEMS_4H"
            )


def test_roots_are_relative_paths() -> None:
    """An absolute root would break ``WorkingDirectory=`` in the systemd
    unit, which is what makes ``./data/...`` resolve on the VM."""
    for name in (
        "MODELS_ROOT_4H",
        "CANDIDATES_ROOT_4H",
        "MODELS_ROOT_1H",
        "MODELS_ROOT_15M",
    ):
        root = getattr(mp, name)
        assert isinstance(root, Path), name
        assert not root.is_absolute(), f"{name} = {root}"


def test_prod_and_candidates_roots_are_distinct_symbols() -> None:
    """Two names, deliberately.

    Э1.1 gives them the same value; Э1.2 moves the production root to
    ``models/prod`` and leaves candidates where they are.  Asserting the
    *names* keeps that a one-line change; asserting their equality would
    have to be deleted in the very next PR.
    """
    assert isinstance(vars(mp).get("MODELS_ROOT_4H"), Path)
    assert isinstance(vars(mp).get("CANDIDATES_ROOT_4H"), Path)
    assert MODELS_ROOT_4H is not None and CANDIDATES_ROOT_4H is not None


# ---------------------------------------------------------------------------
# 4. Linters
# ---------------------------------------------------------------------------

def test_no_model_filename_literals_in_src_and_scripts() -> None:
    found = _violations(_FILENAME_RE)
    assert not found, (
        "artifact filenames must come from model_paths.bundle_filename():\n  "
        + "\n  ".join(found)
    )


def test_no_models_root_literals_in_src_and_scripts() -> None:
    found = _violations(_ROOT_RE)
    assert not found, (
        "models roots must come from model_paths constants:\n  "
        + "\n  ".join(found)
    )


# ---------------------------------------------------------------------------
# 5. Controls — without these the linters above can go vacuously green
# ---------------------------------------------------------------------------

def test_linter_scans_a_nonempty_set_of_files() -> None:
    """A typo in the glob would make both linters pass on an empty set."""
    files = _scanned_files()
    assert len(files) > 50, len(files)
    assert _MODEL_PATHS_MODULE not in files
    total_constants = sum(len(_string_constants(p)) for p in files)
    assert total_constants > 100, total_constants


def test_linter_patterns_match_model_paths_module() -> None:
    """A broken regex would silence both linters everywhere.

    The filename pattern is checked against the module's *output* rather
    than its source: ``bundle_filename`` assembles the name from a template,
    so no complete filename literal exists inside the module by design.
    The root pattern is checked against the source, where the roots do live.
    """
    assert _FILENAME_RE.search(bundle_filename("trend"))
    assert _FILENAME_RE.search(bundle_filename(META_STEM, SUFFIX_V3))
    source = _MODEL_PATHS_MODULE.read_text(encoding="utf-8")
    assert _ROOT_RE.search(source)
