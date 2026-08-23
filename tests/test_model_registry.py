"""Tests — the tracked production model root and its registry.

Two things are asserted here that no unit test of the promotion script
could cover, because they are properties of the repository rather than
of any function: that ``models/prod/*.pkl`` is actually committable
(``.gitignore`` carries a global ``*.pkl`` rule), and that
``deploy/model_registry.json`` describes exactly the artifacts that are
on disk.

Same approach as ``tests/test_install_units_script.py``: shell out to
git with ``check=False`` and skip when this is not a checkout, so the
file does not turn red in an environment that simply has no git.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.models.model_paths import (
    MODELS_ROOT_4H,
    PROD_STEMS_4H,
    REGISTRY_PATH,
    prod_path,
)
from src.models.model_registry import (
    REGISTRY_SCHEMA_VERSION,
    RegistryError,
    load_registry,
    save_registry,
    sha256_file,
    verify_registry,
)

_ROOT = Path(__file__).resolve().parent.parent
_PROD_DIR = _ROOT / MODELS_ROOT_4H


def _check_ignore(rel_path: str, *, verbose: bool = False):
    """``git check-ignore`` on a repo-relative path.

    The path need not exist: check-ignore matches rules, not files, which
    is exactly why this test can run before anything was promoted.

    ``verbose`` is for the failure message only.  Under ``-v`` the exit
    code means "a rule matched", and a *negation* is a rule -- so ``-v``
    exits 0 on a path that is NOT ignored.  The verdict therefore has to
    be taken from the plain form, where 1 means "not ignored".
    """
    argv = ["git", "check-ignore"] + (["-v"] if verbose else []) + [rel_path]
    return subprocess.run(
        argv, cwd=str(_ROOT), capture_output=True, text=True, check=False,
    )


# ---------------------------------------------------------------------------
# 1. The production root is committable
# ---------------------------------------------------------------------------

def test_prod_pkl_is_not_gitignored() -> None:
    """Production bundles must be addable to the index (Э1.2).

    ``.gitignore`` carries a blanket ``*.pkl`` rule for the candidate
    artifacts under the ignored ``data/`` tree.  Without a negation for
    the production root, ``git add`` would silently do nothing and the VM
    would pull a commit with no models in it — a failure whose only
    symptom is a fail-soft "Model not found" line in the journal.

    Deliberately a failure and not a skip when the directory is empty:
    check-ignore matches the rule, so an absent file is no reason to
    stay quiet.
    """
    for stem in PROD_STEMS_4H:
        rel = prod_path(MODELS_ROOT_4H, stem).as_posix()
        proc = _check_ignore(rel)
        if proc.returncode == 128:
            pytest.skip("not a git checkout")
        rule = _check_ignore(rel, verbose=True).stdout.strip()
        assert proc.returncode == 1, (
            f"{rel} is ignored by {rule!r} — it could never be committed. "
            "Add a negation for the production root to .gitignore."
        )


# ---------------------------------------------------------------------------
# 2. The module -- reading, writing, hashing
# ---------------------------------------------------------------------------

def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    """No registry yet is a normal state, not an error.

    A feature branch before the first promotion, and a fresh clone that
    has not pulled the artifacts, both land here.
    """
    registry = load_registry(tmp_path / "nothing-here.json")
    assert registry == {"schema_version": REGISTRY_SCHEMA_VERSION, "models": {}}


def test_load_registry_corrupt_raises(tmp_path: Path) -> None:
    """A present but unreadable registry must not read as empty.

    Silently substituting an empty one would make the next write drop
    every entry it could not parse -- the one failure mode a registry
    exists to prevent.
    """
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(broken)

    wrong_shape = tmp_path / "list.json"
    wrong_shape.write_text("[]", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(wrong_shape)

    future = tmp_path / "future.json"
    future.write_text(
        json.dumps({"schema_version": 99, "models": {}}), encoding="utf-8"
    )
    with pytest.raises(RegistryError):
        load_registry(future)


def test_save_registry_is_atomic_and_stable(tmp_path: Path) -> None:
    """Two writes of the same content give the same bytes, and no litter.

    Byte-stability is what keeps a re-promotion's diff down to the lines
    that changed; a stray .tmp would end up in ``git status``.
    """
    target = tmp_path / "sub" / "registry.json"
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "models": {"trend": {"path": "x", "sha256": "y"}, "aaa": {"path": "z"}},
    }
    save_registry(payload, target)
    first = target.read_bytes()
    save_registry(payload, target)
    assert target.read_bytes() == first
    assert first.endswith(b"\n")
    assert list(target.parent.glob("*.tmp")) == []
    # sort_keys is what makes the bytes stable across dict orderings.
    assert first.index(b'"aaa"') < first.index(b'"trend"')
    assert load_registry(target) == payload


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """Chunked reading must not change the answer.

    Deliberately larger than one 64 KiB chunk, or the loop would never
    run twice and the chunking would go untested.
    """
    blob = tmp_path / "big.bin"
    blob.write_bytes(b"atomicortex" * 20_000)
    assert blob.stat().st_size > 65_536
    assert sha256_file(blob) == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_verify_registry_reports_mismatch_without_raising(tmp_path: Path) -> None:
    """verify_registry diagnoses; it never decides and never raises.

    All four kinds of difference are exercised at once, including the
    case where the directory does not exist at all -- a caller must be
    able to ask about a root that was never created.
    """
    prod = tmp_path / "prod"
    prod.mkdir()
    good = prod / "trend_model.pkl"
    good.write_bytes(b"good bytes")
    orphan = prod / "high_vol_model.pkl"
    orphan.write_bytes(b"nobody claims me")

    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "models": {
            "trend": {
                "path": "models/prod/trend_model.pkl",
                "sha256": "0" * 64,
                "size_bytes": 999,
            },
            "gone": {
                "path": "models/prod/gone_model.pkl",
                "sha256": "1" * 64,
                "size_bytes": 1,
            },
        },
    }
    problems = verify_registry(registry, prod)
    joined = "\n".join(problems)
    assert "trend" in joined and "hashes to" in joined
    assert "999" in joined
    assert "gone" in joined and "not on disk" in joined
    assert "high_vol_model.pkl is on disk but has no registry entry" in problems

    # A root that does not exist is a question, not a crash.
    assert verify_registry(registry, tmp_path / "absent") != []
    # And a registry that agrees with disk is silent.
    agreed = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "models": {
            "trend": {
                "path": "models/prod/trend_model.pkl",
                "sha256": sha256_file(good),
                "size_bytes": good.stat().st_size,
            },
            "high_vol": {
                "path": "models/prod/high_vol_model.pkl",
                "sha256": sha256_file(orphan),
                "size_bytes": orphan.stat().st_size,
            },
        },
    }
    assert verify_registry(agreed, prod) == []


# ---------------------------------------------------------------------------
# 3. The repository -- what is on disk, and what the registry says about it
# ---------------------------------------------------------------------------

# One megabyte per artifact.  The bundles are a few hundred kilobytes and
# the largest tracked file in the tree is about one megabyte, so this
# leaves room without leaving room for a surprise: every promotion adds a
# fresh full copy to history, and pickled boosters delta-compress badly.
_SIZE_CAP_BYTES = 1024 * 1024

_REGISTRY_FILE = _ROOT / REGISTRY_PATH


def _prod_bundles() -> list[Path]:
    return sorted(_PROD_DIR.glob("*.pkl")) if _PROD_DIR.is_dir() else []


def _require_artifacts() -> list[Path]:
    """The promoted bundles, or skip.

    Skipping is right here and only here: these tests assert that the
    registry agrees with the files, which is vacuously true when there
    are none.  A checkout that has not pulled the artifacts, or a branch
    before the first promotion, is not a regression.  The tests below
    that read the registry itself do NOT skip -- that file is tracked
    and is present in every checkout, so its absence is a real fault.
    """
    bundles = _prod_bundles()
    if not bundles:
        pytest.skip(f"{MODELS_ROOT_4H} holds no artifacts -- nothing to verify")
    return bundles


def test_tracked_prod_artifacts_under_size_cap() -> None:
    """No single tracked bundle may exceed the cap.

    Guards the repository rather than the code: git keeps every version
    of a binary forever, LFS is not configured here, and the cost of
    noticing late is a history that cannot be trimmed without a rewrite.
    """
    oversized = [
        f"{p.relative_to(_ROOT)} is {p.stat().st_size} bytes"
        for p in _require_artifacts()
        if p.stat().st_size > _SIZE_CAP_BYTES
    ]
    assert not oversized, (
        f"tracked production bundles over the {_SIZE_CAP_BYTES}-byte cap:\n  "
        + "\n  ".join(oversized)
        + "\nEvery promotion commits a fresh full copy; a binary this large "
        "cannot be removed from history later without rewriting it."
    )


def test_prod_dir_has_no_python_package_marker() -> None:
    """The artifact root must not become an importable package.

    ``sys.path`` carries the repository root in every script and in
    conftest, so a stray ``__init__.py`` here would put a package named
    ``models`` on the path next to ``src.models`` -- an import ambiguity
    that would surface as something entirely unrelated.
    """
    models_root = _ROOT / MODELS_ROOT_4H.parts[0]
    if not models_root.exists():
        return
    stray = sorted(
        str(p.relative_to(_ROOT)) for p in models_root.rglob("*.py")
    )
    assert stray == [], f"python files under the artifact root: {stray}"


def test_registry_is_valid_json_and_schema() -> None:
    """The registry file parses and is a version this code understands.

    A failure, never a skip: the file is tracked, so it exists wherever
    the code does.
    """
    registry = load_registry(_REGISTRY_FILE)
    assert registry["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert isinstance(registry["models"], dict)


def test_registry_entries_cover_prod_stems() -> None:
    """Every stem the strategy loads has an entry."""
    registry = load_registry(_REGISTRY_FILE)
    missing = sorted(set(PROD_STEMS_4H) - set(registry["models"]))
    assert not missing, (
        f"no registry entry for {missing} -- the strategy loads these and "
        "nothing records what they are"
    )


def test_registry_covers_every_tracked_prod_artifact() -> None:
    """Registry and directory describe the same set of files, both ways.

    A file with no entry is an artifact nobody can trace; an entry with
    no file is a registry that promises something the VM will not find.
    """
    bundles = _require_artifacts()
    registry = load_registry(_REGISTRY_FILE)
    on_disk = {p.name for p in bundles}
    recorded = {
        Path(entry["path"]).name for entry in registry["models"].values()
    }
    assert on_disk == recorded, (
        f"on disk but not in the registry: {sorted(on_disk - recorded)}; "
        f"in the registry but not on disk: {sorted(recorded - on_disk)}"
    )


def test_registry_sha256_matches_files_on_disk() -> None:
    """The recorded hash and size are the ones the files actually have.

    This is the check that catches a bundle replaced by hand, and the
    one that would catch a commit where the registry went in and the
    binaries did not.
    """
    _require_artifacts()
    problems = verify_registry(load_registry(_REGISTRY_FILE), _PROD_DIR)
    assert problems == [], "registry disagrees with disk:\n  " + "\n  ".join(
        problems
    )


def test_registry_file_is_tracked_in_git() -> None:
    """The registry is committed, not merely present.

    An untracked registry would describe a deployment nobody else can
    see, and the VM pulls only what is tracked.
    """
    rel = REGISTRY_PATH.as_posix()
    proc = subprocess.run(
        ["git", "ls-files", "--", rel],
        cwd=str(_ROOT), capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        pytest.skip("not a git checkout")
    assert proc.stdout.strip(), (
        f"{rel} is not tracked -- run: git add {rel}"
    )


def test_registry_entries_are_tracked_in_git() -> None:
    """Every registry entry names a file that is in the index.

    The checks above compare the registry against the working directory
    and skip when that directory is empty.  That leaves one commit shape
    fully green: registry in, bundles out -- a broken ``.gitignore``
    negation, a ``git add`` that missed the binaries, a stray
    ``git rm --cached``.  CI passes, the VM pulls, and the only symptom
    is a fail-soft "Model not found" in the journal.

    So this one asks git, not the filesystem.  Both directions: an entry
    with no tracked file is a deployment that will arrive empty, and a
    tracked bundle with no entry is an artifact nobody can trace.

    An empty registry is neither a skip nor a failure -- two empty sets
    are equal, and that is the correct answer on a branch before the
    first promotion.
    """
    proc = subprocess.run(
        ["git", "ls-files", "--", MODELS_ROOT_4H.as_posix()],
        cwd=str(_ROOT), capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        pytest.skip("not a git checkout")

    tracked = {
        line for line in proc.stdout.splitlines()
        if line.strip().endswith(".pkl")
    }
    registry = load_registry(_REGISTRY_FILE)
    recorded = {
        str(entry["path"]) for entry in registry["models"].values()
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }

    untracked = sorted(recorded - tracked)
    unrecorded = sorted(tracked - recorded)
    assert not untracked and not unrecorded, (
        f"in the registry but not tracked by git: {untracked}; "
        f"tracked by git but not in the registry: {unrecorded}. "
        f"Run: git add {MODELS_ROOT_4H.as_posix()}/*.pkl"
    )
