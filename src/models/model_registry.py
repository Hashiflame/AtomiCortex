"""The record of which model bundles are in production and what they are.

A promoted bundle is an opaque pickle: nothing about the file says which
candidate it came from, when it was trained, on what target, or whether
it ever passed a go-live gate.  Git tracks its bytes but not its meaning,
and a binary diff says only "these bytes changed".  This module is the
text that goes beside it.

Split out of the promotion script on purpose.  ``src/`` cannot import
``scripts/`` -- there is no package there and no path entry pointing at
it -- so any logic the loader will eventually need has to live here from
the start, or it would have to be moved on the very next change.

Two responsibilities, kept apart:

* reading and writing the file (:func:`load_registry`, :func:`save_registry`)
* comparing it against what is on disk (:func:`verify_registry`)

The second one never decides anything.  It reports differences and lets
the caller choose what they mean -- a test turns them red, a promotion
run refuses, and a loader may one day refuse to start.  Those are three
different policies over one set of facts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.models.model_paths import REGISTRY_PATH

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Version of THIS file's format, unrelated to the manifest schema stamped
# into a bundle by the trainer.  Bumped when the shape below changes in a
# way a reader could not absorb; readers that see a version they do not
# know must refuse rather than guess.
REGISTRY_SCHEMA_VERSION: int = 1

# 64 KiB: the chunk size already used by the two download verifiers, kept
# the same so there is one answer in the tree to "how do we hash a file".
_CHUNK_BYTES = 65_536


class RegistryError(Exception):
    """The registry file exists but could not be understood.

    Distinct from "there is no registry yet", which is a normal state on
    a fresh checkout and is answered with an empty registry instead.  A
    file that is present and unreadable is the dangerous case: treating
    it as empty would let the next write silently drop every entry it
    could not parse.

    Attributes
    ----------
    path:
        The file that could not be read.
    detail:
        What was wrong with it, in words.
    """

    def __init__(self, path: Path, detail: str) -> None:
        self.path = Path(path)
        self.detail = detail
        super().__init__(f"{self.path}: {detail}")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of a file, read in chunks.

    Chunked rather than ``read_bytes()`` because a bundle is a few
    hundred kilobytes today and nothing guarantees it stays that way.
    ``OSError`` propagates: a caller asking for the hash of a file it
    cannot read has a problem this function must not paper over.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> Path:
    """``path`` relative to the repository root when it is inside it.

    Paths are stored relative so the registry means the same thing in a
    checkout, in CI and on the VM.  A path outside the tree (a tmp
    directory in a test) is returned unchanged rather than forced.
    """
    try:
        return path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def _empty_registry() -> dict[str, Any]:
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "models": {}}


def load_registry(path: Path | str = REGISTRY_PATH) -> dict[str, Any]:
    """Read the registry, or return an empty one when there is no file.

    Raises
    ------
    RegistryError
        The file exists but is not JSON, is not an object, has no
        ``models`` mapping, or carries a ``schema_version`` this code
        does not know.  Every one of those is a state where continuing
        would mean writing over data whose meaning is unknown.
    """
    path = Path(path)
    if not path.exists():
        return _empty_registry()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(path, f"not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RegistryError(path, f"could not be read: {exc}") from exc

    if not isinstance(raw, dict):
        raise RegistryError(path, f"top level is {type(raw).__name__}, not an object")

    version = raw.get("schema_version")
    if version != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(
            path,
            f"schema_version is {version!r}, this code understands "
            f"{REGISTRY_SCHEMA_VERSION!r}",
        )

    models = raw.get("models")
    if not isinstance(models, dict):
        raise RegistryError(
            path, f"'models' is {type(models).__name__}, not an object"
        )
    return raw


def save_registry(
    registry: dict[str, Any], path: Path | str = REGISTRY_PATH,
) -> None:
    """Write the registry atomically, in a form that diffs cleanly.

    ``sort_keys`` and a fixed indent are not cosmetic here: this file is
    under version control, and a re-promotion that reordered untouched
    entries would bury the one line that actually changed.  The trailing
    newline keeps git from reporting "\\ No newline at end of file" on
    every write.

    Written through a sibling temporary file and ``os.replace`` so a
    reader never sees a half-written registry, matching how the other
    JSON state files in the tree are flushed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        registry, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"

    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def registry_entry_for(
    registry: dict[str, Any], stem: str,
) -> dict[str, Any] | None:
    """The entry for one stem, or None when it has never been promoted."""
    entry = registry.get("models", {}).get(stem)
    return entry if isinstance(entry, dict) else None


# ---------------------------------------------------------------------------
# Building an entry
# ---------------------------------------------------------------------------

def build_entry(
    *,
    bundle_path: Path,
    manifest: dict[str, Any],
    promoted_at_utc: str,
    source_path: Path,
    prod_root: Path,
) -> dict[str, Any]:
    """One registry entry describing the bundle now at ``bundle_path``.

    ``prod_root`` is the root the recorded ``path`` is expressed against,
    so an entry written from a temporary directory in a test stays
    self-consistent instead of pretending to live in the repository.

    Every manifest field is read through ``.get`` and may come back
    None.  Bundles on disk were written by more than one version of the
    trainer, and an entry that refused to describe an older one would be
    useless for exactly the artifact whose provenance is least obvious.
    ``barriers`` and ``eval`` are copied whole rather than flattened:
    their contents have changed once already and a copy stays true.
    """
    bundle_path = Path(bundle_path)
    prod_root = Path(prod_root)
    try:
        relative = bundle_path.relative_to(prod_root)
    except ValueError:
        relative = Path(bundle_path.name)
    recorded_path = (_repo_relative(prod_root) / relative).as_posix()

    barriers = manifest.get("barriers")
    evaluation = manifest.get("eval")
    return {
        "path": recorded_path,
        "sha256": sha256_file(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "promoted_at_utc": promoted_at_utc,
        "promoted_from": _repo_relative(Path(source_path)).as_posix(),
        "manifest_schema_version": manifest.get("schema_version"),
        "regime": manifest.get("regime"),
        "interval": manifest.get("interval"),
        "target_kind": manifest.get("target_kind"),
        "trained_at_utc": manifest.get("created_at"),
        "trained_git_commit": manifest.get("git_commit"),
        "feature_columns_hash": manifest.get("feature_columns_hash"),
        "n_features": manifest.get("n_features"),
        "barriers": dict(barriers) if isinstance(barriers, dict) else None,
        "passes": bool(manifest.get("passes")),
        "written_despite_failing": bool(manifest.get("written_despite_failing")),
        "eval": dict(evaluation) if isinstance(evaluation, dict) else None,
        "num_trees": manifest.get("num_trees"),
        "best_iteration": manifest.get("best_iteration"),
    }


# ---------------------------------------------------------------------------
# Verification -- diagnosis only
# ---------------------------------------------------------------------------

def verify_registry(
    registry: dict[str, Any], prod_root: Path | str,
) -> list[str]:
    """Differences between the registry and the files in ``prod_root``.

    Returns a list of one-line descriptions; an empty list means the two
    agree.  **This function never raises.**  It is a diagnosis, not a
    verdict: an unreadable file, a missing directory and a changed hash
    all come back as text, and what any of them means is the caller's
    decision.  A test turns them red, a promotion refuses, a loader may
    one day refuse to start -- three policies over one set of facts, and
    an exception here would force all three to be the same.

    Four kinds of difference are reported: an entry whose file is gone,
    a file with no entry, a size that moved, and a hash that moved.
    """
    problems: list[str] = []
    prod_root = Path(prod_root)

    models = registry.get("models")
    if not isinstance(models, dict):
        return [
            f"registry has no 'models' mapping "
            f"(found {type(models).__name__})"
        ]

    accounted: set[str] = set()
    for stem in sorted(models):
        entry = models[stem]
        if not isinstance(entry, dict):
            problems.append(f"{stem}: entry is {type(entry).__name__}, not an object")
            continue

        recorded = entry.get("path")
        if not isinstance(recorded, str) or not recorded:
            problems.append(f"{stem}: entry has no usable 'path'")
            continue

        # Located by filename under the given root rather than by the
        # recorded path, so a registry written elsewhere (a test, another
        # checkout) still verifies against the root actually in use.
        filename = Path(recorded).name
        accounted.add(filename)
        on_disk = prod_root / filename

        if not on_disk.is_file():
            problems.append(f"{stem}: {recorded} is in the registry but not on disk")
            continue

        expected_size = entry.get("size_bytes")
        try:
            actual_size = on_disk.stat().st_size
        except OSError as exc:
            problems.append(f"{stem}: {recorded} could not be stat'ed: {exc}")
            continue
        if isinstance(expected_size, int) and actual_size != expected_size:
            problems.append(
                f"{stem}: {recorded} is {actual_size} bytes, registry says "
                f"{expected_size}"
            )

        expected_sha = entry.get("sha256")
        try:
            actual_sha = sha256_file(on_disk)
        except OSError as exc:
            problems.append(f"{stem}: {recorded} could not be hashed: {exc}")
            continue
        if not isinstance(expected_sha, str):
            problems.append(f"{stem}: entry has no 'sha256'")
        elif actual_sha != expected_sha:
            problems.append(
                f"{stem}: {recorded} hashes to {actual_sha}, registry says "
                f"{expected_sha}"
            )

    try:
        on_disk_names = (
            sorted(p.name for p in prod_root.glob("*.pkl"))
            if prod_root.is_dir()
            else []
        )
    except OSError as exc:
        problems.append(f"{prod_root} could not be listed: {exc}")
        on_disk_names = []

    for name in on_disk_names:
        if name not in accounted:
            problems.append(f"{name} is on disk but has no registry entry")

    return problems
