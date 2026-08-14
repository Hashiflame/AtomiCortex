"""Tests — systemd unit-file linter.

Verifies that StartLimitIntervalSec / StartLimitBurst live in [Unit]
and NOT in [Service], and that no unit is configured to place real
orders.  Uses a line-by-line section parser (NOT configparser) because
systemd units contain repeated keys like Environment= which configparser
cannot handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
_START_LIMIT_KEYS = {"StartLimitIntervalSec", "StartLimitBurst"}

# Entry points that can submit real orders: they expose --dry-run and
# default it to False.  run_paper*.py / run_watchdog.py / run_reconciler.py
# spell the mode flag differently (--trading-mode) and place no orders.
_ORDER_CAPABLE_SCRIPTS = ("run_live.py", "run_live_15m.py")

# argparse default of --mode in scripts/run_live.py.  A unit may omit
# --mode and inherit this default, because "testnet" cannot reach mainnet
# credentials.  If the parser default ever changes, this must move with it.
_ARGPARSE_DEFAULT_MODE = "testnet"


def _parse_sections(path: Path) -> dict[str, list[str]]:
    """Return ``{section_name: [lines]}`` for a systemd unit file.

    Section names are stored WITH brackets, e.g. ``"[Unit]"``.
    Lines before any section header are stored under ``""``.
    """
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return sections


def _service_files() -> list[Path]:
    """Collect all ``deploy/*.service`` files."""
    files = sorted(_DEPLOY_DIR.glob("*.service"))
    assert files, "No .service files found in deploy/"
    return files


def _exec_start(path: Path) -> str:
    """Return the unit's ExecStart= command as one whitespace-normalised line.

    systemd splits ExecStart across lines with a trailing backslash.
    Reading a single line would yield only the interpreter path, and every
    flag assertion below would then pass vacuously — so continuations are
    joined here before any linter looks at the command.
    """
    collected: list[str] = []
    for line in _parse_sections(path).get("[Service]", []):
        if not collected and not line.startswith("ExecStart="):
            continue
        continued = line.endswith("\\")
        collected.append(line[:-1] if continued else line)
        if not continued:
            break
    if not collected:
        return ""
    return " ".join(" ".join(collected).split()).removeprefix("ExecStart=")


def _is_order_capable(exec_start: str) -> bool:
    """True if this ExecStart runs an entry point that can submit orders."""
    return any(script in exec_start for script in _ORDER_CAPABLE_SCRIPTS)


def _flag_value(exec_start: str, flag: str) -> str | None:
    """Return the token following ``flag``, or None if absent/valueless."""
    tokens = exec_start.split()
    if flag not in tokens:
        return None
    index = tokens.index(flag) + 1
    if index >= len(tokens) or tokens[index].startswith("-"):
        return None
    return tokens[index]


@pytest.mark.parametrize("service_file", _service_files(), ids=lambda p: p.name)
def test_start_limit_keys_in_unit_section(service_file: Path) -> None:
    """StartLimitIntervalSec and StartLimitBurst, when present, MUST be in [Unit]."""
    sections = _parse_sections(service_file)
    full_text = service_file.read_text(encoding="utf-8")
    unit_lines = "\n".join(sections.get("[Unit]", []))
    for key in _START_LIMIT_KEYS:
        if key not in full_text:
            continue  # key not used in this unit (e.g. oneshot timer services)
        assert key in unit_lines, (
            f"{service_file.name}: {key} not found in [Unit] section"
        )


@pytest.mark.parametrize("service_file", _service_files(), ids=lambda p: p.name)
def test_no_start_limit_in_service_section(service_file: Path) -> None:
    """StartLimitIntervalSec and StartLimitBurst MUST NOT be in [Service]."""
    sections = _parse_sections(service_file)
    service_lines = "\n".join(sections.get("[Service]", []))
    for key in _START_LIMIT_KEYS:
        assert key not in service_lines, (
            f"{service_file.name}: {key} found in [Service] section — "
            "must be in [Unit] (systemd 255+ ignores it in [Service])"
        )


@pytest.mark.parametrize("service_file", _service_files(), ids=lambda p: p.name)
def test_no_live_mode_in_any_unit(service_file: Path) -> None:
    """No unit may launch live mode — that requires a human at a TTY."""
    exec_start = _exec_start(service_file)
    for flag in ("--mode", "--trading-mode"):
        assert f"{flag} live" not in exec_start, (
            f"{service_file.name}: ExecStart uses '{flag} live'. Live mode "
            "requires interactive confirmation and must never be run by "
            f"systemd. ExecStart={exec_start!r}"
        )


@pytest.mark.parametrize("service_file", _service_files(), ids=lambda p: p.name)
def test_order_capable_units_require_dry_run(service_file: Path) -> None:
    """An order-capable unit outside testnet MUST pass --dry-run.

    ``--mode paper`` resolves credentials through the non-testnet branch of
    ``LiveTrader.build_node()``, i.e. the mainnet keys.  Without --dry-run
    that configuration sends real orders, and there is no confirmation
    prompt on that path.
    """
    exec_start = _exec_start(service_file)
    if not _is_order_capable(exec_start):
        return
    mode = _flag_value(exec_start, "--mode") or _ARGPARSE_DEFAULT_MODE
    if mode == "testnet":
        return
    assert "--dry-run" in exec_start.split(), (
        f"{service_file.name}: ExecStart runs an order-capable entry point "
        f"with --mode {mode} but without --dry-run — this sends real orders. "
        f"ExecStart={exec_start!r}"
    )


def test_order_capable_set_is_not_empty() -> None:
    """The two linters above must actually apply to something.

    Without this, a renamed script or a broken ExecStart parser would make
    every order-capable check return early and go silently green.
    """
    order_capable = [
        path.name for path in _service_files() if _is_order_capable(_exec_start(path))
    ]
    assert order_capable, (
        "No deploy/*.service was classified as order-capable — the ExecStart "
        f"parser or {_ORDER_CAPABLE_SCRIPTS} is stale, and the dry-run linter "
        "is checking nothing"
    )
    assert "atomicortex-bot.service" in order_capable, (
        "atomicortex-bot.service must be classified as order-capable; got "
        f"{order_capable}"
    )
