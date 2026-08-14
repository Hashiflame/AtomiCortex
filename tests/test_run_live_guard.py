"""Tests — fail-closed guard in scripts/run_live.py.

The guard refuses configurations that would place real orders without an
explicit --dry-run, and it must fire BEFORE the interactive confirmation
prompt, because systemd runs the unit with StandardInput=null where
input() raises EOFError instead of protecting anything.

Nothing here builds a TradingNode, touches the network or reads the real
.env: LiveTrader, get_settings, get_logger, setup_logging, sys.stdin and
builtins.input are all replaced.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

_BOT_UNIT = (
    Path(__file__).resolve().parent.parent / "deploy" / "atomicortex-bot.service"
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _RecordingLog:
    """Stand-in for the loguru logger main() obtains via get_logger()."""

    _LEVELS = ("debug", "info", "warning", "error", "critical", "success")

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        if name not in type(self)._LEVELS:
            raise AttributeError(name)

        def _emit(message: object = "", *args: object, **kwargs: object) -> None:
            self.records.append((name, str(message)))

        return _emit

    def messages(self, level: str) -> list[str]:
        return [message for lvl, message in self.records if lvl == level]


class _FakeStdin:
    """Minimal stdin exposing only what the guard asks for."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _FakeSettings:
    """Only the attribute the divergence check reads."""

    def __init__(self, trading_mode: str) -> None:
        self.trading_mode = trading_mode


@dataclass
class _Outcome:
    log: _RecordingLog
    configs: list[Any] = field(default_factory=list)
    input_calls: list[str] = field(default_factory=list)
    system_exit: SystemExit | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_live_module() -> Any:
    return importlib.import_module("scripts.run_live")


def _exit_config_error() -> int:
    """The guard's exit code, read from its single source of truth.

    Deliberately NOT a literal here: the number must live in exactly one
    place in Python (scripts/run_live.py) and one place in configuration
    (RestartPreventExitStatus= in the unit), and
    test_guard_exit_code_matches_unit_restart_prevention pins those two
    together.
    """
    code = getattr(_run_live_module(), "_EXIT_CONFIG_ERROR", None)
    assert code is not None, (
        "scripts/run_live.py must define _EXIT_CONFIG_ERROR (EX_CONFIG from "
        "sysexits.h) so the guard's exit code and the unit's "
        "RestartPreventExitStatus= cannot drift apart"
    )
    return int(code)


@pytest.fixture
def run_main(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run run_live.main() with every side effect replaced by a double."""

    def _run(
        argv: list[str],
        *,
        env_trading_mode: str = "paper",
        isatty: bool = True,
        input_answer: str = "YES",
    ) -> _Outcome:
        run_live = _run_live_module()
        outcome = _Outcome(log=_RecordingLog())

        class _SpyTrader:
            def __init__(self, config: Any) -> None:
                outcome.configs.append(config)
                self.startup_failed = False

            def run(self) -> None:
                return None

            def stop(self) -> None:
                return None

        def _fake_input(prompt: object = "") -> str:
            outcome.input_calls.append(str(prompt))
            return input_answer

        monkeypatch.setattr(sys, "argv", ["run_live.py", *argv])
        monkeypatch.setattr(sys, "stdin", _FakeStdin(isatty))
        monkeypatch.setattr(run_live, "setup_logging", lambda **kwargs: None)
        monkeypatch.setattr(run_live, "get_logger", lambda name: outcome.log)
        monkeypatch.setattr(
            run_live, "get_settings", lambda: _FakeSettings(env_trading_mode)
        )
        monkeypatch.setattr(run_live, "LiveTrader", _SpyTrader)
        # Leave the process' real SIGINT/SIGTERM handlers alone.
        monkeypatch.setattr(run_live.signal, "signal", lambda *a, **k: None)
        monkeypatch.setattr(builtins, "input", _fake_input)

        try:
            run_live.main()
        except SystemExit as exc:
            outcome.system_exit = exc
        return outcome

    return _run


# ---------------------------------------------------------------------------
# Rule A — paper without --dry-run is never allowed
# ---------------------------------------------------------------------------


def test_guard_refuses_paper_without_dry_run(run_main: Any) -> None:
    """--mode paper resolves to MAINNET keys; without --dry-run it trades."""
    outcome = run_main(["--mode", "paper"])

    assert outcome.system_exit is not None, (
        "--mode paper without --dry-run must be refused, but main() ran to "
        "completion"
    )
    assert outcome.system_exit.code == _exit_config_error()
    assert outcome.configs == [], (
        "LiveTrader must not be constructed after a refusal; got "
        f"{len(outcome.configs)} instance(s)"
    )
    assert outcome.log.messages("critical"), (
        "the refusal must be logged at CRITICAL — it is the only diagnostic "
        "the operator gets from journalctl"
    )


def test_guard_allows_paper_with_dry_run(run_main: Any) -> None:
    """The configuration PR-D actually ships must pass untouched."""
    outcome = run_main(["--mode", "paper", "--dry-run"])

    assert outcome.system_exit is None, (
        "--mode paper --dry-run is the intended production configuration and "
        "must not be refused"
    )
    assert len(outcome.configs) == 1
    config = outcome.configs[0]
    assert config.trading_mode == "paper"
    assert config.dry_run is True


# ---------------------------------------------------------------------------
# Rule B — live without --dry-run needs a TTY
# ---------------------------------------------------------------------------


def test_guard_refuses_live_without_dry_run_when_stdin_not_a_tty(
    run_main: Any,
) -> None:
    """Under systemd stdin is /dev/null: input() would raise EOFError."""
    outcome = run_main(["--mode", "live"], env_trading_mode="live", isatty=False)

    assert outcome.system_exit is not None, (
        "--mode live without --dry-run and without a TTY must be refused"
    )
    assert outcome.system_exit.code == _exit_config_error()
    assert outcome.input_calls == [], (
        "the guard must fire before the confirmation prompt; input() was "
        f"reached with {outcome.input_calls}"
    )
    assert outcome.configs == []


def test_live_without_dry_run_still_prompts_on_tty(run_main: Any) -> None:
    """A human at a terminal keeps the existing 'type YES' confirmation."""
    outcome = run_main(
        ["--mode", "live"],
        env_trading_mode="live",
        isatty=True,
        input_answer="YES",
    )

    assert outcome.system_exit is None, (
        "interactive live must stay available — the guard is scoped to "
        "non-interactive stdin"
    )
    assert len(outcome.input_calls) == 1
    assert len(outcome.configs) == 1


# ---------------------------------------------------------------------------
# Boundary — testnet is fake funds and stays unrestricted
# ---------------------------------------------------------------------------


def test_guard_allows_testnet_without_dry_run(run_main: Any) -> None:
    """testnet cannot reach mainnet credentials, so it needs no --dry-run."""
    outcome = run_main(["--mode", "testnet"], env_trading_mode="testnet")

    assert outcome.system_exit is None, (
        "--mode testnet without --dry-run is documented usage and must not "
        "be refused"
    )
    assert len(outcome.configs) == 1
    assert outcome.configs[0].trading_mode == "testnet"


# ---------------------------------------------------------------------------
# Diagnostics — .env / CLI divergence warns, never refuses
# ---------------------------------------------------------------------------


def test_guard_warns_on_env_cli_mode_divergence(run_main: Any) -> None:
    """settings.trading_mode does not route orders, so it cannot be fatal.

    It only drives the startup banner and the log filename, so a mismatch
    is a legibility problem: warn loudly, keep running.
    """
    outcome = run_main(["--mode", "testnet", "--dry-run"], env_trading_mode="paper")

    assert outcome.system_exit is None, (
        "a TRADING_MODE/--mode mismatch must not kill a correctly configured "
        "unit — .env does not route orders"
    )
    warnings = outcome.log.messages("warning")
    assert any("paper" in msg and "testnet" in msg for msg in warnings), (
        "the divergence between TRADING_MODE=paper and --mode testnet must be "
        f"logged at WARNING with both values; warnings were {warnings}"
    )


# ---------------------------------------------------------------------------
# Code / unit-file coupling
# ---------------------------------------------------------------------------


def test_guard_exit_code_matches_unit_restart_prevention() -> None:
    """The unit must declare the guard's exit code as non-restartable.

    Without RestartPreventExitStatus= the refusal burns all five restarts
    allowed by StartLimitBurst before the unit settles into 'failed', and
    the exhausted counter then blocks the first honest start after the fix.
    """
    declared: set[str] = set()
    for line in _BOT_UNIT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("RestartPreventExitStatus="):
            declared.update(stripped.split("=", 1)[1].split())

    assert declared, (
        f"{_BOT_UNIT.name} must declare RestartPreventExitStatus= so the "
        "fail-closed guard fails once instead of restart-looping"
    )
    assert str(_exit_config_error()) in declared, (
        f"{_BOT_UNIT.name} declares RestartPreventExitStatus={sorted(declared)} "
        f"but the guard exits with {_exit_config_error()} — a restart loop "
        "would survive the guard"
    )
