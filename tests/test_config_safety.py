"""Safety tests for `TRADING_MODE` validation and startup banner.

Guards against the class of bugs where a typo (e.g. ``TRADING_MODE=tesnet``)
or a stray space (``" testnet"``) silently routes the bot to mainnet.
"""
from __future__ import annotations

import io

import pytest
from loguru import logger
from pydantic import ValidationError

from src.config import Settings, _ALLOWED_TRADING_MODES, get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(mode: str) -> Settings:
    """Build a Settings instance from an explicit mode, bypassing .env."""
    return Settings(_env_file=None, TRADING_MODE=mode)


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["tesnet", "livee", "mainnet", "papper", "demo", ""])
def test_typo_raises(bad: str) -> None:
    with pytest.raises(ValidationError) as exc:
        _make(bad)
    assert "TRADING_MODE" in str(exc.value)


def test_typo_includes_suggestion() -> None:
    with pytest.raises(ValidationError) as exc:
        _make("tesnet")
    assert "testnet" in str(exc.value)


@pytest.mark.parametrize("bad", [" testnet", "testnet ", "  live", "paper\t"])
def test_whitespace_raises(bad: str) -> None:
    with pytest.raises(ValidationError) as exc:
        _make(bad)
    assert "whitespace" in str(exc.value).lower()


@pytest.mark.parametrize("bad", ["Testnet", "LIVE", "Paper", "TESTNET"])
def test_wrong_case_raises(bad: str) -> None:
    with pytest.raises(ValidationError) as exc:
        _make(bad)
    assert "lowercase" in str(exc.value).lower()


def test_non_string_raises() -> None:
    with pytest.raises(ValidationError):
        _make(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acceptance cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("good", list(_ALLOWED_TRADING_MODES))
def test_valid_values_accepted(good: str) -> None:
    cfg = _make(good)
    assert cfg.trading_mode == good


def test_is_testnet_and_is_live_consistent() -> None:
    assert _make("testnet").is_testnet is True
    assert _make("testnet").is_live is False
    assert _make("live").is_live is True
    assert _make("live").is_testnet is False
    assert _make("paper").is_testnet is False
    assert _make("paper").is_live is False


def test_lower_comparisons_still_work() -> None:
    """Existing call sites use ``config.trading_mode.lower() == "..."``."""
    cfg = _make("live")
    assert cfg.trading_mode.lower() == "live"


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_logs():
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{level}|{message}")
    try:
        yield sink
    finally:
        logger.remove(handler_id)


def test_banner_live_is_loud(captured_logs: io.StringIO) -> None:
    _make("live").log_startup_banner()
    out = captured_logs.getvalue()
    assert "LIVE TRADING MODE" in out
    assert "REAL MONEY" in out
    assert "WARNING" in out  # logged at WARNING level for visibility
    assert "█" in out


def test_banner_paper(captured_logs: io.StringIO) -> None:
    _make("paper").log_startup_banner()
    out = captured_logs.getvalue()
    assert "PAPER TRADING MODE" in out


def test_banner_testnet(captured_logs: io.StringIO) -> None:
    _make("testnet").log_startup_banner()
    out = captured_logs.getvalue()
    assert "TESTNET MODE" in out


def test_get_settings_emits_banner_once(captured_logs: io.StringIO, monkeypatch) -> None:
    """get_settings() is lru_cached → banner should fire exactly once per process."""
    monkeypatch.setenv("TRADING_MODE", "testnet")
    get_settings.cache_clear()
    try:
        get_settings()
        get_settings()
        get_settings()
        out = captured_logs.getvalue()
        assert out.count("TESTNET MODE") == 1
    finally:
        get_settings.cache_clear()
