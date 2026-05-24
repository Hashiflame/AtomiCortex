"""Tests for Step H11 — env-var typo detection + live-mode credential
fail-fast.

The Settings constructor still accepts ``TRADING_MODE=live`` without
credentials so existing tests that synthesize Settings directly keep
working — the credential check is enforced at the ``get_settings()``
boundary instead.
"""
from __future__ import annotations

import io
import sys

import pytest
from loguru import logger

from src.config import Settings, get_settings


@pytest.fixture
def captured_warnings():
    sink = io.StringIO()
    sink_id = logger.add(sink, level="WARNING", format="{level} | {message}")
    try:
        yield sink
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# warn_env_typos
# ---------------------------------------------------------------------------


class TestEnvTypoDetection:
    def test_correct_var_no_warning(self, captured_warnings, monkeypatch):
        # Strip env then set a single valid alias.
        for k in list(__import__("os").environ.keys()):
            if k.startswith(("BINANCE_", "TRADING_", "REDIS_")):
                monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("BINANCE_API_KEY", "valid-value")
        s = Settings(_env_file=None)
        s.warn_env_typos()
        out = captured_warnings.getvalue()
        assert "BINANCE_API_KEY" not in out

    def test_obvious_binance_typo_flagged(self, captured_warnings, monkeypatch):
        monkeypatch.setenv("BINNANCE_API_KEY", "x")
        s = Settings(_env_file=None)
        s.warn_env_typos()
        out = captured_warnings.getvalue()
        assert "BINNANCE_API_KEY" in out
        assert "BINANCE_API_KEY" in out  # suggestion

    def test_trading_mode_typo_flagged(self, captured_warnings, monkeypatch):
        monkeypatch.setenv("TRADING_MMODE", "live")
        s = Settings(_env_file=None)
        s.warn_env_typos()
        out = captured_warnings.getvalue()
        assert "TRADING_MMODE" in out
        assert "TRADING_MODE" in out

    def test_redis_typo_flagged(self, captured_warnings, monkeypatch):
        monkeypatch.setenv("REDIS_HOTS", "localhost")
        s = Settings(_env_file=None)
        s.warn_env_typos()
        out = captured_warnings.getvalue()
        assert "REDIS_HOTS" in out
        assert "REDIS_HOST" in out

    def test_system_env_vars_not_flagged(self, captured_warnings, monkeypatch):
        """Unrelated system vars (PATH, HOME, LANG) must not produce a
        warning — they're nowhere close to any Settings alias."""
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/tmp")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        s = Settings(_env_file=None)
        s.warn_env_typos()
        out = captured_warnings.getvalue()
        assert "PATH" not in out
        assert "HOME" not in out
        assert "LANG" not in out

    def test_fail_soft_on_internal_error(self, monkeypatch):
        """If something inside warn_env_typos blows up, the call must
        not propagate the exception."""
        s = Settings(_env_file=None)
        # Patch the difflib call inside src.config to raise.
        def _boom(*a, **kw):
            raise RuntimeError("synthetic failure")
        monkeypatch.setattr(
            "src.config.difflib.get_close_matches", _boom,
        )
        s.warn_env_typos()  # must not raise


# ---------------------------------------------------------------------------
# assert_live_credentials_present
# ---------------------------------------------------------------------------


class TestLiveCredentialsGuard:
    def test_live_without_keys_raises(self):
        s = Settings(_env_file=None, TRADING_MODE="live")
        with pytest.raises(RuntimeError) as exc:
            s.assert_live_credentials_present()
        assert "BINANCE_API_KEY" in str(exc.value)
        assert "BINANCE_API_SECRET" in str(exc.value)

    def test_live_with_only_key_raises(self):
        s = Settings(
            _env_file=None,
            TRADING_MODE="live",
            BINANCE_API_KEY="real-key",
        )
        with pytest.raises(RuntimeError) as exc:
            s.assert_live_credentials_present()
        assert "BINANCE_API_SECRET" in str(exc.value)
        assert "BINANCE_API_KEY" not in str(exc.value)  # only the missing one

    def test_live_with_keys_passes(self):
        s = Settings(
            _env_file=None,
            TRADING_MODE="live",
            BINANCE_API_KEY="real-key",
            BINANCE_API_SECRET="real-secret",
        )
        s.assert_live_credentials_present()  # no raise

    def test_testnet_without_keys_passes(self):
        s = Settings(_env_file=None, TRADING_MODE="testnet")
        s.assert_live_credentials_present()  # no raise

    def test_paper_without_keys_passes(self):
        s = Settings(_env_file=None, TRADING_MODE="paper")
        s.assert_live_credentials_present()  # no raise


# ---------------------------------------------------------------------------
# get_settings boundary enforces the guard
# ---------------------------------------------------------------------------


class TestGetSettingsIntegration:
    def test_get_settings_blocks_live_without_keys(self, monkeypatch, tmp_path):
        # cd to an empty dir so the repo's real .env doesn't supply keys.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "live")
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
        get_settings.cache_clear()
        with pytest.raises(RuntimeError) as exc:
            get_settings()
        assert "live" in str(exc.value).lower()
        get_settings.cache_clear()

    def test_get_settings_warns_on_typo(
        self, monkeypatch, captured_warnings, tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "testnet")
        monkeypatch.setenv("BINNANCE_API_KEY", "x")
        get_settings.cache_clear()
        try:
            get_settings()
        finally:
            get_settings.cache_clear()
        assert "BINNANCE_API_KEY" in captured_warnings.getvalue()
