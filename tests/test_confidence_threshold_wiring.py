"""Tests for Step H12 — confidence_threshold single source of truth.

Pre-H12: ``Settings.confidence_threshold`` was decorative; the 4H path
used ``LiveTraderConfig.confidence_threshold = 0.65`` regardless of .env.

Post-H12: ``LiveTraderConfig.confidence_threshold`` defaults to ``None``
and ``build_node`` falls through to ``Settings.confidence_threshold``
when no explicit override was passed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import Settings, get_settings
from src.execution.live_trader import LiveTraderConfig


# ---------------------------------------------------------------------------
# Dataclass default
# ---------------------------------------------------------------------------


class TestLiveTraderConfigDefault:
    def test_confidence_threshold_default_is_sentinel(self):
        """Default must be None so build_node knows to read Settings."""
        cfg = LiveTraderConfig()
        assert cfg.confidence_threshold is None

    def test_explicit_value_overrides_sentinel(self):
        cfg = LiveTraderConfig(confidence_threshold=0.80)
        assert cfg.confidence_threshold == 0.80

    def test_zero_explicit_value_is_respected(self):
        """0.0 is a valid (if degenerate) threshold and must not be
        treated as 'no value' — only None triggers the fallback."""
        cfg = LiveTraderConfig(confidence_threshold=0.0)
        assert cfg.confidence_threshold == 0.0


# ---------------------------------------------------------------------------
# Settings source-of-truth resolution
# ---------------------------------------------------------------------------


class TestSettingsResolution:
    """Mirror the exact fallback logic in build_node(): tests the rule
    rather than the Nautilus-coupled method itself."""

    @staticmethod
    def _resolve(cfg: LiveTraderConfig, settings: Settings) -> float:
        return (
            cfg.confidence_threshold
            if cfg.confidence_threshold is not None
            else settings.confidence_threshold
        )

    def test_default_cfg_pulls_from_settings(self):
        cfg = LiveTraderConfig()
        s = Settings(_env_file=None, CONFIDENCE_THRESHOLD=0.72)
        assert self._resolve(cfg, s) == 0.72

    def test_explicit_cfg_wins_over_settings(self):
        cfg = LiveTraderConfig(confidence_threshold=0.80)
        s = Settings(_env_file=None, CONFIDENCE_THRESHOLD=0.50)
        assert self._resolve(cfg, s) == 0.80

    @pytest.mark.parametrize("env_value", [0.50, 0.65, 0.75, 0.90])
    def test_env_value_flows_through_when_unspecified(self, env_value):
        cfg = LiveTraderConfig()
        s = Settings(_env_file=None, CONFIDENCE_THRESHOLD=env_value)
        assert self._resolve(cfg, s) == env_value


# ---------------------------------------------------------------------------
# get_settings end-to-end: changing the env var changes the resolved value
# ---------------------------------------------------------------------------


class TestEnvVarEndToEnd:
    def test_env_changes_propagate(self, monkeypatch, tmp_path):
        # Avoid pulling the repo's real .env.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADING_MODE", "testnet")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.83")

        get_settings.cache_clear()
        try:
            settings = get_settings()
        finally:
            get_settings.cache_clear()

        cfg = LiveTraderConfig()
        resolved = (
            cfg.confidence_threshold
            if cfg.confidence_threshold is not None
            else settings.confidence_threshold
        )
        assert resolved == pytest.approx(0.83)


# ---------------------------------------------------------------------------
# Backward compatibility: existing LiveTraderConfig usages keep working
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_default_config_still_constructs(self):
        cfg = LiveTraderConfig()
        assert cfg.trading_mode == "testnet"
        assert cfg.symbols == ["BTCUSDT-PERP"]
        # H12: default is now None, not 0.65.
        assert cfg.confidence_threshold is None

    def test_custom_kwargs_still_work(self):
        cfg = LiveTraderConfig(
            confidence_threshold=0.72,
            risk_per_trade=0.02,
            max_leverage=20,
        )
        assert cfg.confidence_threshold == 0.72
        assert cfg.risk_per_trade == 0.02
        assert cfg.max_leverage == 20


# ---------------------------------------------------------------------------
# Settings keeps its public API and default
# ---------------------------------------------------------------------------


class TestSettingsFieldUnchanged:
    def test_settings_default_still_0_65(self):
        s = Settings(_env_file=None)
        assert s.confidence_threshold == 0.65

    def test_settings_accepts_env_override(self):
        s = Settings(_env_file=None, CONFIDENCE_THRESHOLD=0.78)
        assert s.confidence_threshold == 0.78
