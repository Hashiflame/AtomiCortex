"""
AtomiCortex — centralized configuration module.

Reads all settings from a .env file via pydantic-settings.
Use get_settings() everywhere in the application — never instantiate
Settings directly, so the singleton + lru_cache stays intact.
"""

from __future__ import annotations

import difflib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TradingMode = Literal["testnet", "paper", "live"]
_ALLOWED_TRADING_MODES: tuple[str, ...] = ("testnet", "paper", "live")


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Binance
    # ------------------------------------------------------------------
    binance_testnet_api_key: str = Field(default="", alias="BINANCE_TESTNET_API_KEY")
    binance_testnet_api_secret: str = Field(default="", alias="BINANCE_TESTNET_API_SECRET")
    binance_mainnet_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_mainnet_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")

    # ------------------------------------------------------------------
    # Bybit
    # ------------------------------------------------------------------
    bybit_testnet_api_key: str = Field(default="", alias="BYBIT_TESTNET_API_KEY")
    bybit_testnet_api_secret: str = Field(default="", alias="BYBIT_TESTNET_API_SECRET")
    bybit_mainnet_api_key: str = Field(default="", alias="BYBIT_API_KEY")
    bybit_mainnet_api_secret: str = Field(default="", alias="BYBIT_API_SECRET")

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_admin_id: str = Field(default="", alias="TELEGRAM_ADMIN_ID")

    # ------------------------------------------------------------------
    # Payments — Telegram Stars
    # ------------------------------------------------------------------
    premium_price_stars_30d: int = Field(default=500, alias="PREMIUM_PRICE_STARS_30D")
    premium_price_stars_90d: int = Field(default=1200, alias="PREMIUM_PRICE_STARS_90D")

    # ------------------------------------------------------------------
    # Payments — CryptoBot (USDT)
    # ------------------------------------------------------------------
    cryptobot_token: str = Field(default="", alias="CRYPTOBOT_TOKEN")
    premium_price_usdt_30d: float = Field(default=7.00, alias="PREMIUM_PRICE_USDT_30D")
    premium_price_usdt_90d: float = Field(default=18.00, alias="PREMIUM_PRICE_USDT_90D")

    # ------------------------------------------------------------------
    # QuestDB
    # ------------------------------------------------------------------
    questdb_host: str = Field(default="localhost", alias="QUESTDB_HOST")
    questdb_port: int = Field(default=9009, alias="QUESTDB_PORT")
    questdb_http_port: int = Field(default=9000, alias="QUESTDB_HTTP_PORT")

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # ------------------------------------------------------------------
    # Trading parameters
    # ------------------------------------------------------------------
    trading_mode: TradingMode = Field(default="testnet", alias="TRADING_MODE")
    initial_capital: float = Field(default=10_000.0, alias="INITIAL_CAPITAL")
    max_leverage: int = Field(default=10, alias="MAX_LEVERAGE")
    risk_per_trade: float = Field(default=0.01, alias="RISK_PER_TRADE")
    confidence_threshold: float = Field(default=0.65, alias="CONFIDENCE_THRESHOLD")
    max_open_positions: int = Field(default=3, alias="MAX_OPEN_POSITIONS")
    daily_loss_limit: float = Field(default=-0.03, alias="DAILY_LOSS_LIMIT")
    weekly_loss_limit: float = Field(default=-0.08, alias="WEEKLY_LOSS_LIMIT")
    max_drawdown_kill: float = Field(default=-0.15, alias="MAX_DRAWDOWN_KILL")

    # ------------------------------------------------------------------
    # Paths (raw strings — converted to Path objects via validator)
    # ------------------------------------------------------------------
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    logs_dir: Path = Field(default=Path("./logs"), alias="LOGS_DIR")

    # ------------------------------------------------------------------
    # REST API (src/api/main.py) — internal service consumed by website
    # ------------------------------------------------------------------
    atomicortex_api_key: str = Field(default="", alias="ATOMICORTEX_API_KEY")
    api_cors_origins: str = Field(
        default="http://localhost,http://127.0.0.1",
        alias="API_CORS_ORIGINS",
    )
    api_rate_limit_per_minute: int = Field(
        default=60, alias="API_RATE_LIMIT_PER_MINUTE",
    )

    # ------------------------------------------------------------------
    # Symbols — stored as a raw comma-separated string so pydantic-settings
    # does not attempt JSON-parsing; exposed as list[str] via @property.
    # ------------------------------------------------------------------
    symbols_raw: str = Field(
        default="BTC-USDT-PERP,ETH-USDT-PERP,SOL-USDT-PERP",
        alias="SYMBOLS",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _validate_trading_mode(cls, v: Any) -> str:
        """Strictly validate TRADING_MODE — fail-fast on any typo, whitespace, or wrong case.

        Reject silently-normalizable inputs (e.g. " Testnet ") so a typo in .env
        cannot quietly route the bot to mainnet with real funds.
        """
        if not isinstance(v, str):
            raise ValueError(
                f"TRADING_MODE must be a string, got {type(v).__name__}: {v!r}"
            )
        if v.strip() != v:
            raise ValueError(
                f"TRADING_MODE contains leading/trailing whitespace: {v!r}. "
                f"Did you mean {v.strip()!r}?"
            )
        if v != v.lower():
            raise ValueError(
                f"TRADING_MODE must be lowercase, got {v!r}. "
                f"Did you mean {v.lower()!r}?"
            )
        if v not in _ALLOWED_TRADING_MODES:
            hint = difflib.get_close_matches(v, _ALLOWED_TRADING_MODES, n=1)
            suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
            raise ValueError(
                f"TRADING_MODE must be one of {_ALLOWED_TRADING_MODES}, "
                f"got {v!r}.{suggestion}"
            )
        return v

    @field_validator("data_dir", "logs_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: Any) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def create_directories(self) -> "Settings":
        """Ensure data_dir and logs_dir exist on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        """Return the trading symbols as a list, parsed from the raw CSV string."""
        return [s.strip() for s in self.symbols_raw.split(",") if s.strip()]

    @property
    def is_testnet(self) -> bool:
        return self.trading_mode.lower() == "testnet"

    @property
    def is_live(self) -> bool:
        return self.trading_mode.lower() == "live"

    @property
    def binance_api_key(self) -> str:
        """Return the appropriate Binance API key for the current mode."""
        return self.binance_testnet_api_key if self.is_testnet else self.binance_mainnet_api_key

    @property
    def binance_api_secret(self) -> str:
        """Return the appropriate Binance API secret for the current mode."""
        return self.binance_testnet_api_secret if self.is_testnet else self.binance_mainnet_api_secret

    @property
    def bybit_api_key(self) -> str:
        """Return the appropriate Bybit API key for the current mode."""
        return self.bybit_testnet_api_key if self.is_testnet else self.bybit_mainnet_api_key

    @property
    def bybit_api_secret(self) -> str:
        """Return the appropriate Bybit API secret for the current mode."""
        return self.bybit_testnet_api_secret if self.is_testnet else self.bybit_mainnet_api_secret

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _SECRET_FIELDS = {
        "binance_testnet_api_key",
        "binance_testnet_api_secret",
        "binance_mainnet_api_key",
        "binance_mainnet_api_secret",
        "bybit_testnet_api_key",
        "bybit_testnet_api_secret",
        "bybit_mainnet_api_key",
        "bybit_mainnet_api_secret",
        "telegram_bot_token",
        "redis_password",
        "cryptobot_token",
    }

    def log_startup_banner(self) -> None:
        """Emit a highly-visible banner announcing the current trading mode.

        Called once per process from ``get_settings()`` via the lru_cache,
        so the operator cannot miss the active mode — especially ``live``.
        """
        from loguru import logger

        mode = self.trading_mode
        if mode == "live":
            bar = "█" * 72
            banner = (
                "\n" + bar + "\n"
                + "██" + " " * 68 + "██" + "\n"
                + "██" + "⚠️  LIVE TRADING MODE — REAL MONEY AT RISK  ⚠️".center(68) + "██" + "\n"
                + "██" + "ORDERS WILL HIT MAINNET EXCHANGES".center(68) + "██" + "\n"
                + "██" + " " * 68 + "██" + "\n"
                + bar
            )
            logger.warning(banner)
        elif mode == "paper":
            bar = "═" * 60
            logger.info(
                "\n" + bar + "\n"
                + "   📝 PAPER TRADING MODE — simulated, no real orders\n"
                + bar
            )
        else:  # testnet
            bar = "═" * 60
            logger.info(
                "\n" + bar + "\n"
                + "   🧪 TESTNET MODE — Binance/Bybit testnet, fake funds\n"
                + bar
            )

    def safe_dict(self) -> dict[str, Any]:
        """Return settings as a dict with secrets masked."""
        result: dict[str, Any] = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if field_name in self._SECRET_FIELDS:
                result[field_name] = "***" if value else "(not set)"
            else:
                result[field_name] = value
        # Include computed properties
        result["is_testnet"] = self.is_testnet
        result["is_live"] = self.is_live
        result["symbols"] = self.symbols
        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    Emits the trading-mode startup banner on first call so the operator
    sees the active mode exactly once per process.
    """
    settings = Settings()
    settings.log_startup_banner()
    return settings


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    print("\n=== AtomiCortex Configuration ===\n")
    for key, value in settings.safe_dict().items():
        print(f"  {key:<30} = {value}")
    print()
