"""
AtomiCortex — centralized configuration module.

Reads all settings from a .env file via pydantic-settings.
Use get_settings() everywhere in the application — never instantiate
Settings directly, so the singleton + lru_cache stays intact.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    trading_mode: str = Field(default="testnet", alias="TRADING_MODE")
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
    }

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
    """Return the cached singleton Settings instance."""
    return Settings()


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    print("\n=== AtomiCortex Configuration ===\n")
    for key, value in settings.safe_dict().items():
        print(f"  {key:<30} = {value}")
    print()
