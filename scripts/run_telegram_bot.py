#!/usr/bin/env python3
"""
AtomiCortex — Telegram Bot entry point.

Usage:
    python scripts/run_telegram_bot.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.logger import setup_logging, get_logger
from src.telegram_bot.bot import TelegramBot
from src.telegram_bot.roles import OWNER_ID

_log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logging(
        logs_dir=settings.logs_dir,
        trading_mode=settings.trading_mode,
    )

    token = settings.telegram_bot_token
    admin_id_str = settings.telegram_admin_id

    if not token or token == "your_bot_token_here":
        _log.error("TELEGRAM_BOT_TOKEN not configured in .env")
        sys.exit(1)

    # TG-001: strict OWNER_ID validation at startup
    if OWNER_ID is None:
        _log.error(
            "TELEGRAM_ADMIN_ID не установлен или невалиден в .env! "
            "Бот не может работать без owner."
        )
        sys.exit(1)

    try:
        admin_id = int(admin_id_str)
    except (ValueError, TypeError):
        _log.error("TELEGRAM_ADMIN_ID must be a valid integer")
        sys.exit(1)

    db_path = settings.data_dir / "telegram_bot.db"
    bot = TelegramBot(token=token, admin_id=admin_id, db_path=db_path)

    _log.info("Starting AtomiCortex Telegram Bot...")
    bot.run_polling()


if __name__ == "__main__":
    main()
