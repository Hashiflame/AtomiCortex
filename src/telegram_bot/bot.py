"""
AtomiCortex — Telegram Bot Application.

Wires together database, handlers, and broadcaster into a single
``python-telegram-bot`` v21 Application.

Phase 7 — Telegram Bot.
"""

from __future__ import annotations

from datetime import time, timezone
from pathlib import Path

from telegram.ext import Application, CommandHandler, ContextTypes

from src.logger import get_logger
from src.telegram_bot.broadcaster import Broadcaster
from src.telegram_bot.database import Database
from src.telegram_bot.handlers_free import (
    cmd_help,
    cmd_start,
    cmd_stats,
    cmd_subscribe,
)
from src.telegram_bot.handlers_owner import (
    cmd_ban,
    cmd_broadcast,
    cmd_confirm_stop,
    cmd_grant,
    cmd_health,
    cmd_logs,
    cmd_restart_bot,
    cmd_revoke,
    cmd_stats_admin,
    cmd_stop_bot,
    cmd_user,
    cmd_users,
)
from src.telegram_bot.handlers_premium import (
    cmd_funding,
    cmd_history,
    cmd_regime,
    cmd_risk,
    cmd_signal,
)

_log = get_logger(__name__)


class TelegramBot:
    """Main Telegram bot orchestrator.

    Parameters
    ----------
    token:
        Telegram Bot API token.
    admin_id:
        Telegram user ID of the bot owner.
    db_path:
        Path to the SQLite database file.
    """

    def __init__(
        self,
        token: str,
        admin_id: int,
        db_path: str | Path = "data/telegram_bot.db",
    ) -> None:
        self._token = token
        self._admin_id = admin_id
        self._db = Database(db_path)
        self._app: Application | None = None
        self._broadcaster: Broadcaster | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def build(self) -> Application:
        """Build the PTB Application with all handlers."""
        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        # Store DB in bot_data for handler access
        self._app.bot_data["db"] = self._db

        # Create broadcaster
        self._broadcaster = Broadcaster(self._app.bot, self._db)

        # Register command handlers
        self._register_handlers()

        _log.info("TelegramBot application built")
        return self._app

    def _register_handlers(self) -> None:
        """Register all command handlers."""
        app = self._app
        if app is None:
            return

        # Free commands
        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("help", cmd_help))
        app.add_handler(CommandHandler("stats", cmd_stats))
        app.add_handler(CommandHandler("subscribe", cmd_subscribe))

        # Premium commands
        app.add_handler(CommandHandler("signal", cmd_signal))
        app.add_handler(CommandHandler("history", cmd_history))
        app.add_handler(CommandHandler("regime", cmd_regime))
        app.add_handler(CommandHandler("funding", cmd_funding))
        app.add_handler(CommandHandler("risk", cmd_risk))

        # Owner commands
        app.add_handler(CommandHandler("users", cmd_users))
        app.add_handler(CommandHandler("user", cmd_user))
        app.add_handler(CommandHandler("grant", cmd_grant))
        app.add_handler(CommandHandler("revoke", cmd_revoke))
        app.add_handler(CommandHandler("ban", cmd_ban))
        app.add_handler(CommandHandler("broadcast", cmd_broadcast))
        app.add_handler(CommandHandler("health", cmd_health))
        app.add_handler(CommandHandler("stop_bot", cmd_stop_bot))
        app.add_handler(CommandHandler("confirm_stop", cmd_confirm_stop))
        app.add_handler(CommandHandler("restart_bot", cmd_restart_bot))
        app.add_handler(CommandHandler("logs", cmd_logs))
        app.add_handler(CommandHandler("stats_admin", cmd_stats_admin))

        _log.info("Registered %d command handlers", 21)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_polling(self) -> None:
        """Start the bot in polling mode (blocks)."""
        if self._app is None:
            self.build()
        _log.info("Starting Telegram bot polling...")
        self._app.run_polling()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def broadcaster(self) -> Broadcaster | None:
        """Return the Broadcaster instance for external use."""
        return self._broadcaster

    @property
    def database(self) -> Database:
        """Return the Database instance."""
        return self._db

    @property
    def application(self) -> Application | None:
        """Return the PTB Application instance."""
        return self._app
