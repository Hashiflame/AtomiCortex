"""
AtomiCortex — Telegram Bot Application.

Wires together database, handlers, broadcaster, and payment modules
into a single ``python-telegram-bot`` v21 Application.

Phase 7.1 — Payments.
"""

from __future__ import annotations

from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

from src.config import get_settings
from src.logger import get_logger
from src.telegram_bot.broadcaster import Broadcaster
from src.telegram_bot.database import Database
from src.telegram_bot.handlers_free import (
    cmd_help,
    cmd_mystatus,
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
    cmd_payments,
    cmd_restart_bot,
    cmd_revenue,
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
from src.telegram_bot.payments_crypto import CryptoBotPayment
from src.telegram_bot.payments_stars import (
    pre_checkout_handler,
    send_invoice_stars,
    successful_payment_handler,
)
from src.telegram_bot.roles import OWNER_ID

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
        self._crypto_payment: CryptoBotPayment | None = None

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

        # Store DB and prices in bot_data
        self._app.bot_data["db"] = self._db

        settings = get_settings()
        self._app.bot_data["prices"] = {
            "stars_30d": settings.premium_price_stars_30d,
            "stars_90d": settings.premium_price_stars_90d,
            "usdt_30d": settings.premium_price_usdt_30d,
            "usdt_90d": settings.premium_price_usdt_90d,
        }

        # Create broadcaster
        self._broadcaster = Broadcaster(self._app.bot, self._db)

        # Create CryptoBot payment handler (if token configured)
        if settings.cryptobot_token and settings.cryptobot_token != "your_cryptobot_token_here":
            self._crypto_payment = CryptoBotPayment(
                token=settings.cryptobot_token,
                db=self._db,
                bot=self._app.bot,
            )
            self._app.bot_data["crypto_payment"] = self._crypto_payment
            _log.info("CryptoBot payment module initialized")
        else:
            _log.info("CryptoBot token not configured — USDT payments disabled")

        # Register handlers
        self._register_handlers()

        _log.info("TelegramBot application built")
        return self._app

    def _register_handlers(self) -> None:
        """Register all command, callback, and payment handlers."""
        app = self._app
        if app is None:
            return

        # Free commands
        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("help", cmd_help))
        app.add_handler(CommandHandler("stats", cmd_stats))
        app.add_handler(CommandHandler("subscribe", cmd_subscribe))
        app.add_handler(CommandHandler("mystatus", cmd_mystatus))

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
        app.add_handler(CommandHandler("payments", cmd_payments))
        app.add_handler(CommandHandler("revenue", cmd_revenue))

        # Payment handlers — Telegram Stars
        app.add_handler(PreCheckoutQueryHandler(pre_checkout_handler))
        app.add_handler(
            MessageHandler(
                filters.SUCCESSFUL_PAYMENT,
                successful_payment_handler,
            )
        )

        # Callback query handler for inline keyboard buttons
        app.add_handler(CallbackQueryHandler(self._handle_pay_callback))

        _log.info("Registered %d handlers (commands + payments + callbacks)", 25)

    async def _handle_pay_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Route inline keyboard callbacks for payment buttons."""
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        data = query.data or ""
        prices = context.bot_data.get("prices", {})

        if data == "pay_stars_30":
            await send_invoice_stars(
                update, context,
                days=30,
                price_stars=prices.get("stars_30d", 500),
            )

        elif data == "pay_stars_90":
            await send_invoice_stars(
                update, context,
                days=90,
                price_stars=prices.get("stars_90d", 1200),
            )

        elif data == "pay_usdt_30":
            await self._handle_crypto_pay(
                update, context, days=30,
                amount=prices.get("usdt_30d", 7.00),
            )

        elif data == "pay_usdt_90":
            await self._handle_crypto_pay(
                update, context, days=90,
                amount=prices.get("usdt_90d", 18.00),
            )

        elif data == "pay_manual":
            owner_username = None
            if OWNER_ID is not None:
                try:
                    owner_user = self._db.get_user(OWNER_ID)
                    if owner_user:
                        owner_username = owner_user.get("username")
                except Exception:
                    pass
            contact = f"@{owner_username}" if owner_username else "администратору"
            await query.edit_message_text(
                f"✉️ Для ручной оплаты свяжитесь с {contact}.\n\n"
                f"Укажите желаемый срок подписки."
            )

    async def _handle_crypto_pay(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        days: int,
        amount: float,
    ) -> None:
        """Handle CryptoBot USDT payment callback."""
        query = update.callback_query
        crypto: CryptoBotPayment | None = context.bot_data.get("crypto_payment")

        if crypto is None:
            if query:
                await query.edit_message_text(
                    "❌ USDT оплата временно недоступна.\n"
                    "Используйте Telegram Stars или свяжитесь с администратором."
                )
            return

        user_id = update.effective_user.id if update.effective_user else 0
        if user_id == 0:
            return

        pay_url = await crypto.create_invoice(
            user_id=user_id,
            days=days,
            amount_usdt=amount,
        )

        if pay_url and query:
            await query.edit_message_text(
                f"💰 Оплата USDT\n"
                f"{'═' * 30}\n\n"
                f"Сумма: ${amount:.2f} USDT\n"
                f"Срок: {days} дней\n\n"
                f"👉 Перейдите для оплаты:\n{pay_url}\n\n"
                f"После оплаты подписка активируется автоматически."
            )
        elif query:
            await query.edit_message_text(
                "❌ Не удалось создать invoice. Попробуйте позже."
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_polling(self) -> None:
        """Start the bot in polling mode (blocks)."""
        if self._app is None:
            self.build()

        # Start CryptoBot polling as background task via post_init
        if self._crypto_payment is not None:
            crypto = self._crypto_payment

            async def _start_crypto_polling(app: Application) -> None:
                crypto.start_polling(interval=60)
                _log.info("CryptoBot polling started via post_init")

            self._app.post_init = _start_crypto_polling

        _log.info("Starting Telegram bot polling...")
        self._app.run_polling()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def broadcaster(self) -> Broadcaster | None:
        return self._broadcaster

    @property
    def database(self) -> Database:
        return self._db

    @property
    def application(self) -> Application | None:
        return self._app

    @property
    def crypto_payment(self) -> CryptoBotPayment | None:
        return self._crypto_payment
