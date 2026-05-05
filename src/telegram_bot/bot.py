"""
AtomiCortex — Telegram Bot Application.

Wires together database, handlers, broadcaster, and payment modules
into a single ``python-telegram-bot`` v21 Application.

Phase 7.2 — UX menus.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import psutil
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
    USERS_PER_PAGE,
    _build_health_message,
    _build_stats_admin_message,
    _send_users_page,
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
from src.telegram_bot.keyboards import (
    ALL_BUTTON_TEXTS,
    BTN_FUNDING,
    BTN_HEALTH,
    BTN_HELP,
    BTN_HISTORY,
    BTN_REGIME,
    BTN_SIGNAL,
    BTN_STATS,
    BTN_SUBSCRIBE,
    BTN_USERS,
    OWNER_BUTTONS,
    PREMIUM_BUTTONS,
    get_health_buttons,
    get_subscribe_inline_button,
    get_stats_admin_buttons,
    get_users_pagination,
)
from src.telegram_bot.payments_crypto import CryptoBotPayment
from src.telegram_bot.payments_stars import (
    pre_checkout_handler,
    send_invoice_stars,
    successful_payment_handler,
)
from src.telegram_bot.roles import OWNER_ID, _ensure_user

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
        self._signal_poller: Any = None  # SignalPoller (lazy init)

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

        # ReplyKeyboard button handler (text messages matching button labels)
        button_filter = filters.TEXT & filters.Regex(
            "|".join(re_escape(btn) for btn in ALL_BUTTON_TEXTS)
        )
        app.add_handler(MessageHandler(button_filter, self._handle_button_press))

        # Callback query handler for inline keyboard buttons
        app.add_handler(CallbackQueryHandler(self._handle_callback))

        _log.info("Registered handlers (commands + payments + buttons + callbacks)")

    # ------------------------------------------------------------------
    # ReplyKeyboard button router
    # ------------------------------------------------------------------

    async def _handle_button_press(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Route ReplyKeyboard button presses to command handlers."""
        if update.message is None or update.effective_user is None:
            return

        text = update.message.text or ""
        db: Database = context.bot_data["db"]
        user = _ensure_user(db, update)
        role = user.get("role", "free") if user else "free"

        _log.debug(
            "Button press | user={uid} role={r} button={btn}",
            uid=update.effective_user.id,
            r=role,
            btn=text,
        )

        # Check access for premium-only buttons
        if text in PREMIUM_BUTTONS and role not in ("premium", "owner"):
            await update.effective_chat.send_message(
                "🔒 Эта функция только для Premium\n"
                "Нажми ⭐ Подписка чтобы получить доступ",
                reply_markup=get_subscribe_inline_button(),
            )
            return

        # Check access for owner-only buttons
        if text in OWNER_BUTTONS and role != "owner":
            await update.effective_chat.send_message(
                "🔒 Эта функция только для владельца."
            )
            return

        # Route to the corresponding handler
        handler_map = {
            BTN_SIGNAL: cmd_signal,
            BTN_HISTORY: cmd_history,
            BTN_REGIME: cmd_regime,
            BTN_FUNDING: cmd_funding,
            BTN_STATS: cmd_stats,
            BTN_SUBSCRIBE: cmd_subscribe,
            BTN_HELP: cmd_help,
            BTN_USERS: cmd_users,
            BTN_HEALTH: cmd_health,
        }

        handler = handler_map.get(text)
        if handler:
            await handler(update, context)

    # ------------------------------------------------------------------
    # Inline callback router
    # ------------------------------------------------------------------

    async def _handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Route inline keyboard callbacks."""
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        data = query.data or ""
        prices = context.bot_data.get("prices", {})

        # ── Payment callbacks (legacy pay_ prefix) ──
        if data == "pay_stars_30" or data == "buy_stars_30":
            await send_invoice_stars(
                update, context,
                days=30,
                price_stars=prices.get("stars_30d", 500),
            )

        elif data == "pay_stars_90" or data == "buy_stars_90":
            await send_invoice_stars(
                update, context,
                days=90,
                price_stars=prices.get("stars_90d", 1200),
            )

        elif data == "pay_usdt_30" or data == "buy_usdt_30":
            await self._handle_crypto_pay(
                update, context, days=30,
                amount=prices.get("usdt_30d", 7.00),
            )

        elif data == "pay_usdt_90" or data == "buy_usdt_90":
            await self._handle_crypto_pay(
                update, context, days=90,
                amount=prices.get("usdt_90d", 18.00),
            )

        elif data in ("pay_manual", "contact_owner"):
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

        elif data == "show_subscribe":
            # Redirect to subscribe view
            await cmd_subscribe(update, context)

        # ── Health callbacks ──
        elif data == "health_refresh":
            await self._refresh_health(query, context)

        elif data == "health_logs_20":
            await self._send_logs_inline(query, context, n=20)

        elif data == "health_restart":
            await self._restart_bot_inline(query, context)

        # ── Users pagination ──
        elif data.startswith("users_page_"):
            page = int(data.split("_")[-1])
            await self._paginate_users(query, context, page)

        elif data == "users_noop":
            pass  # no-op for the page indicator

        # ── Stats admin period ──
        elif data.startswith("stats_period_"):
            # For now, same as full stats (period filtering can be extended)
            db: Database = context.bot_data["db"]
            msg = _build_stats_admin_message(db)
            await query.edit_message_text(
                msg, reply_markup=get_stats_admin_buttons(),
            )

    # ------------------------------------------------------------------
    # Inline action helpers
    # ------------------------------------------------------------------

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

    async def _refresh_health(self, query, context) -> None:
        """Refresh /health inline."""
        from datetime import datetime, timezone

        db: Database = context.bot_data["db"]
        cpu = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count() or 0
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime = datetime.now(timezone.utc) - boot
        up_days = uptime.days
        up_hours = uptime.seconds // 3600
        up_mins = (uptime.seconds % 3600) // 60

        stats = db.get_stats()
        signals_today = db.get_signals_today_count()
        signals = db.get_signals_history(limit=1)
        regime = signals[0].get("regime", "N/A").upper() if signals else "N/A"
        open_signals = db.get_open_signals()

        msg = _build_health_message(
            cpu, cpu_count, mem, disk, up_days, up_hours, up_mins,
            signals_today, regime, stats, open_signals,
        )

        try:
            await query.edit_message_text(
                msg, reply_markup=get_health_buttons(),
            )
        except Exception:
            pass  # message unchanged

    async def _send_logs_inline(self, query, context, n: int = 20) -> None:
        """Send last N log lines as a follow-up message."""
        import asyncio
        from pathlib import Path

        logs_dir = Path("./logs")
        log_files = sorted(logs_dir.glob("trading_*.log"), reverse=True)

        if not log_files:
            await query.message.reply_text("📭 Логи не найдены.")
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "tail", f"-n{n}", str(log_files[0]),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            output = stdout.decode("utf-8", errors="replace") if stdout else "Пусто."
            if len(output) > 4000:
                output = output[-4000:]
            await query.message.reply_text(f"📋 Последние {n} строк:\n\n{output}")
        except Exception as exc:
            await query.message.reply_text(f"❌ Ошибка чтения логов: {exc}")

    async def _restart_bot_inline(self, query, context) -> None:
        """Restart trading bot via inline button."""
        import asyncio

        await query.edit_message_text("🔄 Перезапускаю торгового бота...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "--user", "restart", "atomicortex-bot",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode == 0:
                await query.message.reply_text("✅ Бот перезапущен.")
            else:
                await query.message.reply_text(
                    f"❌ Ошибка перезапуска (код {proc.returncode})."
                )
        except asyncio.TimeoutError:
            await query.message.reply_text("❌ Таймаут перезапуска (15с).")
        except Exception as exc:
            await query.message.reply_text(f"❌ Ошибка перезапуска: {exc}")

    async def _paginate_users(self, query, context, page: int) -> None:
        """Handle user list pagination callback."""
        db: Database = context.bot_data["db"]
        users = db.get_all_users()
        total_pages = max(1, math.ceil(len(users) / USERS_PER_PAGE))
        page = max(1, min(page, total_pages))
        start = (page - 1) * USERS_PER_PAGE
        end = start + USERS_PER_PAGE
        page_users = users[start:end]

        lines = [f"👥 Пользователи ({len(users)} всего):\n", "ID | Username | Role | Joined"]
        lines.append("─" * 45)
        for u in page_users:
            un = f"@{u['username']}" if u.get("username") else "—"
            joined = (u.get("joined_at") or "")[:10]
            banned = " 🚫" if u.get("is_banned") else ""
            lines.append(f"{u['user_id']} | {un} | {u['role']}{banned} | {joined}")

        pagination = get_users_pagination(page, total_pages) if total_pages > 1 else None
        try:
            await query.edit_message_text(
                "\n".join(lines), reply_markup=pagination,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_polling(self) -> None:
        """Start the bot in polling mode (blocks)."""
        if self._app is None:
            self.build()

        # Capture references for the async post_init closure
        crypto = self._crypto_payment
        broadcaster = self._broadcaster
        shared_db_path = self._get_shared_db_path()
        bot_ref = self  # keep reference for _signal_poller assignment

        async def _post_init(app: Application) -> None:
            # CryptoBot polling
            if crypto is not None:
                crypto.start_polling(interval=60)
                _log.info("CryptoBot polling started")

            # Signal poller
            if broadcaster is not None:
                from src.telegram_bot.signal_poller import SignalPoller
                poller = SignalPoller(
                    db_path=shared_db_path,
                    broadcaster=broadcaster,
                    poll_interval=30,
                )
                bot_ref._signal_poller = poller
                await poller.start()
                _log.info(
                    "SignalPoller started | db={p}",
                    p=shared_db_path,
                )

        self._app.post_init = _post_init

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

    @property
    def signal_poller(self) -> Any:
        return self._signal_poller

    # ------------------------------------------------------------------
    # Shared DB path discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _get_shared_db_path() -> str:
        """Find the shared atomicortex.db used by the trading bot."""
        candidates = [
            Path("data/atomicortex.db"),
            Path("/home/hashiflame/AtomiCortex/data/atomicortex.db"),
        ]
        try:
            settings = get_settings()
            candidates.insert(0, settings.data_dir / "atomicortex.db")
        except Exception:
            pass

        for p in candidates:
            if p.exists():
                return str(p)
        # Default (will be created by SignalBridge on the trading side)
        return str(candidates[0])


# ── Utility ──

import re as _re


def re_escape(s: str) -> str:
    """Escape a string for use in a regex alternation."""
    return _re.escape(s)
