"""AtomiCortex — Free-tier Telegram command handlers."""

from __future__ import annotations

from datetime import datetime, timezone

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from src.telegram_bot.database import Database
from src.telegram_bot.keyboards import (
    get_keyboard_for_role,
    get_renew_button,
    get_subscribe_keyboard,
)
from src.telegram_bot.roles import require_role, _ensure_user


@require_role("free")
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome + auto-register + show role-based ReplyKeyboard."""
    if update.effective_user is None or update.effective_chat is None:
        return

    db: Database = context.bot_data["db"]
    user = _ensure_user(db, update)
    role = user.get("role", "free") if user else "free"
    name = update.effective_user.first_name or "trader"

    keyboard = get_keyboard_for_role(role)

    if role in ("premium", "owner"):
        # Premium / owner greeting
        expires = user.get("expires_at") if user else None
        exp_str = "бессрочно"
        if expires:
            try:
                exp_dt = datetime.fromisoformat(expires)
                exp_str = exp_dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                exp_str = "бессрочно"

        # Last signal info
        signals = db.get_signals_history(limit=1)
        last_signal = "нет данных"
        regime = "N/A"
        if signals:
            s = signals[0]
            d = s.get("direction", "?").upper()
            sym = s.get("symbol", "?")
            last_signal = f"{d} {sym}"
            regime = s.get("regime", "N/A").upper()

        role_badge = "👑" if role == "owner" else "⭐"
        msg = (
            f"👋 Привет, {name}! {role_badge}\n\n"
            f"Premium активен до {exp_str}\n\n"
            f"📊 Последний сигнал: {last_signal}\n"
            f"🌡 Режим рынка: {regime}\n\n"
            f"👇 Используй кнопки меню ниже"
        )
    else:
        # Free greeting
        stats = db.get_stats()
        wr = stats.get("win_rate_30d", 0)
        total = stats.get("total_trades", 0)
        msg = (
            f"👋 Привет, {name}!\n\n"
            f"Добро пожаловать в AtomiCortex — AI торговые "
            f"сигналы крипто фьючерсов.\n\n"
            f"🤖 BTC/ETH/SOL | 4H таймфрейм\n"
            f"📊 Win Rate: {wr:.0%} | {total} сигналов\n\n"
            f"👇 Используй кнопки меню ниже"
        )

    await update.effective_chat.send_message(msg, reply_markup=keyboard)


@require_role("free")
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Role-aware command list."""
    db: Database = context.bot_data["db"]
    user = _ensure_user(db, update)
    role = user.get("role", "free") if user else "free"

    lines = [
        "📋 Доступные команды:\n",
        "═══ FREE ═══",
        "/start — приветствие",
        "/help — список команд",
        "/stats — публичная статистика",
        "/subscribe — оформить подписку",
        "/mystatus — ваш статус",
    ]

    if role in ("premium", "owner"):
        lines += [
            "\n═══ PREMIUM ═══",
            "/signal — последний активный сигнал",
            "/history — последние 10 сигналов",
            "/regime — текущий режим рынка",
            "/funding — экстремальный funding",
            "/risk <капитал> — калькулятор позиции",
        ]

    if role == "owner":
        lines += [
            "\n═══ OWNER ═══",
            "/users — все пользователи",
            "/user <id/@username> — детали пользователя",
            "/grant <id> <role> [срок] — выдать роль",
            "/revoke <id> — сбросить до free",
            "/ban <id> — забанить",
            "/broadcast <msg> — рассылка",
            "/health — состояние системы",
            "/stop_bot — остановка бота",
            "/restart_bot — перезапуск бота",
            "/logs <N> — последние N строк логов",
            "/stats_admin — полная статистика",
            "/payments — история платежей",
            "/revenue — статистика дохода",
        ]

    await update.effective_chat.send_message("\n".join(lines))


@require_role("free")
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Public statistics."""
    db: Database = context.bot_data["db"]
    stats = db.get_stats()
    signals = db.get_signals_history(limit=1)
    regime = signals[0].get("regime", "N/A").upper() if signals else "N/A"

    await update.effective_chat.send_message(
        f"📊 AtomiCortex — Статистика\n"
        f"{'═' * 30}\n"
        f"Win Rate (30д): {stats['win_rate_30d']:.1%}\n"
        f"Всего сигналов: {stats['total_trades']}\n"
        f"  ✅ Win: {stats['wins']}  ❌ Loss: {stats['losses']}  "
        f"🔵 Open: {stats['open']}\n"
        f"Общий P&L: {stats['total_pnl_pct']:+.2f}%\n"
        f"Режим рынка: {regime}\n"
        f"{'═' * 30}"
    )


@require_role("free")
async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show subscription options with inline keyboard buttons."""
    prices = context.bot_data.get("prices", {})
    stars_30 = prices.get("stars_30d", 500)
    stars_90 = prices.get("stars_90d", 1200)
    usdt_30 = prices.get("usdt_30d", 7.00)
    usdt_90 = prices.get("usdt_90d", 18.00)

    keyboard = get_subscribe_keyboard(stars_30, stars_90, usdt_30, usdt_90)

    await update.effective_chat.send_message(
        f"{'━' * 30}\n"
        f"⭐ AtomiCortex Premium\n\n"
        f"Что включено:\n"
        f"🤖 AI сигналы BTC/ETH/SOL\n"
        f"📊 Trend + High_Vol режимы\n"
        f"⚡ Confidence ≥ 65%\n"
        f"📱 Мгновенные алерты 24/7\n\n"
        f"Выбери способ оплаты:\n"
        f"{'━' * 30}",
        reply_markup=keyboard,
    )


@require_role("free")
async def cmd_mystatus(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current subscription status."""
    if update.effective_user is None:
        return
    db: Database = context.bot_data["db"]
    user = db.get_user(update.effective_user.id)

    if user is None:
        await update.effective_chat.send_message("❌ Профиль не найден. Используйте /start.")
        return

    role = user.get("role", "free")
    expires = user.get("expires_at")

    if role == "premium" and expires:
        try:
            exp_dt = datetime.fromisoformat(expires)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            remaining = (exp_dt - datetime.now(timezone.utc)).days
            exp_str = exp_dt.strftime("%Y-%m-%d")

            # Show renewal button if < 7 days remaining
            reply_markup = get_renew_button() if remaining < 7 else None

            await update.effective_chat.send_message(
                f"👤 Ваш статус\n"
                f"{'═' * 30}\n"
                f"Роль: Premium ✅\n"
                f"Истекает: {exp_str} ({remaining} дней)\n"
                f"{'═' * 30}",
                reply_markup=reply_markup,
            )
        except (ValueError, TypeError):
            await update.effective_chat.send_message(
                f"👤 Ваш статус\n"
                f"{'═' * 30}\n"
                f"Роль: Premium ✅\n"
                f"Срок: бессрочно\n"
                f"{'═' * 30}"
            )
    elif role == "owner":
        await update.effective_chat.send_message(
            f"👤 Ваш статус\n"
            f"{'═' * 30}\n"
            f"Роль: Owner 👑\n"
            f"{'═' * 30}"
        )
    else:
        await update.effective_chat.send_message(
            f"👤 Ваш статус\n"
            f"{'═' * 30}\n"
            f"Роль: Free\n\n"
            f"Upgrade: /subscribe\n"
            f"{'═' * 30}"
        )
