"""AtomiCortex — Free-tier Telegram command handlers."""

from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.database import Database
from src.telegram_bot.roles import require_role, _ensure_user


@require_role("free")
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome + auto-register (TG-017: _ensure_user already called by decorator)."""
    name = update.effective_user.first_name if update.effective_user else "trader"
    await update.effective_chat.send_message(
        f"👋 Привет, {name}!\n\n"
        f"Добро пожаловать в AtomiCortex — AI Crypto Futures Trading Bot.\n\n"
        f"📊 /stats — публичная статистика\n"
        f"📋 /help — список команд\n"
        f"⭐ /subscribe — информация о подписке\n\n"
        f"{'═' * 30}"
    )


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
        "/subscribe — информация о подписке",
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
        ]

    await update.effective_chat.send_message("\n".join(lines))


@require_role("free")
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Public statistics."""
    db: Database = context.bot_data["db"]
    stats = db.get_stats()
    # TG-008: read regime from last signal in DB, not bot_data
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
    """Subscription info."""
    await update.effective_chat.send_message(
        "⭐ AtomiCortex Premium\n"
        f"{'═' * 30}\n\n"
        "Что включено:\n"
        "  🔔 Все торговые сигналы в реальном времени\n"
        "  📈 Детальная статистика и история\n"
        "  🎯 Калькулятор размера позиции\n"
        "  📊 Режим рынка, funding, Hurst, ADX\n"
        "  📋 Ежедневные отчёты\n\n"
        "Для получения Premium свяжитесь с @admin\n"
        f"{'═' * 30}"
    )
