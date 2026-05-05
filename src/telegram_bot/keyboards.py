"""
AtomiCortex — Telegram Keyboard Layouts.

Role-based ReplyKeyboard menus and InlineKeyboard builders for
subscription, health, users pagination, and admin stats.

Phase 7.2 — UX.
"""

from __future__ import annotations

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)


# ═══════════════════════════════════════════════════════════════════════════
# Button text constants — used for matching in MessageHandler
# ═══════════════════════════════════════════════════════════════════════════

BTN_SIGNAL = "🟢 Сигнал"
BTN_HISTORY = "📈 История"
BTN_REGIME = "🌡 Режим рынка"
BTN_FUNDING = "💰 Funding"
BTN_STATS = "📊 Статистика"
BTN_SUBSCRIBE = "⭐ Подписка"
BTN_HELP = "❓ Помощь"
BTN_USERS = "👥 Юзеры"
BTN_HEALTH = "🖥 Здоровье"

# Set of all button texts for filter matching
ALL_BUTTON_TEXTS = frozenset({
    BTN_SIGNAL,
    BTN_HISTORY,
    BTN_REGIME,
    BTN_FUNDING,
    BTN_STATS,
    BTN_SUBSCRIBE,
    BTN_HELP,
    BTN_USERS,
    BTN_HEALTH,
})

# Buttons requiring premium access
PREMIUM_BUTTONS = frozenset({
    BTN_SIGNAL,
    BTN_HISTORY,
    BTN_REGIME,
    BTN_FUNDING,
})

# Buttons requiring owner access
OWNER_BUTTONS = frozenset({
    BTN_USERS,
    BTN_HEALTH,
})


# ═══════════════════════════════════════════════════════════════════════════
# ReplyKeyboard builders
# ═══════════════════════════════════════════════════════════════════════════

def get_free_keyboard() -> ReplyKeyboardMarkup:
    """Persistent keyboard for free-tier users.

    ┌─────────────┬─────────────┐
    │ 📊 Статистика│ ⭐ Подписка  │
    ├─────────────┴─────────────┤
    │ ❓ Помощь                  │
    └───────────────────────────┘
    """
    return ReplyKeyboardMarkup(
        [
            [BTN_STATS, BTN_SUBSCRIBE],
            [BTN_HELP],
        ],
        resize_keyboard=True,
    )


def get_premium_keyboard() -> ReplyKeyboardMarkup:
    """Persistent keyboard for premium users.

    ┌──────────────┬────────────┐
    │ 🟢 Сигнал    │ 📈 История  │
    ├──────────────┼────────────┤
    │ 🌡 Режим     │ 💰 Funding  │
    ├──────────────┴────────────┤
    │ 📊 Статистика              │
    └───────────────────────────┘
    """
    return ReplyKeyboardMarkup(
        [
            [BTN_SIGNAL, BTN_HISTORY],
            [BTN_REGIME, BTN_FUNDING],
            [BTN_STATS],
        ],
        resize_keyboard=True,
    )


def get_owner_keyboard() -> ReplyKeyboardMarkup:
    """Persistent keyboard for the bot owner.

    ┌──────────────┬────────────┐
    │ 🟢 Сигнал    │ 📈 История  │
    ├──────────────┼────────────┤
    │ 🌡 Режим     │ 💰 Funding  │
    ├──────────────┼────────────┤
    │ 👥 Юзеры    │ 🖥 Здоровье │
    ├──────────────┴────────────┤
    │ 📊 Статистика              │
    └───────────────────────────┘
    """
    return ReplyKeyboardMarkup(
        [
            [BTN_SIGNAL, BTN_HISTORY],
            [BTN_REGIME, BTN_FUNDING],
            [BTN_USERS, BTN_HEALTH],
            [BTN_STATS],
        ],
        resize_keyboard=True,
    )


def get_keyboard_for_role(role: str) -> ReplyKeyboardMarkup:
    """Return the appropriate keyboard for a user role."""
    if role == "owner":
        return get_owner_keyboard()
    if role == "premium":
        return get_premium_keyboard()
    return get_free_keyboard()


# ═══════════════════════════════════════════════════════════════════════════
# InlineKeyboard builders
# ═══════════════════════════════════════════════════════════════════════════

def get_subscribe_keyboard(
    stars_30: int = 500,
    stars_90: int = 1200,
    usdt_30: float = 7.00,
    usdt_90: float = 18.00,
) -> InlineKeyboardMarkup:
    """InlineKeyboard for /subscribe with payment options."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"⭐ 30 дней — {stars_30} Stars",
                callback_data="buy_stars_30",
            ),
            InlineKeyboardButton(
                f"⭐ 90 дней — {stars_90} Stars",
                callback_data="buy_stars_90",
            ),
        ],
        [
            InlineKeyboardButton(
                f"💰 30 дней — ${usdt_30:.0f} USDT",
                callback_data="buy_usdt_30",
            ),
            InlineKeyboardButton(
                f"💰 90 дней — ${usdt_90:.0f} USDT",
                callback_data="buy_usdt_90",
            ),
        ],
        [
            InlineKeyboardButton(
                "✉️ Написать владельцу",
                callback_data="contact_owner",
            ),
        ],
    ])


def get_subscribe_inline_button() -> InlineKeyboardMarkup:
    """Single "Subscribe" inline button for locked-feature prompts."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⭐ Подписаться", callback_data="show_subscribe")],
    ])


def get_renew_button() -> InlineKeyboardMarkup:
    """Renewal button for expiring premium users."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⭐ Продлить", callback_data="show_subscribe")],
    ])


def get_health_buttons() -> InlineKeyboardMarkup:
    """Inline action buttons for /health."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 Обновить", callback_data="health_refresh"),
            InlineKeyboardButton("📋 Логи 20", callback_data="health_logs_20"),
            InlineKeyboardButton("🔁 Рестарт бота", callback_data="health_restart"),
        ],
    ])


def get_users_pagination(page: int, total_pages: int) -> InlineKeyboardMarkup:
    """Pagination buttons for /users."""
    buttons = []
    if page > 1:
        buttons.append(
            InlineKeyboardButton("◀️ Пред", callback_data=f"users_page_{page - 1}")
        )
    buttons.append(
        InlineKeyboardButton(
            f"Страница {page}/{total_pages}",
            callback_data="users_noop",
        )
    )
    if page < total_pages:
        buttons.append(
            InlineKeyboardButton("▶️ След", callback_data=f"users_page_{page + 1}")
        )
    return InlineKeyboardMarkup([buttons])


def get_stats_admin_buttons() -> InlineKeyboardMarkup:
    """Period filter buttons for /stats_admin."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📅 Сегодня", callback_data="stats_period_today"),
            InlineKeyboardButton("📅 Неделя", callback_data="stats_period_week"),
            InlineKeyboardButton("📅 Месяц", callback_data="stats_period_month"),
        ],
    ])
