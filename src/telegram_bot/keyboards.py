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

from src.telegram_bot.timeframes import (
    active_timeframes,
    get_tf_emoji,
    get_tf_label,
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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7.3 — interactive signal / history keyboards (timeframe-aware)
# ═══════════════════════════════════════════════════════════════════════════

def signals_filter_keyboard(
    active_tfs: list[str], selected: str = "all",
) -> InlineKeyboardMarkup:
    """Timeframe filter for the /signal view. Active TFs only."""
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []

    all_label = "✓ 🔵 Все" if selected == "all" else "🔵 Все"
    row.append(InlineKeyboardButton(all_label, callback_data="signals_tf:all"))

    for tf in active_tfs:
        check = "✓ " if selected == tf else ""
        row.append(InlineKeyboardButton(
            f"{check}{get_tf_emoji(tf)} {get_tf_label(tf)}",
            callback_data=f"signals_tf:{tf}",
        ))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    open_label = "✓ 📂 Открытые" if selected == "open" else "📂 Открытые"
    buttons.append([
        InlineKeyboardButton(open_label, callback_data="signals_tf:open"),
    ])
    return InlineKeyboardMarkup(buttons)


def history_keyboard(
    page: int, total_pages: int, tf: str = "all",
) -> InlineKeyboardMarkup:
    """Pagination + timeframe filter for /history."""
    total_pages = max(1, total_pages)
    page = min(max(1, page), total_pages)
    nav: list[InlineKeyboardButton] = []
    if page > 1:
        nav.append(InlineKeyboardButton(
            "◀️ Пред", callback_data=f"history_page:{page - 1}:{tf}"))
    nav.append(InlineKeyboardButton(
        f"{page}/{total_pages}", callback_data="noop"))
    if page < total_pages:
        nav.append(InlineKeyboardButton(
            "След ▶️", callback_data=f"history_page:{page + 1}:{tf}"))

    all_lbl = "✓ Все" if tf == "all" else "Все"
    tf_row = [InlineKeyboardButton(
        all_lbl, callback_data=f"history_tf:all:1")]
    for active_tf in active_timeframes():
        check = "✓" if tf == active_tf else ""
        tf_row.append(InlineKeyboardButton(
            f"{check}{get_tf_emoji(active_tf)}",
            callback_data=f"history_tf:{active_tf}:1",
        ))
    return InlineKeyboardMarkup([nav, tf_row])


def signal_detail_keyboard(signal_id: int) -> InlineKeyboardMarkup:
    """Actions shown under an individual signal card."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(
            "📋 Подробнее", callback_data=f"signal_detail:{signal_id}"),
        InlineKeyboardButton("🔙 Назад", callback_data="signals_back"),
    ]])
