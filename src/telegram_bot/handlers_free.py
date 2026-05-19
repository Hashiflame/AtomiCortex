"""AtomiCortex — Free-tier Telegram command handlers."""

from __future__ import annotations

from datetime import datetime, timezone

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from src.logger import get_logger
from src.telegram_bot.database import Database
from src.telegram_bot.keyboards import (
    get_keyboard_for_role,
    get_renew_button,
    get_subscribe_keyboard,
)
from src.telegram_bot.roles import require_role, _ensure_user

_log = get_logger(__name__)


def _num(value, spec: str, none: str = "—") -> str:
    """Format a number, or a placeholder when it is None/NULL.

    Cached ``performance_cache`` rows can carry SQL NULLs (sharpe etc.),
    and ``dict.get(k, default)`` returns ``None`` — not the default —
    when the key exists with a None value. Formatting that with a
    numeric spec (``f"{None:.1f}"``) is the TypeError that took /stats
    down. This makes every numeric render None-safe.
    """
    if value is None:
        return none
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return none


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
            "/performance — отчёт по доходности (4H+15m)",
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


def _resolve_stat_dbs(context: ContextTypes.DEFAULT_TYPE) -> list[tuple[str, Database]]:
    """(timeframe_label, Database) for every isolated trading DB.

    Backward compatible: prefers ``shared_db_paths`` (multi-timeframe),
    falls back to the single shared DB, then to the legacy ``db`` —
    so tests passing only ``{"db": db}`` still work.
    """
    paths = context.bot_data.get("shared_db_paths")
    if paths:
        out: list[tuple[str, Database]] = []
        for p in paths:
            name = str(p)
            tf = "15m" if "_15m" in name else "1h" if "_1h" in name else "4h"
            out.append((tf, Database(p)))
        return out
    shared = context.bot_data.get("shared_db")
    if isinstance(shared, Database):
        return [("4h", shared)]
    return [("4h", context.bot_data["db"])]


_BAR_MIN = {"4h": 240, "1h": 60, "15m": 15, "1d": 1440}
_TF_DISPLAY = {"4h": "🟢 4H бот", "15m": "🔵 15m бот",
               "1h": "🟡 1H бот", "1d": "⚪ 1D бот"}


def format_bot_status(context: ContextTypes.DEFAULT_TYPE) -> str:
    """🔧 Статус block: per-timeframe activity, using the latest signal's
    created_at as a 'last bar' proxy (bot_metrics is unreliable).

    Active if the last signal is younger than 3× the bar period.
    Returns '' when no trading DBs are resolvable.
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc)
    for tf, db in _resolve_stat_dbs(context):
        try:
            recent = db.get_recent_signals(limit=1)
        except Exception:
            continue
        name = _TF_DISPLAY.get(tf, f"{tf} бот")
        if not recent or not recent[0].get("created_at"):
            lines.append(f"{name}: ⚪ нет данных")
            continue
        try:
            dt = datetime.fromisoformat(
                str(recent[0]["created_at"]).replace("Z", "+00:00")
            )
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
            age_min = (now - dt).total_seconds() / 60.0
            fresh = age_min <= 3 * _BAR_MIN.get(tf, 240)
            status = "активен" if fresh else "⚠️ нет свежих баров"
            lines.append(
                f"{name}: {status} "
                f"(посл. сигнал {dt.strftime('%d.%m %H:%M')})"
            )
        except (ValueError, TypeError):
            lines.append(f"{name}: ⚪ нет данных")
    if not lines:
        return ""
    return "🔧 Статус:\n" + "\n".join(lines) + "\n"


def fmt_metric(value: float | None, spec: str = ".2f") -> str:
    """Render a risk ratio, or a low-sample placeholder when the
    StatsEngine returned ``None`` (< 10 closed signals)."""
    if value is None:
        return "— (мало данных)"
    return format(value, spec)


def _fmt_strategy_block(
    label: str, icon: str, s: dict, adv: dict | None = None,
) -> str:
    """One strategy section for /stats. ``adv`` adds StatsEngine
    metrics (EV / Sharpe / Max DD) when available."""
    closed = s["closed_signals"]
    if closed > 0:
        wr = (
            f"{s['win_rate']:.0%} ({s['win_count']}W / {s['loss_count']}L)"
        )
        pf = (
            "∞" if s["profit_factor"] == float("inf")
            else f"{s['profit_factor']:.1f}"
        )
    else:
        wr = "— (нет закрытых сделок)"
        pf = "—"
    block = (
        f"{icon} {label} Стратегия:\n"
        f"Сигналов:      {s['total_signals']} "
        f"({s['open_signals']} открытых)\n"
        f"Win Rate:      {wr}\n"
        f"P&L:           {s['total_pnl_pct']:+.1f}%\n"
        f"Profit Factor: {pf}\n"
    )
    if adv:
        block += (
            f"Expected Value: {_num(adv.get('expected_value'), '+.1f')}% / сигнал\n"
            f"Max Drawdown:  {_num(adv.get('max_drawdown'), '.1f')}%\n"
            f"Sharpe:        {fmt_metric(adv.get('sharpe_ratio'))}\n"
        )
    block += f"Avg confidence: {s['avg_confidence']:.0%}\n"
    return block


@require_role("free")
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Public statistics — per-strategy (4H / 15m / 1H) + combined.

    Hard rule: this handler must never crash. Every computation path is
    individually fail-soft; the whole body is additionally wrapped so a
    surprise (None formatting, cache schema drift, DB error) degrades to
    a friendly message instead of a silent dead command.
    """
    try:
        dbs = _resolve_stat_dbs(context)
        _ICONS = {"4h": "🤖", "15m": "🔵", "1h": "🟣"}
        _NAMES = {"4h": "4H", "15m": "15m ORB", "1h": "1H"}

        # Optional StatsEngine metrics (EV / Sharpe / Max DD / equity).
        # Fail-soft: only when shared_db_paths are known (not in the
        # minimal {"db": db} unit-test context).
        adv_by_tf: dict[str, dict] = {}
        eng = None
        db_paths = context.bot_data.get("shared_db_paths")
        if db_paths:
            try:
                from src.analytics.stats_engine import StatsEngine
                eng = StatsEngine(db_paths)
            except Exception:
                eng = None

        parts: list[dict] = []
        blocks: list[str] = []
        for tf, db in dbs:
            try:
                st = db.get_trading_stats(days=30)
            except Exception:
                continue
            if not st:
                continue
            parts.append(st)
            adv = None
            if eng is not None:
                try:
                    adv = eng.compute_performance(timeframe=tf, period_days=30)
                except Exception:
                    adv = None
                adv_by_tf[tf] = adv or {}
            blocks.append(
                _fmt_strategy_block(_NAMES.get(tf, tf.upper()),
                                    _ICONS.get(tf, "•"), st, adv)
            )

        total = Database.merge_stats(parts) if parts else {
            "total_signals": 0, "win_rate": 0.0, "total_pnl_pct": 0.0,
            "closed_signals": 0,
        }
        if total.get("closed_signals", 0) > 0:
            total_wr = _num(total.get("win_rate"), ".0%")
        else:
            total_wr = "— (нет закрытых сделок)"

        today = datetime.now(timezone.utc)
        start = today.fromordinal(today.toordinal() - 30)
        period = f"{start.strftime('%d.%m')} — {today.strftime('%d.%m.%Y')}"

        equity_line = ""
        tracked_line = ""
        if eng is not None:
            try:
                allp = eng.compute_performance(timeframe="all", period_days=30)
                pnl = allp.get("total_pnl_pct") or 0.0
                equity = 10_000.0 * (1.0 + pnl / 100.0)
                equity_line = f"Equity:          ${equity:,.0f}\n"
                if allp.get("days_tracked"):
                    tracked_line = (
                        f"Tracked:         {allp['days_tracked']} дней\n"
                    )
            except Exception:
                pass

        try:
            status_block = format_bot_status(context)
        except Exception:
            status_block = ""

        msg = (
            f"📊 AtomiCortex — Статистика (30 дней)\n"
            f"{'═' * 38}\n"
            + "\n".join(blocks)
            + f"\n📈 Итого:\n"
            f"Всего сигналов:  {total.get('total_signals', 0)}\n"
            f"Общий Win Rate:  {total_wr}\n"
            f"Общий P&L:       {_num(total.get('total_pnl_pct'), '+.1f')}%\n"
            f"{equity_line}{tracked_line}"
            + (f"\n{status_block}" if status_block else "")
            + f"{'═' * 38}\n"
            f"⏰ Период: {period}"
        )
        await update.effective_chat.send_message(msg)
    except Exception as exc:
        _log.error("cmd_stats failed: {e}", e=exc, exc_info=True)
        try:
            await update.effective_chat.send_message(
                "📊 Статистика временно недоступна. Попробуйте позже."
            )
        except Exception:
            pass


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
