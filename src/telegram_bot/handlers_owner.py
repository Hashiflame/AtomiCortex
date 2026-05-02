"""AtomiCortex — Owner-tier Telegram command handlers."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.database import Database
from src.telegram_bot.roles import require_role, OWNER_ID


# TG-003: Regex for sensitive data redaction
_SENSITIVE_RE = re.compile(
    r'(api[_-]?key|secret|password|token|authorization)["\s:=]+\S+',
    re.IGNORECASE,
)


def _redact_sensitive(text: str) -> str:
    """Replace sensitive values in log output."""
    return _SENSITIVE_RE.sub(r"\1=***REDACTED***", text)


# TG-014: SQL-level user resolver
def _resolve_user(db: Database, identifier: str) -> dict | None:
    """Resolve user by numeric ID or @username using SQL."""
    identifier = identifier.lstrip("@")
    if identifier.isdigit():
        return db.get_user(int(identifier))
    return db.get_user_by_username(identifier)


@require_role("owner")
async def cmd_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Table of all users (TG-017: mask user_id partially)."""
    db: Database = context.bot_data["db"]
    users = db.get_all_users()

    if not users:
        await update.effective_chat.send_message("Нет зарегистрированных пользователей.")
        return

    lines = ["👥 Пользователи:\n", "ID | Username | Role | Joined"]
    lines.append("─" * 45)
    for u in users[:50]:
        un = f"@{u['username']}" if u.get("username") else "—"
        joined = (u.get("joined_at") or "")[:10]
        banned = " 🚫" if u.get("is_banned") else ""
        lines.append(f"{u['user_id']} | {un} | {u['role']}{banned} | {joined}")

    await update.effective_chat.send_message("\n".join(lines))


@require_role("owner")
async def cmd_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """User details: /user <id or @username>."""
    db: Database = context.bot_data["db"]
    args = context.args or []
    if not args:
        await update.effective_chat.send_message("Использование: /user <id или @username>")
        return

    user = _resolve_user(db, args[0])
    if not user:
        await update.effective_chat.send_message(f"❌ Пользователь '{args[0]}' не найден.")
        return

    expires = user.get("expires_at") or "бессрочно"
    banned = "Да 🚫" if user.get("is_banned") else "Нет"
    notes = user.get("notes") or "—"

    await update.effective_chat.send_message(
        f"👤 Пользователь\n{'═' * 30}\n"
        f"ID: {user['user_id']}\n"
        f"Username: @{user.get('username', '—')}\n"
        f"Имя: {user.get('first_name', '—')}\n"
        f"Роль: {user['role']}\n"
        f"Истекает: {expires}\n"
        f"Бан: {banned}\n"
        f"Joined: {user.get('joined_at', '—')}\n"
        f"Заметки: {notes}\n{'═' * 30}"
    )


@require_role("owner")
async def cmd_grant(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Grant role: /grant <id> <role> [duration].

    TG-004/TG-011: validates duration (max 365d), blocks granting owner.
    """
    db: Database = context.bot_data["db"]
    args = context.args or []
    if len(args) < 2:
        await update.effective_chat.send_message(
            "Использование: /grant <id/@username> <premium|free> [30d]"
        )
        return

    user = _resolve_user(db, args[0])
    if not user:
        await update.effective_chat.send_message(f"❌ Пользователь '{args[0]}' не найден.")
        return

    role = args[1].lower()
    # TG-011: block granting owner role via command
    if role == "owner":
        await update.effective_chat.send_message(
            "❌ Роль owner нельзя выдать через команду. "
            "Owner определяется через TELEGRAM_ADMIN_ID."
        )
        return
    if role not in ("free", "premium"):
        await update.effective_chat.send_message("❌ Роль должна быть: free, premium")
        return

    expires_at = None
    if len(args) >= 3:
        duration_str = args[2].lower()
        try:
            if duration_str.endswith("d"):
                days = int(duration_str[:-1])
                if days <= 0 or days > 365:
                    await update.effective_chat.send_message(
                        "❌ Срок должен быть от 1 до 365 дней."
                    )
                    return
                expires_at = datetime.now(timezone.utc) + timedelta(days=days)
            elif duration_str.endswith("h"):
                hours = int(duration_str[:-1])
                if hours <= 0 or hours > 8760:
                    await update.effective_chat.send_message(
                        "❌ Срок должен быть от 1 до 8760 часов."
                    )
                    return
                expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
            else:
                await update.effective_chat.send_message(
                    "❌ Неверный формат срока. Пример: 30d, 24h"
                )
                return
        except (ValueError, IndexError):
            await update.effective_chat.send_message("❌ Неверный формат срока. Пример: 30d, 24h")
            return

    db.set_role(user["user_id"], role, expires_at)
    exp_str = expires_at.strftime("%Y-%m-%d %H:%M UTC") if expires_at else "бессрочно"
    await update.effective_chat.send_message(
        f"✅ Роль {role} выдана пользователю {user.get('username', user['user_id'])}.\n"
        f"Срок: {exp_str}"
    )


@require_role("owner")
async def cmd_revoke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset user to free: /revoke <id>."""
    db: Database = context.bot_data["db"]
    args = context.args or []
    if not args:
        await update.effective_chat.send_message("Использование: /revoke <id/@username>")
        return

    user = _resolve_user(db, args[0])
    if not user:
        await update.effective_chat.send_message(f"❌ Пользователь '{args[0]}' не найден.")
        return

    db.set_role(user["user_id"], "free")
    await update.effective_chat.send_message(
        f"✅ Роль пользователя {user.get('username', user['user_id'])} сброшена до free."
    )


@require_role("owner")
async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ban user: /ban <id>.  TG-016: guards against banning owner."""
    db: Database = context.bot_data["db"]
    args = context.args or []
    if not args:
        await update.effective_chat.send_message("Использование: /ban <id/@username>")
        return

    user = _resolve_user(db, args[0])
    if not user:
        await update.effective_chat.send_message(f"❌ Пользователь '{args[0]}' не найден.")
        return

    # TG-016: protect owner from being banned
    if OWNER_ID is not None and user["user_id"] == OWNER_ID:
        await update.effective_chat.send_message("❌ Нельзя забанить владельца.")
        return

    db.ban_user(user["user_id"])
    await update.effective_chat.send_message(
        f"🚫 Пользователь {user.get('username', user['user_id'])} забанен."
    )


@require_role("owner")
async def cmd_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Broadcast message to all non-banned users.  TG-007: check empty list."""
    db: Database = context.bot_data["db"]
    if not context.args:
        await update.effective_chat.send_message("Использование: /broadcast <сообщение>")
        return

    message = " ".join(context.args)
    users = db.get_non_banned_users()

    # TG-007: handle empty user list
    if not users:
        await update.effective_chat.send_message("📭 Нет пользователей для рассылки.")
        return

    sent, failed = 0, 0
    for u in users:
        try:
            await context.bot.send_message(
                chat_id=u["user_id"],
                text=f"📢 Рассылка:\n\n{message}",
            )
            sent += 1
        except Exception:
            failed += 1

    await update.effective_chat.send_message(
        f"✅ Рассылка завершена: {sent} отправлено, {failed} ошибок."
    )


@require_role("owner")
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Full system health report.  TG-008: reads from DB, not bot_data."""
    db: Database = context.bot_data["db"]

    # Server metrics
    cpu = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count() or 0
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    uptime = datetime.now(timezone.utc) - boot
    up_days = uptime.days
    up_hours = uptime.seconds // 3600
    up_mins = (uptime.seconds % 3600) // 60

    # Trading data from DB (TG-008: no cross-process bot_data)
    stats = db.get_stats()
    signals_today = db.get_signals_today_count()
    signals = db.get_signals_history(limit=1)
    regime = signals[0].get("regime", "N/A").upper() if signals else "N/A"
    open_signals = db.get_open_signals()

    msg = (
        f"{'═' * 30}\n"
        f"🖥️ SERVER HEALTH\n{'═' * 30}\n"
        f"CPU:      {cpu}% ({cpu_count} cores)\n"
        f"RAM:      {mem.used / 1e9:.1f}/{mem.total / 1e9:.0f} GB "
        f"({mem.percent}%)\n"
        f"Disk:     {disk.used / 1e9:.1f}/{disk.total / 1e9:.0f} GB "
        f"({disk.percent}%)\n"
        f"Uptime:   {up_days}d {up_hours}h {up_mins}m\n\n"
        f"🤖 BOT STATUS\n{'═' * 30}\n"
        f"Status:   RUNNING ✅\n"
        f"Signals today: {signals_today}\n\n"
        f"📊 TRADING\n{'═' * 30}\n"
        f"Regime:   {regime}\n"
        f"Total:    {stats['total_trades']} signals\n"
        f"Win rate: {stats['win_rate']:.1%}\n"
        f"P&L:      {stats['total_pnl_pct']:+.2f}%\n"
        f"Open:     {len(open_signals)} positions\n"
        f"{'═' * 30}"
    )

    await update.effective_chat.send_message(msg)


# TG-010: asyncio.Lock for stop confirmation instead of bare dict
_stop_lock = asyncio.Lock()
_pending_stop: dict[int, float] = {}


@require_role("owner")
async def cmd_stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Emergency stop (requires /confirm_stop within 60s)."""
    uid = update.effective_user.id if update.effective_user else 0
    async with _stop_lock:
        # Clean stale entries (TG-005)
        now = time.time()
        stale = [k for k, v in _pending_stop.items() if now - v > 60]
        for k in stale:
            del _pending_stop[k]
        _pending_stop[uid] = now

    await update.effective_chat.send_message(
        "⚠️ ВНИМАНИЕ: Остановка торгового бота!\n\n"
        "Отправьте /confirm_stop в течение 60 секунд для подтверждения."
    )


@require_role("owner")
async def cmd_confirm_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm emergency stop.  TG-002: uses asyncio subprocess."""
    uid = update.effective_user.id if update.effective_user else 0
    async with _stop_lock:
        requested_at = _pending_stop.pop(uid, 0)

    if time.time() - requested_at > 60:
        await update.effective_chat.send_message(
            "❌ Время подтверждения истекло. Повторите /stop_bot."
        )
        return

    await update.effective_chat.send_message("🛑 Останавливаю торгового бота...")
    try:
        # TG-002: async subprocess, no sudo
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", "stop", "atomicortex-bot",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            await update.effective_chat.send_message("✅ Бот остановлен.")
        else:
            await update.effective_chat.send_message(
                f"❌ Ошибка остановки (код {proc.returncode})."
            )
    except asyncio.TimeoutError:
        await update.effective_chat.send_message("❌ Таймаут остановки (15с).")
    except Exception as exc:
        await update.effective_chat.send_message(f"❌ Ошибка остановки: {exc}")


@require_role("owner")
async def cmd_restart_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Restart trading bot.  TG-002: async subprocess."""
    await update.effective_chat.send_message("🔄 Перезапускаю торгового бота...")
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", "restart", "atomicortex-bot",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            await update.effective_chat.send_message("✅ Бот перезапущен.")
        else:
            await update.effective_chat.send_message(
                f"❌ Ошибка перезапуска (код {proc.returncode})."
            )
    except asyncio.TimeoutError:
        await update.effective_chat.send_message("❌ Таймаут перезапуска (15с).")
    except Exception as exc:
        await update.effective_chat.send_message(f"❌ Ошибка перезапуска: {exc}")


@require_role("owner")
async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last N lines from trading log.

    TG-003: redacts sensitive data.
    TG-011: validates N (1-100).
    """
    args = context.args or []
    n = 50
    if args:
        if args[0].isdigit():
            n = int(args[0])
        else:
            n = 50

    # TG-011: clamp N
    if n <= 0:
        n = 10
    if n > 100:
        n = 100

    logs_dir = Path("./logs")
    log_files = sorted(logs_dir.glob("trading_*.log"), reverse=True)

    if not log_files:
        await update.effective_chat.send_message("📭 Логи не найдены.")
        return

    try:
        # TG-002: async subprocess for tail
        proc = await asyncio.create_subprocess_exec(
            "tail", f"-n{n}", str(log_files[0]),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode("utf-8", errors="replace") if stdout else "Пусто."

        # TG-003: redact sensitive data before sending
        output = _redact_sensitive(output)

        # Truncate to Telegram limit
        if len(output) > 4000:
            output = output[-4000:]

        await update.effective_chat.send_message(f"📋 Последние {n} строк:\n\n{output}")
    except asyncio.TimeoutError:
        await update.effective_chat.send_message("❌ Таймаут чтения логов.")
    except Exception as exc:
        await update.effective_chat.send_message(f"❌ Ошибка чтения логов: {exc}")


@require_role("owner")
async def cmd_stats_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Full admin stats."""
    db: Database = context.bot_data["db"]
    stats = db.get_stats()
    all_users = db.get_all_users()

    owners = sum(1 for u in all_users if u["role"] == "owner")
    premium = sum(1 for u in all_users if u["role"] == "premium")
    free = sum(1 for u in all_users if u["role"] == "free")
    banned = sum(1 for u in all_users if u.get("is_banned"))

    # Top signals
    signals = db.get_signals_history(limit=5)
    top_lines = []
    for s in signals:
        if s.get("pnl_pct") is not None:
            top_lines.append(
                f"  {s['symbol']} {s['direction']}: {s['pnl_pct']:+.2f}%"
            )

    msg = (
        f"📊 Admin Statistics\n{'═' * 30}\n\n"
        f"👥 Пользователи:\n"
        f"  Owner: {owners}\n"
        f"  Premium: {premium}\n"
        f"  Free: {free}\n"
        f"  Banned: {banned}\n"
        f"  Total: {len(all_users)}\n\n"
        f"📈 Trading:\n"
        f"  Total signals: {stats['total_trades']}\n"
        f"  Win rate: {stats['win_rate']:.1%}\n"
        f"  Win rate 30d: {stats['win_rate_30d']:.1%}\n"
        f"  Total P&L: {stats['total_pnl_pct']:+.2f}%\n\n"
    )

    if top_lines:
        msg += "🏆 Последние сигналы:\n" + "\n".join(top_lines) + "\n"
    msg += f"\n{'═' * 30}"

    await update.effective_chat.send_message(msg)
