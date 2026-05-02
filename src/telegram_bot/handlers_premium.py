"""AtomiCortex — Premium-tier Telegram command handlers."""

from __future__ import annotations

from telegram import Update
from telegram.ext import ContextTypes

from src.telegram_bot.database import Database
from src.telegram_bot.roles import require_role


@require_role("premium")
async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last active signal with full details."""
    db: Database = context.bot_data["db"]
    signals = db.get_open_signals()

    if not signals:
        await update.effective_chat.send_message("📭 Нет активных сигналов.")
        return

    s = signals[0]
    direction = s["direction"].upper() if s["direction"] else "N/A"
    emoji = "🟢" if direction == "LONG" else "🔴"
    symbol = s.get("symbol", "N/A")
    entry = s.get("entry_price", 0)
    sl = s.get("stop_loss", 0)
    tp = s.get("take_profit", 0)
    conf = s.get("confidence", 0)
    regime = s.get("regime", "N/A")

    sl_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
    tp_pct = abs(tp - entry) / entry * 100 if entry > 0 else 0
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else 0

    await update.effective_chat.send_message(
        f"{'═' * 30}\n"
        f"{emoji} {direction} {symbol} PERP\n"
        f"{'═' * 30}\n"
        f"Цена входа: ${entry:,.2f}\n"
        f"Стоп: ${sl:,.2f} (-{sl_pct:.1f}%)\n"
        f"Тейк: ${tp:,.2f} (+{tp_pct:.1f}%)\n"
        f"R:R: 1:{rr:.1f}\n"
        f"Режим: {regime.upper()}\n"
        f"Уверенность: {conf:.0%}\n"
        f"{'═' * 30}"
    )


@require_role("premium")
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last 10 signals with results."""
    db: Database = context.bot_data["db"]
    signals = db.get_signals_history(limit=10)

    if not signals:
        await update.effective_chat.send_message("📭 История сигналов пуста.")
        return

    lines = ["📋 Последние сигналы:\n"]
    for s in signals:
        result_emoji = {"win": "✅", "loss": "❌", "open": "🔵"}.get(
            s.get("result", ""), "⚪"
        )
        d = "L" if s.get("direction", "").lower() == "long" else "S"
        pnl = s.get("pnl_pct")
        pnl_str = f" ({pnl:+.2f}%)" if pnl is not None else ""
        lines.append(
            f"{result_emoji} {d} {s.get('symbol', '?')} "
            f"@ ${s.get('entry_price', 0):,.0f}{pnl_str}"
        )

    await update.effective_chat.send_message("\n".join(lines))


@require_role("premium")
async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Current market regime from DB signals (TG-008: no bot_data dependency)."""
    db: Database = context.bot_data["db"]
    signals = db.get_signals_history(limit=1)
    regime = signals[0].get("regime", "N/A").upper() if signals else "N/A"

    await update.effective_chat.send_message(
        f"📊 Режим рынка\n"
        f"{'═' * 30}\n"
        f"Режим:  {regime}\n"
        f"{'═' * 30}"
    )


@require_role("premium")
async def cmd_funding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Top-3 pairs by extreme funding rate."""
    funding_data = context.bot_data.get("funding_rates", {})

    if not funding_data:
        await update.effective_chat.send_message(
            "📭 Данные по funding rate недоступны."
        )
        return

    sorted_funding = sorted(
        funding_data.items(), key=lambda x: abs(x[1]), reverse=True,
    )[:3]

    lines = [f"💰 Top-3 Extreme Funding\n{'═' * 30}"]
    for sym, rate in sorted_funding:
        emoji = "🔴" if rate > 0.0005 else "🟢" if rate < -0.0005 else "⚪"
        lines.append(f"{emoji} {sym}: {rate:+.4%}")
    lines.append(f"{'═' * 30}")

    await update.effective_chat.send_message("\n".join(lines))


@require_role("premium")
async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Position size calculator: /risk <capital>.

    TG-010: validates negative/zero capital, min $10, max $1,000,000.
    """
    args = context.args or []
    if not args:
        await update.effective_chat.send_message(
            "Использование: /risk <капитал>\nПример: /risk 1000"
        )
        return

    raw = args[0].replace("$", "").replace(",", "")
    try:
        capital = float(raw)
    except ValueError:
        await update.effective_chat.send_message(
            "❌ Неверная сумма. Используй: /risk 1000"
        )
        return

    # TG-010: Validate range
    if capital <= 0:
        await update.effective_chat.send_message("❌ Капитал должен быть положительным.")
        return
    if capital < 10:
        await update.effective_chat.send_message("❌ Минимальный капитал: $10")
        return
    if capital > 1_000_000:
        await update.effective_chat.send_message("❌ Максимальный капитал: $1,000,000")
        return

    risk_pct = 0.01
    dollar_risk = capital * risk_pct
    # TG-008: use sensible defaults (not bot_data from another process)
    atr = 1500.0
    btc_price = 94000.0
    atr_stop = atr * 1.5
    position_size = dollar_risk / atr_stop if atr_stop > 0 else 0
    notional = position_size * btc_price
    leverage = notional / capital if capital > 0 else 0
    stop_price = btc_price - atr_stop

    await update.effective_chat.send_message(
        f"🧮 Калькулятор позиции\n"
        f"{'═' * 30}\n"
        f"Капитал: ${capital:,.0f}\n"
        f"Риск {risk_pct:.0%}: ${dollar_risk:,.0f}\n"
        f"ATR BTC: ~${atr:,.0f}\n"
        f"Позиция: {position_size:.4f} BTC (${notional:,.0f})\n"
        f"Стоп: ${stop_price:,.0f}\n"
        f"Leverage: {leverage:.2f}x\n"
        f"{'═' * 30}"
    )
