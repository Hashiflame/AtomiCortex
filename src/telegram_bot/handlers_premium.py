"""AtomiCortex — Premium-tier Telegram command handlers."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from src.logger import get_logger
from src.telegram_bot.database import Database
from src.telegram_bot.handlers_free import _resolve_stat_dbs
from src.telegram_bot.keyboards import (
    history_keyboard,
    signal_detail_keyboard,
    signals_filter_keyboard,
)
from src.telegram_bot.roles import require_role
from src.telegram_bot.signal_formatter import format_signal_card
from src.telegram_bot.timeframes import active_timeframes, tf_for_db_path

_log = get_logger(__name__)

# Binance USDT-M Futures public endpoint — no auth needed
_BINANCE_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

_REGIME_INFO = {
    "trend_up":   ("📈", "Тренд вверх", "Momentum стратегия активна"),
    "trend_down": ("📉", "Тренд вниз", "Momentum стратегия активна"),
    "high_vol":   ("⚡", "Высокая волатильность", "Defensive mode"),
    "range":      ("↔️", "Боковик", "Mean-reversion режим"),
    "orb_breakout": ("🎯", "ORB пробой", "Breakout стратегия (15m)"),
    "UNKNOWN":    ("❓", "Неизвестно", "Ожидаем первый сигнал"),
}


# ──────────────────────────────────────────────────────────────────────
# Shared multi-DB collectors (mirror the /stats merge pattern). Each
# isolated trading DB is single-timeframe; rows are tagged + merged +
# sorted by created_at DESC. Used by handlers AND inline callbacks.
# ──────────────────────────────────────────────────────────────────────

_RESULT_FILTER_MAP = {"wins": "win", "losses": "loss", "open": "open"}


def _resolve_selector(sel: str | None) -> tuple[str | None, str | None]:
    """History/signal selector → (timeframe, result_filter).

    ``all`` → (None, None); a timeframe → (tf, None);
    ``wins``/``losses``/``open`` → (None, result_filter).
    """
    if not sel or sel == "all":
        return None, None
    if sel in active_timeframes():
        return sel, None
    if sel in _RESULT_FILTER_MAP:
        return None, sel
    return None, None


def _collect_recent(
    context: ContextTypes.DEFAULT_TYPE,
    limit: int = 10,
    timeframe: str | None = None,
    status: str | None = None,
    result_filter: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for tf_label, db in _resolve_stat_dbs(context):
        try:
            part = db.get_recent_signals(limit=200, status=status)
        except Exception:
            continue
        for s in part:
            s.setdefault("timeframe", tf_label)
            if not s.get("timeframe"):
                s["timeframe"] = tf_label
            rows.append(s)
    if timeframe and timeframe not in ("all", "open"):
        rows = [s for s in rows if s.get("timeframe") == timeframe]
    if result_filter in _RESULT_FILTER_MAP:
        want = _RESULT_FILTER_MAP[result_filter]
        rows = [s for s in rows if (s.get("result") or "open") == want]
    rows.sort(key=lambda s: str(s.get("created_at") or ""), reverse=True)
    return rows[:limit]


def _collect_paginated(
    context: ContextTypes.DEFAULT_TYPE,
    page: int,
    per_page: int,
    selector: str = "all",
) -> tuple[list[dict], int]:
    tf, result_filter = _resolve_selector(selector)
    allrows = _collect_recent(
        context, limit=10_000, timeframe=tf, result_filter=result_filter,
    )
    total = len(allrows)
    page = max(1, page)
    start = (page - 1) * per_page
    return allrows[start:start + per_page], total


def render_signals_view(
    context: ContextTypes.DEFAULT_TYPE, selected: str = "all",
) -> tuple[str, object]:
    """(text, keyboard) for the /signal view with a TF filter."""
    status = "open" if selected == "open" else None
    tf = None if selected in ("all", "open") else selected
    sigs = _collect_recent(context, limit=1, timeframe=tf, status=status)
    if not sigs:
        scope = "открытых " if selected == "open" else ""
        text = (
            f"📭 Нет {scope}сигналов.\n"
            "Бот работает и анализирует рынок."
        )
    else:
        text = format_signal_card(
            sigs[0], mode="open" if selected == "open" else "full"
        )
    kb = signals_filter_keyboard(active_timeframes(), selected)
    return text, kb


_SELECTOR_LABEL = {
    "wins": "✅ Wins", "losses": "❌ Losses", "open": "📂 Открытые",
}


def render_history_view(
    context: ContextTypes.DEFAULT_TYPE, page: int = 1, sel: str = "all",
) -> tuple[str, object]:
    per_page = 10
    rows, total = _collect_paginated(context, page, per_page, sel)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(max(1, page), total_pages)
    if not rows:
        text = "📭 История пуста."
    else:
        head = "📋 История сигналов"
        if sel != "all":
            head += f" · {_SELECTOR_LABEL.get(sel, sel.upper())}"
        lines = [head, "━" * 24]
        lines += [format_signal_card(s, mode="compact") for s in rows]
        lines.append(f"\nСтр. {page}/{total_pages} · всего {total}")
        text = "\n".join(lines)
    return text, history_keyboard(page, total_pages, sel)


def find_signal_by_id(
    context: ContextTypes.DEFAULT_TYPE,
    sid: int,
    timeframe: str | None = None,
) -> dict | None:
    """Locate a signal by id across the merged trading DBs.

    M5: ``id`` is per-DB autoincrement so a 4H id=42 and a 15m id=42
    are different rows. When ``timeframe`` is supplied the search is
    scoped to rows whose ``timeframe`` matches — the composite
    ``(timeframe, id)`` is the only globally-unique key without a
    schema change. ``timeframe=None`` keeps legacy first-match
    behaviour for backward compat with older callback_data strings.
    """
    for s in _collect_recent(context, limit=10_000):
        try:
            if int(s.get("id", -1)) != int(sid):
                continue
        except (TypeError, ValueError):
            continue
        if timeframe is None or s.get("timeframe") == timeframe:
            return s
    return None


def _latest_signal(context: ContextTypes.DEFAULT_TYPE) -> dict | None:
    """Most recent signal across all shared DBs (any status), or None."""
    sigs = _collect_recent(context, limit=1)
    return sigs[0] if sigs else None


def _latest_regime(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Regime from the most recent signal across shared DBs; the
    trading bot's bot_metrics is unreliable (often UNKNOWN)."""
    s = _latest_signal(context)
    if s and s.get("regime"):
        # ORB strat encodes "orb:trend_up" — keep the base regime.
        return str(s["regime"]).split(":")[0]
    return "UNKNOWN"


def _format_age(updated_at: str) -> str:
    """Render '5m ago' / '2h ago' / '3d ago' from an ISO timestamp string."""
    try:
        ts = datetime.fromisoformat(updated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "?"


@require_role("premium")
async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Most recent signal (any status) across all trading DBs, with an
    inline timeframe filter. Never crashes — degrades to a message."""
    try:
        text, kb = render_signals_view(context, selected="all")
        await update.effective_chat.send_message(text, reply_markup=kb)
    except Exception as exc:
        _log.error("cmd_signal failed: {e}", e=exc, exc_info=True)
        await update.effective_chat.send_message(
            "📭 Сигналы временно недоступны. Попробуйте позже."
        )


@require_role("premium")
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Paginated signal history (page 1, all timeframes) with inline
    pagination + timeframe filter."""
    try:
        text, kb = render_history_view(context, page=1, sel="all")
        await update.effective_chat.send_message(text, reply_markup=kb)
    except Exception as exc:
        _log.error("cmd_history failed: {e}", e=exc, exc_info=True)
        await update.effective_chat.send_message(
            "📭 История временно недоступна. Попробуйте позже."
        )


@require_role("premium")
async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Performance report: open positions, last 10 closed, equity curve.

    Aggregates every isolated trading DB (4H + 15m + 1H) — reuses the
    same backward-compatible DB resolution as /stats.
    """
    from src.telegram_bot.handlers_free import _resolve_stat_dbs

    dbs = _resolve_stat_dbs(context)

    open_rows: list[tuple[str, dict]] = []
    closed_rows: list[tuple[str, dict]] = []
    for tf, db in dbs:
        try:
            for s in db.get_open_signals():
                open_rows.append((tf, s))
            for s in db.get_signals_history(limit=50):
                if s.get("result") in ("win", "loss", "breakeven"):
                    closed_rows.append((tf, s))
        except Exception:
            continue

    # Newest closed first.
    closed_rows.sort(
        key=lambda r: str(r[1].get("closed_at") or r[1].get("created_at") or ""),
        reverse=True,
    )

    lines = [
        "📈 Performance Report",
        "═" * 38,
        "Период: последние 30 дней",
        "",
        f"Открытые позиции ({len(open_rows)}):",
    ]
    if open_rows:
        for tf, s in open_rows[:10]:
            d = (s.get("direction", "") or "").upper()
            ico = "🟢" if d == "LONG" else "🔴"
            sym = (s.get("symbol", "?") or "?").split("-")[0]
            lines.append(
                f"{ico} {d:<5} {sym} {tf:<3} | Вход: "
                f"${s.get('entry_price', 0):,.0f} | "
                f"Conf: {s.get('confidence', 0):.0%}"
            )
    else:
        lines.append("  — нет открытых позиций")

    lines += ["", "Последние 10 закрытых:"]
    if closed_rows:
        for tf, s in closed_rows[:10]:
            res = s.get("result", "")
            mark = "✅" if res == "win" else "❌" if res == "loss" else "➖"
            d = (s.get("direction", "") or "").upper()
            sym = (s.get("symbol", "?") or "?").split("-")[0]
            pnl = s.get("pnl_pct") or 0.0
            day = str(s.get("closed_at") or s.get("created_at") or "")[:10]
            lines.append(
                f"{mark} {d:<5} {sym} {tf:<3} | {pnl:+.1f}% | {day}"
            )
    else:
        lines.append("  — нет закрытых сделок")

    # Text equity curve: compound closed PnL on a $10k base, oldest→newest.
    chrono = sorted(
        closed_rows,
        key=lambda r: str(r[1].get("closed_at") or r[1].get("created_at") or ""),
    )
    if chrono:
        base = 10_000.0
        eq = base
        pts: list[tuple[str, float]] = []
        for _, s in chrono:
            eq *= 1.0 + (float(s.get("pnl_pct") or 0.0) / 100.0)
            pts.append((str(s.get("closed_at") or s.get("created_at"))[:10], eq))
        # Sample ≤4 points: first, ~1/3, ~2/3, last.
        idx = sorted({0, len(pts) // 3, 2 * len(pts) // 3, len(pts) - 1})
        lines += ["", "Equity curve:"]
        for i in idx:
            day, val = pts[i]
            chg = (val / base - 1.0) * 100.0
            lines.append(f"{day}: ${val:,.0f}  ({chg:+.1f}%)")

    # Risk-adjusted metrics + monthly breakdown via StatsEngine
    # (fail-soft; only when shared_db_paths are known).
    db_paths = context.bot_data.get("shared_db_paths")
    if db_paths:
        try:
            from src.analytics.stats_engine import StatsEngine
            from src.telegram_bot.handlers_free import fmt_metric
            eng = StatsEngine(db_paths)
            perf = eng.compute_performance(timeframe="all", period_days=30)
            pf = perf.get("profit_factor")
            lines[1:1] = [
                "",
                "Risk-adjusted (30д, all):",
                f"  Sharpe:  {fmt_metric(perf.get('sharpe_ratio'))}   "
                f"Sortino: {fmt_metric(perf.get('sortino_ratio'))}",
                f"  Calmar:  {fmt_metric(perf.get('calmar_ratio'))}   "
                f"PF: {'∞' if pf is None else format(pf, '.2f')}",
                f"  Max DD:  {perf.get('max_drawdown', 0.0):.1f}%   "
                f"EV: {perf.get('expected_value', 0.0):+.2f}%/сигнал",
            ]
            monthly = eng.compute_monthly(timeframe="all")
            if monthly:
                lines += ["", "Monthly breakdown:"]
                for m in monthly[-6:]:
                    lines.append(
                        f"  {m['month']}: {m['pnl_pct']:+.1f}% | "
                        f"{m['wins']}W/{m['losses']}L | WR {m['wr']:.0%}"
                    )
        except Exception as exc:
            _log.debug("performance metrics skipped: {e}", e=exc)

    lines.append("═" * 38)
    await update.effective_chat.send_message("\n".join(lines))


@require_role("premium")
async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Current market regime, derived from the most recent signal across
    the trading DBs (bot_metrics.regime is unreliable — often UNKNOWN)."""
    try:
        from src.telegram_bot.signal_formatter import _clean_symbol, _parse_dt
        from src.telegram_bot.timeframes import get_tf_label

        sig = _latest_signal(context)
        regime = (
            str(sig["regime"]).split(":")[0]
            if sig and sig.get("regime") else "UNKNOWN"
        )
        emoji, label, desc = _REGIME_INFO.get(regime, _REGIME_INFO["UNKNOWN"])

        src_lines = ""
        if sig:
            dt = _parse_dt(sig.get("created_at"))
            date_str = (
                dt.strftime("%d.%m.%Y %H:%M UTC") if dt
                else str(sig.get("created_at") or "—")[:16]
            )
            pair = _clean_symbol(sig.get("symbol", "BTC"))
            tf_label = get_tf_label(sig.get("timeframe") or "4h")
            src_lines = (
                f"\n"
                f"📌 Источник: последний сигнал\n"
                f"🕐 {date_str}\n"
                f"💱 Пара: {pair}\n"
                f"⏱ TF: {tf_label}\n"
            )

        await update.effective_chat.send_message(
            f"🌡 Режим рынка\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} {label}\n"
            f"ℹ️ {desc}\n"
            f"{src_lines}"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
    except Exception as exc:
        _log.error("cmd_regime failed: {e}", e=exc, exc_info=True)
        await update.effective_chat.send_message(
            "🌡 Режим временно недоступен. Попробуйте позже."
        )


@require_role("premium")
async def cmd_funding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Top-3 pairs by extreme funding rate — live from Binance USDT-M
    Futures ``/fapi/v1/premiumIndex``.  No DB dependency; works even
    before the trading bot has produced any signals."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_BINANCE_PREMIUM_INDEX_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        _log.warning("cmd_funding: Binance fetch failed: {err}", err=str(exc))
        await update.effective_chat.send_message(
            "📭 Не удалось получить funding rates от Binance."
        )
        return

    if not isinstance(data, list) or not data:
        await update.effective_chat.send_message(
            "📭 Binance вернул пустые данные."
        )
        return

    rates: list[tuple[str, float]] = []
    for item in data:
        try:
            sym = item["symbol"]
            rate = float(item["lastFundingRate"])
            rates.append((sym, rate))
        except (KeyError, TypeError, ValueError):
            continue

    if not rates:
        await update.effective_chat.send_message(
            "📭 Не удалось распарсить ответ Binance."
        )
        return

    top3 = sorted(rates, key=lambda x: abs(x[1]), reverse=True)[:3]

    lines = [f"💰 Top-3 Extreme Funding (live)\n{'═' * 30}"]
    for sym, rate in top3:
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
