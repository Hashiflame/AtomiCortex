"""
Closes orphaned ``open`` signals by replaying historical prices.

Problem: ``SignalBridge.close_signal`` only fires from Nautilus
``on_position_closed``. In paper/testnet these events don't reliably
arrive → signals stay ``open`` forever (3 such rows since 2026-05).

Solution: an *external*, idempotent process reads open signals, fetches
the price path for ``[created_at, now]`` and closes each on first SL/TP
touch. Scope is **ledger correction only** — it does not fix the deeper
data-feed issue that produced those entries.

Safe alongside the live bot:
  * only ``UPDATE … WHERE result='open' AND closed_at IS NULL``
  * skips signals younger than ``skip_recent_bars × bar`` (avoids racing
    a position the bot may still legitimately close)
  * idempotent — a second run is a no-op
  * WAL SQLite (already enabled), short transactions

Price source (per decision): live Binance REST is authoritative for
out-of-sample windows (the real May-2026 signals); local DataStore is
used when it already covers the window (offline / in-sample / tests).
The source is injectable so tests run without network.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from src.logger import get_logger

_log = get_logger(__name__)

# timeframe → (binance interval, bar minutes)
_TF: dict[str, tuple[str, int]] = {
    "4h": ("4h", 240),
    "1h": ("1h", 60),
    "15m": ("15m", 15),
}


def normalize_symbol(symbol: str) -> str:
    """``BTCUSDT-PERP.BINANCE`` / ``BTCUSDT.BINANCE`` → ``BTCUSDT``."""
    m = re.match(r"[A-Z0-9]+", symbol.upper())
    return m.group(0) if m else symbol.upper()


def _parse_dt(value: str) -> datetime:
    """Parse an ISO created_at (tz-aware or naive) → aware UTC."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class PriceSource(Protocol):
    """Returns ascending bars as ``(open_time_ms, high, low)`` tuples."""

    def bars(
        self, symbol: str, interval: str, start: datetime, end: datetime,
    ) -> list[tuple[int, float, float]]:
        ...


class DataStorePriceSource:
    """Local parquet klines (offline / in-sample / tests)."""

    def __init__(self, data_dir: str = "data/features") -> None:
        self._data_dir = data_dir

    def bars(self, symbol, interval, start, end):
        from src.ingestion.data_store import DataStore

        ds = DataStore(Path(self._data_dir))
        try:
            df = ds.get_klines(
                symbol, interval, start, end,
                columns=["open_time", "high", "low"],
            )
        finally:
            ds.close()
        if df.is_empty():
            return []
        return [
            (int(r["open_time"]), float(r["high"]), float(r["low"]))
            for r in df.iter_rows(named=True)
        ]


class BinanceRESTPriceSource:
    """Live Binance USDT-M futures klines via /fapi/v1/klines (paginated)."""

    def __init__(self, trading_mode: str = "live") -> None:
        self._base = (
            "https://testnet.binancefuture.com"
            if trading_mode.lower() == "testnet"
            else "https://fapi.binance.com"
        )

    def bars(self, symbol, interval, start, end):
        import requests

        out: list[tuple[int, float, float]] = []
        cur = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        for _ in range(200):  # hard page cap
            resp = requests.get(
                f"{self._base}/fapi/v1/klines",
                params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1500,
                },
                timeout=15,
            )
            resp.raise_for_status()
            kl = resp.json()
            if not kl:
                break
            for k in kl:
                out.append((int(k[0]), float(k[2]), float(k[3])))  # t, high, low
            nxt = int(kl[-1][0]) + 1
            if len(kl) < 1500 or nxt > end_ms:
                break
            cur = nxt
        return out


class CompositePriceSource:
    """Prefer local DataStore when it covers the window; else Binance.

    'Covers' = DataStore returns bars whose last open_time reaches at
    least ``end - 2 bars`` (so a recent live window falls through to
    Binance, while a fully in-sample 2024-2025 window stays offline).
    """

    def __init__(self, data_dir="data/features", trading_mode="live") -> None:
        self._ds = DataStorePriceSource(data_dir)
        self._bx = BinanceRESTPriceSource(trading_mode)

    def bars(self, symbol, interval, start, end):
        bar_min = _TF.get(interval, ("4h", 240))[1]
        try:
            local = self._ds.bars(symbol, interval, start, end)
        except Exception as exc:
            _log.warning("DataStore price fetch failed ({e}); using Binance", e=exc)
            local = []
        end_ms = int(end.timestamp() * 1000)
        if local and local[-1][0] >= end_ms - 2 * bar_min * 60_000:
            return local
        return self._bx.bars(symbol, interval, start, end)


class SignalReconciler:
    """Closes orphaned ``open`` signals by SL/TP first-touch replay."""

    def __init__(
        self,
        db_path: str,
        data_dir: str = "data/features",
        initial_capital: float = 10_000.0,
        skip_recent_bars: int = 2,
        bar_hours: float = 4.0,
        dry_run: bool = False,
        price_source: PriceSource | None = None,
        trading_mode: str = "live",
    ) -> None:
        self.db_path = db_path
        self.data_dir = data_dir
        self.initial_capital = initial_capital
        self.skip_recent_bars = skip_recent_bars
        self.bar_hours = bar_hours
        self.dry_run = dry_run
        self._src: PriceSource = price_source or CompositePriceSource(
            data_dir, trading_mode
        )

    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _bar_for(self, signal: dict) -> tuple[str, int, float]:
        """(binance_interval, bar_minutes, bar_hours) for a signal —
        prefers the per-signal ``timeframe`` column, else ``bar_hours``."""
        tf = (signal.get("timeframe") or "").strip()
        if tf in _TF:
            iv, mins = _TF[tf]
            return iv, mins, mins / 60.0
        # Fall back to the configured bar size.
        mins = int(round(self.bar_hours * 60))
        iv = {240: "4h", 60: "1h", 15: "15m"}.get(mins, "4h")
        return iv, mins, self.bar_hours

    @staticmethod
    def _evaluate(
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        bars: list[tuple[int, float, float]],
        created_ms: int,
    ) -> dict | None:
        """First-touch SL/TP scan. SL wins on a same-bar tie (conservative).

        Returns close info dict or ``None`` if neither was touched.
        """
        is_long = direction.lower() == "long"
        worst = entry  # for MAE
        best = entry   # for MFE
        for ts, high, low in bars:
            if ts < created_ms:
                continue
            if is_long:
                worst = min(worst, low)
                best = max(best, high)
                sl_hit = low <= sl
                tp_hit = high >= tp
            else:
                worst = max(worst, high)
                best = min(best, low)
                sl_hit = high >= sl
                tp_hit = low <= tp

            if sl_hit or tp_hit:
                # Tie on same bar → SL (loss), conservative.
                if sl_hit:
                    close_px, result = sl, "loss"
                else:
                    close_px, result = tp, "win"
                pnl = (close_px - entry) / entry * 100.0
                if not is_long:
                    pnl = -pnl
                mae = (worst - entry) / entry * 100.0 * (1 if is_long else -1)
                mfe = (best - entry) / entry * 100.0 * (1 if is_long else -1)
                return {
                    "result": result,
                    "close_price": close_px,
                    "pnl_pct": pnl,
                    "close_ms": ts,
                    "mae_pct": mae,
                    "mfe_pct": mfe,
                }
        return None

    # ------------------------------------------------------------------

    def reconcile(self) -> dict:
        now = datetime.now(timezone.utc)
        summary = {
            "checked": 0, "closed_win": 0, "closed_loss": 0,
            "still_open": 0, "skipped_recent": 0, "errors": 0,
            "details": [],
        }
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM signals_log "
                "WHERE result = 'open' AND closed_at IS NULL "
                "ORDER BY id ASC"
            ).fetchall()
            open_signals = [dict(r) for r in rows]
        finally:
            conn.close()

        for sig in open_signals:
            summary["checked"] += 1
            try:
                iv, bar_min, bar_hr = self._bar_for(sig)
                created = _parse_dt(sig["created_at"])
                created_ms = int(created.timestamp() * 1000)

                # Skip very recent signals — the live bot may still own them.
                age_min = (now - created).total_seconds() / 60.0
                if age_min < self.skip_recent_bars * bar_min:
                    summary["skipped_recent"] += 1
                    summary["details"].append(
                        {"id": sig["id"], "action": "skipped_recent",
                         "age_min": round(age_min, 1)}
                    )
                    continue

                sym = normalize_symbol(sig["symbol"])
                entry = float(sig["entry_price"])
                sl = float(sig["stop_loss"])
                tp = float(sig["take_profit"])
                direction = str(sig["direction"])

                bars = self._src.bars(sym, iv, created, now)
                outcome = self._evaluate(
                    direction, entry, sl, tp, bars, created_ms
                )

                if outcome is None:
                    summary["still_open"] += 1
                    summary["details"].append(
                        {"id": sig["id"], "action": "still_open",
                         "bars_seen": len(bars)}
                    )
                    continue

                close_dt = datetime.fromtimestamp(
                    outcome["close_ms"] / 1000, tz=timezone.utc
                )
                dur_min = int((close_dt - created).total_seconds() / 60)
                risk = abs(entry - sl)
                rr = abs(tp - entry) / risk if risk > 0 else None

                detail = {
                    "id": sig["id"], "symbol": sig["symbol"],
                    "direction": direction, "result": outcome["result"],
                    "entry": entry, "close_price": outcome["close_price"],
                    "pnl_pct": round(outcome["pnl_pct"], 4),
                    "duration_minutes": dur_min,
                    "closed_at": close_dt.isoformat(),
                    "action": "dry_run" if self.dry_run else "closed",
                }
                summary["details"].append(detail)
                summary[
                    "closed_win" if outcome["result"] == "win" else "closed_loss"
                ] += 1

                if self.dry_run:
                    continue

                conn = self._connect()
                try:
                    # Idempotency guard: only close if still open.
                    conn.execute(
                        """UPDATE signals_log SET
                               result = ?, close_price = ?, pnl_pct = ?,
                               closed_at = ?, duration_minutes = ?,
                               rr_ratio = ?, mae_pct = ?, mfe_pct = ?
                           WHERE id = ? AND result = 'open'
                                 AND closed_at IS NULL""",
                        (
                            outcome["result"], outcome["close_price"],
                            outcome["pnl_pct"], close_dt.isoformat(),
                            dur_min, rr, outcome["mae_pct"],
                            outcome["mfe_pct"], sig["id"],
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

            except Exception as exc:  # noqa: BLE001 — never abort the batch
                summary["errors"] += 1
                summary["details"].append(
                    {"id": sig.get("id"), "action": "error", "error": str(exc)}
                )
                _log.warning("Reconcile error sig={i}: {e}", i=sig.get("id"), e=exc)

        if not self.dry_run and (summary["closed_win"] or summary["closed_loss"]):
            self._update_daily_stats()
            self._refresh_performance_cache()

        _log.info(
            "Reconcile done | db={db} checked={c} win={w} loss={l} "
            "open={o} skipped={s} err={e}",
            db=self.db_path, c=summary["checked"], w=summary["closed_win"],
            l=summary["closed_loss"], o=summary["still_open"],
            s=summary["skipped_recent"], e=summary["errors"],
        )
        return summary

    # ------------------------------------------------------------------

    def _update_daily_stats(self) -> None:
        """Recompute daily_stats rows from closed signals (idempotent upsert)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT date(closed_at) AS d,
                          COALESCE(timeframe,'4h') AS tf,
                          COUNT(*) AS n,
                          SUM(CASE WHEN result='win'  THEN 1 ELSE 0 END) AS w,
                          SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) AS l,
                          COALESCE(SUM(pnl_pct),0)    AS pnl
                   FROM signals_log
                   WHERE result IN ('win','loss') AND closed_at IS NOT NULL
                   GROUP BY d, tf"""
            ).fetchall()
            eq = self.initial_capital
            for r in rows:
                eq *= 1.0 + (float(r["pnl"]) / 100.0)
                conn.execute(
                    """INSERT INTO daily_stats
                         (date, timeframe, symbol, equity, daily_pnl_pct,
                          signals_count, wins, losses)
                       VALUES (?, ?, 'BTCUSDT', ?, ?, ?, ?, ?)
                       ON CONFLICT(date, timeframe, symbol) DO UPDATE SET
                          equity=excluded.equity,
                          daily_pnl_pct=excluded.daily_pnl_pct,
                          signals_count=excluded.signals_count,
                          wins=excluded.wins, losses=excluded.losses""",
                    (r["d"], r["tf"], eq, float(r["pnl"]),
                     int(r["n"]), int(r["w"]), int(r["l"])),
                )
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            _log.warning("daily_stats update failed: {e}", e=exc)
        finally:
            conn.close()

    def _refresh_performance_cache(self) -> None:
        """Best-effort: let StatsEngine recompute performance_cache."""
        try:
            from src.analytics.stats_engine import StatsEngine

            StatsEngine([self.db_path], self.initial_capital).compute_all()
        except Exception as exc:  # noqa: BLE001 — optional, never fatal
            _log.debug("performance_cache refresh skipped: {e}", e=exc)
