#!/usr/bin/env python
"""
scripts/run_live_feed.py

Run the AtomiCortex live Binance Futures WebSocket feed.

Examples
--------
    # Test for 30 seconds, then stop automatically
    python scripts/run_live_feed.py --symbols BTCUSDT --duration 30

    # Multiple symbols, run until Ctrl-C
    python scripts/run_live_feed.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingestion.live_feed import LiveFeedManager
from src.logger import get_logger, setup_logging


@click.command()
@click.option(
    "--symbols",
    default="BTCUSDT",
    show_default=True,
    help="Comma-separated Binance Futures symbols.",
)
@click.option(
    "--duration",
    default=None,
    type=int,
    help="Stop automatically after N seconds. Default: run until Ctrl-C.",
)
@click.option(
    "--buffer-size",
    default=1_000,
    show_default=True,
    help="Max trades retained per symbol in the in-memory buffer.",
)
def main(symbols: str, duration: int | None, buffer_size: int) -> None:
    """Start Binance Futures live WebSocket feed (TRADES, L2_BOOK, FUNDING)."""
    setup_logging()
    log = get_logger(__name__)

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    dur_label = f"{duration}s" if duration else "until Ctrl-C"
    click.echo(
        f"\n{'─'*56}\n"
        f"  Symbols  : {', '.join(symbol_list)}\n"
        f"  Duration : {dur_label}\n"
        f"  Buffer   : {buffer_size} trades/symbol\n"
        f"{'─'*56}\n"
    )

    manager = LiveFeedManager(max_trades_buffer=buffer_size)

    try:
        manager.run(symbols=symbol_list, duration=duration)
    except KeyboardInterrupt:
        pass

    click.echo(f"\n{'─'*56}")
    click.echo(f"  Reconnects     : {manager.reconnect_count}")

    for sym in symbol_list:
        from src.ingestion.live_feed import _to_cf_symbol
        cf = _to_cf_symbol(sym)
        n_trades = len(manager.tick_buffer.get_recent_trades(cf))
        book = manager.tick_buffer.get_current_book(cf)
        funding = manager.tick_buffer.get_latest_funding(cf)
        best_bid = book["bids"][0][0] if book.get("bids") else "—"
        best_ask = book["asks"][0][0] if book.get("asks") else "—"
        click.echo(
            f"  {sym:<10} trades={n_trades:>5}  "
            f"bid={best_bid}  ask={best_ask}  "
            f"funding={funding:.6f}"
        )

    click.echo(f"{'─'*56}\n")


if __name__ == "__main__":
    main()
