"""AtomiCortex data catalog — loads Parquet data and converts to Nautilus objects."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from nautilus_trader.model.currencies import BTC, ETH, SOL, USDT
from nautilus_trader.model.data import Bar, BarSpecification, BarType, TradeTick
from nautilus_trader.model.enums import (
    AggregationSource,
    AggressorSide,
    BarAggregation,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TradeId
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Price, Quantity

_SPECS: dict[str, dict] = {
    "BTCUSDT": {
        "base": BTC,
        "price_precision": 1,
        "size_precision": 3,
        "price_increment": 0.1,
        "size_increment": 0.001,
    },
    "ETHUSDT": {
        "base": ETH,
        "price_precision": 2,
        "size_precision": 3,
        "price_increment": 0.01,
        "size_increment": 0.001,
    },
    "SOLUSDT": {
        "base": SOL,
        "price_precision": 3,
        "size_precision": 0,
        "price_increment": 0.001,
        "size_increment": 1.0,
    },
}

_INTERVAL_MAP: dict[str, tuple[BarAggregation, int]] = {
    "1m": (BarAggregation.MINUTE, 1),
    "5m": (BarAggregation.MINUTE, 5),
    "15m": (BarAggregation.MINUTE, 15),
    "1h": (BarAggregation.HOUR, 1),
    "4h": (BarAggregation.HOUR, 4),
    "1d": (BarAggregation.DAY, 1),
}


class AtomiCortexCatalog:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def get_instrument(
        self,
        symbol: str,
        maker_fee: Decimal | None = None,
        taker_fee: Decimal | None = None,
    ) -> CryptoPerpetual:
        spec = _SPECS[symbol]
        pp = spec["price_precision"]
        sp = spec["size_precision"]
        instrument_id = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
        return CryptoPerpetual(
            instrument_id=instrument_id,
            raw_symbol=Symbol(f"{symbol}-PERP"),
            base_currency=spec["base"],
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            price_precision=pp,
            size_precision=sp,
            price_increment=Price(spec["price_increment"], precision=pp),
            size_increment=Quantity(spec["size_increment"], precision=sp),
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            ts_event=0,
            ts_init=0,
        )

    def load_bar_data(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        aggregation, step = _INTERVAL_MAP[interval]
        instrument_id = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
        bar_spec = BarSpecification(step, aggregation, PriceType.LAST)
        bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)

        spec = _SPECS[symbol]
        pp = spec["price_precision"]
        sp = spec["size_precision"]
        min_size = 10 ** -sp if sp > 0 else 1.0

        pattern = str(
            self.data_dir
            / f"exchange=BINANCE_UM/symbol={symbol}/klines_{interval}/**/*.parquet"
        )
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        df = (
            pl.scan_parquet(pattern, hive_partitioning=False)
            .filter(
                (pl.col("open_time") >= start_ms) & (pl.col("open_time") < end_ms)
            )
            .select(["open_time", "open", "high", "low", "close", "volume"])
            .sort("open_time")
            .collect()
        )

        bars: list[Bar] = []
        for row in df.iter_rows(named=True):
            ts_ns = row["open_time"] * 1_000_000
            bars.append(
                Bar(
                    bar_type=bar_type,
                    open=Price(row["open"], precision=pp),
                    high=Price(row["high"], precision=pp),
                    low=Price(row["low"], precision=pp),
                    close=Price(row["close"], precision=pp),
                    volume=Quantity(max(row["volume"], min_size), precision=sp),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
            )
        return bars

    def load_trade_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        sample_size: int = 100_000,
    ) -> list[TradeTick]:
        instrument_id = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
        spec = _SPECS[symbol]
        pp = spec["price_precision"]
        sp = spec["size_precision"]
        min_size = 10 ** -sp if sp > 0 else 1.0

        pattern = str(
            self.data_dir
            / f"exchange=BINANCE_UM/symbol={symbol}/agg_trades/**/*.parquet"
        )
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        df = (
            pl.scan_parquet(pattern, hive_partitioning=False)
            .filter(
                (pl.col("transact_time") >= start_ms)
                & (pl.col("transact_time") < end_ms)
            )
            .select(
                ["agg_trade_id", "price", "quantity", "transact_time", "is_buyer_maker"]
            )
            .sort("transact_time")
            .head(sample_size)
            .collect()
        )

        ticks: list[TradeTick] = []
        for row in df.iter_rows(named=True):
            ts_ns = row["transact_time"] * 1_000_000
            aggressor = (
                AggressorSide.SELLER if row["is_buyer_maker"] else AggressorSide.BUYER
            )
            ticks.append(
                TradeTick(
                    instrument_id=instrument_id,
                    price=Price(row["price"], precision=pp),
                    size=Quantity(max(row["quantity"], min_size), precision=sp),
                    aggressor_side=aggressor,
                    trade_id=TradeId(str(row["agg_trade_id"])),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
            )
        return ticks
