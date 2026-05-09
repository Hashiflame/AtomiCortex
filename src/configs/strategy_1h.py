"""
Configuration for 1H ML trading strategy.
All parameters tuned for 1H timeframe specifics.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MLStrategyConfig1H:
    """Strategy configuration for the 1-hour ML trading pipeline.

    Key differences from 4H MLStrategyConfig:
    - Shorter warmup (100 bars vs 300)
    - Lower confidence threshold (0.63 vs 0.65)
    - 2-bar forward prediction (2 hours)
    - Tighter ATR threshold (0.4 vs 0.5)
    - Shorter max hold (6 bars = 6 hours)
    - Stricter fee filter (4.0x vs 3.0x)
    - Separate DB and model paths
    """

    # Timeframe
    timeframe: str = "1h"
    bar_type_str: str = "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL"

    # Warmup
    warmup_bars: int = 100  # 4H uses 300

    # Signal thresholds
    confidence_threshold: float = 0.63  # 4H uses 0.65

    # Target construction
    forward_bars: int = 2              # predict 2H ahead
    atr_threshold_multiplier: float = 0.4  # 4H uses 0.5

    # Position management
    max_hold_bars: int = 6             # max 6 hours
    atr_sl_multiplier: float = 1.3     # 4H uses 1.5
    atr_tp_multiplier: float = 2.0

    # Risk
    max_concurrent_positions: int = 2
    daily_loss_limit: float = -0.04    # 4H uses -0.03
    min_rr_ratio: float = 1.3

    # Fee filter (stricter than 4H)
    min_expected_return_vs_fees: float = 4.0  # 4H uses 3.0

    # Regimes where we trade (must match RegimeDetector output — lowercase)
    valid_regimes: List[str] = field(default_factory=lambda: [
        "trend_up", "trend_down",
        "high_vol",
    ])

    # Regimes to skip (must match RegimeDetector output — lowercase)
    skip_regimes: List[str] = field(default_factory=lambda: [
        "range",
        "unknown",
    ])

    # Session filters
    skip_pre_funding: bool = True      # skip 2H before funding mark

    # Isolation — отдельная БД от 4H!
    signal_db_path: str = "data/atomicortex_1h.db"
    heartbeat_key: str = "bot_1h_heartbeat"

    # Model paths
    trend_model_path: str = "data/models/1h/trend_model_1h.pkl"
    high_vol_model_path: str = "data/models/1h/high_vol_model_1h.pkl"
