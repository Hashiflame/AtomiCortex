"""
Configuration for 15M ML trading strategy.
All parameters tuned for 15M timeframe specifics.

Key differences from 1H:
- 4-bar forward prediction (4 × 15m = 1 hour ahead)
- Tighter ATR threshold (0.35 vs 0.4)
- Higher confidence threshold (0.67 vs 0.63) — compensates for noise
- Only 1 concurrent position (vs 2 on 1H)
- Stricter fee filter (5× vs 4×)
- Two model types: trend + ORB (vs trend + high_vol on 1H)
- ORB breakout and session trap checked at signal time (not via regime string)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MLStrategyConfig15M:
    """Strategy configuration for the 15-minute ML trading pipeline.

    Key differences from 1H MLStrategyConfig1H:
    - Warmup 200 bars (same, but = 50 hours vs 200 hours)
    - Higher confidence (0.67 vs 0.63) — more noise at 15m
    - 4-bar forward prediction (1 hour)
    - Tighter ATR threshold (0.35 vs 0.4)
    - Max hold 8 bars = 2 hours (vs 6 hours)
    - Stricter fee filter (5.0× vs 4.0×)
    - Separate DB and model paths
    - Two model types: trend + orb
    """

    # Timeframe
    timeframe: str = "15m"
    bar_type_str: str = "BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL"

    # Warmup
    warmup_bars: int = 200  # 200 bars × 15m = 50 hours

    # Signal thresholds — higher than 1H (compensate noise)
    confidence_threshold: float = 0.58  # was 0.67; recalibrate after retrain

    # Target construction — fixed-horizon (legacy; still used by the 4H
    # DatasetBuilder path / mlflow param logging)
    forward_bars: int = 4               # 4 × 15min = 1 hour ahead
    atr_threshold_multiplier: float = 0.35  # 1H uses 0.4, 4H uses 0.5

    # Target construction — triple-barrier (AFML Ch.3, active for 15m,
    # both trend + orb). tb_ prefix disambiguates from max_hold_bars.
    tb_pt_multiplier: float = 2.0       # profit-taking barrier = 2.0×ATR
    tb_sl_multiplier: float = 1.0       # stop-loss barrier   = 1.0×ATR
    tb_max_holding_bars: int = 8        # vertical barrier (2 hours)

    # Position management
    max_hold_bars: int = 8              # max 2 hours
    atr_sl_multiplier: float = 1.2      # 1H uses 1.3
    atr_tp_multiplier: float = 2.0

    # Risk — strictest of all timeframes
    max_concurrent_positions: int = 1   # only 1 position at a time
    daily_loss_limit: float = -0.04
    min_rr_ratio: float = 1.3

    # Fee filter (strictest)
    min_expected_return_vs_fees: float = 5.0  # 1H uses 4.0, 4H uses 3.0

    # Session trap: skip first/last N bars of each session
    session_trap_bars: int = 2

    # Regimes where we trade (must match RegimeDetector15M output — lowercase)
    valid_regimes: List[str] = field(default_factory=lambda: [
        "trend_up", "trend_down",
    ])

    # Regimes to skip (must match RegimeDetector15M output — lowercase)
    skip_regimes: List[str] = field(default_factory=lambda: [
        "range",
        "high_vol",
    ])

    # Pre-funding skip
    skip_pre_funding: bool = True

    # Isolation — separate DB from 1H and 4H!
    signal_db_path: str = "data/atomicortex_15m.db"
    heartbeat_key: str = "bot_15m_heartbeat"

    # Model paths — two model types
    trend_model_path: str = "data/models/15m/trend_model_15m.pkl"
    orb_model_path: str = "data/models/15m/orb_model_15m.pkl"
