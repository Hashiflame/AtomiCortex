"""Scratch: train/serve parity for the 4H path.

Offline reference: the training parquet row at time T (full history, warmup-trimmed).
Live emulation: build_from_buffer() on the SAME raw OHLCV,
but truncated to the live buffer depth (bar_buffer_4h maxlen=400),
funding/metrics from the offline frames (best case — live can only be worse).

We compare the regime/vol features that feed BOTH the model and the
regime router, at 60 random timestamps in the last year of data.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/hashiflame/AtomiCortex")
import numpy as np, polars as pl
from src.features.feature_pipeline import FeaturePipeline

TRAIN = pl.read_parquet("/home/hashiflame/AtomiCortex/data/features/ml_features/BTCUSDT_4h_features.parquet")
print("train rows:", len(TRAIN), "cols:", len(TRAIN.columns))

# raw OHLCV needed by build_from_buffer: reconstruct from the training parquet
RAW_COLS = ["open_time","open","high","low","close","volume","taker_buy_volume"]
have = [c for c in RAW_COLS if c in TRAIN.columns]
print("raw cols available:", have)
raw = TRAIN.select([c for c in have])

pipe = FeaturePipeline.__new__(FeaturePipeline)
pipe.data_store = None; pipe.symbol = "BTCUSDT"; pipe.interval = "4h"

CHECK = ["hurst","adx","atr_pct","atr_percentile","trend_strength","regime_confidence",
         "cvd","cvd_slope_12","volume_zscore","vwap_4h","price_to_vwap","returns_6",
         "funding_zscore_30d","oi_zscore","basis_approx"]
CHECK = [c for c in CHECK if c in TRAIN.columns]

rng = np.random.default_rng(3)
n = len(raw)
idxs = sorted(rng.choice(np.arange(n-1500, n-10), size=40, replace=False))
BUF = 400  # live bar_buffer_4h maxlen

rows = []
regime_mismatch = 0
for i in idxs:
    buf = raw.slice(i-BUF+1, BUF)
    live = pipe.build_from_buffer(buf, single_row=True)
    off = TRAIN.slice(i, 1)
    rec = {}
    for c in CHECK:
        lv = live[c][0] if c in live.columns else None
        ov = off[c][0]
        rec[c] = (lv, ov)
    if "regime" in live.columns and "regime" in off.columns:
        if live["regime"][0] != off["regime"][0]:
            regime_mismatch += 1
    rows.append(rec)

print(f"\nregime label mismatch: {regime_mismatch}/{len(idxs)} sampled bars")
print(f"{'feature':<22}{'mean|rel diff|':>14}{'max|rel diff|':>14}{'n_exact':>9}")
for c in CHECK:
    diffs = []
    exact = 0
    for r in rows:
        lv, ov = r[c]
        if lv is None or ov is None: continue
        if lv == ov: exact += 1
        denom = max(abs(ov), 1e-9)
        diffs.append(abs(lv-ov)/denom)
    if diffs:
        print(f"{c:<22}{np.mean(diffs):>14.4f}{np.max(diffs):>14.4f}{exact:>9}")
