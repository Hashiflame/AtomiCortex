import polars as pl
import pickle
import numpy as np

print("=" * 60)
print("ДИАГНОСТИКА: ORB 15m модель")
print("=" * 60)

with open("data/models/15m/orb_model_15m.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["booster"]
print(f"\nregime={bundle.get('regime')}  symbols={bundle.get('symbols')}")
print(f"feature_columns ({len(bundle['feature_columns'])}): {bundle['feature_columns']}")

importance = model.feature_importance(importance_type="gain")
feat_names = model.feature_name()
top20 = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:20]
print("\nТоп-20 фич по gain importance:")
for i, (name, imp) in enumerate(top20, 1):
    print(f"  {i:2d}. {name}: {imp:.1f}")

df_btc = pl.read_parquet("data/features/symbol=BTCUSDT/interval=15m/dataset_orb.parquet", hive_partitioning=False)
print(f"\nBTC датасет: {df_btc.shape[0]} строк, {df_btc.shape[1]} колонок")

suspicious_keywords = ["future", "label", "return", "barrier", "target", "exit", "profit", "pnl", "win"]
suspicious = [c for c in df_btc.columns if any(k in c.lower() for k in suspicious_keywords)]
print(f"\nПодозрительные колонки: {suspicious}")

if "future_return" in df_btc.columns:
    fr = df_btc["future_return"]
    print(f"\nfuture_return:")
    print(f"  min={fr.min():.6f}, max={fr.max():.6f}")
    print(f"  mean={fr.mean():.6f}, std={fr.std():.6f}")
    print(f"  unique values: {fr.n_unique()}")
    target = df_btc["target"].cast(pl.Float64)
    corr = float(np.corrcoef(fr.to_numpy(), target.to_numpy())[0, 1])
    print(f"  корреляция с target: {corr:.4f}")
    if abs(corr) > 0.9:
        print("  ⚠️  КРИТИЧНО: future_return почти идеально коррелирует с target!")

print(f"\nTarget distribution:")
print(df_btc["target"].value_counts().sort("target"))

model_features = bundle["feature_columns"]
print(f"\nМодель обучена на {len(model_features)} фичах")
leaky_in_model = [f for f in model_features if any(k in f.lower() for k in suspicious_keywords)]
print(f"Подозрительные фичи В МОДЕЛИ: {leaky_in_model}")

available_features = [f for f in model_features if f in df_btc.columns]
missing_features = [f for f in model_features if f not in df_btc.columns]
print(f"\nДоступных фич для предсказания: {len(available_features)}/{len(model_features)}")
if missing_features:
    print(f"Отсутствующих фич: {missing_features[:10]}")

if len(available_features) == len(model_features):
    X = df_btc[available_features].to_numpy()
    proba = model.predict(X)
    print(f"\nРаспределение предсказаний (proba P(UP)):")
    print(f"  min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"  mean={proba.mean():.4f}, std={proba.std():.4f}")
    print(f"  > 0.58 (UP signal):  {(proba > 0.58).mean():.1%}")
    print(f"  < 0.42 (DOWN signal): {(proba < 0.42).mean():.1%}")
    print(f"  0.42-0.58 (no signal): {((proba >= 0.42) & (proba <= 0.58)).mean():.1%}")
    target_arr = df_btc["target"].to_numpy()
    pred_direction = np.where(proba > 0.58, 1, np.where(proba < 0.42, -1, 0))
    traded = pred_direction != 0
    if traded.sum() > 0:
        correct = (pred_direction[traded] == target_arr[traded])
        print(f"\nWin rate на traded барах: {correct.mean():.1%} ({traded.sum()} trades)")
    print(f"\nНаправленность:")
    print(f"  UP predictions (>0.5):   {(proba > 0.5).mean():.1%}")
    print(f"  DOWN predictions (<=0.5): {(proba <= 0.5).mean():.1%}")

try:
    df_multi = pl.read_parquet("data/features/symbol=MULTI/interval=15m/dataset_orb.parquet", hive_partitioning=False)
    print(f"\nMULTI датасет: {df_multi.shape}")
    print(f"Target в MULTI:")
    print(df_multi["target"].value_counts().sort("target"))
    if "symbol" in df_multi.columns:
        print(f"\nСимволы в MULTI:")
        print(df_multi["symbol"].value_counts())
    tcol = None
    for c in ["open_time", "ts", "timestamp", "time"]:
        if c in df_multi.columns:
            tcol = c
            break
    print(f"\nВременная колонка: {tcol}")
    if tcol and "symbol" in df_multi.columns:
        print(f"\nПериоды по символам:")
        for sym in df_multi["symbol"].unique().to_list():
            sym_df = df_multi.filter(pl.col("symbol") == sym)
            ot = sym_df[tcol]
            print(f"  {sym}: {ot.min()} → {ot.max()} ({len(sym_df)} rows)")
        # Q3: BTC 2025+ in MULTI train
        df_multi_btc = df_multi.filter(pl.col("symbol") == "BTCUSDT")
        btc_2025 = df_multi_btc.filter(pl.col(tcol) >= 1735689600000)
        print(f"\nQ3: BTC строк в MULTI с 2025+ (epoch ms): {len(btc_2025)}")
except Exception as e:
    import traceback
    print(f"\nMULTI датасет: ошибка {e}")
    traceback.print_exc()

print("\n--- COLUMNS BTC ---")
print(df_btc.columns)
