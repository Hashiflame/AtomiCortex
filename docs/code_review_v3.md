# AtomiCortex v3.0 — Полный Code Review

> Дата начала: 2026-05-23
> Цель: найти все типы ошибок которые тесты не поймают
> Метод: глубокий анализ каждого файла + сравнение с best practices

---

## Раздел 1: ML Pipeline

### Файлы:
- src/models/dataset_builder.py
- src/models/lgbm_trainer.py
- src/models/statistical_tests.py
- src/models/training_pipeline.py
- src/models/ml_validator.py
- src/models/temporal_split.py

---

## 1.1 dataset_builder.py

### [A — Lookahead] `compute_uniqueness_weights` assumes all labels span exactly `max_holding`
**Файл:** src/models/dataset_builder.py:282-342
**Серьёзность:** ВЫСОКАЯ
**Описание:** AFML Ch.4 (López de Prado) defines sample uniqueness using the *actual* event horizon `t1_i` — the bar at which each triple-barrier event exits (could be PT touch, SL touch, or vertical). Here the code hard-codes `[i, i + h - 1]` for every label, i.e. it assumes every label is held the full vertical-barrier horizon. With pt/sl barriers, real exits are usually earlier, so concurrency `c_t` is systematically over-counted, and uniqueness `u_i` is biased toward the same value for every label → the weighting collapses toward uniform.
**Best Practice:** mlfinlab's `mlfinlab.sample_weights.get_av_uniqueness_from_triple_barrier` uses the actual `t1` returned by `triple_barrier.apply_triple_barrier`. The function here even has access to the realized exit bar (it's encoded in `future_return` via the barrier touch loop in `apply_triple_barrier`), but discards it.
**Влияние на trading:** Sample weights don't reflect true label overlap → the model receives a *miscalibrated* effective sample size. AFML shows this materially affects Sharpe and OOS stability of meta-labeling.
**Рекомендация:** Return the touch index from `apply_triple_barrier` (already known internally) and feed it to a true `get_concurrent_events` / `get_av_uniqueness` implementation.

### [A — Lookahead] Uniqueness weights ignore regime filtering
**Файл:** src/models/dataset_builder.py:344-366 (consumed by lgbm_trainer.py:337-344)
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `compute_uniqueness_weights_by_symbol` iterates each symbol and assumes consecutive *rows* in `df` correspond to consecutive *bars in time*. But the trainer applies the regime filter (lgbm_trainer.py:261) **before** computing weights, so the filtered df has time-gaps. The formula `min(t+1, h)` treats row-distance as bar-distance, which is no longer true after filtering. Concurrency estimates are then numerically wrong (over- or under-counted depending on regime density).
**Best Practice:** Concurrency must be computed in the original (unfiltered, contiguous-time) bar space, then restricted to the filtered rows.
**Влияние на trading:** Especially severe for `regime=range` and `high_vol` where filtered rows are sparse — weights become near-meaningless.
**Рекомендация:** Compute uniqueness on the pre-filter df indexed by `open_time`, then look up weights by `open_time` for the filtered rows.

### [A — Lookahead] `create_target` legacy path: `forward_bars` is rounded silently
**Файл:** src/models/dataset_builder.py:204
**Серьёзность:** НИЗКАЯ
**Описание:** `df.head(len(df) - forward_bars)` drops trailing rows where `future_return` is NaN. Fine, but combined with the per-symbol concat in `lgbm_trainer.prepare_data`, the *last* bar of *every* symbol becomes the boundary and uniqueness weights still treat the next symbol's first bar as a temporal neighbor.
**Рекомендация:** Reset/segment all rolling/forward-looking computations on the per-symbol boundary explicitly.

### [E — Architecture] Hard-coded `_EXCLUDE_COLUMNS` is brittle vs. feature evolution
**Файл:** src/models/dataset_builder.py:25-62
**Серьёзность:** СРЕДНЯЯ
**Описание:** Any new leaky/forward-looking feature added in `src/features/*` will silently slip into training because `get_feature_columns` is allow-by-default for numeric dtypes.
**Best Practice:** Use opt-in feature whitelist (the `ModelConfig.feature_whitelist` exists but is optional). For non-whitelist training, add explicit tagging of feature columns at the FeaturePipeline level (e.g., `feature.metadata["leaky"] = True`).

### [B — Math] Uniqueness normalization can over-correct mean to 1.0
**Файл:** src/models/dataset_builder.py:339-342
**Серьёзность:** НИЗКАЯ
**Описание:** Normalizing `u = u / u_mean` forces mean weight = 1.0. AFML keeps weights in `(0, 1]` and uses them as *fractional* uniqueness, with sample-size adjustment via `class_weight`. The normalization changes the absolute scale of weights and changes its interaction with LightGBM's leaf statistics (`min_child_weight`, `min_split_gain`).
**Рекомендация:** Either (a) keep raw `u ∈ (0,1]` and adjust class weights separately, or (b) document why this normalization preserves effective N.

---

## 1.2 lgbm_trainer.py

### [C — LightGBM] `np.nan_to_num` destroys LightGBM's native NaN handling
**Файл:** src/models/lgbm_trainer.py:629
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)` converts every NaN to 0.0 before feeding into LightGBM. LightGBM natively learns optimal split direction for missing values (its core advantage over XGBoost in older versions). Replacing NaN with 0.0 conflates "missing" with "value zero", which is catastrophic for ratio/momentum/z-score features where 0.0 is a meaningful interior value (e.g., a perfectly flat momentum vs. unknown momentum during warm-up).
**Best Practice:** LightGBM docs explicitly: "LightGBM enables the missing value handle by default." Pass NaN through. To compare: every Kaggle financial competition top-3 since 2022 leaves NaN intact (Optiver, Jane Street, etc.).
**Влияние на trading:** Features computed via rolling windows have NaN warm-up periods → those rows now look like "feature = 0" instead of "feature unknown". Boosters likely build spurious splits on rolling-feature NaN→0 sentinels in the early bars of every symbol.
**Рекомендация:** Remove `nan_to_num`. Optionally only handle posinf/neginf (true bugs); leave NaN alone.

### [A — Lookahead] Early-stopping validation set has zero embargo from test
**Файл:** src/models/lgbm_trainer.py:321-326
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Val set is "the last 10-15% of train" with no embargo. With triple-barrier labels of horizon `max_holding=6`, the val tail's labels were constructed using closes that extend up to 6 bars into the *test* set (i.e. across the train/test cut). Result: early-stopping decisions are made on a val set whose labels share information with the test set. This biases best_iteration downward (model trained until just before that future-leakage advantage saturates), inflating OOS metrics.
**Best Practice:** AFML Ch.7 "Cross-validation in finance" — purge + embargo both *between train and val* and *between val and test*. mlfinlab's `PurgedKFold` adds embargo on both sides.
**Влияние на trading:** OOS WR/PF are biased high by ~few percentage points; effect amplifies with larger `max_holding`.
**Рекомендация:** Drop the last `max_holding` rows from val_fit, and drop the last `max_holding` rows from val before train (so train, val, test are mutually purged).

### [A — Lookahead] Regime filter happens *after* target creation but *before* uniqueness/concat — breaks time continuity
**Файл:** src/models/lgbm_trainer.py:259-273
**Серьёзность:** ВЫСОКАЯ
**Описание:** After `_filter_by_regime`, consecutive rows are no longer consecutive bars. Subsequently the per-symbol temporal split (`sym_df.head(train_n)`) is row-based, not time-based — for a regime that bursts in 2024 vs 2025, the row-based 80/20 split can put 2024 trend-bars in train and most 2025 trend-bars in test, *or vice versa*, depending on regime distribution drift. This is not a pure lookahead, but it produces non-comparable train/test windows.
**Best Practice:** Use `temporal_split.temporal_split_multi` (which already exists in this repo!) with a wall-clock `oos_start_ms`. The trainer doesn't use it.
**Влияние на trading:** Test set may be heavily skewed in time, making evaluation results regime-distribution-dependent rather than skill-dependent.
**Рекомендация:** Use `temporal_split_multi(df, oos_start_ms, embargo_bars=max_holding)` in `prepare_data`. Apply regime filter *after* the split.

### [C — LightGBM] Sample weight on val + balanced reweighting on val = distorted early-stopping signal
**Файл:** src/models/lgbm_trainer.py:330, 344
**Серьёзность:** СРЕДНЯЯ
**Описание:** Val weights are `balanced × uniqueness`. Early stopping then minimizes weighted logloss, which no longer corresponds to the OOS logloss the model will face in production (where every bar is weighted equally).
**Best Practice:** Train weights should be class-balanced; val weights should be uniform (or only carry uniqueness, never class balancing). See LightGBM advanced topics docs.
**Рекомендация:** Pass `weight=None` (or `weight=uniq_val` only) to `val_data`.

### [E — Architecture] Pickle of LightGBM Booster is version-brittle
**Файл:** src/models/lgbm_trainer.py:425-426
**Серьёзность:** ВЫСОКАЯ
**Описание:** `pickle.dump({"booster": booster, ...})` ties model loadability to the exact LightGBM minor version. A LightGBM upgrade (e.g., 4.3 → 4.5) can break model loading silently or with a vague error.
**Best Practice:** LightGBM docs explicitly recommend `booster.save_model(path)` + `lgb.Booster(model_file=path)` for cross-version portability. Wrap pickle around the metadata only.
**Влияние на trading:** Production model load may break after `pip install -U lightgbm`. Critical for live trading.
**Рекомендация:** Save booster via `booster.save_model(...)` and pickle only the metadata bundle. Pin LightGBM version in requirements.

### [E — Architecture] MLflow logs `config.lgbm_params` even when MTF_LGBM_PARAMS was used
**Файл:** src/models/lgbm_trainer.py:655-656
**Серьёзность:** СРЕДНЯЯ
**Описание:** When `use_mtf_params=True`, the actual training params come from `MTF_LGBM_PARAMS`, but MLflow logs `self.config.lgbm_params` — i.e. the *unused* defaults. Reproducibility from MLflow is broken for all MTF models.
**Рекомендация:** Log the `raw_params` dict that was actually used (line 370).

### [C — LightGBM] No `scale_pos_weight` audit vs. sample-weight balancing
**Файл:** src/models/lgbm_trainer.py:329 + 56-84 (MTF_LGBM_PARAMS)
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** Currently only `compute_sample_weight("balanced", ...)` is used and MTF params don't set `is_unbalance` or `scale_pos_weight`. Good — no double counting. Worth a unit test guarding against future regressions.

### [D — Time series] `early_stopping_rounds=100` on noisy financial data + 2000 trees
**Файл:** src/models/lgbm_trainer.py:77-79
**Серьёзность:** СРЕДНЯЯ
**Описание:** With 100-round patience and `learning_rate=0.02`, early stopping can wait through long noise plateaus. Combined with leakage-affected val (see above), it tends to overshoot best_iteration in noise. Recent literature (LGBM-style ensemble for finance, e.g. Tang et al. 2024) prefers `lr ≤ 0.01` with patience ≤ 30 on weekly/daily crypto.
**Рекомендация:** Once val embargo is fixed, halve patience and check the new best_iteration distribution.

### [E — Architecture] Validation `reference=train_data` shares categorical bins — fine, but no check that feature dtypes match between train and eval
**Файл:** src/models/lgbm_trainer.py:362-366, 568-631
**Серьёзность:** НИЗКАЯ
**Описание:** `_prepare_xy` casts to float64 + nan_to_num. If a feature dtype changes between training and inference (e.g., an int column becomes float at inference), no contract enforces this. Combined with absence of a feature-schema file alongside the pickle, this is silent breakage risk.
**Рекомендация:** Store dtype map alongside `feature_columns` in the model bundle and validate on load.

---

## 1.3 statistical_tests.py

### [B — Math] DSR uses wrong sample size in SE(SR)
**Файл:** src/models/statistical_tests.py:91-95
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `n_obs = len(sharpe_ratios)` is the *number of strategies/folds*, not the *number of return observations T*. The Mertens (2002) / López de Prado (2014) DSR formula expects `T` = number of *return observations* used to compute each SR (e.g., daily returns count). Dividing by `n_obs=5` instead of `T=hundreds` inflates `se_sr` by ~`sqrt(T/N)` ≈ 10×, which collapses DSR toward 0.5 regardless of true skill. Combined with the proxy path (sharpe_proxies has length = n_folds + n_windows ≈ 5-15), DSR is essentially never significant.
**Best Practice:** Bailey & López de Prado 2014, eq. (9): `SE(SR_hat) = sqrt((1 - γ3·SR + (γ4-1)/4·SR²)/(T-1))` where `T` is the number of return observations.
**Влияние на trading:** DSR scores systematically underestimate model quality — masks both real skill and over-fitting. The "passes ≥ 0.95" criterion likely cannot fire even on genuinely good models.
**Рекомендация:** Pass `T` (length of `per_fold_daily_returns` flattened, or n_bars) as a separate parameter. Compute SE with T-1 in denominator, not number of strategies.

### [B — Math] DSR kurtosis term uses wrong constant
**Файл:** src/models/statistical_tests.py:93
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Formula uses `(kurtosis - 1)/4 * best_sr ** 2`. The correct Mertens (2002) form is `(γ4 - 3)/4 * SR²` with γ4 = *raw* kurtosis (γ4=3 for Normal). The code default `kurtosis=3.0` should make this term zero — but with `(3-1)/4 = 0.5`, the term contributes `0.5 * SR²` of spurious variance, biasing DSR down further. Same constant is used in the real-Sharpe path where `sp_stats.kurtosis(daily_rets, fisher=False)` returns raw kurtosis (~3 for normal). Compounds the DSR collapse.
**Best Practice:** Use `(kurtosis - 3) / 4` with `fisher=False`, OR `kurtosis / 4` with `fisher=True` (excess kurtosis).
**Рекомендация:** Replace `(kurtosis - 1) / 4` with `(kurtosis - 3) / 4`.

### [B — Math] DSR skewness sign convention
**Файл:** src/models/statistical_tests.py:93
**Серьёзность:** ВЫСОКАЯ
**Описание:** Formula `1 - skewness * best_sr + (kurtosis-1)/4 * sr²`. Mertens (2002) is `1 - γ3·SR + (γ4-3)/4·SR²`. The skew sign here matches Mertens, but in combination with the kurtosis bug above the SE is wildly off. Also: code uses `best_sr` in the variance formula — Bailey/LdP use that strategy's *own* SR for the kurtosis correction, which is `best_sr` here, so this is consistent. OK on that detail.
**Рекомендация:** Fix kurtosis bug; skew is correct.

### [D — Time series] PBO implementation is not Combinatorially Symmetric Cross-Validation
**Файл:** src/models/statistical_tests.py:156-234
**Серьёзность:** КРИТИЧЕСКАЯ (methodological)
**Описание:** Bailey, Borwein, López de Prado, Zhu (2014) "The probability of backtest overfitting" defines PBO via CSCV: take T observations of N strategies, partition into S blocks, then for *every* combination of S/2 blocks (IS) vs the other S/2 blocks (OOS), rank strategies, observe if the *best IS* strategy underperforms median OOS. Take the logit, get the distribution. Here we have a single leave-one-out over a small number of folds — not CSCV at all. With n=4-5 folds, leave-one-out PBO is dominated by a single fold being "best" + above-median, which is statistically meaningless.
**Best Practice:** mlfinlab's `mlfinlab.backtest_statistics.probability_of_backtest_overfitting` implements CSCV correctly. The proper input is a matrix of *N strategies × T observations of P&L*, not 5 EvaluationResult summaries.
**Влияние на trading:** The current PBO output is uninterpretable; thresholding it at 0.30 is essentially noise.
**Рекомендация:** Implement CSCV — requires per-bar P&L for each strategy variant, not just one win_rate per fold. Until then, document the metric as "approximate" rather than "PBO".

### [B — Math] DSR proxy from `(WR - 0.5) × PF × 10`
**Файл:** src/models/statistical_tests.py:121-125, 405-413
**Серьёзность:** ВЫСОКАЯ
**Описание:** The fallback "Sharpe proxy" is dimensionally meaningless: `(win_rate_fraction - 0.5) * profit_factor * 10` has no statistical interpretation as a Sharpe ratio. Plugging this into DSR formulas pretends the proxy is annualized SR, which it isn't. Per-fold daily returns path is the only sound one; the proxy path should be removed or labeled "diagnostic-only".
**Рекомендация:** Require `per_fold_daily_returns`; deprecate the proxy or rename outputs to a non-DSR metric.

### [B — Math] t-stat mixes weighted mean with unweighted std
**Файл:** src/models/statistical_tests.py:278-285
**Серьёзность:** СРЕДНЯЯ
**Описание:** `weighted_mean = np.average(wr_arr, weights=n_arr)` but `std_wr = np.std(wr_arr, ddof=1)` (unweighted). The t-stat then mixes apples and oranges. Worse, "n_windows" is used in the denominator instead of "sqrt(total_trades)" — so a window with 1 trade and a window with 1000 trades count equally toward SE.
**Best Practice:** Either (a) compute t-stat on the full pool of per-trade outcomes (n = total trades), or (b) use weighted std consistent with weighted mean.
**Рекомендация:** Compute `successes = sum(round(wr_i/100 * n_i))`, then a single one-proportion z-test against 50%: `z = (p_hat - 0.5)/sqrt(0.5*0.5/n)`.

### [D — Time series] No Bonferroni / multiple-testing correction outside DSR
**Файл:** src/models/statistical_tests.py (whole file)
**Серьёзность:** СРЕДНЯЯ
**Описание:** DSR is supposed to be the multi-testing correction, but with its bugs above it isn't doing its job. Otherwise there's no Bonferroni on t-stat across regimes/timeframes (e.g., 3 regimes × 3 timeframes = 9 hypotheses).
**Рекомендация:** Add a multiplicity-aware p-value for the per-regime t-stats.

---

## 1.4 training_pipeline.py

### [E — Architecture] No persistence of run/config alongside model
**Файл:** src/models/training_pipeline.py:61-76
**Серьёзность:** ВЫСОКАЯ
**Описание:** `ModelConfig(regime=regime, symbols=symbols)` uses *all defaults* (forward_bars=1, no triple barrier, no uniqueness weights). The pipeline silently ignores all v3 features. Models trained via this pipeline will not be v3-compatible — yet they overwrite `{regime}_model.pkl`.
**Best Practice:** A training pipeline should require explicit config, or take a callable factory.
**Рекомендация:** Make `regimes` mapping `regime → ModelConfig` instead of bare list; require explicit config.

### [E — Architecture] Exceptions swallowed per regime — partial state shipped
**Файл:** src/models/training_pipeline.py:77-79
**Серьёзность:** СРЕДНЯЯ
**Описание:** If a regime training raises, the loop continues silently. Old model file from prior run remains. Caller has no easy way to detect partial completion.
**Рекомендация:** Track failed regimes and propagate. Don't silently fall back to stale weights.

---

## 1.5 ml_validator.py

### [A — Lookahead] Walk-forward bypasses triple-barrier when `use_triple_barrier=True`
**Файл:** src/models/ml_validator.py:362-401
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `_load_full_data` always calls `create_target(...)` — the legacy 1-bar sign(return) target — ignoring `trainer.config.use_triple_barrier`. So walk-forward validation of v3 models uses *legacy targets*, while the final production retrain uses *triple-barrier targets*. The walk-forward results therefore evaluate a different model than what ships.
**Влияние на trading:** Reported walk-forward WR/PF do not reflect production model behavior. The 60%-profitable-windows go-live gate is being checked on a different model.
**Рекомендация:** Branch on `trainer.config.use_triple_barrier` and call `create_target_triple_barrier` accordingly.

### [A — Lookahead] PurgedKFoldCV row-based slicing on multi-symbol concatenated data
**Файл:** src/models/ml_validator.py:149-152 + src/execution/walk_forward.py:73-89
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `PurgedKFoldCV.split(df)` slices by row index: `data.slice(0, train_end)`. But `_load_full_data` concatenates `[BTC..., ETH..., SOL...]` (multi-symbol diagonal concat), so row indices are *not* monotonic in time. Fold 1 train = all early BTC; Fold 1 test = late BTC + early ETH (after embargo of 1% rows). Time-ordered purging is *completely defeated*. Same root cause as the ML-018 incident that `temporal_split.py` was created to fix — but PurgedKFoldCV doesn't use that fix.
**Best Practice:** mlfinlab's PurgedKFold operates per-symbol with a proper time index, then aggregates per-fold metrics.
**Влияние на trading:** CV results in this codebase are massively optimistic — the same memorization pathology temporal_split.py was built to prevent is alive and well in CV.
**Рекомендация:** Either (a) call PurgedKFoldCV per symbol then aggregate, or (b) sort df by timestamp globally (interleaving symbols) and embargo by `max_holding` × n_symbols, or (c) implement a proper multi-asset purged CV.

### [A — Lookahead] Embargo of 1% of N rows is undersized vs. triple-barrier horizon
**Файл:** src/models/ml_validator.py:111 (default `embargo_pct=0.01`) + src/execution/walk_forward.py:74
**Серьёзность:** ВЫСОКАЯ
**Описание:** With h=6 bars and N=10k, 1% = 100 bars (OK). But with `regime=range` filtered N=500, 1% = 5 bars — less than `max_holding=6`, so the test fold's first few rows have labels constructed from the last train bars' future windows. Direct label leakage.
**Best Practice:** Embargo ≥ `max_holding` (AFML Ch.7).
**Рекомендация:** `embargo_bars = max(int(N*embargo_pct), max_holding)`.

### [E — Architecture] `confidence_threshold=0.35` default on a binary classifier is dead code
**Файл:** src/models/ml_validator.py:112
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** For binary, `confidence = max(p, 1-p) ≥ 0.5` always. A 0.35 threshold means every bar is a signal. Currently `evaluate()` uses `trainer.config.confidence_threshold` (0.55), so the MLValidator field is unused. Misleading.
**Рекомендация:** Delete the field, or rename + actually wire through.

### [E — Architecture] `n_signals = int(signal_rate × len(test_df))` double-rounds
**Файл:** src/models/ml_validator.py:321
**Серьёзность:** НИЗКАЯ
**Описание:** `signal_rate` was already rounded to 4 decimals in `evaluate`. Multiplying back and casting to int can introduce ±1 errors per window. Feed n_signals straight from `evaluate` (compute exact count there and return it).

---

## 1.6 temporal_split.py

### [A — Lookahead] Embargo applied to train only — correct for forward-looking labels, but not symmetric
**Файл:** src/models/temporal_split.py:70-73
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** `_embargo` drops only the last N rows of train. With triple-barrier (forward-looking only), this is correct — train labels constructed at bar `t-h..t-1` would otherwise peek across cut. No backward leak from test→train exists. Documented intent matches code. ✅

### [E — Architecture] No invariant check that `oos_start_ms` falls strictly inside the data range
**Файл:** src/models/temporal_split.py:42 + 95-99
**Серьёзность:** НИЗКАЯ
**Описание:** `compute_default_oos_start_ms` always produces a valid cutoff, but a user-supplied `oos_start_ms` from CLI might be after dataset end → empty test, caught only by an `assert`. Use a clear ValueError.

### [F — Production] No version stamp on the split decision
**Файл:** src/models/temporal_split.py
**Серьёзность:** СРЕДНЯЯ
**Описание:** The chosen `oos_start_ms` is not persisted with the trained model. Re-running validation later with a different dataset boundary silently changes "OOS" semantics. For reproducibility, the cutoff and embargo should be stamped into the model bundle (the trainer pickles a bundle but doesn't include this).

---

## Summary — ML Pipeline

### Counts by severity
- **КРИТИЧЕСКАЯ:** 8
  - dataset_builder: uniqueness ignores regime filter
  - lgbm_trainer: nan_to_num destroys NaN handling
  - lgbm_trainer: val set leaks into test via triple-barrier horizon
  - statistical_tests: DSR SE(SR) uses wrong sample size
  - statistical_tests: DSR kurtosis constant wrong (off by `0.5·SR²`)
  - statistical_tests: PBO is not CSCV — not the published method
  - ml_validator: walk-forward bypasses triple-barrier targets
  - ml_validator: PurgedKFoldCV row-slices multi-symbol concat (ML-018 pathology in CV)
- **ВЫСОКАЯ:** 8
- **СРЕДНЯЯ:** 6
- **НИЗКАЯ / INFO:** 6

### Top-5 most critical
1. **PurgedKFoldCV multi-symbol row-slicing (ml_validator.py:149 + walk_forward.py:73).** The same memorization-from-time-leakage bug that motivated `temporal_split.py` reappears in CV. All cross-validation results in v3 are likely inflated.
2. **Walk-forward uses legacy target, training uses triple-barrier (ml_validator.py:380).** Validation evaluates a different model than what ships → the go-live gate is meaningless for v3.
3. **`np.nan_to_num` before LightGBM (lgbm_trainer.py:629).** Defeats LightGBM's native missing-value handling; conflates "0" with "unknown". Affects every rolling-window feature warm-up bar.
4. **DSR formula bugs (statistical_tests.py:91-95).** Wrong `T`, wrong kurtosis constant → DSR essentially never significant; the ≥0.95 gate is unreachable even for good models.
5. **Val set leaks into test via triple-barrier horizon (lgbm_trainer.py:321).** Early-stopping picks `best_iteration` on labels that include test-window prices → systematic OOS-metric inflation.

### Overall state
The ML pipeline has **solid structural ambition** (triple-barrier, uniqueness weights, regime-specific models, walk-forward, DSR/PBO) but **the rigor is uneven**: production target construction is correct (`apply_triple_barrier` is causal and well-documented), while the *validation layer* — the very layer that's supposed to catch overfitting and leakage — contains the most severe bugs. The result is that v3 reports plausibly-passing statistical gates while the gates themselves are mis-calibrated or measure the wrong model. **Priority order for fixes: validator (PurgedKFoldCV, walk-forward target), then statistical_tests (DSR/PBO), then trainer NaN handling and val embargo.** Once these are corrected, expect reported OOS metrics to drop materially — but they will then be trustworthy.


---

## Раздел 2: Feature Engineering

### Файлы:
- src/features/feature_pipeline.py
- src/features/derivatives.py
- src/features/microstructure.py
- src/features/regime_detector.py
- src/features/session_features.py
- src/features/orb_features.py
- src/features/mtf_context.py
- src/features/triple_barrier.py
- src/features/agg_15m.py
- src/features/live_feature_state.py
- src/features/utils.py

---

## 2.1 feature_pipeline.py

### [A — Lookahead] `_WARMUP_ROWS = 200` is insufficient for the actual longest-lookback features
**Файл:** src/features/feature_pipeline.py:43, 273
**Серьёзность:** ВЫСОКАЯ
**Описание:** Comment claims "must exceed the longest rolling window (180 for funding_zscore_30d)". But the pipeline also uses `atr_lookback=540` (RegimeDetector.detect_all), `hurst_window=300` and rolling-mean of 720 in `session_cumulative_volume_ratio`. With warmup of 200, the first ~340 rows of regime-related features and ~520 rows of session ratios are computed with truncated history → silently biased (not NaN, but using `len(history)` < lookback).
**Best Practice:** Warmup = max(all lookback windows) + safety margin (≥10%). Drop or NaN-mark rows where any feature was computed on insufficient history. See pandas-ta / mlfinlab convention.
**Влияние на trading:** Early-period regime labels and session ratios are biased — and they participate in *training* (those rows survive the trim). The model sees inconsistent feature distributions in the first ~6 months of training data.
**Рекомендация:** Set `_WARMUP_ROWS = max(540, 300, 720)` = 720, or compute it dynamically from the detector and feature configs.

### [C — Train/Serve Skew] `build_from_buffer` does NOT slice warmup rows; offline `build` does
**Файл:** src/features/feature_pipeline.py:273 vs 515-517
**Серьёзность:** ВЫСОКАЯ
**Описание:** `build` drops `_WARMUP_ROWS` head rows (offline); `build_from_buffer` doesn't (live). The doc claims that with a sufficient buffer the last row's features are warm — but the regime detector's `min_bars=detector.hurst_window` short-circuits Hurst/regime to defaults for *all rows before the threshold*. For 15m with `hurst_window=50` and a buffer of just 60 bars, the last row has only 60 bars of `atr_lookback` history vs. offline's 672. Distribution mismatch is silent.
**Best Practice:** Google ML Rule #29 "Train serve skew": same feature implementation, same data shape, same warmup contract on both sides. Log a `feature_warmth_score` per bar and raise if below threshold.
**Влияние на trading:** Live signals during the first hours/days after a restart fire with features that statistically differ from training distribution. Particularly bad for `atr_percentile`, `oi_zscore`, `funding_zscore_30d`.
**Рекомендация:** Refuse to emit signals until the buffer reaches `max(longest_window) × 1.1`. Add explicit buffer-depth guards.

### [E — Architecture] `get_feature_names` does not include `agg_15m_*` features
**Файл:** src/features/feature_pipeline.py:520-530
**Серьёзность:** СРЕДНЯЯ
**Описание:** `add_15m_aggregated_features` exists and adds 4 columns, but `FEATURE_GROUPS` / `FEATURE_GROUPS_MTF` list them nowhere. NaN audit (line 277) won't check them; downstream feature whitelist generation can't reference them by group.
**Рекомендация:** Add `agg_15m` group.

---

## 2.2 microstructure.py

### [B — Math] `vwap_4h` is timeframe-agnostic (6 bars regardless of TF)
**Файл:** src/features/microstructure.py:58-65
**Серьёзность:** ВЫСОКАЯ
**Описание:** `vwap_4h` is computed as `rolling_sum(window_size=6)`. On 4H: 6 bars = 24h. On 1H: 6 bars = 6h. On 15m: 6 bars = 1.5h. The feature name advertises a 4H VWAP on every timeframe, but the rolling window is in *bars* not in *time*. Since `add_volume_features` is called for every TF in the pipeline (line 260), the 15m model has a feature labelled "vwap_4h" that is in fact a 90-minute VWAP.
**Best Practice:** Timeframe-aware feature implementation. pandas-ta accepts `length` and a TF param; the user is responsible for the conversion.
**Влияние на trading:** Names are lies. Cross-TF model debugging is hindered. Mathematical content differs by TF but is treated as the same feature in `mtf_context` joins (could leak HTF semantics into LTF if joined).
**Рекомендация:** Rename to `vwap_6bar`, OR parameterise the window by TF.

### [B — Math] CVD aggregation: `2 × taker_buy_volume − volume` is the only sane OHLCV proxy, but the comment misses Lee-Ready discussion
**Файл:** src/features/microstructure.py:27-31
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** Binance kline `taker_buy_base_vol` is *exactly* aggressor-buy volume during the bar, so `taker_buy − (volume − taker_buy) = 2·taker_buy − volume` is exact CVD per bar, not an approximation. This is correct. However, `cvd_cum = pl.col("cvd").cum_sum()` over the whole dataset means the absolute level depends on the dataset start date. On a restart, live cvd_cum restarts at the buffer start → completely different scale from training. Severe train/serve skew on this *level* feature (slope/divergence features built from differences are OK).
**Влияние на trading:** `cvd_cum` if it's a feature is essentially useless / mis-scaled live.
**Рекомендация:** Either drop `cvd_cum` from features, or replace with windowed CVD (rolling_sum(window=N)).

### [A — Lookahead] `add_volume_session_features` shift-then-rolling-over has subtle Polars semantics
**Файл:** src/features/microstructure.py:280-302
**Серьёзность:** СРЕДНЯЯ
**Описание:** Expression `pl.col("volume").shift(1).rolling_mean(20, min_periods=1).over("_hour_vsa")` chains shift → rolling → over. Polars semantics: `.over()` partitions by `_hour_vsa`, then within each partition applies `shift(1).rolling_mean(20)`. So "same hour 20 prior observations, excluding current" — that's intended, but the *order* within the hour-of-day partition is determined by the row order at `.over()` time. The function sorts by `open_time` first (line 280), so the partition order is time-ascending — correct. But the contract is fragile; a future change to call order could silently break causality.
**Best Practice:** Materialize the partition order explicitly via `.sort_by()` inside the expression, or use a window function that doesn't depend on outer sort.
**Рекомендация:** Add an explicit `.sort_by("open_time")` inside `.over` or annotate the invariant in a test.

### [E — Architecture] `cvd_cum` listed as a training feature but is non-stationary level
**Файл:** src/features/feature_pipeline.py:48 + microstructure.py:31
**Серьёзность:** ВЫСОКАЯ
**Описание:** LightGBM splits on monotonic raw level → an unbounded cumulative sum gives splits like "cvd_cum > 1.5e9" that depend on dataset start; that boundary is meaningless live. Standard finance ML practice: drop or difference non-stationary features.
**Влияние на trading:** Memorization of dataset epoch instead of structural signal.
**Рекомендация:** Replace with `cvd_rolling_sum(N)` and remove `cvd_cum` from the whitelist.

---

## 2.3 derivatives.py

### [C — Train/Serve Skew] Multiple `rolling_zscore(... , 180)` features assume 4H bar duration
**Файл:** src/features/derivatives.py:109-110, 172-173
**Серьёзность:** ВЫСОКАЯ
**Описание:** Comments declare `funding_zscore_30d` and `oi_zscore` as "180 bars × 4H = 30 days" — but the same `add_funding_features`/`add_oi_features` functions are called for the 1H and 15m pipelines (`feature_pipeline.py:459-460`). On 1H, 180 bars = 7.5 days; on 15m, 180 bars = 1.875 days. The *name* says 30d, the *value* differs by 16×.
**Best Practice:** Per-TF lookback config dict, or pass `bar_duration_ms` and convert.
**Влияние на trading:** Z-scores have different statistical meaning between training pipelines. A `funding_zscore_30d > 2.0` threshold means very different events on 4H vs 15m models.
**Рекомендация:** Add `lookback_30d_bars` param computed from TF, propagate via FeaturePipeline.

### [B — Math] `funding_cum_24h` is a 6-bar rolling sum on every TF
**Файл:** src/features/derivatives.py:115-119
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same root cause: window is in bars, not time. 6 bars × 4H = 24h ✅; × 1H = 6h ❌; × 15m = 1.5h ❌. The cumulative-24h *name* misleads.
**Рекомендация:** Same as above — parameterize by TF.

### [B — Math] `oi_delta_4h` is a 1-bar percent change regardless of TF
**Файл:** src/features/derivatives.py:163-167
**Серьёзность:** ВЫСОКАЯ
**Описание:** `(oi - oi.shift(1)) / oi.shift(1)` is named `oi_delta_4h` but on 1H/15m pipelines is 1H or 15min Δ. `oi_delta_12h` = `shift(3)` — 12h on 4H, 3h on 1H, 45min on 15m. Documented as time-based, computed as bar-based.
**Рекомендация:** Either rename to `oi_delta_1bar` / `oi_delta_3bar` or convert bars based on TF.

### [B — Math] `basis_approx = funding_cum_24h` is a very rough proxy for spot–perp basis
**Файл:** src/features/derivatives.py:191-204
**Серьёзность:** СРЕДНЯЯ
**Описание:** True basis = (perp - spot) / spot. Funding-based proxy is only approximately related (basis drives funding, but with lags, fee floors, and asymmetric clamping). The comment "Daily basis = sum of 3 daily funding payments" is dimensionally wrong: funding rate is per-period (e.g. 0.01%), summing 6 bars of 4H funding rates gives a number that *isn't* a basis in bps.
**Best Practice:** `compute_basis_annualized` exists later in the same file and uses real spot price — use that everywhere or drop the proxy.
**Рекомендация:** Drop `basis_approx` from features, OR rename to `funding_cum_24h_dup` so it's clear it's not a basis.

### [B — Math] `compute_liquidation_proximity` interprets Binance force-order side oddly
**Файл:** src/features/derivatives.py:323-332
**Серьёзность:** СРЕДНЯЯ
**Описание:** Comment says "side=SELL → a LONG was liquidated" — correct per Binance docs. But then `_nearest_above(long_clusters)` searches for long-liquidation clusters *above* current price, when long liquidations sit *below* (longs get stopped on dips). The code's own comment block lines 353-358 acknowledges the contradiction and "honors the spec literally" — but the spec described in code comments points the wrong way relative to standard magnet-toward-liquidations logic. Either the spec is wrong, or the variable naming flips the semantic.
**Рекомендация:** Document with reference; add a test pinning expected output for a synthetic liquidation list with known geography.

### [B — Math] `_zscore` uses population stdev (`pstdev`), not sample stdev
**Файл:** src/features/derivatives.py:231-241
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** With small `series` (n < 30), `pstdev` biases the z-score upward in magnitude relative to a `statistics.stdev` (Bessel). Combined with `basis_zscore_30d` computed on a 30-element sample → systematic bias.
**Рекомендация:** Switch to sample stdev for sample-size-aware z.

---

## 2.4 regime_detector.py

### [B — Math] Hurst computed on windows as small as 50 bars (RegimeDetector15M)
**Файл:** src/features/regime_detector.py:590-601
**Серьёзность:** ВЫСОКАЯ
**Описание:** Academic R/S literature (Weron 2002, Couillard & Davison 2005) shows R/S Hurst is severely biased for N < 100, and noisy for N < 300. With `hurst_window=50`, RegimeDetector15M's Hurst feature is essentially a noisy 0.5±0.2 with biased mean. Couillard & Davison 2005 demonstrate small-sample bias of +0.05 to +0.10 above the true H even for fractional Gaussian noise. For RegimeDetector1H with `hurst_window=100`, mild bias remains.
**Best Practice:** Use DFA (Detrended Fluctuation Analysis) instead of R/S — substantially more robust for small samples (Peng et al. 1994, Kantelhardt 2001). mlfinlab uses DFA. Minimum sample: 100 for DFA, 300 for R/S.
**Влияние на trading:** 15m regime detector's Hurst component contributes noise rather than signal. Since `_classify` doesn't even use Hurst (line 515), it's only a numeric ML feature — but the ML still trains on it.
**Рекомендация:** Either remove Hurst from 15m, or switch to DFA with 100-bar minimum.

### [B — Math] Hurst `_RECOMPUTE_EVERY = 6` hard-coded but meaning differs per TF
**Файл:** src/features/regime_detector.py:400
**Серьёзность:** СРЕДНЯЯ
**Описание:** Comment claims "1 day on 4H" — correct for 4H. For 1H this is 6h; for 15m this is 90 minutes. The amortization frequency is the same number of *bars*, not the same time gap. So 15m has 1.5h-stale Hurst, which on a 50-bar window can shift materially.
**Рекомендация:** TF-aware recompute frequency (e.g., once per day in *time*).

### [D — Regime] `_classify` returns TREND_UP/TREND_DOWN based on *single-bar* last return
**Файл:** src/features/regime_detector.py:521-526
**Серьёзность:** ВЫСОКАЯ
**Описание:** Given ADX > threshold, the up/down decision is `close[idx] >= close[idx-1]`. A single oscillating bar flips the regime label every bar → **flickering**. Master document mentions regime stability but the implementation gives the opposite.
**Best Practice:** Use DI+/DI- crossover (already inside ADX indicator) or EMA-slope sign or median return over N bars. mlfinlab: HMM regime with sticky transition probability.
**Влияние на trading:** Models trained on `regime=='trend_up'` see a near-50/50 mix of true uptrends and oscillation noise; per-regime evaluation conflates regime *quality* with regime *labelling instability*.
**Рекомендация:** Use 3-bar majority sign or DI+/DI− crossover; add minimum-dwell-time (≥ 3 bars).

### [B — Math] ATR percentile uses `searchsorted` without tie handling
**Файл:** src/features/regime_detector.py:163, 423-426
**Серьёзность:** НИЗКАЯ
**Описание:** `np.searchsorted(np.sort(history), current_atr) / len(history)` returns the *left-side* insertion index → leads to slightly biased percentile when many identical ATR values exist (e.g., low-vol regimes). Use `side='right'` for true CDF, or `(left+right)/2` for symmetric.
**Рекомендация:** Use `numpy.percentileofscore` from scipy or implement symmetric tie-handling.

### [D — Regime] No regime hysteresis / minimum-dwell — flickering between bars
**Файл:** src/features/regime_detector.py:499-528
**Серьёзность:** ВЫСОКАЯ
**Описание:** Classification is stateless per bar. No constraint that a regime must persist N bars before switching. Crypto literature (Ang & Bekaert 2002, Hidden Markov Models for crypto regimes 2024) consistently uses sticky transition probabilities.
**Рекомендация:** Add post-processing pass: collapse runs of <3 consecutive same-regime to neighbour majority; or wrap with 3-bar Viterbi smoothing.

---

## 2.5 session_features.py

### [B — Math] `session_cumulative_volume_ratio` uses fixed 720-bar window — 30d only on 1H
**Файл:** src/features/session_features.py:169-171
**Серьёзность:** ВЫСОКАЯ
**Описание:** `rolling_mean(window_size=720)` × 24. On 1H = 30 days × 24 = correct daily volume proxy. On 15m = 7.5 days × 24 → divides cum_vol_today by a number 4× too large. The "× 24" multiplier is wrong on non-1H TFs (should be "bars per day" for that TF).
**Рекомендация:** Replace `× 24` with TF-aware `bars_per_day`, e.g. `1440 / interval_minutes`.

### [B — Math] `is_monday_asia_window` rule may be stale post-ETF
**Файл:** src/features/session_features.py:386-389
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** The "Monday Asia open effect" was widely documented for 2018-2023 BTC perp data, but multiple 2024-2025 studies note that institutional spot ETF flow (US session) has reduced its statistical significance. Feature may now contribute noise. Worth running ablation.
**Рекомендация:** Drop or test feature importance against latest data.

### [E — Architecture] `is_high_vol_day = (dow == Wed) | (dow == Thu)` is hardcoded with no source
**Файл:** src/features/session_features.py:391-392
**Серьёзность:** НИЗКАЯ
**Описание:** No academic citation. Liu et al. 2024 and several pre-print studies find higher BTC volatility on *Tuesdays* and *Sundays* in 2024-2025 (US options/futures expiry effects). Hardcoded Wed/Thu is unsupported.
**Рекомендация:** Derive day-of-week vol empirically from training data, or drop.

### [A — Lookahead] `SessionVWAP.calculate` includes the *current bar's* close/volume in cum_sum
**Файл:** src/features/session_features.py:151-156
**Серьёзность:** СРЕДНЯЯ
**Описание:** Standard VWAP convention is fine *if the bar has closed* — the comment says training builds on closed bars, so OK. But `build_from_buffer` (live) likely calls this with the *current forming bar* as the last row → the last row's `session_vwap` includes a partial bar (its current close), which differs from the offline training distribution (always closed bars).
**Best Practice:** Live inference must use *only closed bars*. Document and enforce.
**Рекомендация:** Add assertion: `assert bar.is_closed` or `df.tail(1)` must reference the last closed bar.

### [E — Architecture] `_PRE_FUNDING_WINDOW_H = 2` makes pre-funding flag unfireable on 4H
**Файл:** src/features/session_features.py:60-61
**Серьёзность:** НИЗКАЯ (INFO)
**Описание:** A 4H bar at hour 0 (00:00 UTC) is 1h from funding (01:00) → falls in pre-funding window (≤2h). At hour 16 → 1h from funding (17:00) → in window. But 4H pipeline never calls `PreFundingDetector` (only 1H/15m/5m/1m do via `build_mtf`). OK then — but the constant comment "On 4H: not applicable" is buried.

---

## 2.6 orb_features.py

### [A — Lookahead] CRITICAL: ORB high/low includes bars *after* the current bar within the ORB-forming hour
**Файл:** src/features/orb_features.py:127-142
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `pl.when(is_orb_bar).then(high).otherwise(None).max().over("_date")` computes the maximum of all ORB bars in the entire day — regardless of whether they have occurred yet. Concretely: for a bar at 00:15 (within Asia ORB window 00:00–00:59), `orb_high_asia` is the max of bars at 00:00, 00:15, 00:30, 00:45 — *including future bars at 00:30 and 00:45*. This is direct lookahead.
**Best Practice:** Compute ORB via `cum_max().over("_date")` filtered by `is_orb_bar`, or compute ORB only after the ORB window closes (e.g., bars from hour `open_hour+1` onward get a forward-filled ORB; bars within the ORB window get NaN/0).
**Влияние на trading:** All "breakout while inside the ORB window" rows are *training on the future*. Since this is one of the headline 15m features, the 15m model's apparent skill could be majority an ORB-leakage artifact.
**Рекомендация:** Replace `.max().over("_date")` with: `pl.when(hour >= open_hour + 1).then(orb_high_from_filtered_max).otherwise(NaN)`. Treat in-ORB bars as having no defined ORB. Also reflect this in `bars_since_session_open`.

### [A — Lookahead] `_compute_session_orb` forward_fill spans across days
**Файл:** src/features/orb_features.py:144-148
**Серьёзность:** ВЫСОКАЯ
**Описание:** After computing per-day ORB then forward-filling, days where the session ORB window had no bars (data gaps) inherit the previous day's ORB. Across multi-day gaps this propagates stale ORB indefinitely. Combined with `dist_to_orb_high_pct` etc., this gives stale features that look fresh to the model.
**Рекомендация:** Forward-fill only within the same day or for at most 1 day; mark stale ORB with a flag.

### [B — Math] `bars_since_session_open` ignores minute-within-hour
**Файл:** src/features/orb_features.py:262-269
**Серьёзность:** НИЗКАЯ
**Описание:** `(hour - start) * 4` assumes bar opens on the hour. For 15m bars, bar 00:00, 00:15, 00:30, 00:45 all have `hour=0`, so all map to `bars_since=0`. Off-by-up-to-3 within the first hour, then off-by-up-to-3 within each subsequent hour — i.e., always the *minimum* bar of the hour. Should use minute as well.
**Рекомендация:** `((hour - start)*4 + minute//15)`.

### [E — Architecture] `current_session` definition disagrees with `SessionEncoder` definition
**Файл:** src/features/orb_features.py:172-182 vs src/features/session_features.py:38-46
**Серьёзность:** СРЕДНЯЯ
**Описание:** SessionEncoder maps 13 → NY, 14-16 → OVERLAP (4), 17-21 → NY again, 22-23 → DEAD. ORBDetector.current_session maps 13-21 → NY (3), 22-23 → DEAD. So a 15m bar at hour 14 has `trading_session=4 (OVERLAP)` from SessionEncoder but `current_session=3 (NY)` from ORBDetector. Same row, two different "session" features that disagree.
**Влияние на trading:** Cross-feature contradictions confuse the model — it can split on one and predict the other.
**Рекомендация:** Unify to a single session enumeration; or rename `current_session` to `current_orb_session` to make scope explicit.

---

## 2.7 mtf_context.py

### [A — Lookahead] CORRECT: `_htf_time = open_time + bar_duration` (close_time)
**Файл:** src/features/mtf_context.py:289-290
**Серьёзность:** INFO (positive finding)
**Описание:** The asof backward join keys on HTF *close_time*, not open_time, so a 4H bar opening at 08:00 is only matched to LTF bars at or after 12:00 — the moment it has closed. Causally correct. ✅

### [E — Architecture] `htf_4h_last_n_bars_dir` duplicates `htf_4h_trend_dir`
**Файл:** src/features/mtf_context.py:312, 329-330
**Серьёзность:** НИЗКАЯ
**Описание:** Both columns are computed via `_trend_direction(close, lookback=3)` from the same HTF df. Identical values. Wastes feature space and inflates MDA importance of "trend direction" by double-counting.
**Рекомендация:** Drop `htf_4h_last_n_bars_dir`, or implement an actually different lookback (e.g., 6 bars).

### [E — Architecture] EMA20 on HTF uses `adjust=False` but `ignore_nulls=True` — early-window bias
**Файл:** src/features/mtf_context.py:47, 308-309
**Серьёзность:** НИЗКАЯ
**Описание:** `adjust=False` gives the recursive EMA `EMA_t = α·x + (1-α)·EMA_{t-1}` seeded by `EMA_0 = x_0`. With short HTF buffers (e.g., 4H buffer of 400 bars), the seed bias propagates 5τ ≈ 100 bars before converging. In `build_from_buffer` with a fresh buffer this affects the first 25 4H bars (≈4 days).
**Рекомендация:** Either drop EMA20 until buffer ≥ 5τ, or use `adjust=True`.

### [A — Lookahead] 1H session_vwap join into 15m uses `close_time` — CORRECT but with caveat
**Файл:** src/features/mtf_context.py:172-188
**Серьёзность:** СРЕДНЯЯ
**Описание:** The join uses `open_time + 1H` as the 1H close-time key — correct, no lookahead. But `session_vwap` on the 1H df is the *intra-1H VWAP including the closing bar's own close*. That's fine after the bar closes, BUT note: a 15m bar at 09:00 gets the *1H bar's VWAP for the 08:00-09:00 hour* — that 1H bar was the *previous* 1H session phase if the 1H pipeline labels 09:00 as a new session-VWAP day. The boundary mechanics are subtle.
**Рекомендация:** Add a unit test that pins the 09:00 15m bar's `htf_1h_vwap_position` to the offline value.

---

## 2.8 agg_15m.py

### [A — Lookahead] CORRECT: bucket-by-floor + parent 4H join
**Файл:** src/features/agg_15m.py:130-153
**Серьёзность:** INFO (positive finding)
**Описание:** Each 4H bar `T` reads only 15m bars in `[T, T+4h)` — the last 15m bar (open at `T+3h45m`) closes exactly at `T+4h`, same instant as the 4H bar. So all 16 children are closed by 4H bar `T`'s close. ✅. The ORB-style breakout inside `_bucket_stats` uses `high[i-4:i].max()` (strictly prior bars) — also causal.

### [E — Architecture] `_bucket_stats` requires `taker_buy_volume` with no fail-soft
**Файл:** src/features/agg_15m.py:58
**Серьёзность:** СРЕДНЯЯ
**Описание:** If the 15m frame lacks `taker_buy_volume` (e.g., legacy datasets with only `taker_buy_base_vol`), the function raises a KeyError instead of using the 50% proxy that `_ensure_taker_buy_volume` provides elsewhere.
**Рекомендация:** Call `_ensure_taker_buy_volume(df_15m)` at the top of `add_15m_aggregated_features`.

### [B — Math] `agg_15m_cvd_slope` normalization divides by `max(|cum|.max(), 1.0)` — sign-flips
**Файл:** src/features/agg_15m.py:73-74
**Серьёзность:** НИЗКАЯ
**Описание:** `denom = max(abs(cum).max(), 1.0)`. When cum_cvd is small (low-activity bucket), `denom = 1.0`; when large, normalized. Two regimes of behavior — the normalized feature isn't continuous as activity grows from "near zero" to "non-trivial".
**Рекомендация:** Normalize by `max(abs(cum).max(), σ_baseline)` where σ_baseline is an estimate of typical bucket scale.

---

## 2.9 live_feature_state.py

### [C — Train/Serve Skew] `funding_rate_history` maxlen=100 vs. training's 180-bar rolling z-score
**Файл:** src/features/live_feature_state.py:51-53
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Funding settles every 8h → 100 records = 33 days ≈ 99 × 8h. After asof-join onto OHLCV bars, the live `funding_rate` series carries forward each settlement to 6 (4H), 8 (1H), 32 (15m) successive bars. So:
- 4H: 100 settlements × 6 bars = 600 bars history → enough for rolling(180). ✅
- 1H: 100 × 8 = 800 bars → enough for rolling(180). ✅
- 15m: 100 × 32 = 3200 bars → enough for rolling(180). ✅
*BUT* the live `funding_zscore_30d` uses only the in-memory buffer of bars, which is limited by `bar_buffer_*` maxlen. 15m buffer = 600 bars = ~6 days → less than 180 bars × 15m × <30d. Cross-checking: 600 15m bars covers ~6.25 days. rolling(180) over 15m = 1.875 days — *fits in 600-bar buffer*. OK. Actually the more pressing issue:
**Влияние на trading:** Less severe than initially feared, but the *funding values* themselves come from `funding_rate_history` (max 100 settlements). On 15m where each settlement repeats 32 times, the unique-value structure of the live z-score differs from training (training has every funding settlement asof-joined, but offline data goes back arbitrarily far).
**Рекомендация:** Increase `funding_rate_history` maxlen to ≥ 200 settlements (~67d); add explicit test that live and offline produce identical `funding_zscore_30d` for the same end-time.

### [C — Train/Serve Skew] `oi_history` maxlen=100 with 5-min poll → 8h history, but training uses 180-bar rolling z-score
**Файл:** src/features/live_feature_state.py:56-58
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** OI poll every 5 minutes × 100 records = 500 minutes ≈ 8.3 hours of OI history. After asof-join onto bars, all bars within this 8.3h window get the closest OI value; older bars get *no* OI data → asof returns null → fill 0. The rolling-180 z-score then sees mostly zeros → degenerate live behavior. Training pipeline ingests months of `metrics_df` from DataStore, so training distribution is far from this.
**Best Practice:** Either persist OI history to disk between restarts, or backfill from REST on startup.
**Влияние на trading:** Live `oi_zscore` is ~0 for the first ~8h after restart and only ~24h of "real" rolling history is ever available even after warmup. The model's OI-dependent signals are effectively neutralized live.
**Рекомендация:** Increase `oi_history` to 30 days worth (8640 records at 5-min poll, or fewer records at lower frequency); backfill on startup.

### [C — Train/Serve Skew] `bar_buffer_15m` maxlen=600 (~6 days) insufficient for atr_lookback=672
**Файл:** src/features/live_feature_state.py:47 vs regime_detector.py:595
**Серьёзность:** ВЫСОКАЯ
**Описание:** RegimeDetector15M's `atr_lookback=672` (1 week of 15m bars). Buffer holds 600. So `atr_percentile` in live always uses ≤ 600 bars of history vs. ≥ 672 in offline training. Always slightly biased high (denominator smaller).
**Рекомендация:** Set `bar_buffer_15m` maxlen ≥ 1000.

---

## 2.10 utils.py

### [B — Math] `safe_divide` doesn't catch `+inf`/`-inf`
**Файл:** src/features/utils.py:12-20
**Серьёзность:** СРЕДНЯЯ
**Описание:** `pl.when(b == 0).then(fill).otherwise(a / b)` guards exact-zero denominators but not denormalized small numbers. For `a=1, b=1e-300`, `a/b = +inf`. The subsequent `.fill_nan(fill).fill_null(fill)` does NOT replace inf (it's neither NaN nor null in IEEE-754). Inf then propagates into downstream features.
**Best Practice:** Polars `is_finite()` check or `clip(-1e12, 1e12)` after division.
**Влияние на trading:** Single inf can blow up LightGBM split heuristics, especially in `cvd_slope_*` features where volume normalization can be tiny.
**Рекомендация:** Add `.replace(float('inf'), fill).replace(float('-inf'), fill)` or post-clamp.

### [B — Math] `rolling_correlation` uses `rolling_mean` for `mean_xy`, `mean_x2`, `mean_y2` without `min_periods` — early values are NaN-propagated
**Файл:** src/features/utils.py:33-56
**Серьёзность:** НИЗКАЯ
**Описание:** Default `min_periods=None` in polars means full window required. For the first `window-1` rows, the result is null → `safe_divide` returns 0 (which may not be the desired neutral value for correlation; "no data" ≠ "zero correlation").
**Рекомендация:** Either set `min_periods=max(2, window//2)` or emit NaN and let downstream decide.

---

## Summary — Feature Engineering

### Counts by severity
- **КРИТИЧЕСКАЯ:** 3
  - orb_features: ORB high/low includes future bars within the ORB-forming hour
  - live_feature_state: `funding_rate_history` undersized (100) vs training's 180-bar windows
  - live_feature_state: `oi_history` only 8h vs training's months-long history
- **ВЫСОКАЯ:** 10
- **СРЕДНЯЯ:** 9
- **НИЗКАЯ / INFO:** 8

### Top-5 most critical
1. **ORB high/low lookahead within the ORB-forming hour (orb_features.py:127-142).** `max().over("_date")` of ORB-window-filtered bars returns the max of *all* 4 ORB bars regardless of position. Every 15m bar at hour 00:00/00:15/00:30 inside Asia ORB sees the full ORB high — direct future leakage. Likely a major source of inflated 15m metrics.
2. **OI live history only 8 hours (live_feature_state.py:56-58).** `oi_zscore` rolling(180) is essentially 0 live for the first day; training-distribution gap is enormous.
3. **Funding live history 100 settlements vs unbounded offline (live_feature_state.py:51-53).** Live `funding_zscore_30d` truncated; train/serve skew on a major derivative feature.
4. **Multiple "X_30d" / "X_24h" features use bar-count windows that mean different time spans per TF (derivatives.py:109-119, 163-173, microstructure.py:58-65, session_features.py:169-171).** Names lie; semantics differ 4-16× between training pipelines.
5. **Regime classifier flickers between TREND_UP/TREND_DOWN on every single-bar oscillation (regime_detector.py:521-526).** Per-regime training labels are noisy; per-regime evaluation conflates regime quality with label instability.

### New lookahead problems not in Section 1
- **ORB intra-window leakage** — critical, in 15m features only.
- **SessionVWAP including the current forming bar live** — bar-closure-contract risk in `build_from_buffer`.
- **ORB forward-fill across data gaps** — stale features look fresh.
- **Confirmed clean (positive findings):** `apply_triple_barrier` (Section 1), `mtf_context` close_time asof join, `agg_15m` parent-bucket aggregation, `add_volume_session_features` shift-then-rolling.

### Overall state
The feature engineering layer is **structurally sound on multi-timeframe joins** (MTF context, agg_15m) but suffers from **two systematic classes of bugs**:

1. **Bar-count-as-time:** Features named "30d", "24h", "4H" are computed in *bars*, leading to severely divergent semantics across the 4H/1H/15m pipelines that share the same `add_*` functions. This is independent of any lookahead but creates real cross-TF feature-distribution mismatch.

2. **Live-buffer too small:** `funding_rate_history` and `oi_history` deques are sized for the legacy 4H pipeline only; on 1H/15m they cap statistical features (z-scores, rolling sums) at a fraction of their training-time span. This is train/serve skew that ML validation can't detect because live data isn't in the offline test set.

Combined with the ORB intra-window lookahead — which can dominate apparent 15m model "skill" — the feature layer needs systematic auditing before any further model retraining is meaningful. **Priority order: fix ORB lookahead → resize live history buffers → parameterize bar-count windows by TF.**

---

## Раздел 3: Execution & Risk

### Файлы:
- src/execution/strategies/ml_strategy.py
- src/execution/strategies/ml_strategy_15m.py
- src/execution/strategies/meta_strategy.py
- src/risk/risk_engine.py
- src/risk/circuit_breaker.py
- src/risk/portfolio_tracker.py
- src/execution/signal_bridge.py
- src/execution/reconciler.py
- src/execution/reconciler_signals.py

> **Pre-finding (most critical):** `CircuitBreaker.check`, `PositionReconciler.reconcile`, and `MetaMLTradingStrategy._apply_meta_gate` are defined but **never invoked from any strategy on_bar / lifecycle path**. They are dead code in live trading. See findings 3.5.1, 3.5.2, 3.3.1 below.

---

## 3.1 ml_strategy.py (4H)

### [A — Position Safety] CRITICAL: No dead-man's switch / heartbeat in 4H strategy
**Файл:** src/execution/strategies/ml_strategy.py:162-287 (whole on_start)
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** The 15m strategy starts a `HeartbeatManager` (ml_strategy_15m.py:208-243) that the parent kill-process can read to auto-close positions if the bot dies. The 4H strategy (the *production* path) does **not** start any heartbeat. If the 4H bot crashes between entry fill (line 583) and SL submission (line 750), there is no external mechanism to close the unprotected position. The 4H bot can stay dead for hours before a human notices.
**Best Practice:** Production trading systems (Hummingbot, Freqtrade, Nautilus production examples) all use either (a) exchange-side OCO orders submitted simultaneously with entry, or (b) a process-external watchdog that closes positions if heartbeat goes silent for N seconds.
**Влияние на trading:** Real-money risk. A crash window of even 30 seconds can be catastrophic on a 10× leveraged BTC position during a fast market.
**Рекомендация:** Wire `HeartbeatManager` into `MLTradingStrategy.on_start` exactly as the 15m subclass does. Add a process-external monitor (`scripts/heartbeat_monitor.py`) that calls `/fapi/v1/openOrders` and force-closes positions on stale heartbeat.

### [A — Position Safety] CRITICAL: Entry-fill → SL-submission window is unprotected
**Файл:** src/execution/strategies/ml_strategy.py:546-595, 645-712, 714-765
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Pattern: `_open_position` submits a market IOC entry; SL is deferred to `on_order_filled` which calls `_submit_stop_loss_with_retry`. Three failure modes:
1. Crash between `on_order_filled` start and `submit_order(stop_order)` → unprotected position.
2. Exchange accepts entry but rejects SL (e.g., reduce_only validation, price-tick mismatch, rate-limit) → 3 fast retries with no backoff → CRITICAL log line → human must respond.
3. `_pending_sl_params` is in-process Python dict — **not persisted**. If the bot crashes after entry submit but before the fill callback runs, the restarted bot has no record that an entry is pending → fill event on reconnect goes to the `else: # Exit fill` branch (line 590) and is mis-classified as an exit. No SL is ever placed.
**Best Practice:** Binance Futures supports OCO via batch order endpoint `/fapi/v1/batchOrders` — submit entry + SL atomically. Or use `STOP_LOSS_MARKET` with `closePosition=true` placed *before* entry (works on testnet). mlfinlab production patterns recommend "always have an exchange-side stop before any market exposure exists."
**Влияние на trading:** Existential. A single crash during a 5-minute SL gap on a 5% drop = 25% account loss at 5× leverage.
**Рекомендация:** Persist `_pending_sl_params` to SQLite/disk on entry submit; on startup, reconcile `openOrders` + `positionRisk` against persisted state. Switch to batch-order submission if possible.

### [A — Position Safety] `_submit_stop_loss_with_retry` retries with zero backoff
**Файл:** src/execution/strategies/ml_strategy.py:738-765
**Серьёзность:** ВЫСОКАЯ
**Описание:** `for attempt in range(1, max_retries + 1)` with no `time.sleep` between attempts. If the rejection cause is rate-limit (`-1003`), all three retries fail within microseconds — then position is unprotected. Also, `submit_order` is asynchronous in Nautilus (queues the order); a successful return does NOT mean acceptance. If the exchange ack arrives 50ms later with a rejection, no retry is triggered.
**Best Practice:** Exponential backoff (50ms → 200ms → 1s). Wait for the order-accepted event before declaring success. Use `OrderStatus` callbacks to confirm.
**Рекомендация:** Add backoff between retries; subscribe to `OrderRejected` event for the SL `client_order_id` and retry on rejection.

### [A — Position Safety] Race condition on `max_open_positions` filter
**Файл:** src/execution/strategies/ml_strategy.py:487 (filter) + 581-583 (tracker update)
**Серьёзность:** ВЫСОКАЯ
**Описание:** `tracker.get_state().open_positions` is read at filter time. Tracker is updated only in `on_order_filled`. Between `submit_order(entry)` and the fill callback, the tracker counts N positions — but N+1 orders may be in-flight. On 15m timeframes or multi-instrument deployments this can exceed `max_open_positions`.
**Best Practice:** Maintain an in-flight-orders counter; include pending entries when checking `open_positions`.
**Влияние на trading:** Exceeds risk budget; can violate leverage limits.
**Рекомендация:** Add `_pending_entries: int`, increment in `_open_position`, decrement in `on_order_filled`; include in `_check_max_positions`.

### [C — Train/Serve / Safety] `_get_funding_rate` falls back to `0.0`, silently bypassing the funding-rate filter
**Файл:** src/execution/strategies/ml_strategy.py:767-786
**Серьёзность:** ВЫСОКАЯ
**Описание:** When `feature_vector is None` or `funding_rate` column is absent, returns `self._last_funding_rate` which initialises to `0.0`. `risk_engine._check_funding_rate` then evaluates `|0.0| > 0.001 → False`, *passing* the filter. A feature-pipeline failure thus disables an important safety filter rather than blocking the trade.
**Рекомендация:** Return `None` on failure; risk engine should treat `None` as "cannot evaluate" and reject the signal.

### [A — Lookahead] `_compute_features_unified` calls `np.nan_to_num` (live)
**Файл:** src/execution/strategies/ml_strategy.py:850
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same issue flagged in Section 1 (trainer:629) but now in live inference. Since trainer also calls nan_to_num, inference and training are at least *consistent* with each other — but both lose LightGBM's native NaN handling. Live warmup bars get `0.0` instead of NaN; spurious "feature=0" splits fire on early-buffer bars.
**Рекомендация:** Remove `nan_to_num` from both trainer and inference.

### [E — Architecture] Equity tracked twice: `_tracker` (manual) vs `self.portfolio.account(venue)` (Nautilus)
**Файл:** src/execution/strategies/ml_strategy.py:1438-1451 vs 184 (tracker)
**Серьёзность:** ВЫСОКАЯ
**Описание:** `_record_equity` reads `self.portfolio.account(venue).balance_total(USDT)` (Nautilus-tracked equity). Risk engine reads `self._tracker.get_state().equity` (manually-tracked equity). The two sources can drift due to:
- Funding payments (Nautilus may not record, tracker definitely doesn't).
- Commission rounding.
- Position-close timing differences.
**Влияние на trading:** Risk decisions made on stale equity; drawdown gate fires at the wrong number.
**Рекомендация:** Single source of truth. Strongly prefer Nautilus's portfolio (exchange-authoritative); deprecate the manual tracker or use it solely for daily/weekly accumulators that Nautilus doesn't expose.

### [C — Train/Serve] Funding settlement detection by `dt.minute == 0` is fragile
**Файл:** src/execution/strategies/ml_strategy.py:336-344
**Серьёзность:** ВЫСОКАЯ
**Описание:** `BinanceFuturesMarkPriceUpdate` streams ~1 update/sec; the settlement-time tick at 01:00:00 UTC requires `dt.minute == 0` AND `dt.hour in (1,9,17)` AND the tick arriving exactly on the minute boundary. In practice settlement ticks arrive at 01:00:00.123, 01:00:00.456 etc. — `dt.minute == 0` survives, but if the stream skips that second (network glitch) the entire settlement is missed.
**Best Practice:** Either pull `fundingRate` from REST `/fapi/v1/fundingRate?limit=1` at 01:00:30 / 09:00:30 / 17:00:30, or use ts-since-last-append > 7.5h as the settlement trigger.
**Влияние на trading:** Missed settlements → `funding_rate_history` has gaps → `funding_zscore_30d` differs from training distribution.
**Рекомендация:** Switch to REST-based settlement append, scheduled 30s post-settlement.

### [E — Architecture] `Price(..., precision=1)` hardcoded in preload
**Файл:** src/execution/strategies/ml_strategy.py:1357-1361, 1409-1413
**Серьёзность:** СРЕДНЯЯ
**Описание:** Precision=1 works for BTCUSDT (0.1 USDT tick) but breaks for symbols with different tick sizes (e.g., SOLUSDT = 0.0001, ETHUSDT = 0.01). The strategy config supports any `instrument_id` but preload bars are constructed with the wrong precision → potential Price-validation errors downstream.
**Рекомендация:** Read `instrument.price_precision` from the cache, or pass through the instrument's `make_price()` helper.

### [E — Architecture] Dead deprecated method `_compute_features` left in code
**Файл:** src/execution/strategies/ml_strategy.py:862-1013
**Серьёзность:** НИЗКАЯ
**Описание:** ~150 lines marked deprecated by Phase 6 but kept "as reference". It's not called, but its presence is a footgun for future maintenance — easy to revive accidentally.
**Рекомендация:** Delete; the git history has it.

### [E — Architecture] `_poll_open_interest` event loop access is fragile
**Файл:** src/execution/strategies/ml_strategy.py:1019-1041
**Серьёзность:** СРЕДНЯЯ
**Описание:** `asyncio.get_event_loop()` is deprecated in Python 3.10+ when no running loop exists; can return a new loop unbound to Nautilus's running loop. `loop.create_task` without storing a reference → garbage collector may cancel the task; exceptions inside the coroutine are swallowed by GC.
**Рекомендация:** Use `asyncio.get_running_loop()` (raises if no loop); store the task reference; add `task.add_done_callback(_log_exc)`.

---

## 3.2 ml_strategy_15m.py

### [C — Train/Serve] HTF context resampled from 15m bars vs trained-on-real-HTF
**Файл:** src/execution/strategies/ml_strategy_15m.py:294-314, 316-342
**Серьёзность:** ВЫСОКАЯ
**Описание:** `_build_feature_row` resamples 15m → 1H and 15m → 4H, then runs `RegimeDetector1H` / `RegimeDetector` on the *resampled* HTF. But training pipelines for the 1H and 4H *real* models use real 1H/4H kline data with `taker_buy_base_vol`, `quote_volume`, etc. — fields lost in 15m-resample (sum of 15m volumes is identical, but taker_buy_volume is summed by `_ensure_taker_buy_volume`'s 50% proxy because Nautilus Bars don't carry it). The HTF feature vectors live differ from training.
**Влияние на trading:** `htf_4h_regime`, `htf_4h_adx`, etc. are computed from a different data source live vs train.
**Рекомендация:** Subscribe to real 1H + 4H bar feeds directly (like the 4H strategy does) instead of resampling.

### [A — Lookahead] Session-trap filter inherits ORB intra-window lookahead from features
**Файл:** src/execution/strategies/ml_strategy_15m.py:391-397
**Серьёзность:** ВЫСОКАЯ
**Описание:** The trap-zone check reads `is_session_trap_zone` from feature output. As flagged in Section 2 (orb_features.py:127-142), `bars_since_session_open` and ORB columns use `.over("_date")` — they see the entire day's bars. In live mode, the *current* (last) bar is the only one without future neighbors, so the effect is smaller, but at exactly the boundary bars (00:00, 08:00, 13:00) the feature has been computed from one bar of data, so it's deterministic. Inconsistent with training where the model learned on look-ahead-contaminated values.
**Рекомендация:** Either fix ORB lookahead at the feature level (Section 2's primary recommendation) or compute trap-zone directly from `bar.ts_event` here.

### [A — Position Safety] 15m max_open_positions=1, but race condition still applies
**Файл:** src/execution/strategies/ml_strategy_15m.py:86
**Серьёзность:** СРЕДНЯЯ
**Описание:** With `max_open_positions=1` the race window from 3.1.4 is less likely to cause harm — a quick double-fire still produces 2 positions. 15m bars arrive every 15 minutes which is far more than fill latency, so practically safe. INFO level.

### [E — Architecture] `_max_bars = max(warmup_bars, 1600)` — but RegimeDetector15M `atr_lookback=672` and resample to 4H needs 16× bars
**Файл:** src/execution/strategies/ml_strategy_15m.py:121, 297 (resample period 4H = 16 15m-bars)
**Серьёзность:** ВЫСОКАЯ
**Описание:** For 4H detector with `hurst_window=300` (4H bars), we need 300 × 16 = 4800 15m bars to resample a valid `htf_4h_hurst`. The 1600 buffer gives only 100 4H bars. The 4H detector falls back to defaults for the first 200 4H bars → `htf_4h_*` features stuck at neutral 0.5/0.0 for ~17 days of live operation.
**Рекомендация:** Raise `_max_bars` to ≥ 5000 OR subscribe to real 4H feed.

### [E — Architecture] Buffer trimming `self._bars[-self._max_bars:]` rebuilds list every bar
**Файл:** src/execution/strategies/ml_strategy_15m.py:364-365
**Серьёзность:** НИЗКАЯ
**Описание:** Slicing creates new list each bar. Use `collections.deque(maxlen=N)` for O(1) bounded append.

---

## 3.3 meta_strategy.py

### [E — Architecture] CRITICAL: `MetaMLTradingStrategy` defines `_apply_meta_gate` but NEVER invokes it
**Файл:** src/execution/strategies/meta_strategy.py:182-232 + grep result (only definitions, no call sites)
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `MetaMLTradingStrategy` inherits `MLTradingStrategy` but does NOT override `on_bar`. Its only added method `_apply_meta_gate` exists but is unreferenced. So when the strategy runs, every signal goes through the *base* model only — the meta-labeling gate **never fires**. The `meta_threshold = 0.60` config is meaningless. Any reported "meta-improved performance" is from the base model alone.
**Влияние на trading:** If meta-labeling was a stated improvement source, the reported metric is wrong. If meta was being used as a stop-loss-relaxation justification, real positions are being taken without the gate's filter.
**Рекомендация:** Override `on_bar` to inject `_apply_meta_gate` between `LGBMTrainer.get_signal` and `risk_engine.evaluate`. Add a unit test asserting the gate is called.

### [F — Production] `meta_threshold = 0.60` hardcoded; selection process undocumented
**Файл:** src/execution/strategies/meta_strategy.py:175
**Серьёзность:** ВЫСОКАЯ
**Описание:** Threshold-tuning on OOS data without held-out CV → over-fit threshold. The wider Section 1 finding about PBO not being CSCV applies double here.
**Рекомендация:** Document threshold-selection methodology; show DSR including the meta threshold as one of the searched hyperparameters.

### [A — Lookahead] `MetaSignalGate.build_feature_vector` `nan_to_num` again
**Файл:** src/execution/strategies/meta_strategy.py:120
**Серьёзность:** СРЕДНЯЯ
**Описание:** Same critique as 1.2 / 3.1.5. The meta-model was likely trained with the same convention, so consistent — but LightGBM's NaN handling is wasted.

### [E — Architecture] `MetaDecision.size_multiplier` linear ramp may give 0× at threshold
**Файл:** src/execution/strategies/meta_strategy.py:160-163
**Серьёзность:** СРЕДНЯЯ
**Описание:** At exactly `proba == threshold`, `raw = 0`, clipped to `min_size`. Default `min_size=0.0` → size_multiplier=0 → effectively skip even though `take=True`. The 15m config sets `min_size=0.25` (good), but if a user constructs a gate with `min_size=0` they get the take=True + size=0 anomaly.
**Рекомендация:** Either return `take=False` when `size_multiplier=0`, or enforce `min_size > 0` in the constructor.

---

## 3.4 risk_engine.py

### [B — Math] `_check_volatility` filter is mis-specified — fixed 1% baseline, not relative
**Файл:** src/risk/risk_engine.py:373-387
**Серьёзность:** ВЫСОКАЯ
**Описание:** Code comment admits confusion: "we approximate 'average' as 1% and flag when atr_pct ≥ 2× that". Translation: blocks signals where `atr_pct > 0.02` (2%). On BTC 4H during 2024-2025, atr_pct often sits at 1.5-3% during normal trend regimes. The filter therefore rejects normal-volatility entries, not vol-spike entries. The actual *spike* detection (current vs. recent average) is delegated to `CircuitBreaker.check` — which is **never called** (see 3.5.1).
**Best Practice:** Real vol-spike detection compares `current_atr / rolling_mean(atr, 20)` against the multiplier. The `RegimeDetector.detect_all` already emits `atr_percentile`; use that.
**Влияние на trading:** Massively over-restricts in normal markets, no protection from actual spikes.
**Рекомендация:** Replace with `atr_percentile > 0.95` (use the regime detector's output) or per-symbol rolling baseline.

### [B — Math] `_check_expected_return` confuses "1×ATR move" with "expected return"
**Файл:** src/risk/risk_engine.py:351
**Серьёзность:** ВЫСОКАЯ
**Описание:** `expected_return_bps = signal.atr_pct * 10_000` — but ATR is *volatility* not *expected return*. The model's expected return is `confidence × R/R × ATR × pt_multiplier` minus loss probability × stop distance. Treating raw ATR as expected return inflates threshold checks: a high-vol regime *trivially* passes the "expected return covers fees" gate even if model has zero edge.
**Best Practice:** Compute true expected return from `(2*p - 1) * R + p * rr_ratio * R - (1-p) * R` where p = confidence, R = stop distance.
**Влияние на trading:** Trades fire in high-vol regimes even when fees-adjusted expectancy is negative.
**Рекомендация:** Real expectancy formula; recalibrate `min_expected_return_bps`.

### [B — Math] `consecutive_losses` counter never decays — pause is "until next win" not "N hours"
**Файл:** src/risk/risk_engine.py:389-410 + portfolio_tracker.py:162-165, 222-230
**Серьёзность:** ВЫСОКАЯ
**Описание:** Filter releases after `elapsed >= pause_hours`, but `consecutive_losses` itself only resets when a win occurs (`portfolio_tracker.close_position` line 165). So after 5 losses, the bot is "paused 4h" — after 4h, filter returns OK — *next loss* makes it 6 (not 1), and filter blocks again immediately because `6 >= 5`. The pause is effectively permanent until a win occurs.
**Best Practice:** Reset `consecutive_losses` to 0 after a successful pause window elapses (mark "pause served"), OR reset on N hours of no losing trades.
**Влияние на trading:** Once 5 losses fire, bot may be paused indefinitely.
**Рекомендация:** In `_check_consecutive_losses`, when `elapsed >= pause_hours`, also call `portfolio_tracker.reset_consecutive_losses()` (new method).

### [A — Position Safety] `_check_max_positions` uses internal tracker count, not exchange
**Файл:** src/risk/risk_engine.py:321-328
**Серьёзность:** ВЫСОКАЯ
**Описание:** If a previous run died and left an orphan position on the exchange, the internal tracker doesn't know about it. The risk engine allows N more positions than the real risk budget.
**Best Practice:** Periodic reconciliation against `/fapi/v2/positionRisk` (and the reconciler already exists — see 3.5.2 for why it isn't run).
**Рекомендация:** Run `PositionReconciler` at startup before `subscribe_bars`, and every N minutes thereafter.

### [F — Production] `daily_loss_limit` checked from `portfolio_tracker._daily_realized_pnl + unrealized`
**Файл:** src/risk/portfolio_tracker.py:198-204 + risk_engine.py:312-319
**Серьёзность:** СРЕДНЯЯ
**Описание:** `_daily_realized_pnl` is reset to 0 on day-rollover via `_roll_periods`. But the initial day-start in `__init__` is `datetime.now(timezone.utc).replace(hour=0,...)` — so after a *restart at 16:00 UTC*, day_start = today's 00:00, but `_daily_realized_pnl = 0` (in-memory init). If the bot lost 2.5% earlier today and was restarted, the new instance shows daily_pnl = 0, can lose another 3% → real daily loss = 5.5%, exceeds the -3% limit.
**Best Practice:** Persist daily/weekly P&L counters to disk; restore on startup.
**Влияние на trading:** Daily loss limit can be bypassed via restart — accidental or otherwise.
**Рекомендация:** Persist `_daily_realized_pnl`, `_weekly_realized_pnl`, `_consecutive_losses`, `_peak_equity` to SQLite on every update; reload on `__init__`.

---

## 3.5 circuit_breaker.py

### [E — Architecture] CRITICAL: `CircuitBreaker.check` is never called from any strategy
**Файл:** src/risk/circuit_breaker.py (whole) + grep on strategies/
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** No strategy invokes `CircuitBreaker.check(...)` or `get_position_size_multiplier(...)`. The breaker is constructed nowhere either. The advertised "kill switch at -15% DD" / "stop trading day at -3%" / "vol spike → halve size" are entirely non-functional. The only actual gating happens inside `RiskEngine` filters which (a) duplicate part of the logic (daily/weekly/drawdown) and (b) have the issues flagged in 3.4.
**Влияние на trading:** Master document claims of cascading circuit breaker protection are not implemented.
**Рекомендация:** Either delete `circuit_breaker.py` (and the master document references) or wire it into `on_bar` immediately after `_tracker.get_state()`.

### [E — Architecture] `_daily_triggered` flag set but logic is stateless
**Файл:** src/risk/circuit_breaker.py:142, 235-238
**Серьёзность:** ВЫСОКАЯ
**Описание:** When daily HARD breaches, `_daily_triggered = True`. But subsequent `check()` calls *re-evaluate* `daily_pnl_pct` from scratch — they don't check the flag. If a position closes profitably and brings daily back above -3%, the breaker silently re-enables. `reset_daily()` exists but no one calls it. The stickiness implied by the design isn't there.
**Рекомендация:** First line of `check()`: if `self._daily_triggered and now < midnight_utc: return triggered_state`. Schedule `reset_daily` via Nautilus timer at 00:00 UTC.

### [B — Math] `MAX_DRAWDOWN_KILL` comparison direction
**Файл:** src/risk/circuit_breaker.py:100-101
**Серьёзность:** НИЗКАЯ
**Описание:** `dd = -portfolio_state.current_drawdown_pct` (negative); compared `dd <= MAX_DRAWDOWN_KILL` (= -0.15). With portfolio at 16% DD: dd = -0.16, -0.16 <= -0.15 → True → trigger. Correct, but cognitive overhead of double-negation invites a future regression.
**Рекомендация:** Compare against positive: `if portfolio_state.current_drawdown_pct >= 0.15`.

---

## 3.6 portfolio_tracker.py

### [B — Math] `close_position` cash math: `self._cash += pos.quantity * pos.avg_entry_price + realized_pnl`
**Файл:** src/risk/portfolio_tracker.py:156
**Серьёзность:** ВЫСОКАЯ
**Описание:** Adds back position notional + PnL. But `update_fill` does NOT subtract notional from cash — it only subtracts fees (line 93). So opening a position never reduces cash by notional, but closing adds notional back. After one round-trip, cash has been *credited* with the entry notional twice. Equity calculation `_cash + unrealized` becomes increasingly wrong after each round trip. Catastrophic accounting bug for any horizon > 1 trade.
**Best Practice:** For futures/perp: positions don't consume cash (only margin). Track margin separately. Or for accounting: `cash -= notional + fee` on open, `cash += notional + pnl - fee` on close (which nets to `cash += pnl - 2×fee`).
**Влияние на trading:** Equity (and therefore position sizing, drawdown gate, daily-loss filter) drifts upward by `notional` per round trip. After 10 trades on $10k account at 5× leverage, reported equity = ~$500k. Drawdown and daily-loss filters become useless.
**Рекомендация:** Audit cash accounting end-to-end. For futures-style: `update_fill` should not touch `_cash` at all except for fees; `close_position` should add only `realized_pnl - fee`. Add invariant test.

### [B — Math] `update_fill` averaging assumes same-direction adds
**Файл:** src/risk/portfolio_tracker.py:97-104
**Серьёзность:** ВЫСОКАЯ
**Описание:** When a fill comes in for an existing symbol, the code averages: `new_avg = (old_avg*old_qty + price*qty) / total_qty`. If the fill is on the *opposite* side (partial close, reversal), this produces a nonsensical average. The check `if total_qty > 0` doesn't distinguish increase-position from reduce-position.
**Рекомендация:** Compare `pos.direction` vs incoming `direction`. If opposite, reduce quantity and realize partial P&L; don't average.

### [B — Math] Daily/weekly PnL fraction uses `_initial_equity` as denominator
**Файл:** src/risk/portfolio_tracker.py:198-212
**Серьёзность:** ВЫСОКАЯ
**Описание:** `daily_pnl / _initial_equity` — but equity changes over time. At week 4 with equity = $11k, `_initial_equity = $10k` → reported daily_pnl_pct = realized/10000 instead of realized/11000. Risk gates fire on a stale denominator.
**Best Practice:** Use start-of-day equity as denominator; persist it on day rollover.
**Рекомендация:** Add `self._day_start_equity`; update in `_roll_periods`; use as denominator.

### [B — Math] `peak_equity` only updated on `close_position`, not on `update_price`
**Файл:** src/risk/portfolio_tracker.py:167-170
**Серьёзность:** ВЫСОКАЯ
**Описание:** Peak equity is set only after a position closes. Unrealized gains never set a new peak. So drawdown is computed as `(peak_at_last_close - current_equity) / peak_at_last_close` — wrong. If position has +10% unrealized then reverses to -5%, drawdown looks like ~5% but actually 15%.
**Рекомендация:** Recompute peak inside `update_price` too.

### [E — Architecture] `update_price` not called from any strategy path
**Файл:** src/risk/portfolio_tracker.py:125-133 + grep on strategies
**Серьёзность:** ВЫСОКАЯ
**Описание:** Unrealized P&L stays at the value set on `update_fill` (line 113 — initially 0 since `current_price=price`). No on_bar handler calls `update_price`. So `get_daily_pnl()`'s "unrealized" component is always 0 until close. Combined with 3.6.4, drawdown gate is structurally broken.
**Рекомендация:** Call `tracker.update_price(symbol, bar.close)` from `on_bar` for each open position symbol.

---

## 3.7 signal_bridge.py

### [E — Architecture] Connection-per-call pattern + cross-process WAL race
**Файл:** src/execution/signal_bridge.py:64-69, 176-208
**Серьёзность:** ВЫСОКАЯ
**Описание:** Every `log_signal` / `close_signal` / `update_metrics` opens and closes a new SQLite connection. With Telegram-bot polling and a 15m strategy writing simultaneously, "database is locked" errors are likely under load. WAL mode helps concurrent reads but writers still serialize across processes; SQLite's default busy_timeout is 0 — the writer immediately errors instead of waiting.
**Best Practice:** Set `PRAGMA busy_timeout=5000` on every connection; reuse one persistent connection per process; batch inserts where possible.
**Рекомендация:** Add `conn.execute("PRAGMA busy_timeout=5000")` in `_connect`.

### [E — Architecture] `threading.Lock` provides no cross-process safety
**Файл:** src/execution/signal_bridge.py:53
**Серьёзность:** СРЕДНЯЯ
**Описание:** Doc says "thread-safe via threading.Lock". This only serializes writes within one process. The Telegram bot is a separate process — the lock doesn't help across processes. WAL + busy_timeout do, but the design comment is misleading.
**Рекомендация:** Document that cross-process safety relies on SQLite WAL + busy_timeout; remove the misleading "thread-safe" comment or scope it correctly.

### [B — Math] `pnl_pct` in `close_signal` from `event.realized_return * 100`
**Файл:** src/execution/strategies/ml_strategy.py:635
**Серьёзность:** СРЕДНЯЯ
**Описание:** Nautilus `realized_return` is computed from fill prices but typically does NOT account for funding-rate payments accumulated during the hold. For a 24-hour hold across 3 funding settlements, this can be ±0.05% of notional that the recorded pnl misses. Stats compounded from these values drift from on-exchange truth.
**Рекомендация:** Read realized_pnl from `/fapi/v1/userTrades` (includes income breakdown) post-close, or query account income.

### [E — Architecture] Schema migration via `ALTER TABLE ADD COLUMN` is one-way
**Файл:** src/execution/signal_bridge.py:136-143
**Серьёзность:** НИЗКАЯ
**Описание:** A subsequent rename or type change can't be done with `ALTER TABLE` in SQLite. Future migrations will need ad-hoc CREATE/COPY scripts. Document the migration policy.

---

## 3.8 reconciler.py

### [E — Architecture] CRITICAL: `PositionReconciler` is never called from any strategy
**Файл:** src/execution/reconciler.py (whole) + grep on strategies
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** No `set_timer` schedules reconciliation; no `on_start` invokes it once. Orphan and ghost position detection — sold as a safety feature — does not run in production. Combined with 3.1.2 (entry/SL gap) this means a crash-induced orphan stays orphan until manual operator intervention.
**Рекомендация:** Schedule `await reconciler.reconcile(...)` at startup and every 15 minutes via Nautilus `set_timer` with an async callback.

### [E — Architecture] `auto_fix=True` doesn't actually fix anything
**Файл:** src/execution/reconciler.py:215-228
**Серьёзность:** ВЫСОКАЯ
**Описание:** `auto_fix` branch only appends strings like `"Would close orphan {symbol}"` to `actions_taken`. No API call is made. The mode label is misleading — it's logging-only.
**Рекомендация:** Either rename to `report_only` or actually implement the orphan close (submit market reduce-only against the orphan position).

### [A — Position Safety] Orphan detection only checks `positionAmt`, ignores exchange-side stop orders
**Файл:** src/execution/reconciler.py:127-138
**Серьёзность:** ВЫСОКАЯ
**Описание:** A position with a perfectly-placed exchange-side SL but missing from `internal_positions` is flagged as orphan. If `auto_fix` were really implemented and closed it, you'd close a properly-protected position. Real reconciler should also enumerate open stop-market orders via `/fapi/v1/openOrders` and only flag an orphan if there's no protective stop.
**Рекомендация:** Enrich reconciliation: fetch openOrders and only treat as orphan if no reduce-only stop exists.

### [E — Architecture] `recvWindow=5000` fixed; no clock-skew handling
**Файл:** src/execution/reconciler.py:242
**Серьёзность:** СРЕДНЯЯ
**Описание:** If the server clock is even ~5s skewed, signed requests fail with `-1021`. The reconciler logs error and returns None → silent reconciliation failure.
**Рекомендация:** Query `/fapi/v1/time` once at init, store the offset, apply to all signed requests.

---

## 3.9 reconciler_signals.py

### [A — Position Safety] Reconciler uses signal's *original* SL/TP, not exchange-side actuals
**Файл:** src/execution/reconciler_signals.py:289-298
**Серьёзность:** ВЫСОКАЯ
**Описание:** Pulls `entry_price`, `stop_loss`, `take_profit` from the `signals_log` row (i.e., the values the strategy *intended* to use at signal time). If the SL was never actually placed on the exchange (3.1.2 scenario) or was placed at a different price (tick rounding, manual modification), the reconciler still books PnL as if the SL fired. Reported metrics diverge from on-exchange truth.
**Рекомендация:** Cross-reference against `/fapi/v1/userTrades` for the symbol+time-window; use real fills for `close_price`.

### [B — Math] `skip_recent_bars = 2` doesn't cover triple-barrier max-holding
**Файл:** src/execution/reconciler_signals.py:281
**Серьёзность:** ВЫСОКАЯ
**Описание:** Skips signals younger than `2 × bar_minutes` minutes. On 4H: 8h; max_holding is typically 6 bars = 24h. So a signal at 4h-old is reconciled but the real trade hasn't completed its barrier window yet. Reconciler may declare "still_open" prematurely or mark a barrier touch that the real bot would have ignored (vertical barrier overrides early SL/TP touch in some triple-barrier variants — though here SL touches always close).
**Best Practice:** Use `max_holding_bars` from the model config, not a constant.
**Рекомендация:** Configure `skip_recent_bars >= max_holding_bars`.

### [B — Math] Same-bar SL/TP tie always resolves to SL
**Файл:** src/execution/reconciler_signals.py:231-236
**Серьёзность:** СРЕДНЯЯ
**Описание:** Conservative choice. But when the real exchange-side SL fired at TP price (or didn't fire at all because TP came first within the bar), the reconciler records a loss for what was actually a win. Inflates loss-rate metric used by `calculate_t_stat` and `EvaluationResult.win_rate` downstream.
**Best Practice:** Use 1-minute klines within the contested bar to disambiguate. Or read actual fill from `/fapi/v1/userTrades`.

### [E — Architecture] `BinanceRESTPriceSource` has `for _ in range(200)` hard page cap
**Файл:** src/execution/reconciler_signals.py:110
**Серьёзность:** НИЗКАЯ
**Описание:** 200 × 1500 = 300k bars max. On 1m bars that's ~7 months — fine. Comment-worthy boundary but not a bug.

---

## Summary — Execution & Risk

### Counts by severity
- **КРИТИЧЕСКАЯ:** 6
  - ml_strategy.py: no dead-man's switch in 4H strategy
  - ml_strategy.py: entry/SL gap + `_pending_sl_params` not persisted
  - meta_strategy.py: `_apply_meta_gate` never called
  - circuit_breaker.py: `CircuitBreaker.check` never called from strategies
  - reconciler.py: `PositionReconciler.reconcile` never scheduled
  - portfolio_tracker.py: cash math credits notional twice per round-trip (3.6.1) — could be re-classed CRITICAL given accounting consequences
- **ВЫСОКАЯ:** 19
- **СРЕДНЯЯ:** 11
- **НИЗКАЯ / INFO:** 4

### Top-5 most critical (capital-risk)
1. **CircuitBreaker dead** (circuit_breaker.py — never called). The "-3% daily / -15% kill switch" advertised in master document doesn't run. The only daily-loss gate is `risk_engine._check_daily_loss` which (a) uses the `_initial_equity` denominator that drifts and (b) reads `daily_pnl_pct` from `_daily_realized_pnl` which **isn't persisted across restart** — restart bypasses the daily limit.
2. **PositionReconciler dead** (reconciler.py — never scheduled). Orphan/ghost detection that's supposed to recover from crashes does not run. Combined with #3 below, post-crash recovery has no safety net.
3. **4H strategy has no heartbeat-based kill** (ml_strategy.py vs 15m). 4H is the production bot; it crashes → unprotected position with no external watchdog. 15m has heartbeat (good); 4H does not.
4. **PortfolioTracker cash math** (3.6.1). Cash is credited `quantity * avg_entry_price + realized_pnl` on close, but never debited on open. After ~10 round-trips, reported equity is wildly inflated → drawdown gate and position sizing both broken on the high side. *This affects every backtest and live run.*
5. **Entry-fill → SL submission gap** (3.1.2). `_pending_sl_params` is in-memory only; a crash between entry submit and SL submit leaves the next process unable to recognize the fill as an entry → no SL is ever placed.

### Capital-threatening problems (yes/no)
**YES — multiple.** The combination of:
- Dead CircuitBreaker + dead Reconciler + missing 4H heartbeat + non-persistent SL params
- Plus PortfolioTracker accounting that mis-reports equity by `notional`-per-trade

means a 4H production run risks at least these failure modes:
- Lose more than -3% / day (filter bypassed via restart)
- Have orphan positions accumulate after any crash
- Operate at miscalibrated position sizes because tracker equity drifts
- Take entries without a stop-loss after a crash window

These are not subtle bugs; each is a known production-trading anti-pattern. **Strong recommendation: pause real-money 4H trading until 3.1.1, 3.1.2, 3.5.1, 3.6.1 are fixed.**

### Other systemic observations
- Three subsystems (`CircuitBreaker`, `PositionReconciler`, `MetaSignalGate`) exist as fully-formed implementations but **none are wired into the live strategy lifecycle**. This is a strong signal that the master-document "safety + meta-labeling" story is more aspirational than implemented.
- The 4H production strategy (`MLTradingStrategy`) is missing safety features its 15m sibling has (heartbeat, etc.) — a clear inversion of risk-priority: the 4H bot trades larger notionals less frequently, so its per-trade tail risk is *higher*.
- Risk-engine filters double-check things CircuitBreaker is supposed to handle, and the duplication has math bugs (volatility filter, expected return). Consolidation would reduce surface area.

---

## Раздел 4: Data Infrastructure

### Файлы:
- src/ingestion/data_store.py
- src/ingestion/binance_downloader.py
- src/ingestion/parquet_converter.py
- src/ingestion/data_quality.py
- src/ingestion/live_feed.py
- src/execution/data_catalog.py

---

## 4.1 binance_downloader.py

### [A — Data Correctness] CRITICAL: `download_funding_rate` requests *daily* URLs but Binance Data Portal serves funding-rate as *monthly* files
**Файл:** src/ingestion/binance_downloader.py:339-360, esp. line 354
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** URL constructed as `/daily/fundingRate/{symbol}/{symbol}-fundingRate-{YYYY-MM-DD}.zip`. Binance Data Portal (`data.binance.vision`) only publishes funding-rate as *monthly* archives at `/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{YYYY-MM}.zip`. Daily funding-rate URLs return 404 for every date. The downloader silently classifies all of them as `not_found` (which the orchestrator counts as "normal").
**Best Practice:** Binance Data Portal docs: kline files are daily OR monthly; *funding rate is monthly only* per https://github.com/binance/binance-public-data#user-content-trading-data.
**Влияние на trading:** **The training pipeline has no historical funding-rate data unless it was sourced elsewhere.** All `funding_*` features (funding_zscore_7d/30d, funding_cum_24h, funding_extreme, etc.) train on zero-filled values (`_zero_funding`), and the model "learns" that funding is always 0. Live inference then gets *real* funding values — a massive train/serve distribution shift on a critical feature.
**Рекомендация:** Switch to monthly URLs and monthly date iteration; OR fetch via REST `/fapi/v1/fundingRate` paginated. Verify by counting actual rows in `data/features/exchange=BINANCE_UM/symbol=BTCUSDT/funding_rate/` parquets.

### [E — Architecture] No live-API rate-limit / weight tracking
**Файл:** src/ingestion/binance_downloader.py (whole, by absence) + src/execution/strategies/ml_strategy.py:259-275, 1099-1131
**Серьёзность:** ВЫСОКАЯ
**Описание:** The downloader handles Data Portal (static CDN, no weights). But the *live strategy* makes uncoordinated calls to `/fapi/v1/fundingRate`, `/fapi/v1/openInterest`, `/fapi/v1/klines` from inside `MLTradingStrategy.on_start` and `_poll_open_interest`. None of these check Binance's `X-MBX-USED-WEIGHT-1m` response header. With multiple strategy instances (4H + 15m) plus OI polls every 5 minutes plus PositionReconciler (if it were wired), it is possible to exceed Binance's 2400 weight/min limit and earn an IP ban.
**Best Practice:** Centralize all REST calls through a shared client that reads response headers and applies token-bucket back-pressure. binance-futures-connector-python provides this; Hummingbot and Freqtrade both use a centralized rate limiter.
**Влияние на trading:** IP ban during live operation → bot disconnected, positions unmanaged.
**Рекомендация:** Introduce a single `BinanceRestClient` wrapper with weight-tracking and use it everywhere.

### [A — Data Correctness] No publication-lag handling — training data ends 1-2 days before "now"
**Файл:** src/ingestion/binance_downloader.py (whole + absence of gap-bridging)
**Серьёзность:** СРЕДНЯЯ
**Описание:** Data Portal files for day D appear on day D+1 (sometimes D+2). Running `download_all` with `end_date=today` always reports many `not_found` for the most recent dates. The training-time/inference-time gap (2 days) isn't bridged anywhere — `MLTradingStrategy._preload_from_parquet` checks `staleness_h > 2 × bar_period_h` (ml_strategy.py:1276) and falls back to REST, but the *training data itself* still ends at D-2. Recently-trained models miss the last ~12 4H bars of behavior.
**Рекомендация:** Either accept the lag explicitly (document training cutoff), or augment training-time builds with REST fetch for the last N days.

### [E — Architecture] Checksum verification only on initial download; CSV truncation not re-detected
**Файл:** src/ingestion/binance_downloader.py:166-169, 79-81
**Серьёзность:** СРЕДНЯЯ
**Описание:** `_do_download` verifies SHA-256 of the *zip*. Then `extract_zip` extracts CSV and **deletes the zip** (line 80). On subsequent runs, if the CSV is corrupted (disk error, partial write, manual edit), the fast-path `csv_path.exists() and st_size > 0` (line 250) treats it as valid and skips. No CSV-level checksum.
**Рекомендация:** Store sidecar `.csv.sha256` after successful extract; verify on re-runs.

### [E — Architecture] `_collect_paths` silently drops `not_found` results
**Файл:** src/ingestion/binance_downloader.py:288-296
**Серьёзность:** НИЗКАЯ
**Описание:** Returned list is "successful paths" — caller can't tell if `not_found` was 0% or 90% of requests. The `download_all` orchestrator does track this in stats, but typed downloaders (`download_klines`, `download_funding_rate`, ...) don't.
**Рекомендация:** Return `(paths, stats)` tuple, or surface a warning when `not_found / total > 0.1`.

---

## 4.2 parquet_converter.py

### [B — Storage] `try_parse_dates=False, ignore_errors=True` silently drops malformed rows
**Файл:** src/ingestion/parquet_converter.py:262-269
**Серьёзность:** ВЫСОКАЯ
**Описание:** `pl.read_csv(... ignore_errors=True)` silently drops rows that fail dtype conversion. No row-count assertion compares against expected (e.g., 6 bars/day for 4H). A partially-corrupted source CSV is converted to a parquet with fewer rows and no warning. Downstream `data_quality.check_completeness` only checks for *missing date partitions*, not for short partitions.
**Best Practice:** Polars convention is `ignore_errors=False` + explicit `null_values`; record drop counts.
**Влияние на trading:** Stealth data loss propagates into training without alarm.
**Рекомендация:** Compute `expected_rows_for_day` based on bar duration; warn if `len(df) < 0.95 × expected`.

### [B — Storage] Monthly funding-rate partitioned with `date=YYYY-MM` but daily kline with `date=YYYY-MM-DD` — DuckDB hive partition type conflict
**Файл:** src/ingestion/parquet_converter.py:200-214, 249-256
**Серьёзность:** ВЫСОКАЯ (conditional on 4.1.1 being fixed)
**Описание:** `extract_date_from_stem` supports both `YYYY-MM` and `YYYY-MM-DD`, so future funding-rate downloads (once URL bug is fixed) will create monthly partitions `date=2024-01/` alongside daily partitions `date=2024-01-01/` in *the same* symbol tree. DuckDB hive partition discovery requires consistent partition value types; mixed daily/monthly within one `data_type` directory may silently misparse the partition value.
**Рекомендация:** Use different partition keys: `year_month=YYYY-MM` for monthly types, `date=YYYY-MM-DD` for daily. Or always re-bucket monthly funding into daily on conversion.

### [E — Architecture] `convert_csv_to_parquet` overwrites `part-0.parquet` — write races possible
**Файл:** src/ingestion/parquet_converter.py:228-329
**Серьёзность:** СРЕДНЯЯ
**Описание:** No atomic write (`tmp_path.rename(final)`). A crash mid-write leaves a corrupted parquet that subsequent reads will fail. ProcessPoolExecutor workers can theoretically race on the same path if a CSV was re-downloaded.
**Рекомендация:** Write to `part-0.parquet.tmp` and `os.replace`.

### [B — Storage] `ROW_GROUP_SIZE=131_072` is bad for small daily 4H/1D parquets
**Файл:** src/ingestion/parquet_converter.py:151
**Серьёзность:** НИЗКАЯ
**Описание:** A daily 4H parquet has 6 rows; daily 1D has 1. Row group size 128k means *every file is a single row group* → predicate pushdown on `open_time` operates at the *file* level (which Hive partitioning already handles). For agg_trades (millions of rows/day), 128k is OK. For metrics (~288 rows/day), one row group per file. So this setting is fine in practice — INFO only.

### [E — Architecture] `_CONFIGS` and `data_quality._TIMESTAMP_COL` are not in sync
**Файл:** src/ingestion/parquet_converter.py:107-149 vs data_quality.py:38-44
**Серьёзность:** ВЫСОКАЯ
**Описание:** Converter accepts any `klines_<interval>` via `_resolve_config`, but `data_quality._TIMESTAMP_COL` only knows `klines_4h` and `klines_1d`. Quality checks on `klines_1h` / `klines_15m` silently skip gap detection. With the v3 multi-timeframe push, the 15m and 1h datasets are entering production *without* gap auditing.
**Рекомендация:** Replace explicit dict with a generic `klines_*` handler in `data_quality`.

### [B — Storage] Metrics CSV datetime parsed as `%Y-%m-%d %H:%M:%S` — no timezone validation
**Файл:** src/ingestion/parquet_converter.py:286-295
**Серьёзность:** СРЕДНЯЯ
**Описание:** Binance metrics CSV `create_time` is in UTC but the format string has no timezone. `pl.col(...).str.to_datetime` produces a tz-naive datetime; `dt.timestamp("ms")` interprets it as UTC, which happens to be correct, but the code is fragile to a future Polars default change.
**Рекомендация:** Explicit `time_zone="UTC"` in `to_datetime`.

### [B — Storage] `quote_volume`, `taker_buy_quote_volume` listed in KLINES_SCHEMA but `data_quality._KEY_COLS` doesn't include them
**Файл:** src/ingestion/parquet_converter.py:40-45 vs data_quality.py:58
**Серьёзность:** НИЗКАЯ
**Описание:** If those columns become all-null due to a Binance schema change, no QC check would notice. Minor.

---

## 4.3 data_store.py

### [A — Data Correctness] `get_klines` end-bound is INCLUSIVE while `data_catalog.load_bar_data` is EXCLUSIVE
**Файл:** src/ingestion/data_store.py:132 (`<=`) vs src/execution/data_catalog.py:115 (`<`)
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same `end=datetime(2024,1,1)` returns the bar at `2024-01-01 00:00 UTC` in `get_klines` but NOT in `load_bar_data`. Tests, training builds, and live preload all use the same temporal endpoints expecting consistent semantics; off-by-one at every cross-system boundary.
**Best Practice:** Standardize on half-open intervals `[start, end)` everywhere (Python / Pandas convention).
**Влияние на trading:** Backtests differ from training-set boundaries by exactly one bar at every cut; for triple-barrier labels with `max_holding=6`, that's also one full label boundary off.
**Рекомендация:** Make `get_klines` use `<` and document the change. Audit all callers.

### [E — Architecture] DuckDB read explicit file list — no partition pruning
**Файл:** src/ingestion/data_store.py:72-88
**Серьёзность:** ВЫСОКАЯ
**Описание:** `read_parquet([file1, file2, ...])` provides DuckDB with an explicit file list. WHERE clauses on `open_time` are evaluated at *row-group* level via parquet stats, but partition-level pruning (skipping entire files based on `date=YYYY-MM-DD`) requires `hive_partitioning=true` and a glob pattern. With 700+ files per symbol per data_type (2-year daily klines), DuckDB opens every parquet footer.
**Best Practice:** `read_parquet('exchange=BINANCE_UM/symbol=X/klines_4h/*/*.parquet', hive_partitioning=true)` then filter on the synthetic `date` partition column to skip files entirely.
**Влияние на trading:** Query slowdown linear in N files — currently 10-100× slower than necessary on large date ranges.
**Рекомендация:** Switch to glob + hive_partitioning; pass date filter as `date BETWEEN ...` in addition to `open_time BETWEEN ...`.

### [E — Architecture] `union_by_name=true` masks schema drift silently
**Файл:** src/ingestion/data_store.py:83
**Серьёзность:** СРЕДНЯЯ
**Описание:** New column added in later parquets → NULL for early parquets. Downstream feature code mostly `.fill_null(0.0)`s → numeric 0 in early data. Old "has column" rows are indistinguishable from "had column but was 0" rows → biased trained model.
**Рекомендация:** Log a warning when union widens the schema.

### [A — Data Correctness] `_to_ms` for tz-naive datetimes assumes UTC
**Файл:** src/ingestion/data_store.py:28-32
**Серьёзность:** СРЕДНЯЯ
**Описание:** Documented behavior but easy to misuse. On a developer machine with non-UTC system TZ, `datetime(2024,1,1)` could mean local time but is forced to UTC. CLI args usually parse to naive datetimes via `dateutil` — single point of subtle off-by-hours bugs.
**Рекомендация:** Raise on tz-naive in production paths.

---

## 4.4 data_quality.py

### [A — Data Correctness] Funding-rate anomaly threshold `|fundingRate| >= 0.05` (5%) is effectively unreachable
**Файл:** src/ingestion/data_quality.py:491
**Серьёзность:** ВЫСОКАЯ
**Описание:** Historical BTC funding has rarely exceeded 0.3% per 8h since 2022. A 5% threshold never fires. Functionally there is no funding-anomaly check.
**Best Practice:** Use rolling-window z-score (|z| > 5) or a tighter absolute floor (e.g., 0.5%).
**Влияние на trading:** Bad funding data (e.g., a stale-feed-induced 1% spike) passes QC into training and feature distribution.
**Рекомендация:** Threshold = 0.005 (0.5%); or compute rolling-30d z-score.

### [A — Data Correctness] Kline anomaly check misses several OHLC invariants
**Файл:** src/ingestion/data_quality.py:489
**Серьёзность:** ВЫСОКАЯ
**Описание:** Only checks `high < low OR close <= 0 OR volume < 0`. Doesn't check:
- `open > high` or `open < low` (impossible)
- `close > high` or `close < low` (impossible)
- `volume == 0` with bar movement (suspicious)
- Per-bar % move > 50% (flash crash candidate that should be flagged for review)
**Best Practice:** OHLC invariants for any quote-driven candle.
**Рекомендация:** Add `OR open > high OR open < low OR close > high OR close < low OR ABS(close - open)/open > 0.5`.

### [A — Data Correctness] `_GAP_THRESHOLD_MS` has no entry for `klines_1h` / `klines_15m`
**Файл:** src/ingestion/data_quality.py:48-54 vs converter supports any `klines_<interval>`
**Серьёзность:** ВЫСОКАЯ
**Описание:** v3 trains 1H and 15m models. `check_gaps` for these data_types returns `{"skipped": True}` (line 202-203). Gap detection — *the* QC for time-series — is disabled silently. With 96 bars/day on 15m, a single 1-hour Binance outage hides as 4 missing rows that produce no completeness or gap signal.
**Рекомендация:** Compute threshold dynamically from interval: e.g. `_threshold_for("klines_15m") = 15*60_000`.

### [F — Production] `THRESHOLD_COMPLETENESS_PCT = 99.9` is brittle vs Binance maintenance windows
**Файл:** src/ingestion/data_quality.py:68
**Серьёзность:** ВЫСОКАЯ
**Описание:** Binance has scheduled maintenance roughly weekly (Tuesday 06:00-09:00 UTC). A 2-3h gap on 15m bars violates 99.9% completeness. The check then triggers `row_passes() = False` and (if wired to alerts) constant noise. Conversely, ignoring this turns the threshold into a no-op.
**Best Practice:** Whitelist known maintenance windows; relax completeness to 99.5% with documented exceptions.
**Рекомендация:** Maintain a `BINANCE_MAINTENANCE.json` list of known down windows; deduct from `expected`.

### [B — Math] `check_clock_drift` samples only `head(1000)` per file — biased toward start-of-day
**Файл:** src/ingestion/data_quality.py:367-374
**Серьёзность:** СРЕДНЯЯ
**Описание:** First 1k rows of a daily agg_trades file = first few seconds of UTC midnight (low-activity period). Late-NY-session drift never observed.
**Рекомендация:** Sample stratified across the file (first/middle/last 333 rows), or use `pl.scan_parquet(...).sample(n=1000, seed=42)`.

### [B — Math] `check_gaps` SQL window function doesn't account for daily file boundaries
**Файл:** src/ingestion/data_quality.py:210-220
**Серьёзность:** СРЕДНЯЯ
**Описание:** `LAG(...) OVER (ORDER BY ts)` operates across the union of all files. For multi-day datasets this is correct, but if the same date partition is written twice (e.g., re-download with extra row at the end), the duplicate row may produce a "gap" of 0ms followed by a "gap" of -X ms. Negative diffs are silently truncated by the WHERE clause.
**Рекомендация:** Add `WHERE diff_ms > 0` explicit + log negative diffs as data corruption.

### [E — Architecture] `THRESHOLD_DRIFT_MS = 50` declared but never used in `row_passes`
**Файл:** src/ingestion/data_quality.py:69, 503-526
**Серьёзность:** НИЗКАЯ
**Описание:** Comment says "live feed only" — but no live-feed file imports it either.

---

## 4.5 live_feed.py

### [E — Architecture] `LiveFeedManager` is orphaned — strategies use Nautilus WS, not this class
**Файл:** src/ingestion/live_feed.py (whole) + grep — no imports from strategies/
**Серьёзность:** ВЫСОКАЯ
**Описание:** No strategy imports `LiveFeedManager`. Production strategies (4H, 15m) get bars via `subscribe_bars(self._bar_type)` (Nautilus's adapter) and funding via `subscribe_data(DataType(BinanceFuturesMarkPriceUpdate))`. This whole module is unused by the live trading path — same pattern as `CircuitBreaker` and `PositionReconciler` (Section 3): infrastructure exists, isn't wired.
**Влияние на trading:** None negative (Nautilus handles it). But all advertised features here (`is_healthy()`, latency monitoring, reconnect counts, funding warning) are not running for the actual trading bot.
**Рекомендация:** Either delete or wire `LiveFeedManager` as a redundant secondary feed for cross-check / drift detection.

### [E — Architecture] `LiveFeedManager.run` creates a *new* event loop — incompatible with Nautilus
**Файл:** src/ingestion/live_feed.py:184-186
**Серьёзность:** СРЕДНЯЯ
**Описание:** `asyncio.new_event_loop()` and `asyncio.set_event_loop(loop)` — would conflict with the loop Nautilus already runs on. Confirms this class is intended for standalone usage only (e.g., a side script).
**Рекомендация:** Document standalone-only; or refactor to accept an external loop.

### [B — Reliability] `retries=-1` in cryptofeed but no exponential backoff log / alert
**Файл:** src/ingestion/live_feed.py:198
**Серьёзность:** СРЕДНЯЯ
**Описание:** Cryptofeed's default retry strategy is internal; this code never observes a reconnect event. `self._reconnects` field is incremented nowhere → always 0. Lost reconnect visibility.
**Рекомендация:** Subscribe to `cryptofeed`'s `EXCEPTION` or connection callback if available; otherwise track `last_tick_time` gaps as proxy.

### [B — Reliability] `LATENCY_WARN_MS = 1000ms` is far too loose for crypto perp
**Файл:** src/ingestion/live_feed.py:127
**Серьёзность:** СРЕДНЯЯ
**Описание:** Normal Binance perp latency: 50-150ms from a co-located VM, 200-400ms from a distant region. 1000ms means "you're already late." A 200ms baseline with 99p ~ 500ms suggests warn threshold 500-700ms.
**Рекомендация:** Set per-deployment based on baseline measurement.

### [E — Architecture] `_last_book_log` dict grows unboundedly with symbol set
**Файл:** src/ingestion/live_feed.py:138, 290
**Серьёзность:** НИЗКАЯ
**Описание:** Tiny leak — only one entry per symbol — but a leak nonetheless.

### [B — Reliability] No persistence of `tick_buffer` across restarts
**Файл:** src/ingestion/live_feed.py:66-89
**Серьёзность:** СРЕДНЯЯ
**Описание:** On crash/restart, all trade/book/funding history is lost. The `max_trades=1000` buffer is also recreated empty — first 1000 ticks after restart are the only available history. Downstream features (VPIN, microstructure) that depend on history have no warm-up.

---

## 4.6 data_catalog.py

### [A — Data Correctness] CRITICAL: `ts_event = open_time × 1e6` — Nautilus convention is bar CLOSE time
**Файл:** src/execution/data_catalog.py:124, 134
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Nautilus's `Bar.ts_event` is the *event timestamp* — for an externally-aggregated bar, this is the time the bar *completed* (= close_time = open_time + bar_duration). Using `open_time` makes Nautilus believe each 4H bar arrived 4 hours earlier than it actually did. Effects:
1. Cross-data ordering: a trade tick at `T + 1h` appears to be *after* a bar that closed at `T + 4h`, but in reality the bar wasn't yet closed.
2. Backtests using both bars and trade-ticks see bars "from the future" relative to ticks — direct lookahead bias.
3. `on_bar` callbacks fire at the bar's open time, not close → strategy logic that depends on bar closure (every strategy in this repo) operates 4h ahead of when it should.
**Best Practice:** Nautilus docs: `ts_event` should equal `close_time` for completed external bars. `ts_init` should equal `ts_event` when the bar arrives at the engine. Most production Nautilus adapters use `(open_time + bar_duration) × 1e6`.
**Влияние на trading:** **Every backtest in this codebase that uses `data_catalog.load_bar_data` is silently lookahead-biased by one full bar.** Reported Sharpe, win-rate, PF are inflated. This is the single highest-impact bug in the data infrastructure.
**Рекомендация:** Change to `ts_ns = (row["open_time"] + bar_duration_ms) * 1_000_000`. Re-run every backtest after fix.

### [A — Data Correctness] `Quantity(max(row["volume"], min_size), ...)` clamps real zero-volume bars to a fake tick
**Файл:** src/execution/data_catalog.py:132, 183
**Серьёзность:** ВЫСОКАЯ
**Описание:** Polite intent (avoid Nautilus rejecting Quantity(0)) but it silently invents `min_size` worth of volume on every truly-empty bar. CVD, volume-ratio, volume-zscore features then see a tiny positive value where reality was zero. Distorts microstructure features for low-activity periods (e.g., weekend overnight 1m bars).
**Best Practice:** Either filter out zero-volume bars or use Nautilus's allowed `Quantity(0)` if the schema permits.
**Рекомендация:** Detect zero-volume rows and emit a flag column; don't fabricate volume.

### [A — Data Correctness] `load_trade_data` uses `.head(sample_size)` — silently truncates chronological tail
**Файл:** src/execution/data_catalog.py:169
**Серьёзность:** ВЫСОКАЯ
**Описание:** With `sample_size=100_000` and a 6-month window of BTC agg_trades (~30M+ rows), the function returns only the first ~hour of trades in the window. Tests/backtests using this think they have full-window trade data but the tail is gone.
**Best Practice:** Documented sample → use random sampling (`pl.col(...).sample(n=sample_size)`) or paginate; never `head` on a sorted time series unless intent is "first N seconds".
**Рекомендация:** Either remove the `head` or rename to `load_trade_data_head`.

### [A — Data Correctness] `load_bar_data` uses `< end_ms` (exclusive) vs `data_store.get_klines` uses `<= end_ms` (inclusive)
**Файл:** src/execution/data_catalog.py:115 vs src/ingestion/data_store.py:132
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same inconsistency as 4.3.1. A backtest spanning `start..end` loads N bars; a training build spans `start..end` loading N+1 bars. Walk-forward validation cross-uses both → off-by-one window boundaries.
**Рекомендация:** Standardize on `<` (half-open).

### [E — Architecture] `_SPECS` hardcoded for 3 symbols — adding a 4th symbol requires code change
**Файл:** src/execution/data_catalog.py:23-45
**Серьёзность:** СРЕДНЯЯ
**Описание:** KeyError at `_SPECS[symbol]` for any unsupported symbol. The wider system has `SYMBOL_ENCODING` in 3 separate places (lgbm_trainer.py, ml_strategy.py, ml_strategy_15m.py) — same data, multiple locations. Drift risk.
**Best Practice:** Fetch from `/fapi/v1/exchangeInfo` once at startup, cache.
**Рекомендация:** Single registry; auto-populated from `exchangeInfo`.

### [B — Storage] `Price(row["open"], precision=pp)` — Nautilus may round-trip through Decimal with precision loss for SOL
**Файл:** src/execution/data_catalog.py:128
**Серьёзность:** НИЗКАЯ
**Описание:** SOLUSDT `price_precision=3` (0.001 increment). Polars stores price as Float64; Nautilus `Price` constructor rounds to precision. For SOL at $100.0005, the parquet's float64 stores correctly but the Price will be `100.001` (rounded). Slight P&L drift on every fill.
**Рекомендация:** Verify by comparing Nautilus-computed P&L vs raw-float P&L on a SOL backtest.

### [E — Architecture] `load_trade_data` doesn't honor `is_buyer_maker` semantic correctly
**Файл:** src/execution/data_catalog.py:177
**Серьёзность:** СРЕДНЯЯ
**Описание:** `is_buyer_maker=True` → SELL was the aggressor (mapped to `AggressorSide.SELLER`). Correct per Binance convention. But this assumes Polars boolean column correctly reflects `True`/`False`; if CSV had "true"/"false" strings, the `pl.Boolean` cast might fail and yield NULLs. With `ignore_errors=True` upstream, these become `False` → mis-attributed aggressor side.
**Рекомендация:** Validate is_buyer_maker null_count in QC.

---

## Summary — Data Infrastructure

### Counts by severity
- **КРИТИЧЕСКАЯ:** 2
  - binance_downloader: `download_funding_rate` uses daily URLs (monthly only on Binance Data Portal)
  - data_catalog: `ts_event = open_time` instead of `close_time` → 1-bar lookahead in every backtest
- **ВЫСОКАЯ:** 16
- **СРЕДНЯЯ:** 11
- **НИЗКАЯ / INFO:** 5

### Top-5 most critical (data-quality / training-validity)

1. **`data_catalog.load_bar_data` sets `ts_event` to bar OPEN time (4.6.1).** Every backtest run via Nautilus + this catalog operates with bars timestamped at the start of the period, not the close — effectively 1 full bar of lookahead. Reported Sharpe/WR/PF in v3 backtests are systematically inflated. This is arguably the most consequential bug found in the entire review.

2. **`download_funding_rate` requests daily URLs but Binance publishes monthly (4.1.1).** Every "fundingTime not_found" silently passes. The training pipeline likely operates on zero-filled funding (via `_zero_funding` fallback in `add_funding_features`), meaning all funding features train on `0.0` while live serves real values. Massive train/serve skew on one of the most important crypto-derivative features.

3. **End-bound inclusive vs exclusive inconsistency (4.3.1 + 4.6.4).** `data_store.get_klines` uses `<=`, `data_catalog.load_bar_data` uses `<`. Same query in different code paths returns different row counts. Walk-forward windows are off-by-one at every boundary.

4. **Live API has no centralized rate limiting (4.1.2).** 4H + 15m strategies + OI polls + reconnect-time klines preloads all hit Binance REST without weight tracking. An IP ban during live trading would disconnect the bot mid-position.

5. **Gap detection unavailable for `klines_1h` / `klines_15m` (4.4.3).** `_GAP_THRESHOLD_MS` only knows 4h/1d. v3 trains on 1h and 15m data with no gap audit — Binance outages of hours go undetected in the training set.

### Are training-data-quality problems present?

**Yes, several with material training impact:**

- **Funding-rate data may be missing entirely** (4.1.1). All `funding_*` features in v3 training likely trained on `0.0`. *Verify immediately by running `wc -l data/features/exchange=BINANCE_UM/symbol=BTCUSDT/funding_rate/*/part-0.parquet | tail`*. If empty, the entire derivatives-feature group is invalid.
- **Backtest results are lookahead-biased by one bar** (4.6.1). Reported v3 metrics need re-running after fix.
- **Zero-volume bars are fabricated with `min_size`** (4.6.2). Distorts CVD/volume features in low-activity periods.
- **`load_trade_data` truncates after 100k trades** (4.6.3). VPIN and microstructure feature precomputation may be using only the first hour of trades per day.
- **Kline/funding anomaly checks too loose** (4.4.1, 4.4.2). Bad-data events pass into training.
- **Gap detection silent on new TFs** (4.4.3). v3 TFs untracked.
- **Schema drift masked** (4.3.3). New columns become NULL→0 in old data, silently.

### Recommended verification commands
```bash
# Check whether funding-rate parquets actually have rows
ls -la /home/hashiflame/AtomiCortex/data/features/exchange=BINANCE_UM/symbol=BTCUSDT/funding_rate/ 2>/dev/null | head
duckdb -c "SELECT COUNT(*) FROM read_parquet('data/features/exchange=BINANCE_UM/symbol=BTCUSDT/funding_rate/**/*.parquet')"

# Confirm ts_event semantics in a recent backtest run
grep -r "ts_event.*open_time\|ts_event.*close_time" src/execution/data_catalog.py
```

### Overall state

The data infrastructure has **two production-impacting bugs** (`ts_event=open_time` lookahead and missing-funding-rate-downloader), plus systematic inconsistencies (inclusive vs exclusive bounds, gap-detection coverage gaps, anomaly thresholds set too loose). The downloader/converter chain is well-designed *structurally* but has *content* bugs that quietly invalidate the data it produces. **Strong recommendation: before any further v3 model retraining or paper-trading metric reporting, (a) audit funding-rate parquet row counts, (b) fix `data_catalog.ts_event`, (c) re-run training and backtests.**

---

## Раздел 5: Telegram Bot

### Файлы:
- src/telegram_bot/bot.py
- src/telegram_bot/database.py
- src/telegram_bot/handlers_free.py
- src/telegram_bot/handlers_premium.py
- src/telegram_bot/handlers_owner.py
- src/telegram_bot/broadcaster.py
- src/telegram_bot/signal_formatter.py
- src/telegram_bot/signal_poller.py
- src/telegram_bot/payments_stars.py
- src/telegram_bot/payments_crypto.py
- src/telegram_bot/roles.py
- src/telegram_bot/timeframes.py

---

## 5.1 roles.py

### [A — Security] `_ensure_user` writes on *every* update — heavy DB churn
**Файл:** src/telegram_bot/roles.py:75-118
**Серьёзность:** ВЫСОКАЯ
**Описание:** Every decorated handler (every command, every button press, every callback) invokes `_ensure_user`, which:
1. Calls `db.get_user(user_id)` — possible UPDATE if downgrading expired premium.
2. Calls `db.create_user(...)` *unconditionally* (line 108-112) even for existing users, just to refresh username/first_name → an `INSERT … ON CONFLICT DO UPDATE` write on every request.
3. May call `db.set_role(...)` if owner.

So each user action triggers 2-3 DB writes. With cross-process WAL but no `busy_timeout` (see 5.2.5), `database is locked` errors are likely under any concurrent load.
**Best Practice:** Update username only when it has changed (compare in-memory cached value); throttle to once per N minutes per user.
**Влияние:** Slowdowns on shared DB under load; potential `database is locked` errors that look intermittent.
**Рекомендация:** Cache user dict for ~60s per user_id; write only on real change.

### [A — Security] `OWNER_ID` resolved at module-import time
**Файл:** src/telegram_bot/roles.py:53
**Серьёзность:** СРЕДНЯЯ
**Описание:** `OWNER_ID = get_owner_id()` runs at import; setting `TELEGRAM_ADMIN_ID` after import has no effect. Tests using `monkeypatch.setenv` after import fail silently — the protection is real but the import order is fragile.
**Рекомендация:** Make `OWNER_ID` a function call, not a module constant; resolve lazily.

### [A — Security] Expired-premium auto-downgrade silently swallows parse errors
**Файл:** src/telegram_bot/database.py:175-193 (called from roles.py)
**Серьёзность:** СРЕДНЯЯ
**Описание:** `try: datetime.fromisoformat(expires_at) ... except (ValueError, TypeError): pass`. A corrupted `expires_at` value keeps the user at premium forever. No alert, no log.
**Рекомендация:** Log + flag user for owner review when expires_at fails to parse.

---

## 5.2 database.py

### [B — Architecture] CRITICAL: One `Database` class writes Telegram-bot schema into the *trading* SQLite DB
**Файл:** src/telegram_bot/database.py:35-148 (init) + src/telegram_bot/bot.py:146 + src/telegram_bot/handlers_free.py:161-162
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `Database.__init__` unconditionally calls `_init_db()`, which:
- Creates `users`, `payments`, `bot_events` tables.
- Adds indexes.
- Runs `ALTER TABLE signals_log ADD COLUMN timeframe`.
- Sets `PRAGMA journal_mode=WAL`.

But the Telegram bot code wraps the *trading-bot's* DBs (`data/atomicortex.db`, `data/atomicortex_15m.db`, etc.) with the same class to read signals — see `TelegramBot.build` line 146 (`self._app.bot_data["shared_db"] = Database(shared_db_path)`) and `_resolve_stat_dbs` line 161-162 (`Database(p)` for each trading DB). Each wrap **writes** to the trading DB:
1. Adds `users` / `payments` tables that don't belong there.
2. Alters `signals_log` schema on a DB another process owns.
3. Runs PRAGMA writes concurrently with the trading process's writes.

**Best Practice:** Strict separation between "owner of the schema" (SignalBridge) and "reader" (Telegram bot). A read-only wrapper must not run DDL.
**Влияние:** SQLite locking races, possibility of writing the bot's schema into the wrong DB, polluted trading DB with irrelevant tables, and altered schema that the trading process didn't expect on next restart.
**Рекомендация:** Split `Database` into `OwnerDatabase` (creates schema, writes users/payments) and `ReadOnlyDatabase` (`PRAGMA query_only=ON` after `_connect`, no `_init_db`). Use the latter for `shared_db` and `_resolve_stat_dbs`.

### [E — Architecture] `_resolve_stat_dbs` instantiates fresh `Database(p)` per call (no caching)
**Файл:** src/telegram_bot/handlers_free.py:155-166
**Серьёзность:** ВЫСОКАЯ
**Описание:** Every call to `cmd_stats`, `format_bot_status`, `cmd_signal`, `cmd_history`, `cmd_performance` creates new `Database` objects. Combined with 5.2.1, each call also re-runs the (write-heavy) `_init_db`. For interactive commands this is one extra round of CREATE/ALTER/PRAGMA per click.
**Рекомендация:** Cache the `Database` instances in `app.bot_data`; instantiate once in `bot.py`.

### [E — Architecture] Connection-per-call without `PRAGMA busy_timeout`
**Файл:** src/telegram_bot/database.py:45-50
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same root cause as Section 3.7.1: `sqlite3.connect(...)` defaults to `busy_timeout=0`. Any concurrent writer (trading bot + telegram bot + telegram poller + reconciler_signals) immediately gets `OperationalError: database is locked`.
**Рекомендация:** `conn.execute("PRAGMA busy_timeout=5000")` in `_connect`.

### [E — Architecture] `get_user` performs UPDATE inside what's nominally a read
**Файл:** src/telegram_bot/database.py:154-197
**Серьёзность:** СРЕДНЯЯ
**Описание:** Side effect in a "get" method (auto-downgrade expired premium). Surprising; concurrent callers may both attempt the UPDATE.
**Рекомендация:** Move auto-downgrade to a scheduled job; keep `get_user` pure.

### [A — Security] `set_role` accepts arbitrary role strings — no validation
**Файл:** src/telegram_bot/database.py:233-249
**Серьёзность:** ВЫСОКАЯ
**Описание:** `db.set_role(user_id, "owner")` will silently promote anyone. Currently `cmd_grant` validates input and blocks `owner` (handlers_owner.py:117), but any other internal code path (including future refactors) could bypass. Defense-in-depth: enforce at the DB layer.
**Рекомендация:** Whitelist role values inside `set_role`; raise `ValueError` for `"owner"` unless caller passes a `force=True` flag.

### [E — Architecture] `pnl_pct` column allows mixed schema: `breakeven` not in `signals_log.result` enum (no enum) yet referenced in queries
**Файл:** src/telegram_bot/database.py:412, 463 (handlers query `breakeven`) but `_init_db` schema doesn't constrain it
**Серьёзность:** НИЗКАЯ
**Описание:** Drift risk — column is free-text. A future bug writing `'BREAKEVEN'` (uppercase) would fail filters silently. Add CHECK constraint.

### [E — Architecture] `merge_stats` aggregates `avg_confidence` weighted by win+loss but ignores open signals
**Файл:** src/telegram_bot/database.py:683-685
**Серьёзность:** НИЗКАЯ
**Описание:** Open signals' confidences are excluded from the weighted average. Mathematically defensible but reported "avg confidence" diverges from a user's intuition that "all displayed signals" contribute.

---

## 5.3 payments_stars.py

### [A — Security] CRITICAL: `pre_checkout_handler` approves any payload that parses — no amount/duplicate check
**Файл:** src/telegram_bot/payments_stars.py:99-118
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `pre_checkout_handler` accepts the checkout if the payload parses to `(days, user_id)` — but doesn't verify:
1. **Amount matches the invoice's expected price** — a manipulated `total_amount` (Telegram should prevent, but defense in depth) would still pass.
2. **Payload is not a replay** — if a user already paid for that exact `(days, user_id)` payload, the checkout proceeds and they're charged again. The dedup check is only in `successful_payment_handler` (line 141), at which point the user has already been billed and needs a refund.
3. **No rate limit** — a stuck or malicious client can spam invoices.

**Best Practice:** Telegram Bot Payments docs recommend rejecting (`ok=False`) when the order is no longer valid. Verify amount, payload integrity, user not already paid for this exact subscription window.
**Влияние:** Possible double-charge → manual refund overhead; bad UX.
**Рекомендация:** Look up `db.get_payment_by_payload(payload)`; if status='paid' → `ok=False, error_message="Already paid"`. Validate amount against the invoice's expected price.

### [A — Security] No `telegram_payment_charge_id` stored — idempotency relies on payload only
**Файл:** src/telegram_bot/payments_stars.py:121-185
**Серьёзность:** ВЫСОКАЯ
**Описание:** Telegram's `SuccessfulPayment` carries `telegram_payment_charge_id` — the canonical idempotency key. The code only checks `db.get_payment_by_payload(payload)`. Payload is `premium_{days}d_{user_id}` — same value for every purchase of the same plan by the same user. So a user buying 30d twice (legitimate: extend) reuses the same payload, and the "already paid" check at line 141 *blocks the second purchase* incorrectly. Conversely, two different `payment_charge_id`s for the same payload look identical.
**Best Practice:** Store and check by `telegram_payment_charge_id`. Payload is for *correlating* invoices, not for dedup.
**Влияние:** Legitimate renewals via Stars are blocked with "already processed" if user purchased before. Or, double-charges aren't caught because dedup is on the wrong key.
**Рекомендация:** Add `telegram_payment_charge_id` column to `payments`; check by it.

### [A — Security] No refund handler
**Файл:** src/telegram_bot/payments_stars.py (whole, by absence)
**Серьёзность:** ВЫСОКАЯ
**Описание:** Telegram Stars supports `refundStarPayment`. If a refund occurs, premium isn't revoked. The user keeps premium access even after their money is returned.
**Рекомендация:** Subscribe to `RefundedPayment` updates (PTB v21 supports via `MessageHandler(filters.SUCCESSFUL_PAYMENT_REFUNDED, ...)` or equivalent); revoke role and downgrade.

### [E — Architecture] `_STARS_TO_USD = 0.013` hardcoded conversion goes stale
**Файл:** src/telegram_bot/payments_stars.py:24
**Серьёзность:** СРЕДНЯЯ
**Описание:** Stars price has changed over time (current rate ~$0.018 in late 2025/2026). Revenue stats and the `amount_usd` column drift from reality.
**Рекомендация:** Make it a config setting; document the date of last refresh.

### [A — Security] Bot may charge for and activate plans with payload `(days, user_id)` mismatched to the invoice's user
**Файл:** src/telegram_bot/payments_stars.py:30-39, 137
**Серьёзность:** СРЕДНЯЯ
**Описание:** `pre_checkout_handler` parses `user_id` from payload but doesn't verify it matches the user making the payment (Telegram delivers the buyer in the update). If anyone could craft a custom invoice link with someone else's user_id payload, they could pay to upgrade a third party. Telegram's own UI normally prevents arbitrary payload tampering, but the server-side code should defensively assert `payload_user_id == query.from_user.id`.
**Рекомендация:** Add equality check in pre_checkout and successful_payment.

---

## 5.4 payments_crypto.py

### [A — Security] HIGH: `_processed_ids` is in-memory only → invoice replay window on restart
**Файл:** src/telegram_bot/payments_crypto.py:54-55, 196-208
**Серьёзность:** ВЫСОКАЯ
**Описание:** `_processed_ids = set()` resides only in process memory. On restart, every paid invoice the API returns is re-considered. The DB-level dedup (`existing["status"] == "paid"`) saves the activation from firing again — *if the payment row exists with status=paid*. But if a paid invoice has a payload the bot has never seen (e.g., manually-created invoice via CryptoBot UI matching `premium_30d_X` format by coincidence or social engineering), it activates premium without any DB row.
**Best Practice:** Persist `_processed_ids` to DB; also verify that the activated `payload` exists in `payments` table with the correct user_id.
**Влияние:** Possible unauthorized premium activation if attacker discovers payload format.
**Рекомендация:** Persist processed_invoice_ids; require pre-existing `payments` row before activation.

### [A — Security] `amount` from CryptoBot response is not verified against the invoice's recorded amount
**Файл:** src/telegram_bot/payments_crypto.py:226, 238-263
**Серьёзность:** ВЫСОКАЯ
**Описание:** `_activate_payment` accepts `amount_usd=float(inv.get("amount", "0"))` from the API response. The `days` count is parsed from payload. CryptoBot itself prevents partial payment, but defensive code should compare the polled `amount` against the DB-stored expected amount and reject mismatches.
**Рекомендация:** Look up the payment row created at invoice time; assert `abs(polled_amount - expected) < 0.01`.

### [E — Architecture] `stop_polling` cancels task but doesn't await — race on shutdown
**Файл:** src/telegram_bot/payments_crypto.py:159-164
**Серьёзность:** СРЕДНЯЯ
**Описание:** `self._polling_task.cancel()` then immediate `self._polling_task = None`. The cancelled coroutine may still be in flight at next interpreter exit → "Task was destroyed but it is pending" warnings; possibly a partial DB write.
**Рекомендация:** `await self._polling_task` inside an async stop method.

### [E — Architecture] `_processed_ids` grows unbounded
**Файл:** src/telegram_bot/payments_crypto.py:55
**Серьёзность:** НИЗКАЯ
**Описание:** Set never trimmed. Memory leak ~50 bytes per invoice; in years, modest but a leak.

### [B — Reliability] `get_paid_invoices` returns up to N invoices (CryptoBot pagination default); old paid invoices may be missed
**Файл:** src/telegram_bot/payments_crypto.py:125-144
**Серьёзность:** СРЕДНЯЯ
**Описание:** API returns most-recent N. If the poller is down for a long time, older paid invoices fall off and never activate.
**Рекомендация:** Use `offset` + paginate, OR pass `count=`/`since=` cursor in API call.

---

## 5.5 broadcaster.py

### [F — Production] HIGH: Telegram global rate limit (30 msg/sec) is not enforced
**Файл:** src/telegram_bot/broadcaster.py:26-30, 256-261
**Серьёзность:** ВЫСОКАЯ
**Описание:** `_BROADCAST_SEMAPHORE = asyncio.Semaphore(25)` controls *concurrency*, not *rate*. 25 in-flight sends can complete in <100ms → easily exceeds Telegram's 30/sec global limit. PTB v21 has built-in rate limiting via `AIORateLimiter` but it's not enabled here. Hitting flood limits triggers a `429` with a `retry_after` of up to several minutes, which the retry logic (`_send_with_retry`) handles with simple `0.5 × 2^attempt` backoff — way less than `retry_after` typically requires.
**Best Practice:** Use PTB v21 `Application.builder().rate_limiter(AIORateLimiter())`; respect `RetryAfter` exceptions; throttle to 25-28 msg/sec for safety.
**Влияние:** Bot temporarily banned by Telegram during signal storms; missed broadcasts.
**Рекомендация:** Add `AIORateLimiter`; catch `RetryAfter` specifically and sleep its `retry_after` value.

### [F — Production] `_send_with_retry` catches all exceptions — retries on permanent 403s
**Файл:** src/telegram_bot/broadcaster.py:221-238
**Серьёзность:** ВЫСОКАЯ
**Описание:** `except Exception` catches `Forbidden` (user blocked the bot), `BadRequest` (invalid chat_id), etc. For these, all 3 retries are wasted, consuming 1.5s + flood-control headroom.
**Best Practice:** Distinguish `Forbidden` / `BadRequest` (don't retry) from `TimedOut` / `NetworkError` / `RetryAfter` (retry).
**Рекомендация:** Match exception types; only retry transient.

### [F — Production] No persistent suppression of blocked users
**Файл:** src/telegram_bot/broadcaster.py (whole, by absence)
**Серьёзность:** СРЕДНЯЯ
**Описание:** A user who blocked the bot still receives 3 retries per broadcast forever. Should mark them `is_banned` or add a `is_blocked` column to skip.
**Рекомендация:** On `Forbidden` exception, call `db.ban_user(uid)` or a dedicated "blocked" flag.

### [F — Production] `_send_to_min_role` and `_send_role_filtered` re-query users from DB per broadcast
**Файл:** src/telegram_bot/broadcaster.py:246, 272
**Серьёзность:** СРЕДНЯЯ
**Описание:** Each broadcast does `db.get_non_banned_users()` — SELECT * over the whole table. For 100s of users, fine; thousands, expensive. Cache.

### [E — Architecture] `broadcast_signal` interprets `direction` as both int and str
**Файл:** src/telegram_bot/broadcaster.py:52-55
**Серьёзность:** НИЗКАЯ
**Описание:** Legacy +1/-1 vs new 'long'/'short' supported. Currently OK but suggests schema migration was incomplete. A future enum/check constraint would clean this up.

---

## 5.6 signal_poller.py

### [F — Production] CRITICAL: closed-signal broadcasts have no deduplication → users receive duplicates
**Файл:** src/telegram_bot/signal_poller.py:240-267
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `_check_closed_signals` queries `closed_at > now()-2min`. With `poll_interval=30s`, a signal closed at 12:00:00 is fetched at 12:00:00, 12:00:30, 12:01:00, 12:01:30 — and broadcast 4 times. There's no per-signal "broadcasted_at" mark; the broadcaster has no dedup memory either. Users receive 4× duplicate "POSITION CLOSED" notifications per close event.
**Best Practice:** Track a `last_close_broadcast_id` per DB, same pattern as `_last_signal_ids` for opens; only broadcast new closes.
**Влияние:** User UX significantly degraded; signal-to-noise ratio drops; possibly hits Telegram flood limits during cascades.
**Рекомендация:** Add `_last_close_signal_ids: dict[str, int]` and filter `WHERE id > ? AND result IN ('win', 'loss', 'breakeven')`.

### [F — Production] HIGH: opened-and-closed-within-30s signals are never broadcast as opened
**Файл:** src/telegram_bot/signal_poller.py:200-216 (filter `result = 'open'`)
**Серьёзность:** ВЫСОКАЯ
**Описание:** `_check_new_signals` queries `WHERE result = 'open'`. If a signal opens and closes within one poll interval (30s, possible on 15m with SL hit quickly), `_check_new_signals` finds it with `result != 'open'` and skips. The user sees only the close, not the original entry signal.
**Рекомендация:** Drop the `result = 'open'` filter; broadcast every new signal regardless of its current status.

### [F — Production] Multi-DB polling is sequential
**Файл:** src/telegram_bot/signal_poller.py:182-194
**Серьёзность:** СРЕДНЯЯ
**Описание:** `for db_path in self._db_paths: await ...`. If the 4H DB hangs for 10s, the 15m DB poll is delayed 10s+. With 3 DBs, worst case is 90s between consecutive polls of the same DB.
**Рекомендация:** `asyncio.gather(*[poll(db) for db in dbs])`.

### [F — Production] `_init_high_water_marks` skips pre-existing records on startup → signal-loss window
**Файл:** src/telegram_bot/signal_poller.py:146-174
**Серьёзность:** ВЫСОКАЯ
**Описание:** When the telegram bot restarts, signals written by the trading bot between (a) trading-bot's write and (b) telegram-bot's startup MAX query are lost — never broadcast. For a 4H signal: ~24h-cadence so low probability per signal. For 15m: ~96/day, real risk.
**Best Practice:** Persist `_last_signal_ids` to disk (or a meta table); restore on startup. Alternatively: broadcast everything ≥ T-5min from startup.
**Рекомендация:** Persist high-water marks; on restart restore from disk, not from max(id).

### [E — Architecture] `signal id` collision across DBs handled by "first match wins"
**Файл:** src/telegram_bot/handlers_premium.py:151-166
**Серьёзность:** СРЕДНЯЯ
**Описание:** `id` is per-DB autoincrement. `find_signal_by_id(sid=42)` could return the 15m DB's signal #42 or the 4H DB's — whichever is newest. User clicks "show details" for a 4H signal and sees a 15m signal. Documented as known limitation but it's a real UX bug.
**Рекомендация:** Use composite key `(timeframe, id)` in callback_data; or globally-unique signal ids (UUID).

---

## 5.7 handlers_owner.py

### [A — Security] HIGH: `cmd_broadcast` sends sequentially with no rate-limiter
**Файл:** src/telegram_bot/handlers_owner.py:210-238
**Серьёзность:** ВЫСОКАЯ
**Описание:** Owner-triggered broadcast loops over `db.get_non_banned_users()` and sends one by one with no retry/throttle. With many users, hits the 30/sec global limit (Section 5.5.1) within ~1 second → subsequent sends are flood-limited (silently fail in this loop's try/except).
**Рекомендация:** Use `Broadcaster._send_to_min_role(message, "free")` instead of re-implementing.

### [A — Security] `cmd_logs` regex redaction misses many secret patterns
**Файл:** src/telegram_bot/handlers_owner.py:28-36
**Серьёзность:** СРЕДНЯЯ
**Описание:** Regex matches `api_key`, `secret`, `password`, `token`, `authorization`. Misses: `private_key`, `auth_token`, `bearer`, `api-key` (with dash), `BINANCE_API_KEY` (env var name shorter form), `session_id`, `cookie`, `client_secret`, `webhook_url`, JWT headers.
**Best Practice:** Use a denylist of explicit env-var names from settings + a broad regex `[A-Z_]+(?:KEY|SECRET|TOKEN|PASSWORD)`.
**Рекомендация:** Expand regex; redact all uppercased env-var-like names containing KEY/SECRET/TOKEN/PASSWORD.

### [A — Security] `cmd_stop_bot` / `cmd_confirm_stop` confirmation state is in module-level dict
**Файл:** src/telegram_bot/handlers_owner.py:284-303
**Серьёзность:** СРЕДНЯЯ
**Описание:** `_pending_stop: dict[int, float]` is process-local. PTB v21 supports concurrent updates; if Application is run with multiple workers in the future, state isn't shared. Currently single-process so works, but fragile.
**Рекомендация:** Persist to DB or use PTB's `chat_data`/`user_data`.

### [E — Architecture] systemctl `--user` calls assume specific deployment
**Файл:** src/telegram_bot/handlers_owner.py:322-326, 345-349, src/telegram_bot/bot.py:551-555
**Серьёзность:** НИЗКАЯ
**Описание:** Hardcoded `systemctl --user atomicortex-bot`. Different deployments (root systemd, docker, supervisord) won't work. Should be configurable.

---

## 5.8 bot.py

### [A — Security] HIGH: Callback handlers `_refresh_health`, `_send_logs_inline`, `_restart_bot_inline` lack role checks
**Файл:** src/telegram_bot/bot.py:370-378 (routing), 486-566 (impls)
**Серьёзность:** ВЫСОКАЯ
**Описание:** Only command handlers carry `@require_role("owner")`. The inline-callback path (`_handle_callback`) routes `health_refresh`, `health_logs_20`, `health_restart` to handlers that perform owner-only actions (restart trading bot, dump logs) **without** any role check. The keyboards are only shown to the owner, but a CallbackQuery with arbitrary `callback_data` from a non-owner user (e.g., by clicking a callback button in a forwarded/group message) would be accepted.

In practice, Telegram delivers a CallbackQuery only to users who can see the message. Owner-only messages are in private chat. **But** if the owner ever forwarded a `/health` message to a group or another chat that contained the inline keyboard, the buttons would work for *every* user in that chat — and the buttons would invoke `_restart_bot_inline` / `_send_logs_inline` without authentication.
**Best Practice:** Defence in depth — every callback handler that performs a privileged action must verify role.
**Влияние:** Possible bot-restart / log-leak by any user who gains access to a forwarded owner message with inline buttons.
**Рекомендация:** Apply `require_role("owner")` to callback handlers, or add an inline role check at the top of each privileged callback branch.

### [A — Security] Same problem for `stats_period_*`, `users_page_*` callbacks
**Файл:** src/telegram_bot/bot.py:380-385, 431-437
**Серьёзность:** СРЕДНЯЯ
**Описание:** Owner-only data (`_build_stats_admin_message`, `_paginate_users` listing all user IDs/usernames) exposed via callbacks with no role check. Same forwarded-message risk.
**Рекомендация:** Same as above.

### [E — Architecture] `Database(shared_db_path)` writes Telegram-bot schema to trading DB at startup
**Файл:** src/telegram_bot/bot.py:146
**Серьёзность:** ВЫСОКАЯ
**Описание:** See 5.2.1 — duplicates that finding.

### [E — Architecture] `_get_shared_db_paths` discovery is one-shot at startup
**Файл:** src/telegram_bot/bot.py:683-699
**Серьёзность:** СРЕДНЯЯ
**Описание:** Existing trading DBs are discovered at bot startup. If the 15m strategy is started later, its DB is invisible until telegram bot is restarted.
**Рекомендация:** Re-discover on each poll cycle in `SignalPoller`.

### [E — Architecture] `re_escape` shadows `re` module imported elsewhere
**Файл:** src/telegram_bot/bot.py:704-709
**Серьёзность:** НИЗКАЯ
**Описание:** Re-importing `re` at the file bottom is unusual style; minor maintainability.

---

## 5.9 handlers_free.py / handlers_premium.py

### [E — Architecture] `_collect_recent` for `/signal` view loads up to 200 rows per DB, then filters in Python
**Файл:** src/telegram_bot/handlers_premium.py:62-86, 105-123
**Серьёзность:** СРЕДНЯЯ
**Описание:** `_collect_recent(..., limit=200)` then Python-side filter and sort. For history with 1000s of signals, this is OK; for `limit=10_000` in `_collect_paginated` (line 97), worse — each call materialises 30k Python dicts.
**Рекомендация:** Push timeframe + result filter into SQL; merge sorted streams.

### [E — Architecture] `cmd_subscribe` marketing claims diverge from runtime config
**Файл:** src/telegram_bot/handlers_free.py:376-384
**Серьёзность:** СРЕДНЯЯ
**Описание:** "Confidence ≥ 65%" — but `RiskConfig.confidence_threshold=0.55` (Section 3.4 + risk_engine.py:46), and the strategy code uses 0.55 (4H) / 0.58 (15m). Marketing claim doesn't match the actual filter. Could be construed as misleading.
**Рекомендация:** Either tighten the threshold to 0.65 OR amend the subscription page.

### [E — Architecture] `cmd_risk` uses hardcoded BTC price `94000.0` and ATR `1500.0`
**Файл:** src/telegram_bot/handlers_premium.py:483-486
**Серьёзность:** НИЗКАЯ
**Описание:** "TG-008: use sensible defaults (not bot_data from another process)". Numbers go stale; if BTC is at $50k or $150k, calculator misleads. Should fetch live price (binance public API, no auth) or read latest from `_latest_signal`.
**Рекомендация:** Use the latest signal's `entry_price` and `atr` (already in DB).

### [E — Architecture] `cmd_funding` calls Binance REST without rate limiting
**Файл:** src/telegram_bot/handlers_premium.py:399-413
**Серьёзность:** НИЗКАЯ
**Описание:** Same broader issue as Section 4.1.2 — no global rate limiter across all Binance REST calls (trading bot + telegram /funding). Each `cmd_funding` invocation costs API weight; multiple premium users hitting it concurrently can compound. /funding's weight is 1 per call, so small.

---

## 5.10 signal_formatter.py / timeframes.py

### [E — Architecture] `signal_formatter.format_signal_card` doesn't escape Telegram special chars
**Файл:** src/telegram_bot/signal_formatter.py (whole)
**Серьёзность:** НИЗКАЯ
**Описание:** `regime`, `symbol` could theoretically contain Markdown reserved chars. Bot sends with default parse mode (None — plain text), so no injection risk. INFO only.

### [E — Architecture] `timeframes.active_timeframes` uses `os.path.exists` — race vs. live DB creation
**Файл:** src/telegram_bot/timeframes.py:64-70
**Серьёзность:** НИЗКАЯ
**Описание:** Result varies between calls during 15m strategy bootstrap. Cache result for the bot's lifetime if you want consistent UI ordering.

---

## Summary — Telegram Bot

### Counts by severity
- **КРИТИЧЕСКАЯ:** 3
  - database: Telegram-bot's Database class writes schema to *trading* DBs every time it's instantiated for a trading DB read
  - payments_stars: pre_checkout approves without amount/duplicate verification
  - signal_poller: closed-signal broadcasts have no dedup → 4× duplicate notifications per close
- **ВЫСОКАЯ:** 16
- **СРЕДНЯЯ:** 14
- **НИЗКАЯ / INFO:** 7

### Top-5 most critical
1. **Trading DBs get Telegram-bot schema written into them on every read (5.2.1).** `Database.__init__` runs `_init_db()` unconditionally, which creates `users` / `payments` tables and runs `ALTER TABLE` on the *trading* DB. Cross-process schema mutation under WAL is a recipe for `database is locked` races and silently-polluted trading DBs.
2. **`pre_checkout_handler` blindly approves payments (5.3.1).** Telegram Stars checkouts approved as long as payload parses; no amount check, no duplicate prevention. A user replaying or constructing an invoice can result in double-charges; refunds require manual intervention.
3. **Duplicate close notifications (5.6.1).** `_check_closed_signals` rebroadcasts the same close event up to 4 times within its 2-minute lookback. Every position close spams users.
4. **Privileged callback handlers without role check (5.8.1).** `health_restart`, `health_logs_20`, `users_page_*`, `stats_period_*` execute owner-only actions via callback queries with no role verification. A forwarded owner message in a group exposes its inline buttons to other chat members.
5. **`_processed_ids` in-memory + missing amount verification in CryptoBot payments (5.4.1 + 5.4.2).** Unauthorized premium activation possible if attacker discovers payload format; partial-amount handling missing.

### Are there security vulnerabilities?

**Yes — several with real impact:**

- **Payment processing:** Both Stars (`pre_checkout` permissive, no `telegram_payment_charge_id` dedup, no refund handler) and Crypto (in-memory processed_ids, no amount verification, payload-only dedup) have weaknesses that could allow double-spends, replay attacks, or premium activation without payment.
- **Privilege escalation via callbacks (5.8.1):** owner-only actions exposed via callback handlers without role verification.
- **Privilege escalation via `set_role` (5.2.5):** any internal code path can promote to owner — no DB-layer guard.
- **Cross-process DB schema mutation (5.2.1):** Telegram-bot writes to the trading DB; could corrupt or race with trading-bot writes.
- **Sensitive-data redaction in logs (5.7.2):** regex misses common secret patterns.
- **Rate-limiting (5.5.1):** missing Telegram-side rate limiter → flood-control bans during broadcast storms.

**Other notable issues:**
- Duplicate/missing broadcasts (5.6.1, 5.6.2) — UX bugs but no security.
- DB churn from `_ensure_user` (5.1.1) — performance.
- Marketing claim mismatch (5.9.2) — minor compliance concern.

### Overall state

Like the previous sections, the Telegram bot is **structurally complete** but the operational details have several real risks. The **two highest-impact themes** are:

1. **Cross-process SQLite sharing without guardrails.** The Telegram bot reads (and inadvertently writes) trading-process DBs as if they were its own. With no `busy_timeout`, no read-only mode, and `_init_db` running on every wrap, every successful run depends on luck of write timing.
2. **Payment-flow simplifications.** Both Stars and CryptoBot implementations skip standard verification steps (telegram_payment_charge_id, amount cross-check, refund handlers). This is not catastrophic at low volume — but at scale, every shortcut becomes a known double-spend vector.

**Recommended fix priority:** 5.2.1 (DB schema isolation) → 5.6.1 (dedup closed signals) → 5.3.1 / 5.4.1 (payment verification) → 5.8.1 (callback role checks).

---

## Раздел 6: Infrastructure & Monitoring

### Файлы:
- src/execution/watchdog.py
- src/execution/heartbeat.py
- src/execution/live_trader.py
- src/execution/paper_trader.py
- src/execution/backtest_runner.py
- src/execution/walk_forward.py
- src/execution/metrics.py
- src/api/main.py
- src/analytics/stats_engine.py
- src/monitoring/metrics_collector.py
- src/monitoring/telegram_reporter.py
- src/config.py

---

## 6.1 watchdog.py + heartbeat.py

### [A — Position Safety] CRITICAL: Watchdog fail-open semantics + 4H bot has no heartbeat → watchdog non-functional for production
**Файл:** src/execution/watchdog.py:355-383 + heartbeat.py + Section 3.1.1
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Three compounding failure modes:
1. Section 3.1.1 — the 4H strategy never starts `HeartbeatManager`; only 15m does. The `atomicortex:heartbeat` key is never written.
2. `_check_heartbeat` returns True (alive) if Redis client is None (line 361) — fail-open; same if heartbeat-check raises (line 383).
3. Default `heartbeat_key="atomicortex:heartbeat"` matches what 4H would write IF wired.

With default config monitoring 4H: either watchdog reads a never-existing key → triggers emergency-close every cycle (15s), or it's been disabled, leaving 4H without a dead-man's switch. Master document's "dead-man's switch" guarantee does not hold.
**Best Practice:** Watchdog should fail-CLOSED on null heartbeat key with explicit "warmup grace period" on startup; emit telemetry per cycle.
**Влияние на trading:** Either constant unwanted emergency closes OR no safety net at all for 4H production.
**Рекомендация:** (a) Wire HeartbeatManager into 4H strategy; (b) fail-closed on null key; (c) explicit grace period after startup/Redis reconnect.

### [A — Position Safety] Redis-restart split-brain: bot live, watchdog flat-closes it
**Файл:** src/execution/watchdog.py:355-383
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Default Redis has no persistence. After a Redis restart, the heartbeat key is gone. Bot continues trading; watchdog sees null → emergency-closes; bot then receives next bar → re-opens positions → double-trade window during recovery.
**Рекомендация:** Track Redis connection uptime via session key; require grace period after Redis reconnect; only fail-closed after grace.

### [A — Position Safety] Emergency close uses MARKET orders without spread sanity check
**Файл:** src/execution/watchdog.py:208-225
**Серьёзность:** ВЫСОКАЯ
**Описание:** `MARKET reduceOnly=true` on a thin book during the crash event that *caused* the bot to die can suffer 1-5% slippage. No spread/depth pre-check.
**Рекомендация:** Limit-IOC at last_price ± 0.3% with fallback to market.

### [A — Position Safety] Emergency close cancels orders AFTER closing positions — reverse-direction risk
**Файл:** src/execution/watchdog.py:208-256
**Серьёзность:** ВЫСОКАЯ
**Описание:** Sequence: GET positions → MARKET close → cancel orders. An existing SL order can fire during the market-close round-trip, double-closing into a reverse position.
**Рекомендация:** Cancel orders FIRST, then close positions, then verify positionAmt=0.

### [E — Architecture] Default `heartbeat_key` shared by 4H but mismatched with 15m's `bot_15m_heartbeat`
**Файл:** src/execution/watchdog.py:69 + ml_strategy_15m.py:89
**Серьёзность:** ВЫСОКАЯ
**Описание:** Per-strategy keys exist but watchdog must be configured per-strategy. No warning if one watchdog instance is started without the correct key.
**Рекомендация:** Require explicit `--strategy=4h|15m` at startup; refuse ambiguous defaults.

### [B — Reliability] After firing, watchdog sleeps without recovery telemetry
**Файл:** src/execution/watchdog.py:336-339
**Серьёзность:** СРЕДНЯЯ
**Описание:** Bot may recover during the post-trigger sleep; watchdog silently re-arms with no "BOT RECOVERED" log.
**Рекомендация:** Poll heartbeat every 5s during sleep; log recovery event.

---

## 6.2 backtest_runner.py

### [A — Lookahead] CRITICAL: Inherits `ts_event = open_time` lookahead from data_catalog (Section 4.6.1)
**Файл:** src/execution/backtest_runner.py:109 → src/execution/data_catalog.py:124
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Every backtest run via `BacktestRunner` operates on bars with `ts_event = open_time`. Nautilus treats this as the bar arriving at its open — strategy's `on_bar` fires with full OHLCV at bar open → trades at `bar.open` knowing the bar's close/high/low. **1 bar of lookahead in every v3 backtest.**
**Влияние на trading:** All reported v3 Sharpe/WR/PF — including master-document go-live numbers and walk-forward statistics — are systematically inflated.
**Рекомендация:** Fix data_catalog (`ts_event = (open_time + bar_duration) × 1e6`); re-run all backtests.

### [B — Math] Sharpe annualisation inconsistency: Nautilus 252 vs internal 365
**Файл:** src/execution/backtest_runner.py:150 vs src/execution/metrics.py:55
**Серьёзность:** ВЫСОКАЯ
**Описание:** Nautilus reports `Sharpe Ratio (252 days)`. Internal `metrics.py` uses 365. Differ by √(365/252) ≈ 1.205 = +20.5% inflation in the internal Sharpe vs Nautilus's. Backtest reports show both labeled as "Sharpe" without distinguishing.
**Рекомендация:** Pick one (365 for crypto 24/7) and standardize.

### [B — Math] `_TYPICAL_FUNDING_RATE = 0.0001` hardcoded for cost estimation
**Файл:** src/execution/backtest_runner.py:262
**Серьёзность:** ВЫСОКАЯ
**Описание:** Real BTC funding during 2024-2025 bull regimes was 0.03-0.10% per 8h — 3-10× the constant. Backtest underestimates funding cost; on long-hold strategies this is several % of return.
**Рекомендация:** Integrate per-bar real funding from feature data.

### [B — Math] Funding cost assumes position held for entire backtest period
**Файл:** src/execution/backtest_runner.py:290-296
**Серьёзность:** ВЫСОКАЯ
**Описание:** `total_hours = (end - start).total_seconds() / 3600` → bills funding on every hour the backtest *exists*, not on hours the bot was *in position*. Massive overstatement of funding cost.
**Рекомендация:** Track position-held-hours from trade ledger; bill only those.

### [B — Math] `_avg_price = (first_open + last_close) / 2` — meaningless for long backtests
**Файл:** src/execution/backtest_runner.py:256-259
**Серьёзность:** СРЕДНЯЯ
**Описание:** For BTC moving $30k → $90k, this gives $60k as "average trade price". Real average trade-time price differs drastically. Bias in fee/slippage cost estimation.
**Рекомендация:** Use mean of all bar closes or volume-weighted price.

---

## 6.3 walk_forward.py

### [A — Lookahead] CRITICAL: No embargo between train end and test start
**Файл:** src/execution/walk_forward.py:167-168
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `test_start = train_end` — zero gap. With triple-barrier `max_holding=6`, the last 6 train bars' labels were constructed using prices that extend into test → direct label leakage in every walk-forward window. Combined with 6.2.1 (`ts_event` lookahead), bias compounds.
**Best Practice:** AFML Ch.7 — embargo ≥ max_holding bars at every train/test boundary.
**Рекомендация:** `test_start = _add_months(train_end + max_holding × bar_duration, 0)` or skip first N test bars.

### [A — Lookahead] CRITICAL: Inherits all PurgedKFoldCV issues from Section 1
**Файл:** src/execution/walk_forward.py:42-89 (same code referenced by ml_validator)
**Серьёзность:** КРИТИЧЕСКАЯ (duplicate)
**Описание:** Row-based slicing on multi-symbol concat (1.5.2) + 1%-of-N embargo (1.5.3). Same code, same bugs.

### [B — Math] `is_profitable = total_return > 0` — binary, ignores severity
**Файл:** src/execution/walk_forward.py:127, 223
**Серьёзность:** СРЕДНЯЯ
**Описание:** A 4-of-6 small-positive / 2-of-6 catastrophic-loss outcome passes the 60% gate even though aggregate P&L is negative. The pass criterion measures consistency of sign only.
**Рекомендация:** Multi-criteria gate: avg(sharpe) ≥ X AND min(sharpe) ≥ Y AND profitable_pct ≥ Z.

---

## 6.4 metrics.py

### [B — Math] Sharpe collapses intraday equity to end-of-day → smoothed Sharpe inflated
**Файл:** src/execution/metrics.py:76-108
**Серьёзность:** ВЫСОКАЯ
**Описание:** For 4H bars (6/day) → collapsed to 1 daily point. For 15m (96/day) → 1 daily point. Loses intraday vol; std drops; Sharpe inflates. For 15m strategies this is likely 3-5× inflated relative to per-bar Sharpe.
**Best Practice:** Compute Sharpe at native bar frequency, annualize by √(bars_per_year).
**Рекомендация:** Use per-bar returns when comparing to bar-based gates.

### [B — Math] Two different `risk_free_rate` defaults in the same module
**Файл:** src/execution/metrics.py:59 (`0.0`) vs 177 (`0.05`)
**Серьёзность:** ВЫСОКАЯ
**Описание:** `calculate_sharpe_ratio` default = 0.0; `calculate_all_metrics` default = 0.05 then passes to `calculate_sharpe_ratio`. Two Sharpe values for same data depending on entry point.
**Рекомендация:** One default (0.0 for crypto convention).

### [B — Math] `passes_minimum_thresholds(sharpe ≥ 1.0)` ambiguous given 6.2.2
**Файл:** src/execution/metrics.py:25-32
**Серьёзность:** СРЕДНЯЯ
**Описание:** Which Sharpe convention? Same model passes/fails differently across reports.

---

## 6.5 api/main.py

### [A — Security] CRITICAL: No authentication on any endpoint + wildcard CORS
**Файл:** src/api/main.py:93-99
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** Every endpoint open with `CORSMiddleware(allow_origins=["*"])`. Public discloses: equity (`/api/v1/live`), open positions count, last signal direction/entry_price, full signal history with PnL, monthly performance. External observer polling `/api/v1/live` learns bot positions/timing → front-running possible. With confidence-based execution and visible entry/SL, an adversary can infer liquidation level.
**Best Practice:** API-key header auth, strict CORS allowlist, localhost-bind default, mask equity for unauthenticated.
**Влияние на trading:** Trade-information leak, possible adversarial front-running, equity disclosure.
**Рекомендация:** Add `Depends(verify_api_key)` to every route; localhost by default; allowlisted CORS.

### [A — Security] No rate limiting — trivial DoS
**Файл:** src/api/main.py (whole, by absence)
**Серьёзность:** ВЫСОКАЯ
**Описание:** Each request opens 1-3 SQLite connections; hammering `/api/v1/live` exhausts connections and contends with trading writes.
**Рекомендация:** `slowapi` middleware with 60 req/min/IP.

### [A — Security] `/api/v1/live` exposes raw equity ($) and daily_pnl_pct
**Файл:** src/api/main.py:158-218
**Серьёзность:** ВЫСОКАЯ
**Описание:** Returns `equity: 10000.0, daily_pnl_pct, open_positions, last_signal {direction, entry_price}`. Sufficient for an attacker to compute leverage, notional, liquidation level.
**Рекомендация:** Require auth; mask raw $ values for unauth.

### [A — Security] Connection-per-call without `busy_timeout`
**Файл:** src/api/main.py:50-53
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same SQLite-locking issue as Section 5.2.3. Contends with trading writes → intermittent "database is locked".
**Рекомендация:** `PRAGMA busy_timeout=5000` + `query_only=ON`.

### [E — Architecture] `/api/v1/live` reads `bot_metrics` but trading DBs may lack it
**Файл:** src/api/main.py:175-188
**Серьёзность:** СРЕДНЯЯ
**Описание:** Section 3.7's SignalBridge writes `bot_metrics` every ~24h on 4H. Fresh deployment shows defaults (equity=10000, regime=UNKNOWN) for first day.
**Рекомендация:** Compute equity from `signals_log` when row missing.

---

## 6.6 stats_engine.py

### [B — Math] `_ANNUALIZE = 252` conflicts with `metrics.py` 365
**Файл:** src/analytics/stats_engine.py:28
**Серьёзность:** ВЫСОКАЯ
**Описание:** Telegram `/stats` (StatsEngine) vs `/backtest` reports (metrics.py) show different Sharpe for the same trades.
**Рекомендация:** Standardize on 365.

### [E — Architecture] `performance_cache` table written into trading DB
**Файл:** src/analytics/stats_engine.py:62-63, 298-347
**Серьёзность:** ВЫСОКАЯ
**Описание:** Same cross-process schema-mutation pattern as Section 5.2.1. StatsEngine writes `INSERT/UPDATE performance_cache` into the trading DB. If the table doesn't exist, write silently fails.
**Рекомендация:** Separate `data/stats_cache.db`.

### [B — Math] `_MIN_RATIO_SAMPLE = 10` blanks ratios for sparse data
**Файл:** src/analytics/stats_engine.py:31, 207-209
**Серьёзность:** СРЕДНЯЯ
**Описание:** 7-day window with 3-5 closed signals returns Sharpe=Sortino=Calmar=None → Telegram shows "— (мало данных)". Statistically correct but unhelpful when newly-deployed timeframes are operating.
**Рекомендация:** Surface `closed_signals` count alongside None.

### [E — Architecture] `compute_equity_curve` always uses `initial_capital=10_000`
**Файл:** src/analytics/stats_engine.py:122-145
**Серьёзность:** СРЕДНЯЯ
**Описание:** Shown to premium Telegram users as their equity curve regardless of actual capital.
**Рекомендация:** Read from `bot_metrics` / `Settings.initial_capital`.

---

## 6.7 config.py

### [A — Security] CRITICAL: `trading_mode` typo silently defaults to MAINNET
**Файл:** src/config.py:82, 132-148
**Серьёзность:** КРИТИЧЕСКАЯ
**Описание:** `trading_mode: str = "testnet"`; `is_testnet = trading_mode.lower() == "testnet"`. Any non-exact value → False → MAINNET keys.

Failure modes:
- `TRADING_MODE=Test` → mainnet (capital "T" → lowered to "test" ≠ "testnet")
- `TRADING_MODE=tesnet` (typo) → mainnet
- `TRADING_MODE=" testnet"` (leading space) → mainnet
- `TRADING_MODE=` (empty) → mainnet

In all cases the bot trades real money without warning.
**Best Practice:** `Literal["testnet", "live"]` or pydantic Enum; raise on unknown.
**Влияние на trading:** Single character typo in `.env` → real money trading.
**Рекомендация:** Convert to `Literal` with strict validator; loud startup banner showing resolved mode.

### [A — Security] `extra="ignore"` silently swallows misspelled env vars
**Файл:** src/config.py:23-28
**Серьёзность:** ВЫСОКАЯ
**Описание:** `BINNANCE_API_KEY=xxx` (typo) is ignored; `binance_mainnet_api_key` falls back to `""`. Binance returns "API key invalid" at runtime — easier to debug if startup fails fast.
**Рекомендация:** `extra="forbid"` or explicit logging of unknown env vars.

### [E — Architecture] `confidence_threshold = 0.65` in config but RiskConfig uses 0.55
**Файл:** src/config.py:86 vs src/risk/risk_engine.py:46
**Серьёзность:** ВЫСОКАЯ
**Описание:** Setting is decorative — never read by the strategy. Strategy uses `MLStrategyConfig.confidence_threshold=0.55`.
**Рекомендация:** Wire through or delete.

### [E — Architecture] Secrets default to empty strings — silent fallback
**Файл:** src/config.py:33-36
**Серьёзность:** СРЕДНЯЯ
**Описание:** Missing `BINANCE_API_KEY` → empty string → 401 at runtime. Should fail fast at startup.
**Рекомендация:** `@model_validator` requiring keys-for-current-mode.

### [E — Architecture] `safe_dict()` correctly masks secrets ✅
**Файл:** src/config.py:164-191
**Серьёзность:** INFO (positive)
**Описание:** `_SECRET_FIELDS` covers all known credentials; safe.

---

## Counts (Section 6)
- **КРИТИЧЕСКАЯ:** 7
- **ВЫСОКАЯ:** 14
- **СРЕДНЯЯ:** 8
- **НИЗКАЯ / INFO:** 2

---

## Итоговый Summary

### Статистика по всем разделам

| Раздел | КРИТИЧЕСКИЕ | ВЫСОКИЕ | СРЕДНИЕ | НИЗКИЕ | Итого |
|--------|:-----------:|:-------:|:-------:|:------:|:-----:|
| 1. ML Pipeline | 8 | 8 | 6 | 6 | 28 |
| 2. Feature Engineering | 3 | 10 | 9 | 8 | 30 |
| 3. Execution & Risk | 6 | 19 | 11 | 4 | 40 |
| 4. Data Infrastructure | 2 | 16 | 11 | 5 | 34 |
| 5. Telegram Bot | 3 | 16 | 14 | 7 | 40 |
| 6. Infrastructure & Monitoring | 7 | 14 | 8 | 2 | 31 |
| **ИТОГО** | **29** | **83** | **59** | **32** | **203** |

### Топ-10 критичных проблем всего проекта

Sorted by impact on trading + capital safety:

1. **`data_catalog.ts_event = open_time`** (4.6.1 / 6.2.1) — каждый v3 backtest получает 1 бар лукэхеда. **Все reported v3 цифры master-документа подлежат пересчёту.**
2. **`config.trading_mode` typo → silent MAINNET** (6.7.1) — опечатка в `.env` (`tesnet`, лишний пробел) → реальные деньги вместо testnet. Прямой риск капитала.
3. **`CircuitBreaker` и `PositionReconciler` — dead code** (3.5.1, 3.8.1) — никогда не вызываются стратегиями. Каскад -2/-3/-15% и orphan-detection в проде не работают. `daily_pnl` не персистится → рестарт обнуляет счётчик.
4. **4H production без heartbeat + Watchdog fail-open** (3.1.1 + 6.1.1) — основной торговый бот без dead-man's switch. Watchdog с дефолтным конфигом либо постоянно flat-закрывает 4H, либо вообще не настроен.
5. **`PortfolioTracker` ломает учёт equity** (3.6.1) — `close_position` начисляет `notional + pnl`, но `update_fill` не списывает notional. После ~10 round-trip equity завышен на N×notional. Drawdown gate, sizing, daily-loss — все на сломанном equity.
6. **ORB intra-window lookahead** (2.6.1) — `max().over("_date")` для ORB-баров включает будущие бары внутри окна формирования. Доминирует apparent "skill" 15m модели.
7. **PurgedKFoldCV row-slicing на multi-symbol concat** (1.5.2, 6.3.2) — тот же ML-018, что чинил `temporal_split.py`, но в CV. Все CV-метрики v3 завышены.
8. **Walk-forward bypasses triple-barrier targets** (1.5.1) — walk-forward валидируется на legacy 1-bar sign(return), production — на triple-barrier. Валидируется не та модель.
9. **`np.nan_to_num(X, nan=0.0)` перед LightGBM** (1.2.1, 3.1.6) — уничтожает родную NaN-обработку; warmup-бары становятся "feature=0" вместо "missing"; обучение и inference на испорченных фичах.
10. **API без auth + wildcard CORS** (6.5.1) + **Entry/SL gap не персистится** (3.1.2) — API раскрывает equity/positions/last_signal публично → front-running риск; `_pending_sl_params` только в памяти → краш = unprotected position.

### Категории проблем

**Lookahead / Data Leakage:**
- `ts_event = open_time` в backtest (4.6.1)
- ORB intra-window leakage (2.6.1)
- Val-set без embargo до test (1.2.2)
- Walk-forward без embargo (6.3.1)
- PurgedKFoldCV row-slicing на multi-symbol (1.5.2)
- SessionVWAP с current bar в live (2.5.x)

**Train/Serve Skew:**
- Funding-rate offline скачан daily вместо monthly → пусто; live = real (4.1.1)
- `funding_rate_history` 100 на live vs unbounded offline (2.9.1)
- `oi_history` 8h на live vs months offline (2.9.2)
- Bar-count-as-time для "30d"/"24h"/"vwap_4h" (2.x)
- Regime filter после target создания (1.2.3)
- 15m HTF resampled из 15m vs trained на real (3.2.1)

**Безопасность капитала:**
- `trading_mode` typo → MAINNET (6.7.1)
- 4H без heartbeat (3.1.1)
- Entry/SL gap с in-memory _pending_sl_params (3.1.2)
- PortfolioTracker cash accounting (3.6.1)
- CircuitBreaker dead code (3.5.1)
- PositionReconciler dead code (3.8.1)
- daily_pnl не персистится (3.4.5)
- consecutive_losses permanent pause без win (3.4.3)
- max_open_positions race condition (3.1.4)

**Безопасность данных и платежей:**
- API без auth + wildcard CORS (6.5.1)
- /api/v1/live раскрывает equity (6.5.3)
- Telegram pre_checkout одобряет любой payload (5.3.1)
- CryptoBot processed_ids in-memory + amount не проверяется (5.4.1-5.4.2)
- Privileged callback handlers без role check (5.8.1)
- Telegram-bot Database пишет схему в trading DB (5.2.1)
- Stars: нет refund handler (5.3.3)

**Архитектурные (мешают качеству, не критичны для денег):**
- DSR формула bugs (1.3.1, 1.3.2)
- PBO не CSCV (1.3.4)
- LightGBM pickle.dump вместо save_model (1.2.5)
- MTF features используют 4H semantics (vwap_4h, funding_cum_24h в барах)
- Sharpe annualisation 252 vs 365 inconsistency (6.2.2, 6.4.1, 6.6.1)
- SQLite connection-per-call без busy_timeout (всюду)
- Три источника equity: tracker / Nautilus portfolio / MetricsCollector (3.1.7)

### Рекомендуемый порядок исправлений

**Фаза 1 (НЕМЕДЛЕННО — риск капитала; не торговать mainnet до выполнения):**

1. `trading_mode` → `Literal["testnet", "live"]` + pydantic validator + startup banner (config.py).
2. Подключить `HeartbeatManager` в 4H стратегии (3.1.1).
3. Починить `PortfolioTracker._cash` accounting + добавить invariant test (3.6.1).
4. Подключить `CircuitBreaker.check` в on_bar (3.5.1).
5. Подключить `PositionReconciler.reconcile` на startup + периодически (3.8.1).
6. Персистировать `_pending_sl_params` на диск (3.1.2).
7. Персистировать daily/weekly/consecutive_losses/peak_equity (3.4.5).
8. Watchdog: fail-closed на null key + grace period после Redis reconnect (6.1.1, 6.1.2).
9. Cancel orders ПЕРЕД close в emergency_close (6.1.4).

**Фаза 2 (до запуска с реальными деньгами):**

10. Починить `data_catalog.ts_event = open_time + bar_duration` (4.6.1) и перезапустить ВСЕ backtests.
11. Аудит funding-rate parquet'ов; если пусто — перекачать через monthly URL (4.1.1).
12. Добавить embargo `max_holding` в WalkForwardValidator (6.3.1).
13. Починить PurgedKFoldCV для multi-symbol (1.5.2 / 6.3.2).
14. Подключить triple-barrier в MLValidator._load_full_data (1.5.1).
15. Убрать `np.nan_to_num` из trainer и live inference (1.2.1, 3.1.6).
16. Починить ORB intra-window lookahead (2.6.1).
17. Авторизация на API + localhost-bind + rate limiting (6.5.1-6.5.4).
18. Telegram pre_checkout: amount check + dedup по telegram_payment_charge_id (5.3.1).
19. Signal poller: dedup закрытий через last_close_signal_ids (5.6.1).
20. require_role на privileged callbacks (5.8.1).
21. Увеличить oi_history / funding_rate_history maxlen (2.9.1, 2.9.2).
22. Telegram Database: разделить OwnerDatabase + ReadOnlyDatabase (5.2.1).

**Фаза 3 (улучшения качества — после первых живых сделок):**

23. Починить DSR (T = число возвратов, kurtosis = γ4-3) (1.3.1-1.3.2).
24. Реализовать PBO через CSCV или удалить (1.3.4).
25. Параметризовать "30d"/"24h"/"vwap_4h" окна по TF (2.x).
26. Стандартизировать Sharpe annualisation = 365 везде (6.2.2, 6.4.1, 6.6.1).
27. `booster.save_model` вместо pickle (1.2.5).
28. MetaSignalGate либо подключить, либо удалить (3.3.1).
29. Регулярный аудит: проверка funding/OI данных vs feature distribution в training.

**Фаза 4 (nice to have):**

30. Расширить `_redact_sensitive` regex (5.7.2).
31. PTB `AIORateLimiter` для broadcaster (5.5.1).
32. Telegram refund handler (5.3.3).
33. LiveFeedManager: подключить как secondary feed или удалить (4.5.1).
34. Регрейм-flickering: 3-bar majority sign (2.4.x).
35. DFA вместо R/S для Hurst на 15m (2.4.1).

### Положительные находки

Что сделано правильно:

1. **`apply_triple_barrier` causal и хорошо документирован** (features/triple_barrier.py) — Lopez de Prado AFML Ch.3 корректно реализован; future_return = real close at touch bar; strictly causal sweep.
2. **`temporal_split_multi`** — корректный fix для ML-018 с invariant assertion + embargo.
3. **MTF context join через close_time** (mtf_context.py:289-290) — backward asof join не допускает HTF lookahead.
4. **`agg_15m` parent-bucket aggregation** — все 16 child баров закрыты к close родительского 4H; causal.
5. **Watchdog как отдельный процесс без Nautilus imports** — правильный принцип изоляции.
6. **`Settings.safe_dict()` маскирует секреты** через `_SECRET_FIELDS`.
7. **Pydantic-settings + `lru_cache` singleton** — стандартный pattern.
8. **DataStore + ParquetConverter Hive partition tree** структурирован правильно.
9. **Idempotent migrations через `ALTER TABLE ... ADD COLUMN` с try/except OperationalError** — корректный pattern.
10. **CostModel — выделенный модуль** с реалистичной slippage-моделью.
11. **Mainstream tech stack:** FastAPI, python-telegram-bot v21, nautilus_trader, LightGBM, polars, DuckDB.
12. **Чистое разделение Section 1-6:** ML / features / execution / data / telegram / infra.
13. **Концептуальное покрытие финтех:** triple-barrier, sample uniqueness, walk-forward, DSR/PBO, regime detection, MTF context, dead-man's switch, circuit breakers, signal bridge. **Главная проблема — не concept gaps, а implementation correctness и operational wiring.**

---

> **Финальная рекомендация:** До исправления Фазы 1 и Фазы 2 — **не торговать реальными деньгами**. Reported метрики master-документа не отражают реальную модель (lookahead в backtest + dead safety code + потенциально отсутствующие funding data). После Фазы 1+2 — перезапустить полную training+validation+backtest последовательность; ожидать снижения Sharpe/WR/PF, но снижённые цифры будут впервые trustworthy.
