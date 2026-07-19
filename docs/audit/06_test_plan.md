# 06 — План тестов (TDD: сначала failing-тест, потом фикс)

> Таксономия по Breck, Cai, Nielsen, Salib, Sculley (Google, 2017),
> *The ML Test Score: A Rubric for ML Production Readiness*:
> [leakage] / [skew] / [stat] / [invariant] / [prod].
> Каждый тест ниже сейчас ДОЛЖЕН УПАСТЬ на текущем коде — это и есть подтверждение
> дефекта. Один PR = один тест + один фикс. Существующие 1443 теста зелёные и
> НЕ ловят ни один из этих дефектов (проверено запуском test_dsr_formula.py,
> test_feature_skew.py — 36 passed).

## PR-кандидаты (приоритетный порядок)

### PR-1 [prod] Fail-fast при неподключённых движках (A1)
```python
# tests/test_node_fail_fast.py
def test_run_live_exits_nonzero_when_engines_not_connected(monkeypatch):
    """run_live must not stay RUNNING when DataEngine/ExecEngine failed to connect."""
    node = FakeNode(data_connected=False, exec_connected=False)
    with pytest.raises(SystemExit) as e:
        run_live_main(node_factory=lambda cfg: node, mode="paper")
    assert e.value.code != 0
```
Фикс: проверка `check_connected()` после старта + `sys.exit(1)`; systemd рестартует.

### PR-2 [stat] DSR: канонический SR0 и per-period SR (A4a-c)
```python
# tests/test_dsr_canonical.py
def test_dsr_noise_is_not_degenerate():
    """100 трейлов чистого шума, T=1000: DSR обязан быть в (0.05, 0.95),
    а не 0.0 (текущий код) и не крэш."""
    rng = np.random.default_rng(0)
    srs = [float(r.mean()/r.std(ddof=1)) for r in rng.normal(0, .01, (100, 1000))]
    dsr = calculate_dsr(srs, n_trials=100, n_obs=1000)   # per-period SRs!
    assert 0.05 < dsr < 0.95

def test_dsr_scales_expected_max_by_trial_std():
    """Умножение всех SR на константу k не должно менять DSR (масштабная
    инвариантность z = (SR−SR0)/SE достигается только при SR0 ∝ std(trials))."""
    base = [0.02, 0.05, 0.03, 0.01, 0.04]
    d1 = calculate_dsr(base, n_trials=50, n_obs=800)
    d2 = calculate_dsr([s*3 for s in base], n_trials=50, n_obs=800)
    assert abs(d1 - d2) < 0.05

def test_dsr_no_crash_on_heavy_annualized_inputs():
    """Регресс сегодняшнего math domain error → молчаливого DSR=0."""
    dsr = calculate_dsr([1.9, 1.2, 0.7], n_trials=100, skewness=0.6,
                        kurtosis=2.9, n_obs=1000)
    assert 0.0 <= dsr <= 1.0
```
Фикс: SR0=std·((1−γ)Z₁+γZ₂); знаменатель (γ4−1)/4; убрать √365 из run_all_tests:403.

### PR-3 [stat] Мертенс-знаменатель (A4b)
```python
def test_variance_term_matches_mertens_2002():
    """(γ4−1)/4 эквивалентен 1+SR²/2−γ3·SR+(γ4−3)/4·SR² — сверка с независимой
    реализацией; при γ4=3, γ3=0 дисперсионный член = 1 + SR²/2, НЕ 1."""
```

### PR-4 [skew] Parity как CI-гейт (A3) — материализация scratch-скрипта
```python
# tests/test_train_serve_parity_4h.py  (медленный, маркер @pytest.mark.parity)
def test_regime_label_parity_offline_vs_buffer():
    """40 случайных исторических баров: build() vs build_from_buffer(буфер
    прод-глубины). Допуск: ≤2/40 расхождений лейбла режима (сейчас 14/40)."""
def test_atr_percentile_parity():
    """KS-тест распределений atr_percentile offline vs buffer, p>0.05
    (сейчас mean rel diff 16.5%)."""
```
Фикс: buffer 4h→700, 15m→740+; затем допуск ужесточить до 0.

### PR-5 [skew] 15m HTF-фичи не константы (A3-15m)
```python
def test_htf_4h_features_are_warm_in_live_15m():
    """build_from_buffer('15m') на прод-объёме буфера: htf_4h_hurst последней
    строки != 0.5 и htf_4h_atr_percentile != 0.5 (сейчас всегда дефолты)."""
```

### PR-6 [leakage] val→test embargo (A12)
```python
def test_val_tail_labels_do_not_reach_test_window():
    """Последний t1_bar валид-набора < первого bar_idx теста."""
```

### PR-7 [invariant] Геометрия сделки = геометрия лейбла (A6)
```python
def test_execution_barriers_match_label_barriers():
    """RiskEngine SL/TP-множители и наличие time-exit обязаны совпадать с
    barrier_pt/sl_multiplier и max_holding модели из манифеста бандла."""
```

### PR-8 [prod] Деплой-гейт (A7)
```python
def test_strategy_refuses_model_without_passing_manifest():
    """Бандл без passes=True (или без манифеста) → стратегия не торгует регим."""
```

### PR-9 [stat] Cost model порядок величины (A9)
```python
def test_sqrt_impact_uses_daily_sigma():
    """$10k на $1B ADV, σ_ann=0.6: slippage_bps в [0.1, 2.0], не ~9.5."""
```

### PR-10 [stat] Издержки в eval-гейте (A8)
```python
def test_eval_profit_factor_is_net_of_costs():
    """PF в EvaluationResult при fee=0 строго больше, чем при taker-fee 5 bps —
    и гейт passes_minimum_thresholds документирован как net."""
```

### PR-11 [leakage] Лейблы по high/low (A11) — metamorphic
```python
def test_intrabar_stop_is_labeled_as_stop():
    """Синтетический бар: low пробивает SL-барьер, close возвращается выше.
    Текущий код лейблит +1/0; после фикса обязан быть −1 (SL-first)."""
```

### PR-12 [prod] Rejected-ордер не является сделкой (A10)
```python
def test_rejected_order_marks_signal_rejected():
    """submit_order → REJECTED: signals_log.result == 'rejected', pnl NULL."""
```

### PR-13 [stat] t-stat / proportional z (A15) + PF cap (A16)
Синтетика с известным z; PF=None при n<30.

### PR-14 [prod] Watchdog смоук (A1)
Интеграционный: heartbeat протух → watchdog реагирует; сервис в CI-checklist деплоя.

## Существующие тесты — почему зелёные при сломанном DSR
`test_dsr_formula.py` проверяет только (T−1)-скейлинг, знак эффектов и монотонность —
все утверждения инвариантны к отсутствию √V[SR]-множителя и к аннуализации,
потому что сравнивают DSR с самим собой при вариации одного параметра.
Урок: property-тесты статистики должны включать **абсолютную** калибровку на синтетике
с известным аналитическим ответом (шум → DSR~U(0,1); сильный скилл → DSR→1).
