# 08 — Библиография (с привязкой к находкам)

## Статьи (первоисточники формул)
- **Bailey, D., López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality.** J. Portfolio Management 40(5).
  [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551).
  → A4a-c (SR0 = √V[SR]·(…), знаменатель (γ4−1)/4, per-period SR). Независимые сверки:
  [marti.ai разбор](https://marti.ai/qfin/2018/05/30/deflated-sharpe-ratio.html),
  [Wikipedia: Deflated Sharpe ratio](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio).
- **Mertens, E. (2002). Comments on variance of the IID estimator of Sharpe ratio.** → A4b.
- **Bailey, Borwein, López de Prado, Zhu (2014). The Probability of Backtest Overfitting.**
  J. Computational Finance. → A5 (CSCV — единственное каноническое определение PBO).
- **Harvey, C., Liu, Y. (2015). Backtesting.** J. Portfolio Management. → честный подсчёт N трейлов (P0.3).
- **Easley, López de Prado, O'Hara (2012). Flow Toxicity and Liquidity in a High-Frequency
  World.** RFS 25(5). → A20 (VPIN: последовательные volume-buckets; горизонт эффектов — HFT/минуты, не 4H).
- **Tóth et al. (2011). Anomalous price impact and the critical nature of liquidity.** PRX;
  **Kyle, Obizhaeva (2016). Market Microstructure Invariance.** Econometrica;
  **Donier, Bonart (2014). A million metaorder analysis of market impact on the Bitcoin.**
  [arXiv:1412.4503](https://arxiv.org/abs/1412.4503). → A9 (I = Y·σ_daily·√(Q/ADV), Y≈0.5–1;
  подтверждено на BTC). Сводка: [The two square root laws of market impact](https://arxiv.org/pdf/2311.18283).
- **Grinsztajn, Oyallon, Varoquaux (2022). Why do tree-based models still outperform deep
  learning on tabular data?** NeurIPS. → 04 §4.7 (отказ от Transformer при 2.4k строк).
- **Breck, Cai, Nielsen, Salib, Sculley (2017). The ML Test Score: A Rubric for ML
  Production Readiness.** Google, IEEE BigData. → 06 (таксономия), A1/A3 (skew-тесты, monitoring).
- Weron (2002); Couillard & Davison (2005) — малосэмпловый bias R/S-Хёрста (контекст hurst_window=50 на 15m, остаётся 🟡).

## Книги
- **López de Prado. Advances in Financial Machine Learning (2018).** Ch.3 (triple-barrier —
  A6/A11: лейбл = торгуемая стратегия), Ch.4 (uniqueness — сверено, закрыто), Ch.7
  (purge/embargo — A12), Ch.11-14 (backtest c издержками — A8).
- **López de Prado. Machine Learning for Asset Managers (2020).** Clustered feature
  importance → P2.
- **Carver. Systematic Trading (2015); Advanced Futures Trading Strategies (2023).**
  Диверсификация/IDM, vol-targeting, издержки → P3, A8.
- **Cartea, Jaimungal, Penalva. Algorithmic and HF Trading (2015)**; **Lehalle, Laruelle.
  Market Microstructure in Practice (2018).** → импакт-модели (A9), OFI-горизонты (P2).
- **Jansen. ML for Algorithmic Trading, 2e (2020).** Ch.5 (метрики/тесты значимости — A15).
- **Thorp; MacLean, Thorp, Ziemba (Kelly Capital Growth, 2011).** → fractional Kelly ≤0.25 (P3).

## Реализации для сверки
- mlfinlab: `deflated_sharpe_ratio`, `probability_of_backtest_overfitting` (CSCV) — эталон A4/A5.
- Balaena Quant Insights, [DSR issue 24](https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464) — вторая независимая реализация DSR.
- Nautilus Trader GitHub (reconnect/`check_connected` семантика) — A1.

## Внутренние артефакты аудита
- `docs/audit/scratch/dsr_synthetic_check.py` — синтетика DSR (noise/skill/proxy + крэш).
- `docs/audit/scratch/parity_4h_check.py` — parity offline vs live-buffer (35% mismatch).
- Команды-источники всех чисел указаны в 00_index.md построчно.
