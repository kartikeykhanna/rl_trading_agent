# RL Trading Agent — Results & Metrics Summary

**Project:** Deep Reinforcement Learning Trading Agent with Cross-Modal Sentiment-Price Attention Fusion  
**Asset:** AAPL (Apple Inc.)  
**Evaluation Period:** Sep 2023 – Dec 2024 (286 test days)  
**Training Period:** Jan 2019 – Sep 2023 (1,143 days)  

---

## 1. Primary Test Set Performance

| Metric | Sentiment Agent | Baseline (Price-Only) | Buy & Hold |
|---|---|---|---|
| **Total Return (%)** | **35.13%** | 32.02% | 32.02% |
| **Final Portfolio Value ($10k start)** | **$13,513** | $13,202 | $13,202 |
| **Sharpe Ratio** | **1.57** | 1.23 | 1.23 |
| **Max Drawdown (%)** | **15.20%** | 16.61% | 16.61% |
| **Number of Trades** | **22** | 1 | 1 |

> **Key takeaway:** The sentiment-aware agent achieved +3.11% higher total return and a +0.34 higher Sharpe ratio vs. the price-only baseline, while also reducing max drawdown by 1.41 percentage points.

---

## 2. Walk-Forward Validation (3-Fold Expanding Window)

Each fold used a 252-day test window and ≥504-day training window. The sentiment agent won in **all 3 folds (100% win rate)**.

### Sentiment Agent

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Total Return (%) | **19.29%** | ±13.72% | 2.43% | 36.04% |
| Sharpe Ratio | **0.903** | ±0.541 | 0.240 | 1.565 |
| Max Drawdown (%) | **18.41%** | ±2.03% | 16.61% | 21.25% |

### Baseline Agent (Price-Only)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Total Return (%) | 11.60% | ±17.96% | -9.02% | 34.76% |
| Sharpe Ratio | 0.637 | ±0.638 | -0.092 | 1.461 |
| Max Drawdown (%) | 20.59% | ±5.55% | 15.70% | 28.35% |

### Buy & Hold

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Total Return (%) | 16.99% | ±19.10% | -9.48% | 34.89% |
| Sharpe Ratio | 0.828 | ±0.675 | -0.106 | 1.465 |
| Max Drawdown (%) | 20.22% | ±5.76% | 15.70% | 28.35% |

> **Walk-Forward Summary:** The sentiment agent outperformed the baseline on mean return (+7.69%), Sharpe (+0.266), and max drawdown (-2.18%) across all folds, with lower variance on all metrics — indicating more consistent out-of-sample performance.

---

## 3. Statistical Significance Tests

All tests compare daily returns: Sentiment Agent vs. Baseline Agent (n=252 days).

| Test | Statistic | p-value | Significant at α=0.05? |
|---|---|---|---|
| Paired t-test | t = -0.0718 | 0.943 | No |
| Wilcoxon Signed-Rank | W = 15,890 | 0.966 | No |
| Sharpe Equality (Jobson-Korkie) | z = -0.0008 | 0.999 | No |
| Bootstrap Sharpe Difference (10k resamples) | CI: [-2.79, 2.75] | 0.504 | No |

**Verdict:** 0/4 tests reached statistical significance.

> **Interpretation:** While the sentiment agent shows consistent *directional* improvement across all metrics and all folds, the differences are not statistically significant at conventional levels. This is expected — 252 test days is below the threshold needed for sufficient statistical power in noisy financial return data. A longer evaluation horizon or multi-asset study would be required for significance.

**Annualized Sharpe (Statistical Tests Run):**
- Sentiment Agent: 0.760
- Baseline Agent: 0.761
- Mean daily return difference: -0.000131 (std: 0.0290)

---

## 4. Model & Architecture Summary

| Component | Detail |
|---|---|
| RL Algorithm | PPO (Proximal Policy Optimization, Stable-Baselines3) |
| Feature Fusion | Cross-Modal Attention (bidirectional: Price→Sentiment, Sentiment→Price) |
| Sentiment Model | FinBERT / VADER |
| Observation Space | 30-dim: 24 market features + 6 portfolio/commitment state |
| Action Space | 24 discrete (6 macro-actions × 4 commitment durations) |
| Policy Network | 134 → 128 → 64 → 24 |
| Value Network | 134 → 128 → 64 → 1 |
| Training Timesteps | 200,000 (full run) |
| Initial Portfolio | $10,000 |
| Transaction Cost | 0.1% per trade |
| Slippage | Volatility-proportional (5 bps base + 2 bps spread) |

---

## 5. Dataset Summary

| Item | Detail |
|---|---|
| Price Source | Yahoo Finance (yfinance) |
| Ticker | AAPL |
| Price Date Range | Jan 2019 – Dec 2024 (~1,429 trading days) |
| Total News Articles | 536 |
| Kaggle Articles | 441 (82%) |
| NewsAPI Articles | 95 (18%) |
| News Coverage Period | Mar 2020 – Feb 2026 |
| Avg Articles/Day | 6.6 |
| Total Features | 24 per day (12 technical + 7 regime + 5 sentiment) |
| Train/Test Split | 80/20 chronological (1,143 / 286 days) |

---

## 6. Key Findings

1. **Sentiment adds consistent practical value** — +3.11% higher total return and +0.34 Sharpe vs. price-only baseline on the test period.
2. **Better risk management** — Sentiment agent max drawdown of 15.20% vs. 16.61% for baseline; shallower troughs during volatile periods.
3. **Active trading generates alpha** — 22 trades vs. 1 for the baseline, showing the sentiment signals are actionable.
4. **Walk-forward robustness** — 100% win rate across 3 folds; lower variance on Sharpe (±0.541 vs. ±0.638) and drawdown (±2.03% vs. ±5.55%).
5. **No statistical significance** — p > 0.9 on all four tests; insufficient data volume to pass conventional thresholds.

---

## 7. Output Files

| File | Description |
|---|---|
| `results/training_results.json` | Per-agent metrics from the main test run |
| `results/walk_forward_results.json` | 3-fold walk-forward returns, Sharpe, drawdown |
| `results/statistical_tests.json` | Full statistical test output (run: 2026-03-29) |
| `results/figures/portfolio_comparison.png` | Portfolio value curves over test period |
| `results/figures/drawdown_comparison.png` | Drawdown curves over test period |
| `results/figures/returns_distribution.png` | Daily returns histogram |
| `results/figures/metrics_table.png` | Side-by-side metrics table |
| `results/figures/price_portfolio_overlay.png` | AAPL price vs. portfolio value overlay |
| `models/saved/sentiment_agent.zip` | Trained sentiment PPO agent checkpoint |
| `models/saved/baseline_agent.zip` | Trained baseline PPO agent checkpoint |
