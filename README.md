# DRL Trading Agent — Cross-Modal Sentiment-Price Attention Fusion

A deep reinforcement learning trading agent (PPO) that fuses **financial-news sentiment** with **technical price signals** through a bidirectional cross-modal attention mechanism, and rigorously benchmarks the result against a price-only baseline and buy-and-hold — with walk-forward validation and statistical significance testing.

> **TL;DR:** On a held-out 2023–2024 test period (AAPL), the sentiment-aware agent returned **35.1% vs 32.0%** for a price-only baseline, with a higher **Sharpe (1.57 vs 1.23)** and lower **max drawdown (15.2% vs 16.6%)**. It won all **3 folds** of walk-forward validation — though the improvement is *not* statistically significant on 252 test days (reported honestly below).

---

## Highlights

- **Cross-modal attention fusion** — bidirectional attention (Price→Sentiment and Sentiment→Price) over technical indicators and FinBERT/VADER news sentiment.
- **Hierarchical PPO agent** — 24 discrete actions (6 macro-strategies × 4 commitment horizons) in a custom Gymnasium environment with realistic transaction costs and volatility-proportional slippage.
- **Honest evaluation** — walk-forward validation (3 expanding folds) *and* four statistical significance tests (paired t-test, Wilcoxon, Jobson-Korkie Sharpe equality, bootstrap), with results reported as-is.
- **Served via FastAPI** — a REST API + minimal web UI that loads the trained agent and returns trade actions from a 30-feature market state.
- **Reproducible** — single YAML config, fixed seed, saved model checkpoints, and committed result artifacts.

---

## Results

### Primary test set — AAPL, Sep 2023 – Dec 2024 (286 days)

| Metric | Sentiment Agent | Price-Only Baseline | Buy & Hold |
|---|---|---|---|
| **Total Return** | **35.1%** | 32.0% | 32.0% |
| **Sharpe Ratio** | **1.57** | 1.23 | 1.23 |
| **Max Drawdown** | **15.2%** | 16.6% | 16.6% |
| **Trades** | 22 | 1 | 1 |

The sentiment agent added **+3.1 pp of return** and **+0.34 Sharpe** while *reducing* drawdown — i.e. better risk-adjusted performance, not just higher risk.

### Walk-forward validation — 3 expanding folds (252-day test windows)

| Metric (mean ± std) | Sentiment Agent | Price-Only Baseline |
|---|---|---|
| Total Return | **19.3% ± 13.7%** | 11.6% ± 18.0% |
| Sharpe Ratio | **0.90 ± 0.54** | 0.64 ± 0.64 |
| Max Drawdown | **18.4% ± 2.0%** | 20.6% ± 5.5% |

The sentiment agent outperformed on mean return, Sharpe, and drawdown in **all 3 folds**, with **lower variance** on every metric — the more meaningful signal than any single test run.

### Statistical significance — reported honestly

Across four tests on 252 daily returns (paired t-test, Wilcoxon signed-rank, Jobson-Korkie Sharpe equality, and a 10k-resample bootstrap), **0 of 4 reached significance at α = 0.05** (all p > 0.9). 252 test days is well below the sample size needed for statistical power on noisy financial returns. The improvement is *consistent and directional* across folds but should **not** be read as statistically proven; a longer horizon or multi-asset study would be required to establish significance.

![Portfolio comparison](results/plots/portfolio_comparison.png)

*Additional figures in [`results/plots/`](results/plots/): drawdown comparison, returns distribution, price/portfolio overlay, and a side-by-side metrics table.*

---

## Architecture

```
News (Kaggle + NewsAPI)          Prices (Yahoo Finance)
        │                                │
   FinBERT / VADER              Technical indicators
   sentiment scoring          (RSI, MACD, BB, ATR, regime)
        │                                │
        └──────────► Cross-Modal Attention ◄──────────┘
                   (Price↔Sentiment, bidirectional)
                             │
                    30-dim market state
                             │
                   Hierarchical PPO Agent
              (6 macro-actions × 4 horizons = 24)
                             │
          Trading env (0.1% cost + vol-scaled slippage)
                             │
              Evaluation · Walk-forward · Stat tests
                             │
                   FastAPI serving layer
```

| Component | Detail |
|---|---|
| RL algorithm | PPO (Stable-Baselines3); DQN implemented for comparison |
| Sentiment | FinBERT (`ProsusAI/finbert`) + VADER |
| Fusion | Bidirectional cross-modal attention |
| Observation | 30-dim (24 market features + 6 portfolio/commitment state) |
| Action space | 24 discrete (6 macro-strategies × {1,3,5,10}-day horizons) |
| Training | 200k timesteps, seed 42, $10k start, 0.1% cost, proportional slippage |
| Serving | FastAPI REST API + Jinja2 web UI |

---

## Tech Stack

**Language:** Python
**ML/RL:** PyTorch, Stable-Baselines3, Gymnasium, Hugging Face Transformers (FinBERT), NLTK/VADER
**Data & analysis:** pandas, NumPy, SciPy, scikit-learn, `ta`, yfinance
**Serving:** FastAPI, Uvicorn, Jinja2
**Tooling:** pytest, PyYAML, Matplotlib/Seaborn

---

## Project Structure

```
rl_trading_agent/
├── src/
│   ├── data/            # Data loading, feature engineering, sentiment extraction
│   ├── models/          # PPO/DQN agents, cross-modal attention, trading env
│   ├── training/        # Trainer + walk-forward orchestration
│   ├── evaluation/      # Metrics, statistical tests, plotting
│   └── explainability/  # Attention visualization
├── configs/config.yaml  # Single source of truth for all hyperparameters
├── results/             # Metrics JSON + plots
├── models/saved/        # Trained agent checkpoints
├── app.py               # FastAPI serving layer
├── tests/               # pytest suite
└── requirements.txt
```

---

## Getting Started

```bash
# 1. Install
pip install -r requirements.txt

# 2. (Optional) set news API keys for fresh data; falls back to bundled Kaggle data
export FINNHUB_API_KEY=your_key
export NEWSAPI_KEY=your_key

# 3. Run the full experiment (train, walk-forward, stat tests, plots)
python -m src.training.run_full_experiment

# 4. Serve the trained agent
uvicorn app:app --reload
# → GET /            health check
# → GET /web         web UI
# → POST /web/predict  action from a 30-feature state

# 5. Run tests
pytest
```

Experiment depth is configurable in `configs/config.yaml` (`quick` ≈ 10 min, `full` ≈ 60 min, `extensive` ≈ 3 hrs).

---

## Key Takeaways

1. **Sentiment adds consistent practical value** — higher return and Sharpe with lower drawdown across the test period and all walk-forward folds.
2. **Risk management improved, not just returns** — shallower drawdowns and lower cross-fold variance.
3. **Rigor over hype** — the pipeline reports negative statistical results honestly rather than overclaiming; the walk-forward consistency, not a single run, is the real evidence.

---

## Disclaimer

This is a research project for educational purposes. It is **not** financial advice and is not intended for live trading. Past backtested performance does not indicate future results.
