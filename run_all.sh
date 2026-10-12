#!/usr/bin/env bash
# ------------------------------------------------------------------
# Full end-to-end run of the RL trading agent
#   1. Python environment + dependencies
#   2. Price data (Yahoo Finance) + technical features
#   3. Sector news sentiment (FinBERT) from the news saved in data/raw/news
#   4. Feature matrix
#   5. Training (main train/test split) + walk-forward + statistical tests
#   6. Tests
#
# Usage:   bash run_all.sh            (full mode, ~1 hour)
#          MODE=quick bash run_all.sh (quick mode, ~10 min)
# ------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
MODE="${MODE:-full}"
mkdir -p logs data/raw data/processed models/saved results/plots
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=================================================="; echo ">>> $*"; echo "=================================================="; }

# ---- 1. Python environment ----
step "1/7  Setting up Python environment"
PY=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "Python 3 not found. Install it from python.org first."; exit 1; }
echo "Using $($PY --version)"
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
# jupyter is only needed for notebooks, skip it to save time
grep -v -i "jupyter" requirements.txt > .requirements_run.txt
python -m pip install -q -r .requirements_run.txt
python -m pip install -q feedparser || true
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
rm -f .requirements_run.txt

# ---- 2. Prices ----
step "2/7  Downloading AAPL prices from Yahoo Finance"
rm -f data/raw/prices.csv
python - <<'EOF'
from src.data.data_loader import PriceDataLoader
loader = PriceDataLoader()
df = loader.download(force=True)
print(f"Rows: {len(df)}  |  {df.index.min()} -> {df.index.max()}")
EOF

# ---- 3. Technical features ----
step "3/7  Technical + regime features"
python -m src.data.feature_engineering

# ---- 4. Sentiment ----
step "4/7  FinBERT sector sentiment"
python - <<'EOF'
import os
from pathlib import Path
import pandas as pd
from src.data.sector_sentiment import SectorSentimentPipeline

pipe = SectorSentimentPipeline()
have_live = (os.getenv("FINNHUB_API_KEY") or os.getenv("NEWSAPI_KEY")
             or Path("data/raw/raw_analyst_ratings.csv").exists())
cache = Path("data/raw/news/AAPL_sector_raw.csv")

if not have_live and cache.exists():
    # No API keys / Kaggle file: use the real headlines already saved in the repo
    print(f"Using saved real news: {cache}")
    def from_cache():
        df = pd.read_csv(cache)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pipe.config["data"]["start_date"]) &
                (df["date"] <= pipe.config["data"]["end_date"])]
        print(f"  {len(df)} saved headlines in date range")
        return df.reset_index(drop=True)
    pipe.fetch_all_sector_news = from_cache

out = pipe.run(use_finbert=True)
assert out is not None and not out.empty, "Sentiment step produced no features"
EOF

# ---- 5. Combine ----
step "5/7  Combining features"
python -m src.data.feature_combiner

# ---- 6. Training + evaluation ----
step "6/7a Training sentiment + baseline agents (main split, mode=$MODE)"
python -m src.training.run_experiment --mode "$MODE"

step "6/7b Walk-forward validation + statistical tests (mode=$MODE)"
python -m src.training.run_full_experiment --mode "$MODE"

# ---- 7. Tests ----
step "7/7  Running tests"
python -m pytest -q tests || echo "(some tests failed, see above)"

step "DONE"
echo "Results:  results/  (JSON metrics + plots)"
echo "Models:   models/saved/"
echo "Log:      $LOG"
echo
echo "To start the API:  source .venv/bin/activate && uvicorn app:app --reload"
