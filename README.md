# OpenAI-Enhanced RL Trading Toolkit
> Derived from the FinRL open-source project. Goal: Extends the original with OpenAI LLM integration (sentiment & risk), intraday support, and streamlined scripts. Future extensions and improvements are planned.

## Background & Upstream Roadmap

The upstream FINRL project has recently consolidated the DeepSeek workflow into its core repository and published a comprehensive Developer Roadmap outlining planned enhancements and best practices. This toolkit will draws from those official updates—leveraging FINRL's unified data pipelines and environment abstractions—while adding OpenAI-powered sentiment and risk scoring and exploring additional feature improvements inspired by the roadmap.

This folder contains a self-contained toolkit to:
1. Score financial news headlines for sentiment and risk using OpenAI LLMs.
2. Prepare merged datasets for RL training.
3. Train PPO/CPPO agents on sentiment and risk-enhanced data.
4. Backtest trained agents.

## Setup
1. Ensure you are in this directory (project root).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

## Workflow

### 1. Score News Headlines
- Sentiment:
  ```bash
  python score_sentiment_openai.py --input data/headlines.csv \
    --output sentiment_scored.csv --model o4-mini
  ```
- Risk:
  ```bash
  python score_risk_openai.py --input data/headlines.csv \
    --output risk_scored.csv --model o4-mini
  ```

### 2. Prepare Dataset
```bash
python prepare_dataset_openai.py \
  --price-data data/intraday.csv \
  --sentiment sentiment_scored.csv \
  --risk risk_scored.csv \
  --output merged_dataset.csv
```

### 3. Train Agents
```bash
bash train_openai.sh merged_dataset.csv ppo sentiment
bash train_openai.sh merged_dataset.csv cppo risk
```

### 4. Backtest
```bash
python backtest_openai.py --data merged_dataset.csv \
  --model trained_models/agent_ppo_llm_100_epochs_sentiment.pth \
  --env sentiment --output-plot equity.png
```

## Project Structure
- `score_*.py`: headline scoring.
- `prepare_dataset_openai.py`: merges features & signals.
- `train_openai.sh`: wrapper to train agents.
- `backtest_openai.py`: runs backtest and plots.
- `train_ppo_llm.py`, `train_cppo_llm_risk.py`: core training scripts.
- `env_stocktrading_llm.py`, `env_stocktrading_llm_risk.py`: environment definitions.
- `requirements.txt`: Python dependencies.