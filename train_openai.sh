#!/usr/bin/env bash
# Wrapper script to train RL agents on OpenAI-enhanced dataset
# Usage: train_openai.sh <merged_dataset.csv> <algorithm> <mode>
#   algorithm: ppo or cppo
#   mode: sentiment or risk

set -e
if [ $# -ne 3 ]; then
  echo "Usage: $0 <merged_dataset.csv> <algorithm> <mode>"
  echo "  algorithm: ppo or cppo"
  echo "  mode: sentiment or risk"
  exit 1
fi

DATA=$1
ALG=$2
MODE=$3

echo "Training $ALG with $MODE signals using dataset $DATA"

case "$ALG" in
  ppo)
    if [ "$MODE" = "sentiment" ]; then
      cp "$DATA" train_data_deepseek_sentiment_2013_2018.csv
      mpirun -np 8 python train_ppo_llm.py
    else
      echo "PPO mode 'risk' not supported; use 'sentiment'"
      exit 1
    fi
    ;;
  cppo)
    if [ "$MODE" = "risk" ]; then
      cp "$DATA" train_data_deepseek_risk_2013_2018.csv
      mpirun -np 8 python train_cppo_llm_risk.py
    else
      echo "CPPO mode 'sentiment' not supported; use 'risk'"
      exit 1
    fi
    ;;
  *)
    echo "Unknown algorithm: $ALG (choose ppo or cppo)"
    exit 1
    ;;
esac