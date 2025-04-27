#!/usr/bin/env python3
"""
Score downside risk of financial news headlines using OpenAI LLMs.
"""
import os
import argparse
import time
import json
import logging
from typing import Optional

import pandas as pd
import openai

# System prompt for risk scoring
SYSTEM_PROMPT = """
You are a financial risk officer.
Score each headline for downside risk of holding the stock:
 1 = very low risk
 2 = low risk
 3 = moderate / unknown (default)
 4 = high risk
 5 = very high / catastrophic risk
Return ONLY valid JSON: {"risks": [r1, r2, ...]}. No other keys, no explanation.
Use 3 when risk cannot be inferred.
"""

def score_headline(headline: str, symbol: str, model: str, retry: int = 3, pause: float = 0.5) -> Optional[int]:
    """
    Call OpenAI ChatCompletion to score one headline for risk.
    Returns integer risk or None on failure.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TICKER: {symbol}\nHEADLINES:\n1. {headline}"}
    ]
    for attempt in range(1, retry + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)
            if isinstance(data, dict) and "risks" in data and isinstance(data["risks"], list):
                return data["risks"][0]
            logging.warning(f"Unexpected response format: {data}")
            return None
        except Exception as e:
            logging.error(f"Attempt {attempt}/{retry} failed: {e}")
            time.sleep(pause * attempt)
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Score downside risk for financial news headlines using OpenAI"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV with columns: symbol, headline"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output CSV; adds 'risk_score' column"
    )
    parser.add_argument(
        "--model", default="o4-mini",
        help="OpenAI model name (e.g., o4-mini, gpt-4.1, o3)"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key

    df = pd.read_csv(args.input)
    if not all(col in df.columns for col in ["symbol", "headline"]):
        parser.error("Input CSV must contain 'symbol' and 'headline' columns")

    risks = []
    for idx, row in df.iterrows():
        sym = row["symbol"]
        text = row["headline"]
        risk = score_headline(text, sym, args.model)
        risks.append(risk)
        time.sleep(0.1)

    df["risk_score"] = risks
    df.to_csv(args.output, index=False)
    print(f"Wrote output with risk scores to {args.output}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()