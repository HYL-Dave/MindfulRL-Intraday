#!/usr/bin/env python3
"""
Score sentiment of financial news headlines using OpenAI LLMs.
"""
import os
import argparse
import time
import json
import logging
from typing import Optional

import pandas as pd
import openai

# System prompt for sentiment scoring
SYSTEM_PROMPT = """
You are a sell-side equity strategist.
For each news headline about one stock, assign an integer sentiment score:
 1 = very bearish  (likely >5 % drop)
 2 = bearish       (2–5 % drop)
 3 = neutral / not relevant
 4 = bullish       (2–5 % rise)
 5 = very bullish  (>5 % rise)
Return ONLY valid JSON: {"scores": [s1, s2, ...]}. No other keys, no explanation.
If information is insufficient, use 3.
"""

def score_headline(headline: str, symbol: str, model: str, retry: int = 3, pause: float = 0.5) -> Optional[int]:
    """
    Call OpenAI ChatCompletion to score one headline.
    Returns integer score or None on failure.
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
            if isinstance(data, dict) and "scores" in data and isinstance(data["scores"], list):
                return data["scores"][0]
            logging.warning(f"Unexpected response format: {data}")
            return None
        except Exception as e:
            logging.error(f"Attempt {attempt}/{retry} failed: {e}")
            time.sleep(pause * attempt)
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Score sentiment for financial news headlines using OpenAI"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV with columns: symbol, headline"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output CSV; adds 'sentiment_score' column"
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

    scores = []
    for idx, row in df.iterrows():
        sym = row["symbol"]
        text = row["headline"]
        score = score_headline(text, sym, args.model)
        scores.append(score)
        time.sleep(0.1)

    df["sentiment_score"] = scores
    df.to_csv(args.output, index=False)
    print(f"Wrote output with sentiment scores to {args.output}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()