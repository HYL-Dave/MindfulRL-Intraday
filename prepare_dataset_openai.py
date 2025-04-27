#!/usr/bin/env python3
"""
Merge price+indicator data with sentiment and risk scores for RL training/backtest.
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Prepare RL dataset by merging price, indicators, sentiment, and risk"
    )
    parser.add_argument(
        "--price-data", required=True,
        help="CSV file with price and technical indicators; must include date and symbol columns"
    )
    parser.add_argument(
        "--sentiment", required=False,
        help="CSV with sentiment scores; must include date, symbol, sentiment_score"
    )
    parser.add_argument(
        "--risk", required=False,
        help="CSV with risk scores; must include date, symbol, risk_score"
    )
    parser.add_argument(
        "--date-col", default="date",
        help="Name of the date column in CSVs"
    )
    parser.add_argument(
        "--symbol-col", default="symbol",
        help="Name of the symbol column in CSVs"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output merged CSV"
    )
    args = parser.parse_args()

    price = pd.read_csv(args.price_data, parse_dates=[args.date_col])
    df = price.copy()

    # Merge sentiment
    if args.sentiment:
        sent = pd.read_csv(args.sentiment, parse_dates=[args.date_col])
        sent = sent.rename(columns={"sentiment_score": "llm_sentiment"})
        df = df.merge(
            sent[[args.date_col, args.symbol_col, "llm_sentiment"]],
            on=[args.date_col, args.symbol_col], how="left"
        )
    else:
        df["llm_sentiment"] = 3

    # Merge risk
    if args.risk:
        risk = pd.read_csv(args.risk, parse_dates=[args.date_col])
        risk = risk.rename(columns={"risk_score": "llm_risk"})
        df = df.merge(
            risk[[args.date_col, args.symbol_col, "llm_risk"]],
            on=[args.date_col, args.symbol_col], how="left"
        )
    else:
        df["llm_risk"] = 3

    # Fill missing
    df["llm_sentiment"].fillna(3, inplace=True)
    df["llm_risk"].fillna(3, inplace=True)

    df.to_csv(args.output, index=False)
    print(f"Merged dataset saved to {args.output}")

if __name__ == "__main__":
    main()