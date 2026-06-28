import pandas as pd
import numpy as np
import sys
import os

# Add the shivalik_v2 root to path so we can import from data/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_universe import fetch_stock_data, get_closing_prices, NIFTY_TICKERS


def calculate_price_momentum(price_table, lookback_months=6):
    monthly_returns = price_table.pct_change()
    cumulative_return = (1 + monthly_returns).rolling(
        window=lookback_months
    ).apply(lambda x: x.prod() - 1, raw=True)
    volatility = monthly_returns.rolling(window=lookback_months).std()
    momentum_score = cumulative_return / (volatility + 1e-10)
    return momentum_score


def rank_by_momentum(momentum_scores):
    scores = momentum_scores.iloc[-1].dropna()
    ranked = scores.sort_values(ascending=False)
    return ranked


if __name__ == "__main__":
    print("Fetching Nifty data...")
    nifty_data = fetch_stock_data(NIFTY_TICKERS)
    nifty_prices = get_closing_prices(nifty_data)

    print("Calculating momentum scores...")
    momentum_scores = calculate_price_momentum(nifty_prices)

    print("Ranking stocks...")
    ranked = rank_by_momentum(momentum_scores)

    print("\n" + "=" * 50)
    print("TOP 10 NIFTY STOCKS BY MOMENTUM")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked.head(10).items()):
        print(f"  {i+1}. {ticker:<20} Score: {score:.4f}")

    print("\n" + "=" * 50)
    print("BOTTOM 5 NIFTY STOCKS BY MOMENTUM")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked.tail(5).items()):
        print(f"  {ticker:<20} Score: {score:.4f}")