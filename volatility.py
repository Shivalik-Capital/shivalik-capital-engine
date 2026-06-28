import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_universe import fetch_stock_data, get_closing_prices, NIFTY_TICKERS


def calculate_volatility_score(price_table, lookback_months=6):
    """
    Calculates a volatility score for each stock.

    Lower volatility = higher score (we want smooth movers)
    Higher volatility = lower score (we penalize wild swings)

    Returns a dataframe of volatility scores
    """

    monthly_returns = price_table.pct_change()

    # Standard deviation of returns over lookback period
    # Higher std dev = more volatile = noisier signal
    volatility = monthly_returns.rolling(window=lookback_months).std()

    # Invert so lower volatility = higher score
    # Add small number to avoid dividing by zero
    volatility_score = 1 / (volatility + 1e-10)

    # Normalize so scores are comparable across stocks
    volatility_score = volatility_score.div(
        volatility_score.rolling(window=lookback_months).mean()
    )

    return volatility_score


def rank_by_volatility(volatility_score):
    """
    Ranks stocks by volatility score — highest score (lowest volatility) first
    """
    scores = volatility_score.iloc[-1].dropna()
    ranked = scores.sort_values(ascending=False)
    return ranked


if __name__ == "__main__":
    print("Fetching Nifty data...")
    nifty_data = fetch_stock_data(NIFTY_TICKERS)
    nifty_prices = get_closing_prices(nifty_data)

    print("Calculating volatility scores...")
    volatility_scores = calculate_volatility_score(nifty_prices)

    ranked = rank_by_volatility(volatility_scores)

    print("\n" + "=" * 50)
    print("TOP 10 NIFTY STOCKS BY LOW VOLATILITY")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked.head(10).items()):
        print(f"  {i + 1}. {ticker:<20} Score: {score:.4f}")

    print("\n" + "=" * 50)
    print("MOST VOLATILE (BOTTOM 5)")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked.tail(5).items()):
        print(f"  {ticker:<20} Score: {score:.4f}")