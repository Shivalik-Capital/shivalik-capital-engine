import pandas as pd
import numpy as np


def calculate_price_momentum(price_table, lookback_months=6):
    """
    Calculates risk-adjusted price momentum for each stock.

    For each stock:
    - Calculate return over the past 'lookback_months' months
    - Divide by the stock's volatility over the same period
    - Result: momentum score (higher = stronger momentum)

    price_table: dataframe where rows = months, columns = stock tickers
    lookback_months: how far back we look (default 6 months)

    Returns a dataframe of momentum scores — same shape as price_table
    """

    # Step 1 — Calculate monthly returns for every stock
    # pct_change() gives us month over month % change
    monthly_returns = price_table.pct_change()

    # Step 2 — Calculate 6 month cumulative return for each stock
    # Rolling 6 month product of (1 + monthly return) - 1
    # This is the mathematically correct way to compound returns
    cumulative_return = (1 + monthly_returns).rolling(
        window=lookback_months
    ).apply(lambda x: x.prod() - 1, raw=True)

    # Step 3 — Calculate volatility over the same 6 month window
    # Volatility = standard deviation of monthly returns
    # Higher std dev = more volatile = noisier momentum signal
    volatility = monthly_returns.rolling(window=lookback_months).std()

    # Step 4 — Risk adjust: divide return by volatility
    # This penalizes stocks that had wild swings to get their return
    # A small number to avoid dividing by zero
    momentum_score = cumulative_return / (volatility + 1e-10)

    return momentum_score


def rank_by_momentum(momentum_score, date=None):
    """
    Takes momentum scores and ranks stocks from best to worst.

    date: which month to rank for (default = most recent month)

    Returns a series of stocks ranked by momentum score
    highest score = rank 1 (strongest momentum)
    """

    if date is None:
        # Use the most recent month
        scores = momentum_score.iloc[-1]
    else:
        scores = momentum_score.loc[date]

    # Drop stocks with missing scores
    scores = scores.dropna()

    # Rank from highest to lowest
    ranked = scores.sort_values(ascending=False)

    return ranked


# --- TEST IT ---
if __name__ == "__main__":
    import sys
    import os

    # Add parent folder to path so we can import fetch_universe
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fetch_universe import fetch_stock_data, get_closing_prices, NIFTY_TICKERS

    print("Fetching Nifty data for momentum test...")
    nifty_data = fetch_stock_data(NIFTY_TICKERS)
    nifty_prices = get_closing_prices(nifty_data)

    print("\nCalculating momentum scores...")
    momentum_scores = calculate_price_momentum(nifty_prices)

    print("\nRanking stocks by momentum (most recent month)...")
    ranked_stocks = rank_by_momentum(momentum_scores)

    print("\n" + "=" * 50)
    print("TOP 10 NIFTY STOCKS BY MOMENTUM")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked_stocks.head(10).items()):
        print(f"  {i + 1}. {ticker:<20} Score: {score:.4f}")

    print("\n" + "=" * 50)
    print("BOTTOM 5 NIFTY STOCKS BY MOMENTUM")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked_stocks.tail(5).items()):
        print(f"  {ticker:<20} Score: {score:.4f}")