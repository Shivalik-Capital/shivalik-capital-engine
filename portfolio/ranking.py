import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.fetch_universe
import signals.momentum
import signals.volume
import signals.volatility

# Signal weights for composite score
WEIGHTS = {
    "momentum": 0.50,
    "volume": 0.30,
    "volatility": 0.20
}


def normalize(df):
    """
    Rank-based normalization to [0, 1] percentile scale.
    Ensures signals with different magnitudes are comparable.
    """
    return df.rank(axis=1, pct=True)


def calculate_composite_score(price_table, volume_table):
    """
    Combines momentum, volume, and volatility signals into a single
    weighted composite score per stock per month.
    """

    # Calculate raw signals
    momentum_scores = signals.momentum.calculate_price_momentum(price_table)
    volume_scores = signals.volume.calculate_volume_momentum(volume_table)
    volatility_scores = signals.volatility.calculate_volatility_score(price_table)

    # Normalize all to 0-1 scale using percentile ranks
    momentum_norm = normalize(momentum_scores)
    volume_norm = normalize(volume_scores)
    volatility_norm = normalize(volatility_scores)

    # Align all dataframes to same dates
    common_index = momentum_norm.index.intersection(
        volume_norm.index).intersection(volatility_norm.index)

    momentum_norm = momentum_norm.loc[common_index]
    volume_norm = volume_norm.loc[common_index]
    volatility_norm = volatility_norm.loc[common_index]

    # Weighted combination
    composite = (
            momentum_norm * WEIGHTS["momentum"] +
            volume_norm * WEIGHTS["volume"] +
            volatility_norm * WEIGHTS["volatility"]
    )

    return composite


def get_top_stocks(composite_scores, top_n=10):
    """Returns top N stocks by composite score for the most recent month."""
    latest_scores = composite_scores.iloc[-1].dropna()
    top_stocks = latest_scores.sort_values(ascending=False).head(top_n)
    return top_stocks


if __name__ == "__main__":
    print("=" * 60)
    print("SHIVALIK CAPITAL — COMPOSITE MOMENTUM RANKING")
    print("=" * 60)

    print("\nFetching Nifty 50 universe...")
    nifty_data = data.fetch_universe.fetch_stock_data(data.fetch_universe.NIFTY_TICKERS)
    nifty_prices = data.fetch_universe.get_closing_prices(nifty_data)
    nifty_volumes = signals.volume.get_volume_tables(nifty_data)

    print("\nCalculating composite scores...")
    nifty_composite = calculate_composite_score(nifty_prices, nifty_volumes)

    top_nifty = get_top_stocks(nifty_composite, top_n=10)

    print("\n" + "=" * 60)
    print("TOP 10 NIFTY STOCKS — COMPOSITE SCORE")
    print("Price Momentum 50% | Volume 30% | Volatility 20%")
    print("=" * 60)
    for i, (ticker, score) in enumerate(top_nifty.items()):
        print(f"  {i + 1}. {ticker:<20} Composite: {score:.4f}")

    print("\nFetching S&P 500 universe...")
    sp500_data = data.fetch_universe.fetch_stock_data(data.fetch_universe.SP500_TICKERS)
    sp500_prices = data.fetch_universe.get_closing_prices(sp500_data)
    sp500_volumes = signals.volume.get_volume_tables(sp500_data)

    print("\nCalculating composite scores...")
    sp500_composite = calculate_composite_score(sp500_prices, sp500_volumes)

    top_sp500 = get_top_stocks(sp500_composite, top_n=10)

    print("\n" + "=" * 60)
    print("TOP 10 S&P 500 STOCKS — COMPOSITE SCORE")
    print("Price Momentum 50% | Volume 30% | Volatility 20%")
    print("=" * 60)
    for i, (ticker, score) in enumerate(top_sp500.items()):
        print(f"  {i + 1}. {ticker:<20} Composite: {score:.4f}")