import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_universe import fetch_stock_data, get_closing_prices, NIFTY_TICKERS


def get_volume_tables(all_data):
    """Extracts volume data into a single DataFrame (rows=months, cols=tickers)."""
    volumes = {}
    for ticker, df in all_data.items():
        if "Volume" in df.columns:
            volumes[ticker] = df["Volume"].squeeze()

    volume_table = pd.DataFrame(volumes)
    volume_table = volume_table.dropna(thresh=int(len(volume_table.columns) * 0.8))
    return volume_table


def calculate_volume_momentum(volume_table, lookback_months=6):
    """
    Computes volume ratio: current month volume / rolling 6-month average.
    Ratio > 1 indicates above-average activity (positive signal).
    """
    avg_volume = volume_table.rolling(window=lookback_months).mean()
    volume_ratio = volume_table / (avg_volume + 1e-10)
    return volume_ratio


def rank_by_volume(volume_ratio):
    """Ranks stocks by volume momentum for the latest month."""
    scores = volume_ratio.iloc[-1].dropna()
    ranked = scores.sort_values(ascending=False)
    return ranked


if __name__ == "__main__":
    print("Fetching Nifty data...")
    nifty_data = fetch_stock_data(NIFTY_TICKERS)

    print("Extracting volume data...")
    volume_table = get_volume_tables(nifty_data)

    print("Calculating volume momentum...")
    volume_ratio = calculate_volume_momentum(volume_table)

    ranked = rank_by_volume(volume_ratio)

    print("\n" + "=" * 50)
    print("TOP 10 NIFTY STOCKS BY VOLUME MOMENTUM")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked.head(10).items()):
        print(f"  {i + 1}. {ticker:<20} Ratio: {score:.4f}")

    print("\n" + "=" * 50)
    print("BOTTOM 5 BY VOLUME MOMENTUM")
    print("=" * 50)
    for i, (ticker, score) in enumerate(ranked.tail(5).items()):
        print(f"  {ticker:<20} Ratio: {score:.4f}")