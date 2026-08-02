import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_universe import fetch_stock_data, get_closing_prices, NIFTY_TICKERS, SP500_TICKERS
from portfolio.ranking import calculate_composite_score, get_top_stocks
from signals.volume import get_volume_tables


def save_live_tracking(nifty_top_10, sp500_top_10, nifty_prices, sp500_prices):
    """
    Save today's portfolio recommendation and prices to JSON.
    """
    tracking_file = os.path.join(os.path.dirname(__file__), "live_tracking.json")
    today = datetime.now().strftime("%Y-%m-%d")
    
    nifty_current_prices = {}
    for stock in nifty_top_10.index:
        if stock in nifty_prices.columns:
            price = nifty_prices[stock].iloc[-1]
            nifty_current_prices[stock] = float(price)
    
    sp500_current_prices = {}
    for stock in sp500_top_10.index:
        if stock in sp500_prices.columns:
            price = sp500_prices[stock].iloc[-1]
            sp500_current_prices[stock] = float(price)
    
    today_entry = {
        "date": today,
        "nifty_portfolio": {stock: float(score) for stock, score in nifty_top_10.items()},
        "nifty_prices": nifty_current_prices,
        "sp500_portfolio": {stock: float(score) for stock, score in sp500_top_10.items()},
        "sp500_prices": sp500_current_prices
    }
    
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    else:
        tracking_data = []
    
    # Skip if we already have an entry for today (prevents duplicates)
    if any(entry["date"] == today for entry in tracking_data):
        print(f"[SKIP] Entry for {today} already exists, no duplicate created")
        return tracking_data
    
    tracking_data.append(today_entry)
    
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"[OK] Live tracking saved for {today}")
    return tracking_data


def calculate_live_performance(tracking_data):
    """
    Calculate cumulative performance and metrics from live tracking data.
    """
    if len(tracking_data) < 2:
        print("Need at least 2 days of data to calculate returns")
        return None, None
    
    portfolio_values = []
    
    for i in range(1, len(tracking_data)):
        prev_day = tracking_data[i-1]
        curr_day = tracking_data[i]
        
        # Calculate Nifty portfolio return
        nifty_return = 0
        for stock in prev_day["nifty_portfolio"].keys():
            if stock in prev_day["nifty_prices"] and stock in curr_day["nifty_prices"]:
                prev_price = prev_day["nifty_prices"][stock]
                curr_price = curr_day["nifty_prices"][stock]
                stock_return = (curr_price - prev_price) / prev_price
                nifty_return += (stock_return * 0.05)
        
        # Calculate S&P 500 portfolio return
        sp500_return = 0
        for stock in prev_day["sp500_portfolio"].keys():
            if stock in prev_day["sp500_prices"] and stock in curr_day["sp500_prices"]:
                prev_price = prev_day["sp500_prices"][stock]
                curr_price = curr_day["sp500_prices"][stock]
                stock_return = (curr_price - prev_price) / prev_price
                sp500_return += (stock_return * 0.05)
        
        # Combined daily return (50% Nifty, 50% S&P)
        combined_return = (nifty_return * 0.5) + (sp500_return * 0.5)
        
        portfolio_values.append({
            "date": curr_day["date"],
            "daily_return": combined_return
        })
    
    # Calculate cumulative and metrics
    cumulative_value = 1.0
    for entry in portfolio_values:
        cumulative_value *= (1 + entry["daily_return"])
        entry["cumulative_value"] = cumulative_value
    
    # Calculate metrics
    returns_array = np.array([e["daily_return"] for e in portfolio_values])
    cumulative_array = np.array([e["cumulative_value"] for e in portfolio_values])
    
    total_return = cumulative_array[-1] - 1
    daily_vol = returns_array.std()
    sharpe = (returns_array.mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
    
    running_max = np.maximum.accumulate(cumulative_array)
    drawdown = (cumulative_array - running_max) / running_max
    max_drawdown = drawdown.min()
    
    wins = (returns_array > 0).sum()
    win_rate = wins / len(returns_array)
    
    metrics = {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Days Tracked": len(portfolio_values)
    }
    
    return portfolio_values, metrics


if __name__ == "__main__":
    print("=" * 70)
    print("SHIVALIK CAPITAL — LIVE PERFORMANCE TRACKER")
    print("=" * 70)
    
    print("\nFetching latest Nifty 50 data...")
    nifty_data = fetch_stock_data(NIFTY_TICKERS)
    nifty_prices = get_closing_prices(nifty_data)
    nifty_volumes = get_volume_tables(nifty_data)
    
    print("Calculating Nifty composite scores...")
    nifty_composite = calculate_composite_score(nifty_prices, nifty_volumes)
    nifty_top_10 = get_top_stocks(nifty_composite, top_n=10)
    
    print("Top 10 Nifty stocks for today's portfolio:")
    for i, (stock, score) in enumerate(nifty_top_10.items()):
        print(f"  {i+1}. {stock}: {score:.4f}")
    
    print("\nFetching latest S&P 500 data...")
    sp500_data = fetch_stock_data(SP500_TICKERS)
    sp500_prices = get_closing_prices(sp500_data)
    sp500_volumes = get_volume_tables(sp500_data)
    
    print("Calculating S&P 500 composite scores...")
    sp500_composite = calculate_composite_score(sp500_prices, sp500_volumes)
    sp500_top_10 = get_top_stocks(sp500_composite, top_n=10)
    
    print("Top 10 S&P 500 stocks for today's portfolio:")
    for i, (stock, score) in enumerate(sp500_top_10.items()):
        print(f"  {i+1}. {stock}: {score:.4f}")
    
    # Save today's portfolio and prices
    tracking_data = save_live_tracking(nifty_top_10, sp500_top_10, nifty_prices, sp500_prices)
    
    # Calculate and display live performance
    print("\n" + "=" * 70)
    print("LIVE PERFORMANCE (since July 29, 2026)")
    print("=" * 70)
    
    live_performance, live_metrics = calculate_live_performance(tracking_data)
    
    if live_performance and live_metrics:
        print("\nDaily Performance (last 5 days):")
        for entry in live_performance[-5:]:
            print(f"  {entry['date']}: Daily {entry['daily_return']:>7.2%} | Cumulative {entry['cumulative_value']:>7.4f}")
        
        print("\n" + "=" * 70)
        print("LIVE PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Total Return:        {live_metrics['Total Return']:>8.2%}")
        print(f"Sharpe Ratio:        {live_metrics['Sharpe Ratio']:>8.2f}")
        print(f"Max Drawdown:        {live_metrics['Max Drawdown']:>8.2%}")
        print(f"Win Rate:            {live_metrics['Win Rate']:>8.2%}")
        print(f"Days Tracked:        {live_metrics['Days Tracked']:>8.0f}")
    else:
        print("First day recorded. Run again tomorrow to see performance.")