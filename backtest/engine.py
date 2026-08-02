import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_universe import fetch_stock_data, get_closing_prices, NIFTY_TICKERS, SP500_TICKERS
from signals.momentum import calculate_price_momentum
from signals.volume import calculate_volume_momentum, get_volume_tables
from signals.volatility import calculate_volatility_score
from portfolio.ranking import calculate_composite_score, WEIGHTS


def build_monthly_portfolios(price_table, composite_scores, top_n=10):
    """
    For each month, select top N stocks and build equal-weight portfolio.
    Returns a dataframe where each row is a month, each column is a stock,
    value is the portfolio weight (0.05 for top 10, 0 otherwise)
    """
    
    portfolio_weights = pd.DataFrame(0.0, index=price_table.index, columns=price_table.columns)
    
    # Loop through each month
    for month_idx in range(1, len(composite_scores)):
        # Get scores for this month
        scores_this_month = composite_scores.iloc[month_idx].dropna()
        
        # Get top N stocks
        top_stocks = scores_this_month.sort_values(ascending=False).head(top_n).index
        
        # Assign equal weight (1/N) to each top stock
        for stock in top_stocks:
            portfolio_weights.iloc[month_idx, portfolio_weights.columns.get_loc(stock)] = 1.0 / top_n
    
    return portfolio_weights


def calculate_portfolio_returns(price_table, portfolio_weights):
    """
    Calculate monthly returns of the rebalanced portfolio.
    """
    
    # Monthly price changes
    monthly_returns = price_table.pct_change()
    
    # Portfolio return each month = sum of (weight * stock return)
    portfolio_returns = (monthly_returns * portfolio_weights).sum(axis=1)
    
    return portfolio_returns


def calculate_metrics(returns_series, benchmark_returns):
    """
    Calculate performance metrics.
    """
    
    # Total return
    total_return = (1 + returns_series).prod() - 1
    
    # Annualized return
    years = len(returns_series) / 12
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # Sharpe ratio (assuming 0% risk-free rate)
    monthly_vol = returns_series.std()
    sharpe = (returns_series.mean() / monthly_vol) * np.sqrt(12) if monthly_vol > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate vs benchmark
    wins = (returns_series > benchmark_returns).sum()
    win_rate = wins / len(returns_series)
    
    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SHIVALIK CAPITAL V2 — 5 YEAR BACKTEST")
    print("Composite Momentum Model (50/30/20 weights)")
    print("=" * 70)
    
    # ==================== NIFTY 50 BACKTEST ====================
    print("\nFetching Nifty 50 universe (60 months)...")
    nifty_data = fetch_stock_data(NIFTY_TICKERS)
    nifty_prices = get_closing_prices(nifty_data)
    nifty_volumes = get_volume_tables(nifty_data)
    
    print("Calculating composite scores...")
    nifty_composite = calculate_composite_score(nifty_prices, nifty_volumes)
    
    print("Building monthly portfolios...")
    nifty_portfolio_weights = build_monthly_portfolios(nifty_prices, nifty_composite, top_n=10)
    
    print("Calculating returns...")
    nifty_strategy_returns = calculate_portfolio_returns(nifty_prices, nifty_portfolio_weights)
    nifty_benchmark_returns = nifty_prices.pct_change().mean(axis=1)  # Equal-weight all stocks
    
    nifty_metrics = calculate_metrics(nifty_strategy_returns, nifty_benchmark_returns)
    
    print("\n" + "=" * 70)
    print("NIFTY 50 — STRATEGY vs EQUAL-WEIGHT BENCHMARK")
    print("=" * 70)
    print(f"Total Return:        {nifty_metrics['Total Return']:>8.2%}")
    print(f"Annual Return:       {nifty_metrics['Annual Return']:>8.2%}")
    print(f"Sharpe Ratio:        {nifty_metrics['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:        {nifty_metrics['Max Drawdown']:>8.2%}")
    print(f"Win Rate:            {nifty_metrics['Win Rate']:>8.2%}")
    
    # ==================== S&P 500 BACKTEST ====================
    print("\n\nFetching S&P 500 universe (60 months)...")
    sp500_data = fetch_stock_data(SP500_TICKERS)
    sp500_prices = get_closing_prices(sp500_data)
    sp500_volumes = get_volume_tables(sp500_data)
    
    print("Calculating composite scores...")
    sp500_composite = calculate_composite_score(sp500_prices, sp500_volumes)
    
    print("Building monthly portfolios...")
    sp500_portfolio_weights = build_monthly_portfolios(sp500_prices, sp500_composite, top_n=10)
    
    print("Calculating returns...")
    sp500_strategy_returns = calculate_portfolio_returns(sp500_prices, sp500_portfolio_weights)
    sp500_benchmark_returns = sp500_prices.pct_change().mean(axis=1)
    
    sp500_metrics = calculate_metrics(sp500_strategy_returns, sp500_benchmark_returns)
    
    print("\n" + "=" * 70)
    print("S&P 500 — STRATEGY vs EQUAL-WEIGHT BENCHMARK")
    print("=" * 70)
    print(f"Total Return:        {sp500_metrics['Total Return']:>8.2%}")
    print(f"Annual Return:       {sp500_metrics['Annual Return']:>8.2%}")
    print(f"Sharpe Ratio:        {sp500_metrics['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:        {sp500_metrics['Max Drawdown']:>8.2%}")
    print(f"Win Rate:            {sp500_metrics['Win Rate']:>8.2%}")
    
    print("\n" + "=" * 70)
    print("CROSS-MARKET SUMMARY")
    print("=" * 70)
    combined_total = nifty_metrics['Total Return'] + sp500_metrics['Total Return']
    combined_annual = (nifty_metrics['Annual Return'] + sp500_metrics['Annual Return']) / 2
    print(f"Combined Total:      {combined_total:>8.2%}")
    print(f"Average Annual:      {combined_annual:>8.2%}")