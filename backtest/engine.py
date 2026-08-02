import pandas as pd
import numpy as np
import sys
import os
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_universe import fetch_stock_data, get_closing_prices
from data.historical_constituents import get_all_unique_tickers, get_sp500_constituents, get_nifty_constituents
from signals.momentum import calculate_price_momentum
from signals.volume import calculate_volume_momentum, get_volume_tables
from signals.volatility import calculate_volatility_score
from portfolio.ranking import calculate_composite_score

from backtest.analytics import (
    run_fama_french_regression,
    analyze_market_regimes,
    calculate_turnover,
    analyze_sector_concentration
)

def build_monthly_portfolios(price_table, composite_scores, market="sp500", top_n=10):
    portfolio_weights = pd.DataFrame(0.0, index=price_table.index, columns=price_table.columns)
    
    for month_idx in range(len(composite_scores)):
        current_date = composite_scores.index[month_idx]
        
        # 1. Fetch active constituents for THIS SPECIFIC MONTH (Fixes Survivorship Bias)
        if market == "sp500":
            active_tickers = get_sp500_constituents(current_date)
        else:
            active_tickers = get_nifty_constituents(current_date)
            
        scores_this_month = composite_scores.iloc[month_idx]
        
        # 2. Only rank stocks that were actually in the index at that time
        available_active = list(set(active_tickers).intersection(scores_this_month.dropna().index))
        valid_scores = scores_this_month[available_active]
        
        if len(valid_scores) > 0:
            top_stocks = valid_scores.sort_values(ascending=False).head(top_n).index
            
            for stock in top_stocks:
                portfolio_weights.iloc[month_idx, portfolio_weights.columns.get_loc(stock)] = 1.0 / top_n
                
    return portfolio_weights

def calculate_portfolio_returns(price_table, portfolio_weights, market="sp500"):
    # 1. Shift weights by 1 month. If we calculate weights on Jan 31, we earn Feb returns.
    # (Fixes Look-Ahead Bias)
    actual_weights = portfolio_weights.shift(1).fillna(0.0)
    
    monthly_returns = price_table.pct_change()
    gross_returns = (monthly_returns * actual_weights).sum(axis=1)
    
    # 2. Transaction Costs (Fixes zero-cost illusion)
    cost_bps = 0.001 if market == "india" else 0.0005
    turnover = portfolio_weights.diff().abs().sum(axis=1) / 2.0  # One-sided
    costs = turnover * cost_bps
    
    net_returns = gross_returns - costs.fillna(0.0)
    
    return gross_returns, net_returns, costs

def calculate_metrics(returns_series, benchmark_returns):
    total_return = (1 + returns_series).prod() - 1
    
    years = len(returns_series) / 12
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    
    monthly_vol = returns_series.std()
    sharpe = (returns_series.mean() / monthly_vol) * np.sqrt(12) if monthly_vol > 0 else 0
    
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    wins = (returns_series > benchmark_returns).sum()
    win_rate = wins / len(returns_series) if len(returns_series) > 0 else 0
    
    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }

def run_backtest_for_market(market_name, start_date, end_date):
    print(f"\n==================================================")
    print(f"RUNNING BACKTEST: {market_name.upper()}")
    print(f"==================================================")
    
    is_us = (market_name == "sp500")
    
    print(f"1. Determining historical universe ({start_date} to {end_date})...")
    tickers = get_all_unique_tickers(market_name, start_date, end_date)
    
    print(f"2. Fetching price and volume data for {len(tickers)} stocks...")
    stock_data = fetch_stock_data(tickers, period="5y", interval="1mo")
    prices = get_closing_prices(stock_data)
    volumes = get_volume_tables(stock_data)
    
    print("3. Fetching actual index benchmark...")
    bench_ticker = "^SPXTR" if is_us else "NIFTYBEES.NS"
    bench_data = yf.download(bench_ticker, start=start_date, end=end_date, interval="1mo", progress=False)
    if not bench_data.empty and "Close" in bench_data.columns:
        bench_prices = bench_data["Close"].squeeze()
        # Some yfinance weirdness with single ticker downloads returns DataFrame, squeeze handles it
        if isinstance(bench_prices, pd.DataFrame):
            bench_prices = bench_prices.iloc[:, 0]
    else:
        print(f"Failed to fetch {bench_ticker}, falling back to fake benchmark.")
        bench_prices = prices.mean(axis=1) # absolute fallback
    
    bench_returns = bench_prices.pct_change().dropna()
    
    print("4. Calculating momentum, volume, and volatility signals...")
    composite = calculate_composite_score(prices, volumes)
    
    print("5. Building point-in-time monthly portfolios (Fixing Survivorship Bias)...")
    weights = build_monthly_portfolios(prices, composite, market=market_name, top_n=10)
    
    print("6. Calculating returns and applying transaction costs...")
    gross_ret, net_ret, costs = calculate_portfolio_returns(prices, weights, market="us" if is_us else "india")
    
    # Align dates
    net_ret = net_ret.loc[bench_returns.index.intersection(net_ret.index)]
    bench_returns = bench_returns.loc[net_ret.index]
    
    # 7. Metrics & Analytics
    metrics = calculate_metrics(net_ret, bench_returns)
    alpha_total = metrics['Total Return'] - ((1 + bench_returns).prod() - 1)
    
    # Walk-forward (First 36 months In-Sample, Rest Out-of-Sample)
    split_idx = 36 if len(net_ret) > 36 else len(net_ret) // 2
    is_ret = net_ret.iloc[:split_idx]
    oos_ret = net_ret.iloc[split_idx:]
    
    is_metrics = calculate_metrics(is_ret, bench_returns.iloc[:split_idx])
    oos_metrics = calculate_metrics(oos_ret, bench_returns.iloc[split_idx:])
    
    # Advanced Analytics
    ff_res = run_fama_french_regression(net_ret, bench_returns, is_us_market=is_us)
    regimes = analyze_market_regimes(net_ret, bench_returns)
    avg_turnover = calculate_turnover(weights)
    sector_analysis = analyze_sector_concentration(weights)
    
    # 8. Output Report
    print("\n" + "=" * 60)
    print(f"{market_name.upper()} — STRATEGY vs ACTUAL {bench_ticker}")
    print("=" * 60)
    
    print("--- PERFORMANCE (Net of Costs) ---")
    print(f"Total Return:            {metrics['Total Return']:>8.2%}")
    print(f"Benchmark Return:        {((1 + bench_returns).prod() - 1):>8.2%}")
    print(f"Excess Return (Alpha):   {alpha_total:>8.2%}")
    print(f"Annualized Return:       {metrics['Annual Return']:>8.2%}")
    print(f"Sharpe Ratio:            {metrics['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:            {metrics['Max Drawdown']:>8.2%}")
    print(f"Win Rate vs Bench:       {metrics['Win Rate']:>8.2%}")
    print(f"Total Est. Costs:        {costs.sum():>8.2%}")
    
    print("\n--- WALK-FORWARD VALIDATION ---")
    print(f"In-Sample Sharpe:        {is_metrics['Sharpe Ratio']:>8.2f} (Months 1-{split_idx})")
    print(f"Out-of-Sample Sharpe:    {oos_metrics['Sharpe Ratio']:>8.2f} (Months {split_idx+1}-{len(net_ret)})")
    if is_metrics['Sharpe Ratio'] > 0:
        decay = (oos_metrics['Sharpe Ratio'] / is_metrics['Sharpe Ratio'])
        print(f"Sharpe Retention:        {decay:>8.2%} (Target > 60%)")
        
    print("\n--- REGIME ANALYSIS ---")
    for reg, stats in regimes.items():
        print(f"{reg.capitalize():<8s} ({stats['n']:2d} mos):  Strat: {stats['strat_avg']:>6.2%} | Bench: {stats['bench_avg']:>6.2%} | Alpha: {stats['alpha']:>6.2%}")
        
    print("\n--- REGRESSION ANALYSIS ---")
    print(f"Monthly Alpha (reg):     {ff_res['alpha_monthly']:>8.2%} (p={ff_res['alpha_pvalue']:.3f})")
    print(f"Market Beta:             {ff_res['beta_market']:>8.2f}")
    if ff_res['is_ff3']:
        print(f"Size Exp (SMB):          {ff_res['beta_smb']:>8.2f}")
        print(f"Value Exp (HML):         {ff_res['beta_hml']:>8.2f}")
    print(f"R-squared:               {ff_res['r_squared']:>8.2f}")
    
    print("\n--- PORTFOLIO HEALTH ---")
    print(f"Avg Monthly Turnover:    {avg_turnover:>8.2%}")
    print(f"Avg Sectors/Month:       {sector_analysis['avg_sectors_per_month']:>8.1f} / 10")
    print(f"Max Single Sector Wgt:   {sector_analysis['max_concentration']:>8.1%} ({sector_analysis['max_concentration_sector']})")
    print("Most Frequent Sectors:")
    for sec, count in sector_analysis['top_sectors']:
        print(f"  - {sec}: {count} appearances")

if __name__ == "__main__":
    start = "2021-07-01"
    end = "2026-07-01"
    
    print("=" * 70)
    print("SHIVALIK CAPITAL V2 — 5 YEAR BACKTEST (BIAS-CORRECTED)")
    print("=" * 70)
    
    run_backtest_for_market("nifty", start, end)
    run_backtest_for_market("sp500", start, end)