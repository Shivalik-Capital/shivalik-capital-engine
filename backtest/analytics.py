import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
import zipfile
import io
import yfinance as yf
import os
import time

def get_fama_french_factors():
    """Downloads and parses Fama-French 3-factor monthly data."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                lines = f.readlines()
                start_idx = 0
                for i, line in enumerate(lines):
                    if b"Mkt-RF" in line:
                        start_idx = i
                        break
                
                end_idx = len(lines)
                for i, line in enumerate(lines[start_idx+1:]):
                    if not line.strip() or b"Annual Factors" in line or b"Annual" in line:
                        end_idx = start_idx + 1 + i
                        break
                        
                f.seek(0)
                df = pd.read_csv(f, skiprows=start_idx, nrows=(end_idx - start_idx - 1))
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'].astype(str), format="%Y%m", errors='coerce')
                df = df.dropna(subset=['Date'])
                # FF provides beginning of month, we align by year-month
                df['YearMonth'] = df['Date'].dt.to_period('M')
                df.set_index('YearMonth', inplace=True)
                df.drop(columns=['Date'], inplace=True)
                
                # Convert from percentages to decimals
                return df / 100.0
    except Exception as e:
        print(f"Warning: Could not fetch Fama-French data: {e}")
        return None

def run_fama_french_regression(strategy_returns, benchmark_returns, is_us_market=True):
    """
    Runs regression analysis. For US, uses FF3 factors. 
    For non-US, falls back to CAPM (1-factor) against benchmark.
    Returns dict of results.
    """
    # Align to YearMonth to ensure joining works properly
    strat = strategy_returns.copy()
    strat.index = strat.index.to_period('M')
    bench = benchmark_returns.copy()
    bench.index = bench.index.to_period('M')
    
    df = pd.concat([strat, bench], axis=1).dropna()
    df.columns = ['Strategy', 'Benchmark']
    
    res = {
        "alpha_monthly": 0.0,
        "alpha_pvalue": 1.0,
        "beta_market": 0.0,
        "beta_smb": None,
        "beta_hml": None,
        "r_squared": 0.0,
        "is_ff3": False
    }
    
    if is_us_market:
        ff = get_fama_french_factors()
        if ff is not None:
            # Join data
            data = df.join(ff, how='inner').dropna()
            if len(data) > 10:
                y = data['Strategy'] - data['RF']
                X = data[['Mkt-RF', 'SMB', 'HML']]
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                res["alpha_monthly"] = model.params['const']
                res["alpha_pvalue"] = model.pvalues['const']
                res["beta_market"] = model.params['Mkt-RF']
                res["beta_smb"] = model.params['SMB']
                res["beta_hml"] = model.params['HML']
                res["r_squared"] = model.rsquared
                res["is_ff3"] = True
                return res
    
    # Fallback to CAPM
    if len(df) > 10:
        y = df['Strategy']
        X = df['Benchmark']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        res["alpha_monthly"] = model.params['const']
        res["alpha_pvalue"] = model.pvalues['const']
        res["beta_market"] = model.params['Benchmark']
        res["r_squared"] = model.rsquared
        
    return res

def analyze_market_regimes(strategy_returns, benchmark_returns):
    """
    Classifies months into Bull, Bear, and Sideways, and calculates average returns.
    """
    df = pd.DataFrame({
        'Strategy': strategy_returns,
        'Benchmark': benchmark_returns
    }).dropna()
    
    # Define regimes
    # Bull: benchmark > 2%
    # Bear: benchmark < -2%
    # Sideways: between -2% and 2%
    bull_mask = df['Benchmark'] > 0.02
    bear_mask = df['Benchmark'] < -0.02
    side_mask = (df['Benchmark'] >= -0.02) & (df['Benchmark'] <= 0.02)
    
    def get_stats(mask):
        subset = df[mask]
        if len(subset) == 0:
            return 0, 0.0, 0.0, 0.0
        s_avg = subset['Strategy'].mean()
        b_avg = subset['Benchmark'].mean()
        return len(subset), s_avg, b_avg, s_avg - b_avg
        
    bull_n, bull_s, bull_b, bull_a = get_stats(bull_mask)
    bear_n, bear_s, bear_b, bear_a = get_stats(bear_mask)
    side_n, side_s, side_b, side_a = get_stats(side_mask)
    
    return {
        "bull": {"n": bull_n, "strat_avg": bull_s, "bench_avg": bull_b, "alpha": bull_a},
        "bear": {"n": bear_n, "strat_avg": bear_s, "bench_avg": bear_b, "alpha": bear_a},
        "sideways": {"n": side_n, "strat_avg": side_s, "bench_avg": side_b, "alpha": side_a},
    }

def calculate_turnover(portfolio_weights):
    """
    Calculates average monthly turnover (weight changes).
    Since we are equal-weight (e.g. 10 stocks at 10% each), 
    changing 1 stock = 10% sold + 10% bought = 20% turnover.
    Generally, we report one-sided turnover: 
    Total turnover / 2
    """
    # Fill NA with 0
    w = portfolio_weights.fillna(0.0)
    # diff between months
    diff = w.diff().abs()
    # sum absolute weight changes across stocks per month
    total_change = diff.sum(axis=1)
    
    # drop the first month (NaN)
    total_change = total_change.dropna()
    
    # One-sided turnover
    one_sided_turnover = total_change / 2.0
    
    # Top 10 = 0.1 per stock. If one sided turnover is 0.1, we replaced 1 stock.
    # Replace count = one_sided_turnover / (1/N)
    # Average over time:
    avg_turnover = one_sided_turnover.mean()
    
    return avg_turnover

def analyze_sector_concentration(portfolio_weights):
    """
    For the historical portfolio weights, looks up sector of each ticker 
    and checks if it's over-concentrated.
    """
    all_tickers = portfolio_weights.columns.tolist()
    sector_map = {}
    
    print("Fetching sector information for portfolio stocks... (this takes a moment)")
    for ticker in all_tickers:
        # Only fetch for stocks that actually got picked at least once
        if portfolio_weights[ticker].sum() > 0:
            try:
                # Small sleep to prevent rate limiting
                time.sleep(0.1)
                info = yf.Ticker(ticker).info
                sector = info.get('sector', 'Unknown')
                sector_map[ticker] = sector
            except:
                sector_map[ticker] = 'Unknown'
                
    # Reconstruct portfolio by sector
    monthly_sectors = []
    
    for i in range(len(portfolio_weights)):
        row = portfolio_weights.iloc[i]
        active_stocks = row[row > 0]
        
        sector_counts = {}
        for ticker, weight in active_stocks.items():
            sec = sector_map.get(ticker, 'Unknown')
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            
        monthly_sectors.append(sector_counts)
        
    # Analyze across time
    max_concentration = 0.0
    max_concentration_sector = ""
    avg_sectors_per_month = 0.0
    
    total_sector_counts = {}
    
    valid_months = 0
    for counts in monthly_sectors:
        if not counts:
            continue
        valid_months += 1
        num_sectors = len(counts)
        avg_sectors_per_month += num_sectors
        
        for sec, count in counts.items():
            total_sector_counts[sec] = total_sector_counts.get(sec, 0) + count
            pct = count / sum(counts.values())
            if pct > max_concentration:
                max_concentration = pct
                max_concentration_sector = sec
                
    if valid_months > 0:
        avg_sectors_per_month /= valid_months
        
    # Sort most frequent
    sorted_sectors = sorted(total_sector_counts.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_sectors[:3]
    
    return {
        "avg_sectors_per_month": avg_sectors_per_month,
        "max_concentration": max_concentration,
        "max_concentration_sector": max_concentration_sector,
        "top_sectors": top_3,
        "warning": max_concentration > 0.4  # more than 40% in one sector
    }
