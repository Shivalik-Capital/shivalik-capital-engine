import pandas as pd
import requests
import io
import os
from datetime import datetime

# Optional dependency, used for fallback
from data.fetch_universe import NIFTY_TICKERS, SP500_TICKERS

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SP500_CACHE_FILE = os.path.join(DATA_DIR, "sp500_historical.csv")
NIFTY_CSV_FILE = os.path.join(DATA_DIR, "Historical_Nifty_50_Constituent_Weights.csv")

def _download_sp500_historical():
    """Downloads SP500 historical constituents from GitHub dataset if not cached."""
    if os.path.exists(SP500_CACHE_FILE):
        return pd.read_csv(SP500_CACHE_FILE)
        
    url = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
    try:
        print("Downloading S&P 500 historical constituents from fja05680/sp500...")
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv(SP500_CACHE_FILE, index=False)
        return df
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
        return None

def get_sp500_constituents(target_date):
    """
    Returns list of S&P 500 tickers active on a given date.
    target_date should be a string 'YYYY-MM-DD' or datetime object.
    """
    df = _download_sp500_historical()
    
    if df is None:
        print("WARNING: Falling back to static S&P 500 tickers.")
        return SP500_TICKERS
        
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
        
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the most recent row at or before target_date
    past_dates = df[df['date'] <= target_date]
    
    if past_dates.empty:
        print(f"WARNING: No S&P 500 data available before {target_date}. Falling back to static.")
        return SP500_TICKERS
        
    latest_row = past_dates.iloc[-1]
    tickers = latest_row['tickers'].split(',')
    
    return [t.strip() for t in tickers if t.strip()]

def get_nifty_constituents(target_date):
    """
    Returns list of Nifty 50 tickers active on a given date.
    Requires Kaggle CSV in data/ directory.
    target_date should be a string 'YYYY-MM-DD' or datetime object.
    """
    if not os.path.exists(NIFTY_CSV_FILE):
        print("WARNING: Kaggle Nifty 50 historical CSV not found.")
        print(f"Please place {NIFTY_CSV_FILE} in the data folder.")
        print("Falling back to static Nifty 50 tickers.")
        return NIFTY_TICKERS
        
    try:
        df = pd.read_csv(NIFTY_CSV_FILE)
        
        # The Kaggle CSV is in WIDE format: 'DATE' column, and then Ticker columns with weights
        date_col = next((col for col in df.columns if col.lower() in ['date', 'month', 'period']), None)
        
        if not date_col:
            return NIFTY_TICKERS
            
        df[date_col] = pd.to_datetime(df[date_col])
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        past_dates = df[df[date_col] <= target_date]
        if past_dates.empty:
            return NIFTY_TICKERS
            
        latest_date = past_dates[date_col].max()
        current_row = df[df[date_col] == latest_date].iloc[0]
        
        # Tickers are all columns except the date column where value > 0
        tickers = []
        for col in df.columns:
            if col != date_col:
                # If weight > 0, it was in the index
                val = current_row[col]
                if pd.notna(val) and float(val) > 0:
                    tickers.append(col)
        
        # Ensure .NS suffix is present for yfinance
        ns_tickers = []
        for t in tickers:
            t_str = str(t).strip()
            if not t_str.endswith('.NS'):
                t_str += '.NS'
            ns_tickers.append(t_str)
            
        return ns_tickers
    except Exception as e:
        print(f"Error reading Nifty 50 data: {e}. Falling back to static.")
        return NIFTY_TICKERS

def get_all_unique_tickers(market, start_date, end_date):
    """
    Returns union of all tickers that were in the index at any point between start_date and end_date.
    This is useful for downloading the superset of price data upfront.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    months = pd.date_range(start=start, end=end, freq='MS')
    all_tickers = set()
    
    if market.lower() == 'sp500':
        for dt in months:
            all_tickers.update(get_sp500_constituents(dt))
        # Add fallback just in case
        if not all_tickers:
            all_tickers.update(SP500_TICKERS)
    else:
        for dt in months:
            all_tickers.update(get_nifty_constituents(dt))
        if not all_tickers:
            all_tickers.update(NIFTY_TICKERS)
            
    return list(all_tickers)
