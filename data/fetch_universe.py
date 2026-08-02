import yfinance as yf
import pandas as pd

# Nifty 50 universe — NSE tickers (Yahoo Finance .NS suffix)

NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS", "ONGC.NS",
    "DRREDDY.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "COALINDIA.NS"
]

# S&P 500 universe — top 30 by market cap

SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA",
    "PG", "COST", "JNJ", "HD", "ABBV",
    "BAC", "KO", "MRK", "CVX", "CRM",
    "NFLX", "AMD", "PEP", "TMO", "ADBE"
]


def fetch_stock_data(tickers, period="5y", interval="1mo"):
    """
    Downloads monthly price and volume data for a list of stocks.
    Returns dict of ticker -> DataFrame with OHLCV history.
    """

    print(f"Fetching data for {len(tickers)} stocks...")

    all_data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            if len(df) > 0:
                all_data[ticker] = df
                print(f"  [OK] {ticker} -- {len(df)} months of data")
            else:
                print(f"  [FAIL] {ticker} -- no data found, skipping")

        except Exception as e:
            print(f"  [FAIL] {ticker} -- error: {e}, skipping")

    print(f"\nSuccessfully fetched {len(all_data)} out of {len(tickers)} stocks")
    return all_data


def get_closing_prices(all_data):
    """
    Extracts closing prices into a single DataFrame.
    Rows = months, Columns = tickers.
    """

    closes = {}

    for ticker, df in all_data.items():
        closes[ticker] = df["Close"].squeeze()

    # Combine into one dataframe
    price_table = pd.DataFrame(closes)

    # We intentionally do not dropna() here because with historical constituents, 
    # some stocks may not have data for the full 5 years. The ranking engine
    # handles NaNs on a per-month basis.

    return price_table


# --- TEST IT ---
if __name__ == "__main__":
    print("=" * 50)
    print("FETCHING NIFTY 50 STOCKS")
    print("=" * 50)
    nifty_data = fetch_stock_data(NIFTY_TICKERS)
    nifty_prices = get_closing_prices(nifty_data)
    print(f"\nNifty price table shape: {nifty_prices.shape}")
    print(nifty_prices.tail(3))

    print("\n" + "=" * 50)
    print("FETCHING S&P 500 STOCKS")
    print("=" * 50)
    sp500_data = fetch_stock_data(SP500_TICKERS)
    sp500_prices = get_closing_prices(sp500_data)
    print(f"\nS&P 500 price table shape: {sp500_prices.shape}")
    print(sp500_prices.tail(3))