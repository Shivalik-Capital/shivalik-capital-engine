import yfinance as yf
import pandas as pd

# --- NIFTY 50 STOCKS ---
# These are the ticker symbols for top Nifty 50 stocks on Yahoo Finance
# Yahoo Finance adds .NS at the end for NSE listed Indian stocks

NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS", "ONGC.NS",
    "DRREDDY.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "COALINDIA.NS"
]

# --- S&P 500 STOCKS ---
# Top 30 S&P 500 stocks by market cap
# No suffix needed for US stocks on Yahoo Finance

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

    period = how far back we go (5 years)
    interval = one candle per month (monthly data for momentum)

    Returns a dictionary:
    ticker → dataframe of that stock's price history
    """

    print(f"Fetching data for {len(tickers)} stocks...")

    all_data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            if len(df) > 0:
                all_data[ticker] = df
                print(f"  ✓ {ticker} — {len(df)} months of data")
            else:
                print(f"  ✗ {ticker} — no data found, skipping")

        except Exception as e:
            print(f"  ✗ {ticker} — error: {e}, skipping")

    print(f"\nSuccessfully fetched {len(all_data)} out of {len(tickers)} stocks")
    return all_data


def get_closing_prices(all_data):
    """
    Extracts just the closing prices from all stocks
    and combines them into one clean table.

    Rows = months
    Columns = each stock ticker
    """

    closes = {}

    for ticker, df in all_data.items():
        closes[ticker] = df["Close"].squeeze()

    # Combine into one dataframe
    price_table = pd.DataFrame(closes)

    # Drop any months where too many stocks have missing data
    price_table = price_table.dropna(thresh=int(len(price_table.columns) * 0.8))

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