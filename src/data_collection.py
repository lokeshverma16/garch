import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TICKERS = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}
START_DATE = '2014-01-01'
END_DATE = '2023-12-31'


def download_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}. Check if the ticker is valid and data is available.")
        return df
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def calculate_log_returns(df, ticker):
    # Handle multi-index columns from yfinance
    print(f"Downloaded columns for {ticker}: {df.columns}")
    if isinstance(df.columns, pd.MultiIndex):
        if ('Adj Close', ticker) in df.columns:
            adj_close = df[('Adj Close', ticker)]
        elif ('Close', ticker) in df.columns:
            adj_close = df[('Close', ticker)]
        else:
            raise ValueError(f"'Adj Close' or 'Close' column not found for {ticker}. Columns: {df.columns}")
    else:
        if 'Adj Close' in df.columns:
            adj_close = df['Adj Close']
        elif 'Close' in df.columns:
            adj_close = df['Close']
        else:
            raise ValueError(f"'Adj Close' or 'Close' column not found for {ticker}. Columns: {df.columns}")
    log_returns = np.log(adj_close / adj_close.shift(1))
    log_returns = log_returns.dropna()
    return adj_close, log_returns

def check_outliers(series, threshold=5):
    mean = series.mean()
    std = series.std()
    outliers = series[np.abs(series - mean) > threshold * std]
    return outliers

def check_missing(series):
    return series.isnull().sum()

def check_continuity(df):
    # Check for missing business days
    all_days = pd.date_range(df.index.min(), df.index.max(), freq='B')
    missing_days = all_days.difference(df.index)
    return missing_days

def save_to_csv(df, filename):
    try:
        df.to_csv(os.path.join(DATA_DIR, filename))
        print(f"Saved cleaned data to {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def summary_statistics(series):
    stats = series.describe()
    stats['skew'] = series.skew()
    stats['kurtosis'] = series.kurtosis()
    return stats

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    summary_tables = {}
    for ticker, name in TICKERS.items():
        print(f"\nProcessing {name} ({ticker})...")
        df = download_data(ticker, START_DATE, END_DATE)
        if df is None:
            continue
        # Forward fill missing values (weekends/holidays)
        df = df.asfreq('B')
        df = df.ffill()
        # Calculate log returns
        try:
            adj_close, log_ret = calculate_log_returns(df, ticker)
        except Exception as e:
            print(f"Error calculating log returns for {ticker}: {e}")
            continue
        # Data quality checks
        outliers = check_outliers(log_ret)
        n_missing = check_missing(log_ret)
        missing_days = check_continuity(df)
        # Feedback
        if not outliers.empty:
            print(f"Warning: {len(outliers)} outlier(s) detected (>5 std dev). Indices: {outliers.index.tolist()}")
        if n_missing > 0:
            print(f"Warning: {n_missing} missing value(s) in log returns.")
        if len(missing_days) > 0:
            print(f"Warning: {len(missing_days)} missing business day(s) in data continuity. Dates: {missing_days.date.tolist()}")
        else:
            print("Data continuity check passed.")
        # Save cleaned data
        cleaned = pd.DataFrame({
            'Adj Close': adj_close,
            'Log Return': log_ret
        })
        save_to_csv(cleaned, f"{ticker}_cleaned.csv")
        # Summary statistics
        stats = summary_statistics(log_ret)
        summary_tables[ticker] = stats
        print(f"Summary statistics for {ticker} log returns:\n{stats}\n")
    # Save summary table
    try:
        summary_df = pd.DataFrame(summary_tables)
        save_to_csv(summary_df, "summary_statistics.csv")
    except Exception as e:
        print(f"Error saving summary statistics: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 