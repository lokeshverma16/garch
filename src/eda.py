import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import pearsonr
import warnings

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}

# Helper to load cleaned data
def load_data(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_cleaned.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

def savefig(fig, name):
    try:
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, name), dpi=300)
        plt.close(fig)
        print(f"Saved figure: {name}")
    except Exception as e:
        print(f"Error saving figure {name}: {e}")
        plt.close(fig)

def get_price_column(df):
    if 'Adj Close' in df.columns:
        return df['Adj Close']
    elif 'Close' in df.columns:
        return df['Close']
    else:
        raise ValueError(f"Neither 'Adj Close' nor 'Close' found in columns: {df.columns}")

def plot_price_levels(dfs):
    try:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, ticker in enumerate(TICKERS):
            price = get_price_column(dfs[ticker])
            price.plot(ax=ax[i], label=NAMES[ticker])
            ax[i].set_ylabel('Price')
            ax[i].set_title(f"{NAMES[ticker]} Price Level")
            ax[i].legend()
            ax[i].grid(True)
        ax[1].set_xlabel('Date')
        savefig(fig, 'price_levels.png')
    except Exception as e:
        print(f"Error in plot_price_levels: {e}")

def plot_log_returns(dfs):
    try:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, ticker in enumerate(TICKERS):
            dfs[ticker]['Log Return'].plot(ax=ax[i], label=NAMES[ticker], color='tab:blue')
            ax[i].set_ylabel('Log Return')
            ax[i].set_title(f"{NAMES[ticker]} Log Returns")
            ax[i].legend()
            ax[i].grid(True)
        ax[1].set_xlabel('Date')
        savefig(fig, 'log_returns.png')
    except Exception as e:
        print(f"Error in plot_log_returns: {e}")

def plot_rolling_volatility(dfs, window=21):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        for ticker in TICKERS:
            rolling_vol = dfs[ticker]['Log Return'].rolling(window).std() * np.sqrt(252)
            rolling_vol.plot(ax=ax, label=f"{NAMES[ticker]} {window}d Volatility")
        # Example: shade 2020 COVID recession (Feb 2020 - Apr 2020)
        ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30'), color='gray', alpha=0.2, label='COVID Recession')
        ax.set_title('21-Day Rolling Volatility (Annualized)')
        ax.set_ylabel('Volatility')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        savefig(fig, 'rolling_volatility.png')
    except Exception as e:
        print(f"Error in plot_rolling_volatility: {e}")

def correlation_analysis(dfs):
    try:
        ret1 = dfs[TICKERS[0]]['Log Return']
        ret2 = dfs[TICKERS[1]]['Log Return']
        # Align indices and drop NaNs/infs
        ret1, ret2 = ret1.align(ret2, join='inner')
        valid = (~ret1.isna()) & (~ret2.isna()) & (~ret1.isin([np.inf, -np.inf])) & (~ret2.isin([np.inf, -np.inf]))
        ret1 = ret1[valid]
        ret2 = ret2[valid]
        corr, pval = pearsonr(ret1, ret2)
        print(f"Correlation between {NAMES[TICKERS[0]]} and {NAMES[TICKERS[1]]} returns: {corr:.4f} (p={pval:.4g})")
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x=ret1, y=ret2, ax=ax, scatter_kws={'alpha':0.5})
        ax.set_title(f"Correlation of Log Returns\nPearson r={corr:.3f}, p={pval:.2g}")
        ax.set_xlabel(f"{NAMES[TICKERS[0]]} Log Return")
        ax.set_ylabel(f"{NAMES[TICKERS[1]]} Log Return")
        ax.grid(True)
        savefig(fig, 'correlation_returns.png')
        return corr, pval
    except Exception as e:
        print(f"Error in correlation_analysis: {e}")
        return None, None

def descriptive_stats(dfs):
    stats = {}
    for ticker in TICKERS:
        s = dfs[ticker]['Log Return'].dropna()
        jb_stat, jb_p, _, _ = jarque_bera(s)
        stats[ticker] = {
            'mean': s.mean(),
            'std': s.std(),
            'skew': s.skew(),
            'kurtosis': s.kurtosis(),
            'Jarque-Bera': jb_stat,
            'JB p-value': jb_p
        }
    df_stats = pd.DataFrame(stats).T
    print("\nDescriptive statistics:")
    print(df_stats)
    try:
        df_stats.to_csv(os.path.join(FIG_DIR, 'descriptive_statistics.csv'))
    except Exception as e:
        print(f"Error saving descriptive statistics: {e}")
    return df_stats

def box_plots(dfs):
    try:
        data = [dfs[ticker]['Log Return'].dropna() for ticker in TICKERS]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data, labels=[NAMES[t] for t in TICKERS], patch_artist=True)
        ax.set_title('Box Plot of Log Returns')
        ax.set_ylabel('Log Return')
        ax.grid(True)
        savefig(fig, 'boxplot_returns.png')
    except Exception as e:
        print(f"Error in box_plots: {e}")

def acf_pacf_plots(dfs):
    try:
        for ticker in TICKERS:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            s = dfs[ticker]['Log Return'].dropna()
            sq = s ** 2
            plot_acf(s, ax=axes[0,0], lags=40, title=f"{NAMES[ticker]} Log Returns ACF")
            plot_pacf(s, ax=axes[0,1], lags=40, title=f"{NAMES[ticker]} Log Returns PACF")
            plot_acf(sq, ax=axes[1,0], lags=40, title=f"{NAMES[ticker]} Squared Returns ACF")
            plot_pacf(sq, ax=axes[1,1], lags=40, title=f"{NAMES[ticker]} Squared Returns PACF")
            for axx in axes.flatten():
                axx.grid(True)
            fig.suptitle(f"ACF/PACF for {NAMES[ticker]}")
            savefig(fig, f"acf_pacf_{ticker}.png")
    except Exception as e:
        print(f"Error in acf_pacf_plots: {e}")

def joint_distribution_plot(dfs):
    try:
        ret1 = dfs[TICKERS[0]]['Log Return']
        ret2 = dfs[TICKERS[1]]['Log Return']
        # Align indices
        ret1, ret2 = ret1.align(ret2, join='inner')
        fig = plt.figure(figsize=(8, 8))
        g = sns.jointplot(x=ret1, y=ret2, kind='scatter', marginal_kws=dict(bins=30, fill=True), color='tab:blue', alpha=0.5)
        g.set_axis_labels(f"{NAMES[TICKERS[0]]} Log Return", f"{NAMES[TICKERS[1]]} Log Return")
        plt.suptitle('Joint Distribution of Log Returns', y=1.02)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'joint_distribution.png'), dpi=300)
        plt.close()
        print("Saved figure: joint_distribution.png")
    except Exception as e:
        print(f"Error in joint_distribution_plot: {e}")

def main():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    # Load data
    dfs = {ticker: load_data(ticker) for ticker in TICKERS}
    if any(df is None for df in dfs.values()):
        print("Error: Could not load all data files. Exiting.")
        sys.exit(1)
    # Plots and analysis
    plot_price_levels(dfs)
    plot_log_returns(dfs)
    plot_rolling_volatility(dfs)
    correlation_analysis(dfs)  # Only returns two values, but we don't need to unpack here
    descriptive_stats(dfs)
    box_plots(dfs)
    acf_pacf_plots(dfs)
    joint_distribution_plot(dfs)
    print("\nEDA complete. All plots and tables saved in 'figures/'.")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 