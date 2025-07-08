import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}
EVENTS = [
    {'name': 'COVID-19 Crash', 'start': '2020-03-01', 'end': '2020-03-31'},
    {'name': 'AI Boom', 'start': '2023-01-01', 'end': '2023-12-31'},
    {'name': 'Fed Policy Change', 'start': '2022-03-01', 'end': '2022-06-30'}
]

# Helper to load GARCH model

def load_model(ticker, order=(1,1)):
    path = os.path.join(MODEL_DIR, f"garch_{ticker}_{order}.pkl")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model for {ticker} GARCH{order}: {e}")
        return None

def load_returns(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_cleaned.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df['Log Return']
    except Exception as e:
        print(f"Error loading returns for {ticker}: {e}")
        return None

def annualize_volatility(vol):
    return vol * np.sqrt(252)

def plot_volatility(vols, lowers, uppers, returns, ticker):
    try:
        fig, ax = plt.subplots(figsize=(14,6))
        vols.plot(ax=ax, label='Annualized Volatility', color='tab:blue')
        ax.fill_between(vols.index, lowers, uppers, color='gray', alpha=0.2, label='95% CI')
        mean = vols.mean()
        std = vols.std()
        high_vol = vols[vols > mean + 2*std]
        ax.scatter(high_vol.index, high_vol, color='red', label='High Volatility (>2σ)', zorder=5)
        # Highlight events
        for event in EVENTS:
            ax.axvspan(pd.Timestamp(event['start']), pd.Timestamp(event['end']), alpha=0.2, label=event['name'])
        ax.set_title(f"Annualized Conditional Volatility: {NAMES[ticker]} (GARCH(1,1))")
        ax.set_ylabel('Annualized Volatility')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"volatility_{ticker}.png"), dpi=300)
        plt.close(fig)
        print(f"Saved volatility plot for {ticker}")
    except Exception as e:
        print(f"Error plotting volatility for {ticker}: {e}")

def volatility_persistence(model):
    # Persistence = alpha + beta
    try:
        alpha = model.params.get('alpha[1]', np.nan)
        beta = model.params.get('beta[1]', np.nan)
        persistence = alpha + beta
        # Half-life = ln(0.5)/ln(alpha+beta)
        if persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf
        return persistence, half_life
    except Exception as e:
        print(f"Error calculating persistence: {e}")
        return np.nan, np.nan

def volatility_clustering_plot(vols, returns, ticker):
    try:
        # Align indices to ensure same length
        aligned_returns, aligned_vols = returns.align(vols, join='inner')
        fig, ax1 = plt.subplots(figsize=(14,6))
        ax1.plot(aligned_vols.index, aligned_returns, color='tab:gray', alpha=0.5, label='Log Returns')
        ax2 = ax1.twinx()
        ax2.plot(aligned_vols.index, aligned_vols, color='tab:blue', label='Annualized Volatility')
        ax1.set_ylabel('Log Returns')
        ax2.set_ylabel('Annualized Volatility')
        ax1.set_title(f"Volatility Clustering: {NAMES[ticker]}")
        ax1.grid(True)
        fig.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"vol_clustering_{ticker}.png"), dpi=300)
        plt.close(fig)
        print(f"Saved volatility clustering plot for {ticker}")
    except Exception as e:
        print(f"Error plotting volatility clustering for {ticker}: {e}")

def volatility_summary(vols, ticker):
    stats = vols.describe()
    stats['skew'] = vols.skew()
    stats['kurtosis'] = vols.kurtosis()
    return stats

def regime_analysis(vols, ticker):
    # Simple regime: high (>mean+std), normal (mean±std), low (<mean-std)
    mean = vols.mean()
    std = vols.std()
    regime = pd.Series('Normal', index=vols.index)
    regime[vols > mean + std] = 'High'
    regime[vols < mean - std] = 'Low'
    return regime

def main():
    all_stats = {}
    for ticker in TICKERS:
        print(f"\nVolatility Analysis for {NAMES[ticker]} ({ticker})")
        model = load_model(ticker, order=(1,1))
        returns = load_returns(ticker)
        if model is None or returns is None:
            continue
        vols = pd.Series(model.conditional_volatility, index=returns.index[-len(model.conditional_volatility):])
        vols = annualize_volatility(vols)
        lower = vols - 1.96 * vols.std()
        upper = vols + 1.96 * vols.std()
        plot_volatility(vols, lower, upper, returns, ticker)
        persistence, half_life = volatility_persistence(model)
        print(f"Persistence: {persistence:.4f}, Half-life: {half_life:.2f} days")
        volatility_clustering_plot(vols, returns, ticker)
        stats = volatility_summary(vols, ticker)
        regime = regime_analysis(vols, ticker)
        # Save regime and stats
        try:
            df = pd.DataFrame({'Volatility': vols, 'Regime': regime})
            df.to_csv(os.path.join(FIG_DIR, f"volatility_regime_{ticker}.csv"))
            stats.to_csv(os.path.join(FIG_DIR, f"volatility_stats_{ticker}.csv"))
        except Exception as e:
            print(f"Error saving stats/regime for {ticker}: {e}")
        all_stats[ticker] = stats
    print("\nVolatility analysis complete. All plots and tables saved in 'figures/'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 