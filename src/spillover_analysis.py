import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}
EVENTS = [
    {'name': 'COVID-19 Crash', 'start': '2020-02-01', 'end': '2020-04-30'},
    {'name': 'AI Boom', 'start': '2023-01-01', 'end': '2023-12-31'},
    {'name': 'Fed Policy', 'start': '2022-03-01', 'end': '2022-06-30'},
    # Add major NVIDIA earnings (example dates)
    {'name': 'NVDA Earnings Q1 2020', 'start': '2020-02-13', 'end': '2020-02-13'},
    {'name': 'NVDA Earnings Q2 2023', 'start': '2023-05-24', 'end': '2023-05-24'},
    {'name': 'NVDA Earnings Q3 2023', 'start': '2023-08-23', 'end': '2023-08-23'},
    {'name': 'NVDA Earnings Q4 2023', 'start': '2023-11-21', 'end': '2023-11-21'}
]

# Helper to load volatility series
def load_volatility(ticker):
    path = os.path.join(FIG_DIR, f"volatility_regime_{ticker}.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df['Volatility']
    except Exception as e:
        print(f"Error loading volatility for {ticker}: {e}")
        return None

def rolling_correlation(vol1, vol2, window=21):
    return vol1.rolling(window).corr(vol2)

def volatility_ratio(vol1, vol2):
    return vol1 / vol2

def plot_spillover(vol1, vol2, roll_corr, ratio, events):
    try:
        fig, ax1 = plt.subplots(figsize=(16,8))
        ax1.plot(vol1.index, vol1, label=f'{NAMES[TICKERS[0]]} Volatility', color='tab:blue')
        ax1.plot(vol2.index, vol2, label=f'{NAMES[TICKERS[1]]} Volatility', color='tab:orange')
        ax1.set_ylabel('Annualized Volatility')
        ax1.set_xlabel('Date')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        # Rolling correlation (secondary axis)
        ax2 = ax1.twinx()
        ax2.plot(roll_corr.index, roll_corr, label='Rolling Corr (21d)', color='tab:green', alpha=0.5)
        ax2.set_ylabel('Rolling Correlation')
        ax2.legend(loc='upper right')
        # Annotate events
        for event in events:
            start = pd.Timestamp(event['start'])
            end = pd.Timestamp(event['end'])
            if start == end:
                ax1.axvline(start, color='red', linestyle='--', alpha=0.7)
                ax1.annotate(event['name'], xy=(start, ax1.get_ylim()[1]), xycoords='data', xytext=(0,10), textcoords='offset points', rotation=90, color='red', fontsize=9, ha='center', va='bottom')
            else:
                ax1.axvspan(start, end, color='gray', alpha=0.2)
                ax1.annotate(event['name'], xy=(start, ax1.get_ylim()[1]), xycoords='data', xytext=(0,10), textcoords='offset points', color='black', fontsize=10, ha='left', va='bottom')
        plt.title('Volatility Spillover and Rolling Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'spillover_annotated.png'), dpi=300)
        plt.close(fig)
        print("Saved annotated spillover plot.")
    except Exception as e:
        print(f"Error plotting spillover: {e}")

def plot_heatmap(data, title, filename):
    try:
        fig, ax = plt.subplots(figsize=(14,6))
        sns.heatmap(data.T, cmap='coolwarm', cbar=True, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
        plt.close(fig)
        print(f"Saved heatmap: {filename}")
    except Exception as e:
        print(f"Error plotting heatmap: {e}")

def spillover_intensity(roll_corr, ratio):
    # Intensity: abs(rolling corr) * abs(log(vol ratio))
    return np.abs(roll_corr) * np.abs(np.log(ratio))

def transmission_speed(roll_corr):
    # Speed: how quickly correlation rises above threshold (e.g., 0.5)
    above = roll_corr > 0.5
    if above.any():
        first = above.idxmax()
        return first
    return None

def directional_spillover(vol1, vol2, window=21):
    # NVDA→NDX: lead NVDA, lag NDX
    lead_nvda = vol1.shift(window)
    spill_nvda_to_ndx = lead_nvda.corr(vol2)
    # NDX→NVDA: lead NDX, lag NVDA
    lead_ndx = vol2.shift(window)
    spill_ndx_to_nvda = lead_ndx.corr(vol1)
    return spill_nvda_to_ndx, spill_ndx_to_nvda

def statistical_significance(series, event_periods):
    # Compare mean in event vs. non-event
    results = []
    for event in event_periods:
        start = pd.Timestamp(event['start'])
        end = pd.Timestamp(event['end'])
        event_vals = series.loc[start:end]
        non_event_vals = series.drop(event_vals.index)
        tstat, pval = ttest_ind(event_vals.dropna(), non_event_vals.dropna(), equal_var=False)
        results.append({'event': event['name'], 't-stat': tstat, 'p-value': pval})
    return pd.DataFrame(results)

def summary_stats(series, threshold=0.5):
    # Count episodes above threshold
    n_episodes = (series > threshold).sum()
    mean_intensity = series[series > threshold].mean()
    return {'n_episodes': n_episodes, 'mean_intensity': mean_intensity}

def main():
    vol1 = load_volatility(TICKERS[0])
    vol2 = load_volatility(TICKERS[1])
    if vol1 is None or vol2 is None:
        print("Error: Could not load volatility data.")
        sys.exit(1)
    # Align
    df = pd.DataFrame({TICKERS[0]: vol1, TICKERS[1]: vol2}).dropna()
    vol1 = df[TICKERS[0]]
    vol2 = df[TICKERS[1]]
    # Rolling correlation and ratio
    roll_corr = rolling_correlation(vol1, vol2, window=21)
    ratio = volatility_ratio(vol1, vol2)
    # Plot annotated spillover
    plot_spillover(vol1, vol2, roll_corr, ratio, EVENTS)
    # Spillover intensity and speed
    intensity = spillover_intensity(roll_corr, ratio)
    speed = transmission_speed(roll_corr)
    print(f"First high spillover (corr>0.5) at: {speed}")
    # Heatmap
    heatmap_data = roll_corr.rolling(63).mean().to_frame().T  # 3-month rolling mean
    plot_heatmap(heatmap_data, 'Spillover Heatmap (3M Rolling Corr)', 'spillover_heatmap.png')
    # Summary stats
    stats = summary_stats(intensity)
    print(f"Spillover episodes above threshold: {stats['n_episodes']}, mean intensity: {stats['mean_intensity']:.4f}")
    # Directional spillover
    nvda_to_ndx, ndx_to_nvda = directional_spillover(vol1, vol2, window=21)
    print(f"Directional spillover NVDA→NDX: {nvda_to_ndx:.4f}, NDX→NVDA: {ndx_to_nvda:.4f}")
    # Statistical significance for events
    sig_df = statistical_significance(intensity, EVENTS)
    try:
        sig_df.to_csv(os.path.join(FIG_DIR, 'spillover_significance.csv'), index=False)
    except Exception as e:
        print(f"Error saving significance results: {e}")
    print("\nSpillover analysis complete. All plots and tables saved in 'figures/'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 