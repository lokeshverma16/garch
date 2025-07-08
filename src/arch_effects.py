import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import warnings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}

SIGNIFICANCE_LEVELS = [0.01, 0.05, 0.10]

# Helper to load cleaned data
def load_data(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_cleaned.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

def ljung_box_test(series, lags=20, squared=False):
    if squared:
        series = series ** 2
    lb = acorr_ljungbox(series, lags=[lags], return_df=True)
    stat = lb['lb_stat'].iloc[0]
    pval = lb['lb_pvalue'].iloc[0]
    return stat, pval

def arch_lm_test(series, lags):
    results = {}
    for lag in lags:
        test = het_arch(series, nlags=lag)
        stat, pval = test[0], test[1]
        results[lag] = (stat, pval)
    return results

def plot_squared_returns(series, ticker):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        sq = series ** 2
        ax.plot(sq, label='Squared Returns', color='tab:blue')
        mean = sq.mean()
        std = sq.std()
        upper = mean + 1.96 * std
        lower = mean - 1.96 * std
        ax.axhline(mean, color='black', linestyle='--', label='Mean')
        ax.fill_between(sq.index, lower, upper, color='gray', alpha=0.2, label='95% Confidence Band')
        ax.set_title(f"{NAMES[ticker]} Squared Returns with 95% Confidence Bands")
        ax.set_xlabel('Date')
        ax.set_ylabel('Squared Log Return')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"squared_returns_{ticker}.png"), dpi=300)
        plt.close(fig)
        print(f"Saved squared returns plot for {ticker}")
    except Exception as e:
        print(f"Error plotting squared returns for {ticker}: {e}")

def interpret_test(name, stat, pval, sig_levels):
    interp = f"{name}: Test statistic = {stat:.4f}, p-value = {pval:.4g}. "
    for alpha in sig_levels:
        if pval < alpha:
            interp += f"Reject H0 at {int(alpha*100)}% (evidence of effect). "
        else:
            interp += f"Fail to reject H0 at {int(alpha*100)}%. "
    return interp

def main():
    results = []
    for ticker in TICKERS:
        print(f"\nARCH Effects Testing for {NAMES[ticker]} ({ticker})")
        df = load_data(ticker)
        if df is None:
            continue
        returns = df['Log Return'].dropna()
        # 1. Ljung-Box on returns
        lb_stat_r, lb_pval_r = ljung_box_test(returns, lags=20, squared=False)
        # 2. Ljung-Box on squared returns
        lb_stat_sq, lb_pval_sq = ljung_box_test(returns, lags=20, squared=True)
        # 3. ARCH-LM test
        arch_lags = [1, 5, 10]
        arch_results = arch_lm_test(returns, arch_lags)
        # 4. Plot squared returns
        plot_squared_returns(returns, ticker)
        # 5. Results summary
        row = {
            'Ticker': ticker,
            'Ljung-Box Returns Stat': lb_stat_r,
            'Ljung-Box Returns p': lb_pval_r,
            'Ljung-Box SqReturns Stat': lb_stat_sq,
            'Ljung-Box SqReturns p': lb_pval_sq,
        }
        for lag in arch_lags:
            row[f'ARCH-LM Stat (lag {lag})'] = arch_results[lag][0]
            row[f'ARCH-LM p (lag {lag})'] = arch_results[lag][1]
        results.append(row)
        # 6. Interpretation
        print("\n--- Hypothesis Statements ---")
        print("Ljung-Box on returns:")
        print("  H0: No autocorrelation in returns up to lag 20.")
        print("  H1: Autocorrelation exists in returns up to lag 20.")
        print("Ljung-Box on squared returns:")
        print("  H0: No autocorrelation in squared returns (no ARCH effects) up to lag 20.")
        print("  H1: Autocorrelation exists in squared returns (ARCH effects) up to lag 20.")
        print("ARCH-LM test:")
        print("  H0: No ARCH effects (no conditional heteroscedasticity).")
        print("  H1: Presence of ARCH effects (conditional heteroscedasticity).\n")
        print("--- Test Results and Interpretation ---")
        print(interpret_test("Ljung-Box on returns", lb_stat_r, lb_pval_r, SIGNIFICANCE_LEVELS))
        print(interpret_test("Ljung-Box on squared returns", lb_stat_sq, lb_pval_sq, SIGNIFICANCE_LEVELS))
        for lag in arch_lags:
            stat, pval = arch_results[lag]
            print(interpret_test(f"ARCH-LM (lag {lag})", stat, pval, SIGNIFICANCE_LEVELS))
        print("\nIf squared returns and/or ARCH-LM tests are significant, GARCH modeling is justified.")
    # Save results summary
    try:
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(FIG_DIR, 'arch_effects_summary.csv'), index=False)
        print("\nARCH effects test summary saved to 'figures/arch_effects_summary.csv'.")
    except Exception as e:
        print(f"Error saving results summary: {e}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 