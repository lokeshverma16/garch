import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import warnings

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}

# Helper to load volatility series
def load_volatility(ticker):
    path = os.path.join(FIG_DIR, f"volatility_regime_{ticker}.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if 'Volatility' in df.columns:
            return df['Volatility']
        else:
            raise ValueError(f"'Volatility' column not found in {path}. Columns: {df.columns}")
    except Exception as e:
        print(f"Error loading volatility for {ticker}: {e}")
        return None

def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"ADF test for {name}: p-value={result[1]:.4g}")
    return result[1] < 0.05

def select_optimal_lag(df, max_lag=10):
    # Use VAR model to select lag by AIC/BIC
    model = VAR(df)
    lag_order_result = model.select_order(maxlags=max_lag)
    best_aic = lag_order_result.selected_orders['aic']
    best_bic = lag_order_result.selected_orders['bic']
    print(f"Optimal lag by AIC: {best_aic}, by BIC: {best_bic}")
    return best_aic, best_bic

def granger_test(df, cause, effect, max_lag=10):
    results = {}
    for lag in range(1, max_lag+1):
        try:
            test = grangercausalitytests(df[[effect, cause]].dropna(), maxlag=lag, verbose=False)
            fstat = test[lag][0]['ssr_ftest'][0]
            pval = test[lag][0]['ssr_ftest'][1]
            results[lag] = {'F-stat': fstat, 'p-value': pval}
        except Exception as e:
            results[lag] = {'F-stat': np.nan, 'p-value': np.nan}
    return results

def create_results_matrix(results, cause, effect):
    df = pd.DataFrame(results).T
    df['Significant (5%)'] = df['p-value'] < 0.05
    df['Direction'] = f'{cause} → {effect}'
    return df

def var_analysis(df, lags):
    try:
        model = VAR(df)
        res = model.fit(lags)
        return res
    except Exception as e:
        print(f"Error fitting VAR: {e}")
        return None

def plot_irf(var_res, steps, ticker1, ticker2):
    try:
        irf = var_res.irf(steps)
        fig = irf.plot(orth=False)
        plt.suptitle(f"Impulse Response Functions: {NAMES[ticker1]} & {NAMES[ticker2]}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"irf_{ticker1}_{ticker2}.png"), dpi=300)
        plt.close()
        print("Saved IRF plot.")
    except Exception as e:
        print(f"Error plotting IRF: {e}")

def main():
    warnings.filterwarnings('ignore')
    # Load volatility series
    vols = {}
    for ticker in TICKERS:
        vols[ticker] = load_volatility(ticker)
    # Align and dropna
    df = pd.DataFrame({t: vols[t] for t in TICKERS}).dropna()
    # Check stationarity
    for t in TICKERS:
        if not check_stationarity(df[t], NAMES[t]):
            print(f"Warning: {NAMES[t]} volatility is non-stationary. Consider differencing or transformation.")
    # Optimal lag selection
    best_aic, best_bic = select_optimal_lag(df, max_lag=10)
    # Granger causality tests (bidirectional)
    results_nvda_to_ndx = granger_test(df, 'NVDA', '^NDX', max_lag=10)
    results_ndx_to_nvda = granger_test(df, '^NDX', 'NVDA', max_lag=10)
    mat_nvda_to_ndx = create_results_matrix(results_nvda_to_ndx, 'NVDA', '^NDX')
    mat_ndx_to_nvda = create_results_matrix(results_ndx_to_nvda, '^NDX', 'NVDA')
    # Save results
    try:
        mat_nvda_to_ndx.to_csv(os.path.join(FIG_DIR, 'granger_nvda_to_ndx.csv'))
        mat_ndx_to_nvda.to_csv(os.path.join(FIG_DIR, 'granger_ndx_to_nvda.csv'))
    except Exception as e:
        print(f"Error saving Granger results: {e}")
    # VAR model and IRF
    var_res = var_analysis(df, best_aic)
    if var_res is not None:
        plot_irf(var_res, 10, 'NVDA', '^NDX')
        # Save summary
        try:
            with open(os.path.join(FIG_DIR, 'var_summary.txt'), 'w') as f:
                f.write(str(var_res.summary()))
        except Exception as e:
            print(f"Error saving VAR summary: {e}")
    # Results table
    try:
        summary = pd.concat([mat_nvda_to_ndx, mat_ndx_to_nvda], axis=0)
        summary.to_csv(os.path.join(FIG_DIR, 'granger_causality_summary.csv'))
        print("Granger causality summary saved.")
    except Exception as e:
        print(f"Error saving summary: {e}")
    print("\nGranger causality and VAR analysis complete. All results saved in 'figures/'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 