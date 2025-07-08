import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, normaltest, ttest_ind, probplot
from datetime import datetime
import warnings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}

TRAIN_END = '2021-12-31'
TEST_START = '2022-01-01'
TEST_END = '2023-12-31'

# Helper to load cleaned data
def load_data(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_cleaned.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

def load_model(ticker, order=(1,1), dist='normal', spec='GARCH'):
    fname = f"garch_{ticker}_{order}.pkl"
    path = os.path.join(MODEL_DIR, fname)
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model for {ticker} {spec}{order}: {e}")
        return None

def residual_diagnostics(res, ticker, order, dist, spec):
    try:
        std_resid = res.std_resid
        # Normality tests
        jb_stat, jb_p = jarque_bera(std_resid)
        norm_stat, norm_p = normaltest(std_resid)
        # Ljung-Box
        lb = acorr_ljungbox(std_resid, lags=[20], return_df=True)
        lb_stat = lb['lb_stat'].iloc[0]
        lb_pval = lb['lb_pvalue'].iloc[0]
        # ARCH-LM
        arch_stat, arch_p = het_arch(std_resid, nlags=10)[:2]
        # Q-Q plot
        fig, ax = plt.subplots(figsize=(6,6))
        probplot(std_resid, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: Std Residuals {NAMES[ticker]} {spec}{order} ({dist})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"qqplot_{ticker}_{spec}{order}_{dist}.png"), dpi=300)
        plt.close(fig)
        return {
            'jb_stat': jb_stat, 'jb_p': jb_p,
            'norm_stat': norm_stat, 'norm_p': norm_p,
            'lb_stat': lb_stat, 'lb_pval': lb_pval,
            'arch_stat': arch_stat, 'arch_p': arch_p
        }
    except Exception as e:
        print(f"Error in residual diagnostics for {ticker} {spec}{order}: {e}")
        return None

def naive_forecast(test_len, train_vol):
    # Historical average
    hist_avg = np.full(test_len, np.mean(train_vol))
    # EWMA (lambda=0.94)
    ewma = [train_vol[-1]]
    alpha = 0.06
    for _ in range(test_len-1):
        ewma.append(alpha * train_vol[-1] + (1-alpha) * ewma[-1])
    return hist_avg, np.array(ewma)

def forecast_accuracy(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    return {'MSE': mse, 'MAE': mae}

def fit_and_forecast(train, test, order, dist, spec):
    try:
        model = arch_model(train, mean='constant', vol=spec, p=order[0], q=order[1], dist=dist, rescale=False)
        res = model.fit(disp='off')
        forecasts = res.forecast(horizon=len(test), start=None, reindex=False)
        fcast_vol = np.sqrt(forecasts.variance.values[-1, :])
        return res, fcast_vol
    except Exception as e:
        print(f"Error fitting/forecasting {spec}{order} ({dist}): {e}")
        return None, None

def robustness_checks(train, test, ticker):
    specs = [('GARCH', (1,1)), ('GARCH', (1,2)), ('GARCH', (2,1)), ('GJR-GARCH', (1,1)), ('EGARCH', (1,1))]
    dists = ['normal', 't', 'skewt']
    results = []
    for spec, order in specs:
        for dist in dists:
            res, fcast_vol = fit_and_forecast(train, test, order, dist, spec)
            if res is None or fcast_vol is None:
                continue
            acc = forecast_accuracy(test, fcast_vol)
            diag = residual_diagnostics(res, ticker, order, dist, spec)
            results.append({
                'spec': spec, 'order': order, 'dist': dist,
                'MSE': acc['MSE'], 'MAE': acc['MAE'],
                'jb_p': diag['jb_p'] if diag else np.nan,
                'lb_pval': diag['lb_pval'] if diag else np.nan,
                'arch_p': diag['arch_p'] if diag else np.nan
            })
    return pd.DataFrame(results)

def subsample_analysis(returns, ticker, order, dist, spec):
    # Split into two halves
    n = len(returns)
    mid = n // 2
    halves = [returns[:mid], returns[mid:]]
    stats = []
    for i, sub in enumerate(halves):
        res, _ = fit_and_forecast(sub, sub, order, dist, spec)
        if res is None:
            continue
        diag = residual_diagnostics(res, ticker, order, dist, spec)
        stats.append({'half': i+1, 'jb_p': diag['jb_p'] if diag else np.nan, 'lb_pval': diag['lb_pval'] if diag else np.nan, 'arch_p': diag['arch_p'] if diag else np.nan})
    return pd.DataFrame(stats)

def main():
    for ticker in TICKERS:
        print(f"\nDiagnostics for {NAMES[ticker]} ({ticker})")
        df = load_data(ticker)
        if df is None:
            continue
        returns = df['Log Return'].dropna() * 100
        # Split train/test
        train = returns.loc[:TRAIN_END]
        test = returns.loc[TEST_START:TEST_END]
        # Fit GARCH(1,1) normal
        res, fcast_vol = fit_and_forecast(train, test, (1,1), 'normal', 'GARCH')
        if res is None or fcast_vol is None:
            continue
        # Residual diagnostics
        diag = residual_diagnostics(res, ticker, (1,1), 'normal', 'GARCH')
        print(f"Residual diagnostics: {diag}")
        # Out-of-sample accuracy
        acc = forecast_accuracy(test, fcast_vol)
        print(f"Forecast accuracy: {acc}")
        # Naive models
        hist_avg, ewma = naive_forecast(len(test), train.values)
        acc_hist = forecast_accuracy(test, hist_avg)
        acc_ewma = forecast_accuracy(test, ewma)
        print(f"Naive (Hist Avg) accuracy: {acc_hist}")
        print(f"Naive (EWMA) accuracy: {acc_ewma}")
        # Robustness checks
        robust_df = robustness_checks(train, test, ticker)
        robust_df.to_csv(os.path.join(FIG_DIR, f"robustness_{ticker}.csv"))
        # Subsample analysis
        subsample_df = subsample_analysis(returns, ticker, (1,1), 'normal', 'GARCH')
        subsample_df.to_csv(os.path.join(FIG_DIR, f"subsample_{ticker}.csv"))
        print("Diagnostics, robustness, and subsample results saved.")
    print("\nModel diagnostics and validation complete. See figures/ for plots and tables.")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 