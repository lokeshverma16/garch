import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

TICKERS = ['NVDA', '^NDX']
NAMES = {'NVDA': 'NVIDIA', '^NDX': 'NASDAQ-100'}
GARCH_ORDERS = [(1,1), (1,2), (2,1)]

# Helper to load cleaned data
def load_data(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_cleaned.csv")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

def fit_garch_model(returns, p, q, ticker, dist='normal'):
    try:
        model = arch_model(returns, mean='constant', vol='GARCH', p=p, q=q, dist=dist, rescale=False)
        res = model.fit(disp='off')
        return res, None
    except Exception as e:
        err = f"Model fitting failed: {e}"
        return None, err
    return None, 'Model did not converge.'

def residual_diagnostics(res, ticker, order):
    try:
        std_resid = res.std_resid
        # Q-Q plot
        fig, ax = plt.subplots(figsize=(6,6))
        stats.probplot(std_resid, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: Std Residuals {NAMES[ticker]} GARCH{order}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"qqplot_{ticker}_garch{order}.png"), dpi=300)
        plt.close(fig)
        # Ljung-Box test
        lb = acorr_ljungbox(std_resid, lags=[20], return_df=True)
        lb_stat = lb['lb_stat'].iloc[0]
        lb_pval = lb['lb_pvalue'].iloc[0]
        return std_resid, lb_stat, lb_pval
    except Exception as e:
        print(f"Error in residual diagnostics for {ticker} GARCH{order}: {e}")
        return None, None, None

def extract_volatility(res):
    cond_vol = res.conditional_volatility
    lower = cond_vol - 1.96 * cond_vol.std()
    upper = cond_vol + 1.96 * cond_vol.std()
    return cond_vol, lower, upper

def save_model(res, ticker, order):
    try:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        with open(os.path.join(MODEL_DIR, f"garch_{ticker}_{order}.pkl"), 'wb') as f:
            pickle.dump(res, f)
        print(f"Saved model: garch_{ticker}_{order}.pkl")
    except Exception as e:
        print(f"Error saving model for {ticker} GARCH{order}: {e}")

def main():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model_comparison = []
    for ticker in TICKERS:
        print(f"\nGARCH Modeling for {NAMES[ticker]} ({ticker})")
        df = load_data(ticker)
        if df is None:
            continue
        returns = df['Log Return'].dropna() * 100  # arch expects percent returns
        for order in GARCH_ORDERS:
            p, q = order
            print(f"\nFitting GARCH({p},{q})...")
            res, err = fit_garch_model(returns, p, q, ticker)
            if res is None:
                print(f"Model fitting failed for {ticker} GARCH{order}: {err}")
                continue
            # Extract results
            params = res.params
            std_err = res.std_err
            tvals = res.tvalues
            pvals = res.pvalues
            llf = res.loglikelihood
            aic = res.aic
            bic = res.bic
            # Residual diagnostics
            std_resid, lb_stat, lb_pval = residual_diagnostics(res, ticker, order)
            # Volatility extraction
            cond_vol, lower, upper = extract_volatility(res)
            # Plot conditional volatility
            try:
                fig, ax = plt.subplots(figsize=(12,6))
                cond_vol.plot(ax=ax, label='Conditional Volatility')
                ax.fill_between(cond_vol.index, lower, upper, color='gray', alpha=0.2, label='95% CI')
                ax.set_title(f"Conditional Volatility {NAMES[ticker]} GARCH{order}")
                ax.set_xlabel('Date')
                ax.set_ylabel('Volatility')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_DIR, f"cond_vol_{ticker}_garch{order}.png"), dpi=300)
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting conditional volatility for {ticker} GARCH{order}: {e}")
            # Save model
            save_model(res, ticker, order)
            # Model comparison table
            model_comparison.append({
                'Ticker': ticker,
                'Order': f'GARCH{order}',
                'AIC': aic,
                'BIC': bic,
                'LogLik': llf,
                'LBQ Stat': lb_stat,
                'LBQ p': lb_pval,
                'Params': params.to_dict(),
                'StdErr': std_err.to_dict(),
                't-Stat': tvals.to_dict(),
                'p-Value': pvals.to_dict()
            })
    # Save model comparison table
    try:
        df_comp = pd.DataFrame(model_comparison)
        df_comp.to_csv(os.path.join(FIG_DIR, 'garch_model_comparison.csv'), index=False)
        print("\nModel comparison table saved to 'figures/garch_model_comparison.csv'.")
    except Exception as e:
        print(f"Error saving model comparison table: {e}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 