# Data Dictionary & Variable Definitions

## Data Sources

### Raw Data
- **Source**: Yahoo Finance via `yfinance` API
- **Tickers**: NVDA (NVIDIA Corporation), ^NDX (NASDAQ-100 Index)
- **Period**: 2014-01-01 to 2023-12-31
- **Frequency**: Daily

### Data Fields

#### Price Data
| Variable | Description | Unit | Source |
|----------|-------------|------|--------|
| `Open` | Opening price | USD | Yahoo Finance |
| `High` | Daily high price | USD | Yahoo Finance |
| `Low` | Daily low price | USD | Yahoo Finance |
| `Close` | Closing price | USD | Yahoo Finance |
| `Adj Close` | Adjusted closing price | USD | Yahoo Finance |
| `Volume` | Trading volume | Shares | Yahoo Finance |

#### Derived Variables
| Variable | Description | Formula | Unit |
|----------|-------------|---------|------|
| `Log Return` | Logarithmic return | ln(P_t / P_{t-1}) | Decimal |
| `Squared Return` | Squared log return | (Log Return)² | Decimal |
| `Conditional Volatility` | GARCH conditional volatility | σ_t from GARCH model | Decimal |
| `Annualized Volatility` | Annualized conditional volatility | σ_t × √252 | Decimal |

## Statistical Tests & Metrics

### ARCH Effects Testing
| Test | Null Hypothesis | Alternative Hypothesis | Interpretation |
|------|----------------|----------------------|----------------|
| **Ljung-Box (Returns)** | No autocorrelation in returns | Autocorrelation exists | Serial correlation test |
| **Ljung-Box (Squared Returns)** | No ARCH effects | ARCH effects present | Heteroscedasticity test |
| **ARCH-LM** | No ARCH effects | ARCH effects present | Lagrange multiplier test |

### GARCH Model Parameters
| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| **Alpha (α)** | α₁ | ARCH coefficient | [0, 1] |
| **Beta (β)** | β₁ | GARCH coefficient | [0, 1] |
| **Persistence** | α + β | Volatility persistence | [0, 1] |
| **Half-life** | ln(0.5)/ln(α+β) | Time to 50% decay | Days |

### Model Selection Criteria
| Criterion | Description | Formula | Interpretation |
|-----------|-------------|---------|----------------|
| **AIC** | Akaike Information Criterion | -2×LogLik + 2k | Lower is better |
| **BIC** | Bayesian Information Criterion | -2×LogLik + k×ln(n) | Lower is better |
| **Log-Likelihood** | Model fit measure | Σ log(f(r_t)) | Higher is better |

### Granger Causality
| Statistic | Description | Interpretation |
|-----------|-------------|----------------|
| **F-statistic** | Test statistic | Higher = stronger causality |
| **p-value** | Significance level | < 0.05 = significant causality |
| **Optimal Lag** | Best lag length | From AIC/BIC criteria |

### Spillover Metrics
| Metric | Description | Formula | Interpretation |
|--------|-------------|---------|----------------|
| **Rolling Correlation** | 21-day correlation | corr(vol₁, vol₂) | [-1, 1] |
| **Volatility Ratio** | Relative volatility | vol₁ / vol₂ | > 1 = higher vol₁ |
| **Spillover Intensity** | Combined measure | |corr| × |ln(ratio)| | Higher = more spillover |
| **Transmission Speed** | Time to threshold | Days to corr > 0.5 | Faster = quicker transmission |

## File Structure & Formats

### Data Files (CSV Format)
| File | Description | Key Columns |
|------|-------------|-------------|
| `NVDA_cleaned.csv` | Cleaned NVIDIA data | Date, Adj Close, Log Return |
| `^NDX_cleaned.csv` | Cleaned NASDAQ data | Date, Adj Close, Log Return |
| `summary_statistics.csv` | Descriptive statistics | Ticker, Mean, Std, Skew, Kurtosis |

### Analysis Results (CSV Format)
| File | Description | Key Columns |
|------|-------------|-------------|
| `arch_effects_summary.csv` | ARCH test results | Ticker, LB_stat, LB_p, ARCH_stat, ARCH_p |
| `garch_model_comparison.csv` | Model comparison | Ticker, Order, AIC, BIC, LogLik |
| `granger_causality_summary.csv` | Causality tests | Direction, Lag, F-stat, p-value |
| `spillover_significance.csv` | Event significance | Event, t-stat, p-value |

### Model Objects (Pickle Format)
| File | Description | Contents |
|------|-------------|----------|
| `garch_NVDA_(1,1).pkl` | NVIDIA GARCH model | Fitted model object |
| `garch_^NDX_(1,1).pkl` | NASDAQ GARCH model | Fitted model object |

### Visualization Files (PNG Format)
| File | Description | Content |
|------|-------------|---------|
| `price_levels.png` | Price time series | Adjusted close prices |
| `log_returns.png` | Return series | Log returns over time |
| `rolling_volatility.png` | Volatility plots | 21-day rolling volatility |
| `spillover_annotated.png` | Spillover analysis | Annotated spillover events |
| `volatility_*.png` | Conditional volatility | GARCH volatility extraction |

## Statistical Significance Levels

### Hypothesis Testing
| Level | α | Interpretation |
|-------|---|----------------|
| **1%** | 0.01 | Highly significant |
| **5%** | 0.05 | Significant |
| **10%** | 0.10 | Marginally significant |

### Model Diagnostics
| Test | Good Model (p-value) | Poor Model (p-value) |
|------|---------------------|---------------------|
| **Ljung-Box (residuals)** | > 0.05 | < 0.05 |
| **ARCH-LM (residuals)** | > 0.05 | < 0.05 |
| **Jarque-Bera (normality)** | > 0.05 | < 0.05 |

## Economic Events Timeline

### Major Market Events
| Event | Period | Description |
|-------|--------|-------------|
| **COVID-19 Crash** | 2020-02-01 to 2020-04-30 | Market crash and recovery |
| **AI Boom** | 2023-01-01 to 2023-12-31 | AI-driven market surge |
| **Fed Policy Changes** | 2022-03-01 to 2022-06-30 | Interest rate adjustments |
| **NVIDIA Earnings** | Various dates | Quarterly earnings releases |

### Volatility Regimes
| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| **High** | Volatility > μ + σ | Above-normal volatility |
| **Normal** | μ - σ ≤ Volatility ≤ μ + σ | Normal volatility range |
| **Low** | Volatility < μ - σ | Below-normal volatility |

## Quality Checks

### Data Validation
- Missing values: Forward-filled for weekends/holidays
- Outliers: Identified as > 5 standard deviations
- Continuity: Business day frequency maintained
- Alignment: All series synchronized by date

### Model Validation
- Convergence: All models checked for optimization convergence
- Residuals: Diagnostic tests for model adequacy
- Out-of-sample: 2022-2023 held out for validation
- Robustness: Multiple specifications tested

This data dictionary provides comprehensive documentation for all variables, metrics, and files used in the analysis. 