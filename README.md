# Financial Time Series Analysis: GARCH Modeling of NVDA & NASDAQ-100

## Project Overview

This project provides a comprehensive analysis of volatility dynamics, spillover effects, and causal relationships between NVIDIA (NVDA) and the NASDAQ-100 (^NDX) from 2014 to 2023. Using advanced econometric models including GARCH, VAR, and Granger causality tests, we quantify risk transmission and provide actionable insights for portfolio management and risk assessment.

## Objectives

- **Volatility Modeling**: Fit and validate GARCH models to capture time-varying volatility
- **Spillover Analysis**: Identify and quantify volatility spillovers between NVDA and NASDAQ-100
- **Causality Testing**: Test for Granger causality and lead-lag relationships
- **Risk Assessment**: Provide comprehensive risk metrics and persistence measures
- **Model Validation**: Conduct out-of-sample testing and robustness checks

## Data Sources & Methodology

### Data
- **Source**: Yahoo Finance via `yfinance` API
- **Assets**: NVIDIA (NVDA) and NASDAQ-100 Index (^NDX)
- **Period**: January 1, 2014 to December 31, 2023
- **Frequency**: Daily adjusted close prices, converted to log returns

### Methodology
- **Volatility Modeling**: GARCH(1,1), GARCH(1,2), GARCH(2,1), GJR-GARCH, EGARCH
- **Distributions**: Normal, Student-t, Skewed-t
- **Causality Testing**: Granger causality tests with optimal lag selection
- **Spillover Analysis**: Rolling correlations, volatility ratios, VAR impulse responses
- **Validation**: Out-of-sample forecasting, residual diagnostics, robustness checks

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd garch
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python src/main.py
   ```

## Usage Guide

### Quick Start

1. **Run complete analysis**:
   ```bash
   python run_analysis.py
   ```

2. **Individual components**:
   ```bash
   # Data collection
   python src/data_collection.py
   
   # Exploratory analysis
   python src/eda.py
   
   # ARCH effects testing
   python src/arch_effects.py
   
   # GARCH modeling
   python src/garch_modeling.py
   
   # Volatility analysis
   python src/volatility_analysis.py
   
   # Granger causality
   python src/granger_causality.py
   
   # Spillover analysis
   python src/spillover_analysis.py
   
   # Model diagnostics
   python src/model_diagnostics.py
   ```

### Interactive Analysis

Use the Jupyter notebook for interactive exploration:
```bash
cd notebooks/
jupyter notebook garch_interactive_analysis.ipynb
```

The notebook provides:
- Interactive data visualization
- Step-by-step analysis walkthrough
- Access to all GARCH results
- Custom analysis playground

### Expected Outputs

After running the analysis, you'll find:
- **Data**: Cleaned datasets in `data/` folder
- **Figures**: All plots and visualizations in `figures/` folder
- **Results**: Model outputs and summary tables in `results/` folder
- **Report**: Comprehensive markdown report in `results/summary_report.md`

## File Structure

```
garch/
├── README.md                          # This file
├── requirements.txt                   # Package dependencies
├── run_analysis.py                    # Master script to run all analyses
├── data/                              # Data storage
│   ├── NVDA_cleaned.csv              # Cleaned NVIDIA data
│   ├── ^NDX_cleaned.csv              # Cleaned NASDAQ-100 data
│   └── summary_statistics.csv        # Descriptive statistics
├── src/                               # Source code
│   ├── main.py                       # Environment setup and validation
│   ├── data_collection.py            # Data download and cleaning
│   ├── eda.py                        # Exploratory data analysis
│   ├── arch_effects.py               # ARCH effects testing
│   ├── garch_modeling.py             # GARCH model estimation
│   ├── volatility_analysis.py        # Volatility extraction and analysis
│   ├── granger_causality.py          # Granger causality testing
│   ├── spillover_analysis.py         # Spillover effects analysis
│   └── model_diagnostics.py          # Model validation and diagnostics
├── notebooks/                         # Jupyter notebooks
│   └── garch_interactive_analysis.ipynb  # Interactive analysis notebook
├── figures/                           # Generated plots and tables
│   ├── price_levels.png              # Price level plots
│   ├── log_returns.png               # Returns time series
│   ├── rolling_volatility.png        # Volatility plots
│   ├── spillover_annotated.png       # Spillover analysis
│   └── ...                          # Other generated figures
└── results/                           # Analysis results
    ├── summary_report.md              # Comprehensive analysis report
    ├── garch_*.pkl                   # Saved GARCH models
    └── ...                           # Other result files
```

## Results Interpretation

### Key Metrics

- **Volatility Persistence**: α + β coefficient (closer to 1 = higher persistence)
- **Half-life**: Time for volatility shock to decay by 50%
- **Spillover Intensity**: Correlation × log(volatility ratio)
- **Granger Causality**: F-statistic and p-value for causal relationships

### Statistical Significance

- **1% level**: Highly significant (p < 0.01)
- **5% level**: Significant (p < 0.05)
- **10% level**: Marginally significant (p < 0.10)

### Model Selection

- **AIC/BIC**: Lower values indicate better model fit
- **Ljung-Box**: p > 0.05 indicates no autocorrelation in residuals
- **ARCH-LM**: p > 0.05 indicates no remaining ARCH effects

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'yfinance'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

2. **Data Download Failures**:
   ```
   Error downloading data for NVDA
   ```
   **Solution**: Check internet connection and Yahoo Finance availability

3. **GARCH Convergence Issues**:
   ```
   Model fitting failed for NVDA GARCH(1,1)
   ```
   **Solution**: Try different optimizers or starting values (handled automatically)

4. **Memory Issues**:
   ```
   MemoryError during model fitting
   ```
   **Solution**: Reduce data sample size or use more powerful hardware

5. **Missing Files**:
   ```
   Error loading model for NVDA GARCH(1,1)
   ```
   **Solution**: Run data collection and GARCH modeling first

### Performance Tips

- Use virtual environment to avoid package conflicts
- Run analysis in sequence for first-time setup
- Check data quality before proceeding to modeling
- Monitor console output for warnings and errors

### Getting Help

1. Check console output for specific error messages
2. Verify all dependencies are correctly installed
3. Ensure data files exist before running analysis
4. Review the methodology section for parameter explanations

## Dependencies

See `requirements.txt` for complete list. Key packages:
- `yfinance`: Financial data download
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `arch`: GARCH modeling
- `statsmodels`: Statistical tests
- `matplotlib/seaborn`: Visualization
- `scipy`: Statistical functions

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or support, please create an issue in the repository. 