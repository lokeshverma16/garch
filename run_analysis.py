#!/usr/bin/env python3
"""
Master Analysis Script for GARCH Financial Time Series Analysis
Runs complete analysis pipeline from data collection to final report generation.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header for each analysis step."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {description}")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            print("Error:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ Failed to run {description}: {e}")
        return False

def check_files_exist(file_paths, description):
    """Check if required files exist before proceeding."""
    missing = [f for f in file_paths if not os.path.exists(f)]
    if missing:
        print(f"⚠ Warning: Missing files for {description}: {missing}")
        return False
    print(f"✓ All required files found for {description}")
    return True

def main():
    """Main analysis pipeline."""
    start_time = time.time()
    
    print_header("GARCH FINANCIAL TIME SERIES ANALYSIS")
    print("Starting comprehensive analysis pipeline...")
    print(f"Working directory: {os.getcwd()}")
    
    # Analysis steps with dependencies
    analysis_steps = [
        {
            'script': 'src/main.py',
            'description': 'Environment Setup & Validation',
            'required_files': [],
            'creates': []
        },
        {
            'script': 'src/data_collection.py',
            'description': 'Data Collection & Cleaning',
            'required_files': [],
            'creates': ['data/NVDA_cleaned.csv', 'data/^NDX_cleaned.csv']
        },
        {
            'script': 'src/eda.py',
            'description': 'Exploratory Data Analysis',
            'required_files': ['data/NVDA_cleaned.csv', 'data/^NDX_cleaned.csv'],
            'creates': ['figures/price_levels.png', 'figures/log_returns.png']
        },
        {
            'script': 'src/arch_effects.py',
            'description': 'ARCH Effects Testing',
            'required_files': ['data/NVDA_cleaned.csv', 'data/^NDX_cleaned.csv'],
            'creates': ['figures/arch_effects_summary.csv']
        },
        {
            'script': 'src/garch_modeling.py',
            'description': 'GARCH Model Estimation',
            'required_files': ['data/NVDA_cleaned.csv', 'data/^NDX_cleaned.csv'],
            'creates': ['results/garch_NVDA_(1, 1).pkl', 'figures/garch_model_comparison.csv']
        },
        {
            'script': 'src/volatility_analysis.py',
            'description': 'Volatility Analysis & Extraction',
            'required_files': ['results/garch_NVDA_(1, 1).pkl'],
            'creates': ['figures/volatility_NVDA.png', 'figures/volatility_regime_NVDA.csv']
        },
        {
            'script': 'src/granger_causality.py',
            'description': 'Granger Causality Testing',
            'required_files': ['figures/volatility_regime_NVDA.csv', 'figures/volatility_regime_^NDX.csv'],
            'creates': ['figures/granger_causality_summary.csv']
        },
        {
            'script': 'src/spillover_analysis.py',
            'description': 'Spillover Effects Analysis',
            'required_files': ['figures/volatility_regime_NVDA.csv', 'figures/volatility_regime_^NDX.csv'],
            'creates': ['figures/spillover_annotated.png']
        },
        {
            'script': 'src/model_diagnostics.py',
            'description': 'Model Diagnostics & Validation',
            'required_files': ['data/NVDA_cleaned.csv', 'data/^NDX_cleaned.csv'],
            'creates': ['figures/robustness_NVDA.csv']
        }
    ]
    
    # Execute analysis pipeline
    successful_steps = 0
    total_steps = len(analysis_steps)
    
    for i, step in enumerate(analysis_steps, 1):
        print_header(f"STEP {i}/{total_steps}: {step['description']}")
        
        # Check prerequisites
        if step['required_files']:
            if not check_files_exist(step['required_files'], step['description']):
                print(f"⚠ Proceeding anyway, files may be created during execution...")
        
        # Run the analysis step
        success = run_script(step['script'], step['description'])
        
        if success:
            successful_steps += 1
            # Verify expected outputs
            if step['creates']:
                time.sleep(1)  # Allow file system to sync
                created = [f for f in step['creates'] if os.path.exists(f)]
                if created:
                    print(f"✓ Created files: {created}")
                else:
                    print(f"⚠ Expected files not found: {step['creates']}")
        else:
            print(f"⚠ Step {i} failed, but continuing with remaining analysis...")
            continue
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_header("ANALYSIS COMPLETE")
    print(f"Successfully completed: {successful_steps}/{total_steps} steps")
    print(f"Total runtime: {elapsed_time:.1f} seconds")
    
    if successful_steps == total_steps:
        print("🎉 All analysis steps completed successfully!")
        print("\nGenerated outputs:")
        print("- Data files: data/")
        print("- Figures and tables: figures/")
        print("- Model results: results/")
        print("- Summary report: results/summary_report.md")
        print("\nNext steps:")
        print("1. Review results/summary_report.md for key findings")
        print("2. Examine figures/ for visualizations")
        print("3. Check figures/ CSV files for detailed statistics")
    else:
        print(f"⚠ Analysis completed with {total_steps - successful_steps} failed steps")
        print("Check error messages above for troubleshooting guidance")
    
    return successful_steps == total_steps

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error in analysis pipeline: {e}")
        sys.exit(1) 