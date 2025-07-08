#!/usr/bin/env python3
"""
Environment Setup and Validation for GARCH Financial Analysis
Validates that all required packages are installed and working correctly.
"""

import sys
import os
import importlib
import warnings

def check_python_version():
    """Check if Python version is compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print(f"✓ Python version: {sys.version.split()[0]} (compatible)")
        return True
    else:
        print(f"✗ Python version: {sys.version.split()[0]} (requires >= {min_version[0]}.{min_version[1]})")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "unknown"
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: Not installed")
        return False

def check_directories():
    """Check if required directories exist and create them if needed."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    required_dirs = ['data', 'figures', 'results']
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ Directory '{dir_name}' exists")
        else:
            try:
                os.makedirs(dir_path)
                print(f"✓ Directory '{dir_name}' created")
            except Exception as e:
                print(f"✗ Failed to create directory '{dir_name}': {e}")
                all_exist = False
    
    return all_exist

def validate_data_access():
    """Test basic data operations."""
    try:
        import pandas as pd
        import numpy as np
        
        # Test basic operations
        df = pd.DataFrame({'test': np.random.randn(10)})
        result = df.describe()
        print("✓ Basic pandas/numpy operations working")
        return True
    except Exception as e:
        print(f"✗ Basic data operations failed: {e}")
        return False

def validate_financial_packages():
    """Test financial analysis packages."""
    try:
        import yfinance as yf
        
        # Test basic yfinance functionality
        ticker = yf.Ticker("AAPL")
        # Don't actually download data, just test the object creation
        print("✓ yfinance package accessible")
        return True
    except Exception as e:
        print(f"✗ Financial packages validation failed: {e}")
        return False

def validate_modeling_packages():
    """Test statistical modeling packages."""
    try:
        from arch import arch_model
        from statsmodels.tsa.api import VAR
        
        print("✓ Statistical modeling packages accessible")
        return True
    except Exception as e:
        print(f"✗ Modeling packages validation failed: {e}")
        return False

def main():
    """Main validation routine."""
    print("=" * 60)
    print(" GARCH Financial Analysis - Environment Validation")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    all_checks = []
    
    # Check Python version
    all_checks.append(check_python_version())
    
    print("\nChecking required packages...")
    required_packages = [
        ('yfinance', 'yfinance'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('arch', 'arch'),
        ('statsmodels', 'statsmodels'),
        ('scipy', 'scipy')
    ]
    
    for package_name, import_name in required_packages:
        all_checks.append(check_package(package_name, import_name))
    
    print("\nChecking directories...")
    all_checks.append(check_directories())
    
    print("\nValidating functionality...")
    all_checks.append(validate_data_access())
    all_checks.append(validate_financial_packages())
    all_checks.append(validate_modeling_packages())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(all_checks)
    total = len(all_checks)
    
    if passed == total:
        print(f"🎉 Environment validation successful! ({passed}/{total} checks passed)")
        print("The system is ready for GARCH financial analysis.")
        return True
    else:
        print(f"⚠ Environment validation incomplete! ({passed}/{total} checks passed)")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error during environment validation: {e}")
        sys.exit(1) 