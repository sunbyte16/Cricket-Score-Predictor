#!/usr/bin/env python3
"""
Setup script for Cricket Score Predictor
Handles installation of dependencies with fallbacks for different Python versions
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main setup function"""
    print("Setting up Cricket Score Predictor...")
    print(f"Python version: {sys.version}")
    
    # Core packages that should work with most Python versions
    core_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "requests"
    ]
    
    # Optional packages
    optional_packages = [
        "xgboost",
        "jupyter",
        "notebook"
    ]
    
    print("\nInstalling core packages...")
    failed_core = []
    for package in core_packages:
        print(f"Installing {package}...")
        if not install_package(package):
            failed_core.append(package)
            print(f"Failed to install {package}")
        else:
            print(f"Successfully installed {package}")
    
    print("\nInstalling optional packages...")
    failed_optional = []
    for package in optional_packages:
        print(f"Installing {package}...")
        if not install_package(package):
            failed_optional.append(package)
            print(f"Failed to install {package} (optional)")
        else:
            print(f"Successfully installed {package}")
    
    # Try to install kagglehub separately
    print("\nTrying to install kagglehub...")
    kagglehub_installed = install_package("kagglehub")
    if not kagglehub_installed:
        print("kagglehub installation failed - will use sample data instead")
    
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    if failed_core:
        print(f"❌ Failed core packages: {', '.join(failed_core)}")
        print("⚠️  Some core functionality may not work properly")
    else:
        print("✅ All core packages installed successfully")
    
    if failed_optional:
        print(f"⚠️  Failed optional packages: {', '.join(failed_optional)}")
    else:
        print("✅ All optional packages installed successfully")
    
    if kagglehub_installed:
        print("✅ kagglehub installed - can load real dataset")
    else:
        print("⚠️  kagglehub not installed - will use sample data")
    
    print("\nYou can now run:")
    print("  python main.py")
    
    if "jupyter" not in failed_optional and "notebook" not in failed_optional:
        print("  jupyter notebook cricket_analysis.ipynb")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()