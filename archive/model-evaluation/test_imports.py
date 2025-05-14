#!/usr/bin/env python
"""Test script to debug failed imports"""

print("Testing imports...")

# Test ModernLorentzian
try:
    from modern_pytorch_implementation import ModernLorentzian

    print("ModernLorentzian imported successfully")
except Exception as e:
    print(f"ModernLorentzian import error: {e}")

# Test AnalysisImplementation
try:
    from analysis_implementation import AnalysisImplementation

    print("AnalysisImplementation imported successfully")
except Exception as e:
    print(f"AnalysisImplementation import error: {e}")

# Test YourLogisticRegression
try:
    from your_logistic_regression import LogisticRegression

    print("LogisticRegression imported successfully")
except Exception as e:
    print(f"LogisticRegression import error: {e}")

# Test YourChandelierExit
try:
    from your_chandelier_exit import ChandelierExitIndicator

    print("ChandelierExitIndicator imported successfully")
except Exception as e:
    print(f"ChandelierExitIndicator import error: {e}")

print("Import tests completed")
