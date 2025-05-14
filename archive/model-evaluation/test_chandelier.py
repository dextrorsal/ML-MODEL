#!/usr/bin/env python
"""Test ChandelierExit import"""

try:
    from your_chandelier_exit import ChandelierExitIndicator

    print("ChandelierExitIndicator imported successfully")
except Exception as e:
    print(f"Error importing ChandelierExitIndicator: {e}")
    import traceback

    traceback.print_exc()
