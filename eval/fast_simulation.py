"""
Alias module for the optimized MockFund.
"""
from eval.backtest import MockFund as FastMockFund

# This module is maintained for backward compatibility. 
# All optimizations (O(1) price lookup, batched inference, thresholds) 
# have been merged into the base MockFund class.
