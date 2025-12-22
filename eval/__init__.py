"""
Evaluation module for stock trading model.

This module provides tools for:
- Backtesting trading strategies
- Performance analysis
- Mock fund simulation
"""

from eval.backtest import MockFund, BacktestResults
from eval.analysis import analyze_results, plot_performance

__all__ = [
    "MockFund",
    "BacktestResults",
    "analyze_results",
    "plot_performance",
]

