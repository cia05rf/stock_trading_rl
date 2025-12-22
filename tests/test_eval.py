"""
Tests for evaluation module.

Tests backtesting, analysis, and reporting functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBacktestResults:
    """Tests for BacktestResults class."""
    
    def test_results_creation(self):
        """Test BacktestResults creation."""
        from eval.backtest import BacktestResults
        
        results = BacktestResults(
            ledger=[{'trade': 1}, {'trade': 2}],
            final_balance=10500,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_return=500,
        )
        
        assert results.final_balance == 10500
        assert results.total_trades == 10
        assert results.winning_trades == 6
    
    def test_results_to_dataframe(self):
        """Test converting results to DataFrame."""
        from eval.backtest import BacktestResults
        
        ledger = [
            {'date': '2024-01-01', 'ticker': 'AAPL', 'action': 'buy'},
            {'date': '2024-01-02', 'ticker': 'AAPL', 'action': 'sell'},
        ]
        
        results = BacktestResults(
            ledger=ledger,
            final_balance=10500,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            total_return=500,
        )
        
        df = results.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_results_summary(self):
        """Test results summary generation."""
        from eval.backtest import BacktestResults
        
        results = BacktestResults(
            ledger=[],
            final_balance=10500,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_return=500,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
        )
        
        summary = results.summary()
        
        assert 'final_balance' in summary
        assert 'win_rate' in summary
        assert 'sharpe_ratio' in summary
        assert summary['win_rate'] == 0.6


class TestMockFund:
    """Tests for MockFund class."""
    
    def test_ledger_entry_creation(self):
        """Test creating ledger entries."""
        from eval.backtest import MockFund
        
        # Mock the dependencies
        with patch('eval.backtest.Infer') as MockInfer, \
             patch('eval.backtest.Ingestion'):
            
            mock_infer = MockInfer.return_value
            mock_infer.env.actions = {
                'buy25': (None, {'proportion': 0.25}, 'buy'),
                'sell25': (None, {'proportion': 0.25}, 'sell'),
                'hold': (None, {}, 'hold'),
            }
            
            fund = MockFund(infer=mock_infer, initial_balance=10000)
            
            entry = fund._create_ledger_entry(
                date='2024-01-01',
                ticker='AAPL',
                action='buy',
                action_type='buy25',
                action_prob=0.8,
                price=100.0,
                quantity=10,
                stamp_duty=5.0,
                spread_cost=10.0,
                trade_value=1000.0,
                balance_pre=10000.0,
                balance_post=8985.0,
            )
            
            assert entry['ticker'] == 'AAPL'
            assert entry['quantity'] == 10
    
    def test_buy_execution(self):
        """Test buy order execution."""
        from eval.backtest import MockFund
        
        with patch('eval.backtest.Infer') as MockInfer, \
             patch('eval.backtest.Ingestion'):
            
            mock_infer = MockInfer.return_value
            mock_infer.env.actions = {
                'buy25': (None, {'proportion': 0.25}, 'buy'),
            }
            
            fund = MockFund(infer=mock_infer, initial_balance=10000)
            
            fund.buy('buy25', 0.8, '2024-01-01', 'AAPL', 100.0, 1000.0)
            
            assert len(fund.ledger) == 1
            assert 'AAPL' in fund.holdings
            assert fund.balance < 10000
    
    def test_sell_execution(self):
        """Test sell order execution."""
        from eval.backtest import MockFund
        
        with patch('eval.backtest.Infer') as MockInfer, \
             patch('eval.backtest.Ingestion'):
            
            mock_infer = MockInfer.return_value
            mock_infer.env.actions = {
                'buy25': (None, {'proportion': 0.25}, 'buy'),
                'sell25': (None, {'proportion': 0.25}, 'sell'),
            }
            
            fund = MockFund(infer=mock_infer, initial_balance=10000)
            
            # First buy
            fund.buy('buy25', 0.8, '2024-01-01', 'AAPL', 100.0, 1000.0)
            initial_holding_qty = fund.holdings['AAPL']['total_quantity']
            
            # Then sell
            fund.sell('sell25', 0.7, '2024-01-02', 'AAPL', 110.0, 5)
            
            assert len(fund.ledger) == 2
            if 'AAPL' in fund.holdings:
                assert fund.holdings['AAPL']['total_quantity'] < initial_holding_qty


class TestAnalysis:
    """Tests for analysis functions."""
    
    def test_analyze_results_empty(self):
        """Test analysis with empty ledger."""
        from eval.analysis import analyze_results
        
        df = pd.DataFrame()
        result = analyze_results(df)
        
        assert 'error' in result
    
    def test_analyze_results(self):
        """Test analysis with valid data."""
        from eval.analysis import analyze_results
        
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'action': ['buy', 'hold', 'sell'],
            'action_type': ['buy', 'hold', 'sell'],
            'balance_pre_trade': [10000, 9900, 9900],
            'balance_post_trade': [9900, 9900, 10100],
        })
        
        result = analyze_results(df)
        
        assert result['total_trades'] == 3
        assert result['initial_balance'] == 10000
        assert result['final_balance'] == 10100
        assert result['total_return'] == 100
    
    def test_analyze_results_metrics(self):
        """Test that all expected metrics are present."""
        from eval.analysis import analyze_results
        
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'action': ['buy', 'sell'],
            'action_type': ['buy', 'sell'],
            'balance_pre_trade': [10000, 9900],
            'balance_post_trade': [9900, 10100],
        })
        
        result = analyze_results(df)
        
        expected_keys = [
            'total_trades', 'buy_trades', 'sell_trades',
            'initial_balance', 'final_balance', 'total_return',
            'return_pct', 'win_rate', 'sharpe_ratio', 'max_drawdown',
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestPlotPerformance:
    """Tests for plotting functions."""
    
    def test_plot_performance_runs(self):
        """Test that plot_performance runs without error."""
        from eval.analysis import plot_performance
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'action': ['buy', 'sell'] * 5,
            'action_type': ['buy', 'sell'] * 5,
            'balance_pre_trade': [10000 + i * 10 for i in range(10)],
            'balance_post_trade': [10010 + i * 10 for i in range(10)],
        })
        
        fig = plot_performance(df, save_path=None)
        
        assert fig is not None
    
    def test_compare_strategies(self):
        """Test strategy comparison plotting."""
        from eval.analysis import compare_strategies
        import matplotlib
        matplotlib.use('Agg')
        
        df1 = pd.DataFrame({
            'balance_pre_trade': [10000, 10050, 10100],
            'balance_post_trade': [10050, 10100, 10150],
        })
        
        df2 = pd.DataFrame({
            'balance_pre_trade': [10000, 9950, 9900],
            'balance_post_trade': [9950, 9900, 9850],
        })
        
        results = [('Strategy A', df1), ('Strategy B', df2)]
        
        fig = compare_strategies(results)
        
        assert fig is not None


class TestGenerateReport:
    """Tests for report generation."""
    
    def test_generate_report(self):
        """Test report generation."""
        from eval.analysis import generate_report
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'action': ['buy', 'sell'] * 5,
            'action_type': ['buy', 'sell'] * 5,
            'balance_pre_trade': [10000 + i * 10 for i in range(10)],
            'balance_post_trade': [10010 + i * 10 for i in range(10)],
        })
        
        report = generate_report(df, strategy_name="Test Strategy")
        
        assert 'Test Strategy' in report
        assert 'Total Trades' in report
        assert 'Final Balance' in report
    
    def test_generate_report_with_output(self, tmp_path):
        """Test report generation with file output."""
        from eval.analysis import generate_report
        import matplotlib
        matplotlib.use('Agg')
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'action': ['buy', 'sell'] * 5,
            'action_type': ['buy', 'sell'] * 5,
            'balance_pre_trade': [10000 + i * 10 for i in range(10)],
            'balance_post_trade': [10010 + i * 10 for i in range(10)],
        })
        
        report = generate_report(df, output_dir=str(tmp_path), strategy_name="Test")
        
        assert (tmp_path / "Test_report.md").exists()
        assert (tmp_path / "Test_performance.png").exists()

