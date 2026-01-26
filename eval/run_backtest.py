"""
Backtest CLI for Stock Trading Agent.

This script runs backtests on trained models to evaluate trading performance.

Usage:
    python -m eval.run_backtest --start-date 2024-01-01 --end-date 2024-06-30
    python -m eval.run_backtest --model training/models/ppo_stock_trading_20251211_2110.zip
    python -m eval.run_backtest --start-date 2024-01-01 --end-date 2024-12-01 --output-dir ./reports
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.config import get_config
from shared.logging_config import setup_logging, get_logger
from eval.backtest import MockFund, run_backtest, BacktestResults
from eval.analysis import (
    analyze_results,
    plot_performance,
    generate_report,
    monte_carlo_significance_test,
)
from training.inference import Infer

logger = get_logger(__name__)


def run_evaluation(
    start_date: str,
    end_date: str,
    model_path: Optional[str] = None,
    initial_balance: float = 10000.0,
    buy_threshold: float = 0.3,
    sell_threshold: float = -0.3,
    max_positions: int = 5,
    output_dir: Optional[str] = None,
    generate_plots: bool = True,
    run_significance_test: bool = True,
    verbose: bool = True,
) -> BacktestResults:
    """
    Run a complete backtest evaluation.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model_path: Path to model file (None = use latest)
        initial_balance: Starting capital
        buy_threshold: Signal strength required to buy
        sell_threshold: Signal strength required to sell
        max_positions: Maximum concurrent positions
        output_dir: Directory to save results
        generate_plots: Whether to generate visualization
        run_significance_test: Whether to run Monte Carlo test
        verbose: Print detailed output
    
    Returns:
        BacktestResults object
    """
    config = get_config()
    
    # Load model
    if model_path:
        logger.info(f"Loading model from: {model_path}")
        infer = Infer(model_path=model_path)
    else:
        logger.info("Loading latest model...")
        infer = Infer()  # Uses load_latest_model() internally
    
    # Create fund
    fund = MockFund(
        infer=infer,
        initial_balance=initial_balance,
        max_holding_count=max_positions,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )
    
    # Run backtest
    logger.info(f"Running backtest: {start_date} to {end_date}")
    logger.info(f"Initial balance: £{initial_balance:,.2f}")
    logger.info(f"Thresholds -> Buy: >{buy_threshold}, Sell: <{sell_threshold}")
    
    results = fund.run_backtest(start_date, end_date)
    
    # Analyze results
    ledger_df = results.to_dataframe()
    metrics = analyze_results(ledger_df)
    
    # Helper for formatting ratios
    def fmt_ratio(value):
        if value == float('inf'):
            return "∞"
        elif value == float('-inf'):
            return "-∞"
        else:
            return f"{value:.2f}"
    
    # Print summary
    if verbose:
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"Period:              {start_date} to {end_date}")
        print(f"Model:               {model_path or 'latest'}")
        print("-" * 70)
        print(f"Initial Balance:     £{initial_balance:,.2f}")
        print(f"Final Balance:       £{results.final_balance:,.2f}")
        print(f"Total Return:        £{results.total_return:,.2f} ({metrics.get('return_pct', 0):.2f}%)")
        print("-" * 70)
        print(f"Total Trades:        {results.total_trades}")
        print(f"Winning Trades:      {results.winning_trades}")
        print(f"Losing Trades:       {results.losing_trades}")
        print(f"Win Rate:            {metrics.get('win_rate', 0):.1%}")
        print("-" * 70)
        print(f"Sharpe Ratio:        {fmt_ratio(metrics.get('sharpe_ratio', 0))}")
        print(f"Sortino Ratio:       {fmt_ratio(metrics.get('sortino_ratio', 0))}")
        print(f"Calmar Ratio:        {fmt_ratio(metrics.get('calmar_ratio', 0))}")
        print(f"Profit Factor:       {fmt_ratio(metrics.get('profit_factor', 0))}")
        print("-" * 70)
        print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Expectancy:          £{metrics.get('expectancy', 0):.2f} per trade")
        print("-" * 70)
        
        # Benchmark comparison
        if results.benchmark_return != 0:
            alpha = metrics.get('return_pct', 0) - results.benchmark_return
            print(f"Benchmark Return:    {results.benchmark_return:.2f}%")
            print(f"Alpha:               {alpha:+.2f}%")
            print("-" * 70)
        
        print("=" * 70)
    
    # Run significance test
    if run_significance_test and verbose:
        print("\nRunning Monte Carlo significance test...")
        significance = monte_carlo_significance_test(ledger_df)
        if "error" in significance:
            print(f"Skipping significance test: {significance['error']}")
        else:
            print(f"\n{significance['interpretation']}")
            print(f"P-value: {significance['p_value_return']:.4f}")
            print()
    
    # Save outputs
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ledger CSV
        csv_path = output_path / f"backtest_ledger_{timestamp}.csv"
        ledger_df.to_csv(csv_path, index=False)
        logger.info(f"Ledger saved to: {csv_path}")
        
        # Save equity curve if available
        if results.equity_curve:
            equity_df = results.equity_to_dataframe()
            equity_path = output_path / f"equity_curve_{timestamp}.csv"
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"Equity curve saved to: {equity_path}")
        
        # Generate report
        if generate_plots:
            equity_df = results.equity_to_dataframe() if results.equity_curve else None
            generate_report(
                ledger_df,
                output_dir=str(output_path),
                strategy_name=f"Backtest_{timestamp}",
                include_significance_test=run_significance_test,
                equity_df=equity_df,
            )
            logger.info(f"Report saved to: {output_path}")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run backtest evaluation on trained trading model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Date range (required)
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        required=True,
        help="Start date for backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", "-e",
        type=str,
        required=True,
        help="End date for backtest (YYYY-MM-DD)",
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model file (.zip). Defaults to latest model in training/models/",
    )
    
    # Trading parameters
    parser.add_argument(
        "--balance", "-b",
        type=float,
        default=10000.0,
        help="Initial balance in GBP",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.3,
        help="Signal strength required to buy",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.3,
        help="Signal strength required to sell",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum number of concurrent positions",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save results (CSV, plots, report)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots and reports",
    )
    parser.add_argument(
        "--no-significance-test",
        action="store_true",
        help="Skip Monte Carlo significance test",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output",
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        parser.error(f"Invalid date format: {e}. Use YYYY-MM-DD")
    
    if args.start_date >= args.end_date:
        parser.error("start-date must be before end-date")
    
    # Setup logging
    config = get_config()
    setup_logging(level=config.LOG_LEVEL, log_dir=config.LOGS_DIR)
    
    # Run evaluation
    run_evaluation(
        start_date=args.start_date,
        end_date=args.end_date,
        model_path=args.model,
        initial_balance=args.balance,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        max_positions=args.max_positions,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        run_significance_test=not args.no_significance_test,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

