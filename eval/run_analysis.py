"""
CLI wrapper for `eval.analysis` to generate reports from a ledger CSV.

Example:
    python -m eval.run_analysis --ledger ./reports/backtest_ledger.csv \
        --output-dir ./reports --strategy-name "PPO Trading Strategy"
"""

import argparse
from pathlib import Path

import pandas as pd

from shared.config import get_config
from shared.logging_config import setup_logging, get_logger
from eval.analysis import analyze_results, generate_report

logger = get_logger(__name__)

# Minimum columns needed for analyze_results()
REQUIRED_COLUMNS = {
    "balance_pre_trade",
    "balance_post_trade",
    "action",
    "action_type",
}


def load_ledger(path: Path) -> pd.DataFrame:
    """Load and validate a ledger CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Ledger file not found: {path}")
    
    ledger_df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(ledger_df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Ledger missing required columns: {missing_cols}")
    
    if "date" in ledger_df.columns:
        ledger_df["date"] = pd.to_datetime(ledger_df["date"])
    
    return ledger_df


def load_equity(path: Path) -> pd.DataFrame:
    """Load optional equity curve CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Equity file not found: {path}")
    equity_df = pd.read_csv(path)
    if "date" in equity_df.columns:
        equity_df["date"] = pd.to_datetime(equity_df["date"])
    return equity_df


def print_summary(metrics: dict, strategy_name: str) -> None:
    """Pretty-print core metrics to stdout."""
    def fmt_ratio(value: float) -> str:
        if value == float("inf"):
            return "∞"
        if value == float("-inf"):
            return "-∞"
        return f"{value:.2f}"
    
    print("\n" + "=" * 70)
    print(f"{strategy_name} ANALYSIS")
    print("=" * 70)
    print(f"Total Trades:        {metrics.get('total_trades', 0)}")
    print(f"Winning Trades:      {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades:       {metrics.get('losing_trades', 0)}")
    print(f"Win Rate:            {metrics.get('win_rate', 0):.1%}")
    print("-" * 70)
    print(f"Initial Balance:     £{metrics.get('initial_balance', 0):,.2f}")
    print(f"Final Balance:       £{metrics.get('final_balance', 0):,.2f}")
    print(f"Total Return:        £{metrics.get('total_return', 0):,.2f} ({metrics.get('return_pct', 0):.2f}%)")
    print("-" * 70)
    print(f"Sharpe Ratio:        {fmt_ratio(metrics.get('sharpe_ratio', 0))}")
    print(f"Sortino Ratio:       {fmt_ratio(metrics.get('sortino_ratio', 0))}")
    print(f"Calmar Ratio:        {fmt_ratio(metrics.get('calmar_ratio', 0))}")
    print(f"Profit Factor:       {fmt_ratio(metrics.get('profit_factor', 0))}")
    print("-" * 70)
    print(f"Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Expectancy:          £{metrics.get('expectancy', 0):,.2f} per trade")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate analysis report from a backtest ledger CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--ledger", "-l",
        required=True,
        type=str,
        help="Path to ledger CSV produced by a backtest run",
    )
    parser.add_argument(
        "--equity", "-q",
        type=str,
        default=None,
        help="Optional equity curve CSV to include in the report",
    )
    parser.add_argument(
        "--strategy-name", "-n",
        type=str,
        default=None,
        help="Friendly name for the strategy used in report headings",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./reports",
        help="Directory to write the markdown report and performance chart",
    )
    parser.add_argument(
        "--no-significance-test",
        action="store_true",
        help="Skip the Monte Carlo significance test section",
    )
    
    args = parser.parse_args()
    
    config = get_config()
    setup_logging(level=config.LOG_LEVEL, log_dir=config.LOGS_DIR)
    
    ledger_path = Path(args.ledger)
    equity_path = Path(args.equity) if args.equity else None
    strategy_name = args.strategy_name or ledger_path.stem
    
    logger.info(f"Loading ledger from {ledger_path}")
    ledger_df = load_ledger(ledger_path)
    
    equity_df = None
    if equity_path:
        logger.info(f"Loading equity curve from {equity_path}")
        equity_df = load_equity(equity_path)
    
    metrics = analyze_results(ledger_df)
    print_summary(metrics, strategy_name)
    
    report_text = generate_report(
        ledger_df,
        output_dir=args.output_dir,
        strategy_name=strategy_name,
        include_significance_test=not args.no_significance_test,
        equity_df=equity_df,
    )
    
    report_path = Path(args.output_dir) / f"{strategy_name.replace(' ', '_')}_report.md"
    if args.output_dir:
        print(f"Report and figures written to: {Path(args.output_dir).resolve()}")
    else:
        # If no output directory requested, emit report to stdout for convenience.
        print(report_text)
        print_summary(metrics, strategy_name)
        print(f"(Report path would be: {report_path})")


if __name__ == "__main__":
    main()

