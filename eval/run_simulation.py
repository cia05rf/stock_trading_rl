"""
CLI tool for running optimized trading simulations with configurable thresholds.

Usage:
    python -m eval.run_simulation --start-date 2024-01-01 --end-date 2024-02-01 \
        --balance 50000 --buy-threshold 0.4 --max-positions 10
"""
import argparse
import sys
from pathlib import Path
from eval.fast_simulation import FastMockFund
from eval.analysis import analyze_results, print_summary
from training.inference import Infer
from shared.logging_config import setup_logging, get_logger
from shared.config import get_config

logger = get_logger(__name__)

def run_simulation_cli():
    parser = argparse.ArgumentParser(description="Run High-Performance Trading Simulation")
    
    # Standard Backtest Args
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    
    # User-Requested Config Variables
    parser.add_argument("--balance", type=float, default=10000.0, help="Nominal starting amount")
    parser.add_argument("--buy-threshold", type=float, default=0.3, help="Signal strength required to buy (e.g. 0.3)")
    parser.add_argument("--sell-threshold", type=float, default=-0.3, help="Signal strength required to sell (e.g. -0.3)")
    parser.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions")
    parser.add_argument("--prioritization", choices=["score", "signal"], default="score", help="How to prioritize competing signals")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./reports", help="Where to save CSV/Report")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

    args = parser.parse_args()
    
    # Setup
    config = get_config()
    setup_logging(level="DEBUG" if args.verbose else "INFO", log_dir=config.LOGS_DIR)
    
    logger.info(f"Initializing Simulation with Balance: £{args.balance:,.2f}")
    logger.info(f"Thresholds -> Buy: >{args.buy_threshold}, Sell: <{args.sell_threshold}")
    
    # Initialize Inference
    infer = Infer(model_path=args.model)
    
    # Initialize Optimized Fund
    fund = FastMockFund(
        infer=infer,
        initial_balance=args.balance,
        max_holding_count=args.max_positions,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        prioritize_by=args.prioritization
    )
    
    # Run
    try:
        results = fund.run_backtest(args.start_date, args.end_date)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Analysis
    ledger_df = results.to_dataframe()
    metrics = analyze_results(ledger_df)
    
    print("\n" + "="*30)
    print(" SIMULATION CONFIGURATION ")
    print("="*30)
    print(f"Start Balance:   £{args.balance:,.0f}")
    print(f"Buy Threshold:   {args.buy_threshold}")
    print(f"Sell Threshold:  {args.sell_threshold}")
    print(f"Max Positions:   {args.max_positions}")
    
    print_summary(metrics, "Simulation Results")
    
    # Save Results
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        csv_path = out_path / f"sim_ledger_{args.start_date}_{args.end_date}.csv"
        ledger_df.to_csv(csv_path, index=False)
        print(f"Ledger saved to: {csv_path}")

if __name__ == "__main__":
    run_simulation_cli()