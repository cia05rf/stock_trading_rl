"""
Backtesting module for stock trading strategies.

This module provides tools for simulating trading strategies on historical
data to evaluate model performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from shared.config import get_config
from shared.logging_config import get_logger
from training.inference import Infer
from training.data_ingestion import Ingestion, PricesDf

logger = get_logger(__name__)


@dataclass
class BacktestResults:
    """Container for backtest results."""
    
    ledger: List[Dict]
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    initial_balance: float = 10000.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    benchmark_return: float = 0.0
    equity_curve: List[Dict] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert ledger to DataFrame."""
        return pd.DataFrame(self.ledger)
    
    def equity_to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        return pd.DataFrame(self.equity_curve)
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "total_return": self.total_return,
            "return_pct": self.total_return / self.initial_balance * 100,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "benchmark_return": self.benchmark_return,
            "alpha": (self.total_return / self.initial_balance * 100) - self.benchmark_return,
        }


class MockFund:
    """
    Simulated fund for backtesting trading strategies.
    
    Simulates trading based on model predictions with realistic
    transaction costs and position sizing.
    
    Args:
        infer: Inference object with loaded model
        initial_balance: Starting capital
        tickers: List of tickers to trade (None for all)
        spread: Bid-ask spread percentage
        stamp_duty: Stamp duty percentage
        max_holding_count: Maximum number of positions
    """

    def __init__(
        self,
        infer: Optional[Infer] = None,
        initial_balance: Optional[float] = None,
        tickers: Optional[List[str]] = None,
        spread: float = 0.00,
        stamp_duty: float = 0.005,
        max_holding_count: int = 5,
    ):
        config = get_config()
        
        self.infer = infer or Infer()
        self.ingest = Ingestion()
        self.balance = initial_balance or config.INITIAL_BALANCE
        self.initial_balance = self.balance
        self.tickers = tickers
        self.spread = spread
        self.stamp_duty = stamp_duty
        self.max_holding_count = max_holding_count
        
        # Trading records
        self.ledger: List[Dict] = []
        self.holdings: Dict[str, Dict] = {}
        self.equity_curve: List[Dict] = []
        
        # Get action proportions from environment
        self.actions = {
            k: v[1].get("proportion", 1.0)
            for k, v in self.infer.env.actions.items()
        }
        
        logger.info(f"MockFund initialized with balance: £{self.balance:,.2f}")

    def _calculate_equity(self, date, prices_df: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate total equity including unrealized P&L.
        
        Args:
            date: Current date
            prices_df: Optional prices DataFrame (to avoid re-fetching)
        
        Returns:
            Total equity (cash + holdings value)
        """
        equity = self.balance
        
        if not self.holdings:
            return equity
        
        # Get current prices for holdings
        if prices_df is None:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
            self.ingest.min_date = date_str
            self.ingest.max_date = date_str
            prices_df = self.ingest.read_prices(list(self.holdings.keys()))
        
        if "date" in prices_df.columns:
            prices_df = prices_df[prices_df["date"] <= pd.to_datetime(date)]
        
        for ticker, holding in self.holdings.items():
            price_row = prices_df[prices_df["ticker"] == ticker]
            if len(price_row) > 0:
                # Use latest price up to the timestamp
                current_price = price_row.sort_values("date").iloc[-1]["close"]
                equity += holding["total_quantity"] * current_price
        
        return equity

    def _record_equity(self, date, prices_df: Optional[pd.DataFrame] = None) -> None:
        """Record equity for the current date."""
        equity = self._calculate_equity(date, prices_df)
        unrealized_pnl = equity - self.balance - sum(
            h["total_quantity"] * h["avco"] for h in self.holdings.values()
        ) if self.holdings else 0
        
        ts = pd.to_datetime(date)
        self.equity_curve.append({
            "date": ts,
            "cash": self.balance,
            "holdings_value": equity - self.balance,
            "total_equity": equity,
            "unrealized_pnl": unrealized_pnl,
            "num_positions": len(self.holdings),
        })

    def _create_ledger_entry(
        self,
        date,
        ticker: str,
        action: str,
        action_type: str,
        action_prob: float,
        price: float,
        quantity: int,
        stamp_duty: float,
        spread_cost: float,
        trade_value: float,
        balance_pre: float,
        balance_post: float,
    ) -> Dict:
        """Create a ledger entry."""
        ts = pd.to_datetime(date)
        return {
            "date": ts,
            "ticker": ticker,
            "action": action,
            "action_type": action_type,
            "action_prob": action_prob,
            "price": price,
            "quantity": quantity,
            "stamp_duty": stamp_duty,
            "spread_cost": spread_cost,
            "trade_value": trade_value,
            "balance_pre_trade": balance_pre,
            "balance_post_trade": balance_post,
        }

    def buy(
        self,
        action_type: str,
        action_prob: float,
        date: str,
        ticker: str,
        price: float,
        value: float,
    ) -> None:
        """Execute a buy order."""
        if value <= 0 or price <= 0:
            return
        
        # Calculate costs
        stamp_duty = value * self.stamp_duty
        spread_cost = value * self.spread
        value_available = value - stamp_duty - spread_cost
        
        quantity = int(value_available / price)
        if quantity <= 0:
            return
        
        trade_value = quantity * price
        stamp_duty = trade_value * self.stamp_duty
        spread_cost = trade_value * self.spread
        total_cost = trade_value + stamp_duty + spread_cost
        
        balance_pre = self.balance
        balance_post = balance_pre - total_cost
        
        # Record trade
        self.ledger.append(self._create_ledger_entry(
            date, ticker, "buy", action_type, action_prob,
            price, quantity, stamp_duty, spread_cost, trade_value,
            balance_pre, balance_post
        ))
        
        self.balance = balance_post
        
        # Update holdings
        if ticker in self.holdings:
            h = self.holdings[ticker]
            h["quantities"].append(quantity)
            h["prices"].append(price)
            h["total_quantity"] += quantity
            h["avco"] = (
                sum(q * p for q, p in zip(h["quantities"], h["prices"]))
                / h["total_quantity"]
            ) if h["total_quantity"] > 0 else price
        else:
            self.holdings[ticker] = {
                "ticker": ticker,
                "quantities": [quantity],
                "prices": [price],
                "total_quantity": quantity,
                "avco": price,
            }

    def sell(
        self,
        action_type: str,
        action_prob: float,
        date: str,
        ticker: str,
        price: float,
        quantity: int,
    ) -> None:
        """Execute a sell order."""
        if ticker not in self.holdings or quantity <= 0:
            return
        
        holding = self.holdings[ticker]
        quantity = min(quantity, holding["total_quantity"])
        
        trade_value = quantity * price
        stamp_duty = 0  # No stamp duty on sales
        spread_cost = trade_value * self.spread
        
        balance_pre = self.balance
        balance_post = balance_pre + trade_value - spread_cost
        
        # Record trade
        self.ledger.append(self._create_ledger_entry(
            date, ticker, "sell", action_type, action_prob,
            price, quantity, stamp_duty, spread_cost, trade_value,
            balance_pre, balance_post
        ))
        
        self.balance = balance_post
        
        # Update holdings
        holding["quantities"].append(-quantity)
        holding["prices"].append(-price)
        holding["total_quantity"] -= quantity
        
        if holding["total_quantity"] <= 0:
            del self.holdings[ticker]
        else:
            holding["avco"] = (
                sum(q * p for q, p in zip(holding["quantities"], holding["prices"]))
                / holding["total_quantity"]
            )

    def _run_timestamp(
        self,
        timestamp: pd.Timestamp,
        prices_by_ticker: Dict[str, PricesDf],
        preds: Optional[pd.DataFrame] = None,
    ) -> None:
        """Run trading for a single timestamp (e.g., 15m bar)."""
        if preds is None:
            preds = self.infer.predict_for_timestamp(prices_by_ticker, timestamp)
        if preds.empty:
            return
        
        # Get top recommendations
        top_buys = preds[preds["action_type"] == "buy"].sort_values("action_prob", ascending=False)
        top_sells = preds[preds["action_type"] == "sell"].sort_values("action_prob", ascending=False)
        
        # Execute sells first
        for _, row in top_sells.iterrows():
            if row["ticker"] in self.holdings:
                h = self.holdings[row["ticker"]]
                proportion = self.actions.get(row["action"], 0.25)
                qty = max(1, int(h["total_quantity"] * proportion))
                qty = min(qty, h["total_quantity"])
                self.sell(
                    row["action_type"], row["action_prob"],
                    timestamp, row["ticker"], row["close"], qty
                )
        
        # Execute buys
        for _, row in top_buys.iterrows():
            if self.balance < row["close"]:
                continue
            
            proportion = self.actions.get(row["action"], 0.25)
            value = int((self.balance / self.max_holding_count) * proportion)
            
            if value > 0:
                self.buy(
                    row["action_type"], row["action_prob"],
                    timestamp, row["ticker"], row["close"], value
                )
        
        # Record equity at this timestamp
        price_snapshot = preds[["ticker", "close", "date"]].rename(columns={"date": "timestamp"})
        # Align schema for _record_equity (expects 'date' column)
        price_snapshot = price_snapshot.rename(columns={"timestamp": "date"})
        self._record_equity(timestamp, price_snapshot)

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
    ) -> BacktestResults:
        """
        Run a backtest over a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            BacktestResults with trading results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Preload price data with a prefix buffer for initial windows
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        buffer_days = max(5, int(self.max_holding_count))  # simple buffer heuristic
        load_start = (start_ts - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        self.ingest.min_date = load_start
        self.ingest.max_date = end_date
        prices = self.ingest.read_prices(self.tickers)
        prices = prices.sort_values(["ticker", "date"])
        
        # Filter to requested window (timestamps inside range) but keep prefix data in memory
        in_range_prices = prices[
            (prices["date"] >= start_ts) &
            (prices["date"] <= end_ts)
        ]
        
        if len(in_range_prices) == 0:
            logger.warning("No price data found for the requested backtest window.")
            return BacktestResults(
                ledger=[],
                final_balance=self.balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_return=0.0,
                initial_balance=self.initial_balance,
                benchmark_return=0.0,
            )
        
        # Prepare per-ticker preprocessed data (adds technicals and normalization)
        prices_by_ticker: Dict[str, PricesDf] = {}
        for ticker, df_t in prices.groupby("ticker"):
            prepared = PricesDf(df_t.copy()).prep_data().sort_values("date")
            prices_by_ticker[ticker] = prepared
        
        # Precompute predictions for all timestamps to avoid repeated inference
        preds_all = self.infer.predict_timespan(
            start=start_ts,
            end=end_ts,
            tickers=self.tickers,
            buffer_days=buffer_days,
            deterministic=False,
        )

        # SUmmarise the number of predictions by action type
        logger.info(f"Number of predictions by action type: {preds_all.groupby('action_type').size()}")
        preds_all["date"] = pd.to_datetime(preds_all["date"])
        preds_by_ts = {
            ts: df for ts, df in preds_all.groupby("date")
        } if len(preds_all) else {}
        
        # Unique timestamps across all tickers in the requested window, sorted
        timestamps = sorted(in_range_prices["date"].unique())
        
        # Run simulation over all timestamps (e.g., 15m bars)
        for ts in tqdm(timestamps, desc="Trading over timestamps with predictions"):
            ts_dt = pd.to_datetime(ts)
            preds_ts = preds_by_ts.get(ts_dt)
            self._run_timestamp(ts_dt, prices_by_ticker, preds_ts)
        
        # Close all positions at end (using last available date)
        final_ts = pd.to_datetime(timestamps[-1])
        self._close_all_positions(final_ts)
        
        # Calculate results
        return self._calculate_results(start_date, end_date)

    def _close_all_positions(self, date) -> None:
        """Close all open positions."""
        if not self.holdings:
            return
        
        ts = pd.to_datetime(date)
        date_str = ts.strftime("%Y-%m-%d")
        logger.info(f"Closing all positions on {date_str}")
        
        self.ingest.min_date = date_str
        self.ingest.max_date = date_str
        
        prices = self.ingest.read_prices(list(self.holdings.keys()))
        prices = prices[["ticker", "close"]].drop_duplicates()
        
        for ticker, holding in list(self.holdings.items()):
            price_row = prices[prices["ticker"] == ticker]
            if len(price_row) > 0:
                self.sell(
                    "close", 1.0, ts, ticker,
                    price_row["close"].iloc[0],
                    holding["total_quantity"]
                )

    def _calculate_benchmark(self, start_date: str, end_date: str) -> float:
        """
        Calculate buy-and-hold benchmark return.
        
        Returns equal-weighted average return of all traded stocks.
        """
        try:
            self.ingest.min_date = start_date
            self.ingest.max_date = end_date
            prices = self.ingest.read_prices(self.tickers)
            
            if len(prices) == 0:
                return 0.0
            
            # Get first and last prices for each ticker
            first_prices = prices.groupby("ticker")["close"].first()
            last_prices = prices.groupby("ticker")["close"].last()
            
            # Calculate returns
            stock_returns = (last_prices - first_prices) / first_prices
            benchmark_return = stock_returns.mean() * 100
            
            return benchmark_return
        except Exception as e:
            logger.warning(f"Failed to calculate benchmark: {e}")
            return 0.0

    def _calculate_results(self, start_date: str, end_date: str) -> BacktestResults:
        """Calculate backtest results."""
        ledger_df = pd.DataFrame(self.ledger)
        
        # Calculate benchmark
        benchmark_return = self._calculate_benchmark(start_date, end_date)
        
        if len(ledger_df) == 0:
            return BacktestResults(
                ledger=self.ledger,
                final_balance=self.balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_return=0,
                initial_balance=self.initial_balance,
                benchmark_return=benchmark_return,
                equity_curve=self.equity_curve,
            )
        
        # Calculate trade P&L
        buy_trades = ledger_df[ledger_df["action"] == "buy"].copy()
        sell_trades = ledger_df[ledger_df["action"] == "sell"].copy()
        
        total_trades = len(buy_trades) + len(sell_trades)
        
        # Calculate returns per trade (simplified)
        winning = len(ledger_df[ledger_df["balance_post_trade"] > ledger_df["balance_pre_trade"]])
        losing = total_trades - winning
        
        total_return = self.balance - self.initial_balance
        
        # Calculate max drawdown from equity curve if available
        if self.equity_curve:
            equities = [e["total_equity"] for e in self.equity_curve]
            peak = equities[0]
            max_drawdown = 0
            for equity in equities:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            # Fallback to balance-based calculation
            balances = ledger_df["balance_post_trade"].values
            peak = balances[0]
            max_drawdown = 0
            for balance in balances:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return BacktestResults(
            ledger=self.ledger,
            final_balance=self.balance,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            total_return=total_return,
            initial_balance=self.initial_balance,
            max_drawdown=max_drawdown,
            benchmark_return=benchmark_return,
            equity_curve=self.equity_curve,
        )


def run_backtest(
    start_date: str,
    end_date: str,
    initial_balance: float = 10000,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> BacktestResults:
    """
    Convenience function to run a backtest.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_balance: Starting capital
        model_path: Path to model (optional)
        output_path: Path to save results CSV (optional)
    
    Returns:
        BacktestResults
    """
    infer = Infer(model_path=model_path)
    fund = MockFund(infer, initial_balance=initial_balance)
    results = fund.run_backtest(start_date, end_date)
    
    if output_path:
        results.to_dataframe().to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    return results

