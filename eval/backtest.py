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
    orders_placed: int = 0
    orders_filled: int = 0
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
            "fill_rate": self.orders_filled / max(self.orders_placed, 1),
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "total_return": self.total_return,
            "return_pct": self.total_return / self.initial_balance * 100,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "benchmark_return": self.benchmark_return,
            "alpha": (self.total_return / self.initial_balance * 100) - self.benchmark_return,
        }


class MockFund:
    """
    Simulated fund for backtesting Limit Order trading strategies.
    
    Architecture:
    - Maintains ActiveLimitOrders list.
    - Tracks OpenPositions dict.
    - Simulates fills by checking High/Low of current candle.
    - Ranks multiple trade signals by (Signal * Predicted_Discount).
    """

    def __init__(
        self,
        infer: Optional[Infer] = None,
        initial_balance: Optional[float] = None,
        tickers: Optional[List[str]] = None,
        max_holding_count: int = 5,
    ):
        self.config = get_config()
        self.infer = infer or Infer()
        self.ingest = Ingestion()
        self.balance = initial_balance or self.config.INITIAL_BALANCE
        self.initial_balance = self.balance
        self.tickers = tickers
        self.max_holding_count = max_holding_count
        
        # State
        self.active_limit_orders: List[Dict] = []
        self.open_positions: Dict[str, Dict] = {} # ticker -> position
        
        # Stats
        self.orders_placed = 0
        self.orders_filled = 0
        self.ledger: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        logger.info(f"MockFund (Limit Order) initialized with balance: £{self.balance:,.2f}")

    def _calculate_limit_price(self, close_price: float, action_val: float, order_type: str = "buy") -> float:
        """Formula: Limit_Price = Close * (1 - (abs(Action[1]) * MAX_LIMIT_OFFSET))"""
        offset = abs(action_val) * self.config.MAX_LIMIT_OFFSET
        if order_type == "buy":
            return close_price * (1 - offset)
        else: # sell
            return close_price * (1 + offset)

    def _calculate_stop_price(self, limit_price: float, action_val: float) -> float:
        """Formula: Stop_Price = Limit_Price * (1 - (abs(Action[2]) * MAX_STOP_LOSS))"""
        return limit_price * (1 - (abs(action_val) * self.config.MAX_STOP_LOSS))

    def _create_ledger_entry(
        self,
        date,
        ticker: str,
        action: str, # buy/sell/close/stop_loss
        price: float,
        quantity: int,
        trade_value: float,
        balance_post: float,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
    ) -> Dict:
        """Create a ledger entry."""
        return {
            "date": pd.to_datetime(date),
            "ticker": ticker,
            "action": action,
            "price": price,
            "quantity": quantity,
            "trade_value": trade_value,
            "balance": balance_post,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }

    def _run_timestamp(
        self,
        timestamp: pd.Timestamp,
        prices_by_ticker: Dict[str, PricesDf],
        preds: Optional[pd.DataFrame] = None,
    ) -> None:
        """Run simulation for a single timestamp (15m candle)."""
        
        # 1. Update Existing Limit Orders (TTL)
        for order in self.active_limit_orders[:]:
            order["ttl"] -= 1
            if order["ttl"] <= 0:
                self.active_limit_orders.remove(order)

        # 2. Check for Fills and Stop Losses using current candle
        # Logic: Stop Loss checked first (pessimistic)
        for ticker in list(self.open_positions.keys()) + [o["ticker"] for o in self.active_limit_orders]:
            if ticker not in prices_by_ticker:
                continue
            
            ticker_df = prices_by_ticker[ticker]
            candle_row = ticker_df[ticker_df["date"] == timestamp]
            if candle_row.empty:
                continue
            
            candle = candle_row.iloc[0]
            high = candle["high"]
            low = candle["low"]
            
            # 2a. Check Stop Loss for Open Positions
            if ticker in self.open_positions:
                pos = self.open_positions[ticker]
                if low <= pos["stop_loss"]:
                    # Stop loss hit: Pessimistic exit at stop_loss price or high if gap
                    exit_price = min(pos["stop_loss"], high)
                    self._liquidate(ticker, exit_price, timestamp, reason="stop_loss")
            
            # 2b. Check Fills for Active Limit Orders (Only if no stop_loss hit just now)
            if ticker not in self.open_positions:
                ticker_orders = [o for o in self.active_limit_orders if o["ticker"] == ticker]
                for order in ticker_orders:
                    fill_price = order["price"]
                    filled = False
                    if order["type"] == "buy":
                        if low <= fill_price:
                            filled = True
                            fill_price = min(fill_price, high)
                    else: # sell (Exit)
                        if high >= fill_price:
                            filled = True
                            fill_price = max(fill_price, low)
                    
                    if filled:
                        if order["type"] == "buy":
                            self._execute_fill(order, fill_price, timestamp)
                        else: # sell fill
                            self._liquidate(ticker, fill_price, timestamp, reason="limit_fill_sell")
                        
                        if order in self.active_limit_orders:
                            self.active_limit_orders.remove(order)

        # 3. Process New Predictions
        if preds is not None and not preds.empty:
            # First, handle cancellations for tickers with pending orders
            pending_tickers = {o["ticker"] for o in self.active_limit_orders}
            for _, row in preds.iterrows():
                ticker = row["ticker"]
                signal = row["signal"]
                
                if ticker in pending_tickers:
                    # Cancel if signal is neutral or opposite of order
                    # For buy order: cancel if signal <= 0.3
                    # For sell order: cancel if signal >= -0.3
                    order = next(o for o in self.active_limit_orders if o["ticker"] == ticker)
                    if (order["type"] == "buy" and signal <= 0.3) or \
                       (order["type"] == "sell" and signal >= -0.3):
                        self.active_limit_orders = [o for o in self.active_limit_orders if o["ticker"] != ticker]
            
            # Second, identify new order candidates
            new_candidates = []
            for _, row in preds.iterrows():
                ticker = row["ticker"]
                signal = row["signal"]
                
                # Case A: No Position -> Potential Buy Limit
                if ticker not in self.open_positions and ticker not in pending_tickers:
                    if signal > 0.3:
                        new_candidates.append(row)
                
                # Case B: Active Position -> Potential Sell Limit
                elif ticker in self.open_positions and ticker not in pending_tickers:
                    if signal < -0.3:
                        new_candidates.append(row)
            
            if new_candidates:
                candidate_df = pd.DataFrame(new_candidates)
                # Calculate ranking score: abs(Signal) * Predicted_Discount
                candidate_df["predicted_discount"] = candidate_df.apply(
                    lambda r: (
                        (r["close"] - self._calculate_limit_price(
                            r["close"], r["limit_offset_action"], "buy"
                        )) / r["close"]
                        if r["signal"] > 0.3 else
                        (self._calculate_limit_price(
                            r["close"], r["limit_offset_action"], "sell"
                        ) - r["close"]) / r["close"]
                    ),
                    axis=1
                )
                candidate_df["score"] = (
                    candidate_df["signal"].abs() * 
                    candidate_df["predicted_discount"]
                )
                candidate_df = candidate_df.sort_values("score", ascending=False)
                
                available_slots = (
                    self.max_holding_count - len(self.open_positions)
                )
                
                for _, row in candidate_df.iterrows():
                    ticker = row["ticker"]
                    if row["signal"] > 0.3: # Buy candidate
                        if available_slots > 0:
                            limit_price = self._calculate_limit_price(
                                row["close"], row["limit_offset_action"], "buy"
                            )
                            stop_loss = self._calculate_stop_price(
                                limit_price, row["stop_loss_action"]
                            )
                            self.active_limit_orders.append({
                                "ticker": ticker,
                                "type": "buy",
                                "price": limit_price,
                                "stop_loss": stop_loss,
                                "ttl": self.config.ORDER_TTL,
                                "timestamp": timestamp
                            })
                            self.orders_placed += 1
                            available_slots -= 1
                    else: # Sell candidate
                        limit_price = self._calculate_limit_price(
                            row["close"], row["limit_offset_action"], "sell"
                        )
                        self.active_limit_orders.append({
                            "ticker": ticker,
                            "type": "sell",
                            "price": limit_price,
                            "ttl": self.config.ORDER_TTL,
                            "timestamp": timestamp
                        })
                        self.orders_placed += 1

        # 4. Record Equity Snapshot
        self._record_equity(timestamp, prices_by_ticker)

    def _execute_fill(self, order: Dict, fill_price: float, timestamp: pd.Timestamp) -> None:
        """Convert filled order to position."""
        current_fee = self.config.TARGET_TRANSACTION_FEE # Use target fee for backtest
        
        # Max value per position
        max_pos_value = self.balance / max(1, (self.max_holding_count - len(self.open_positions)))
        
        if order["type"] == "buy":
            cost_per_share = fill_price * (1 + current_fee)
            quantity = int(max_pos_value / cost_per_share)
            if quantity > 0:
                trade_value = quantity * fill_price
                total_cost = quantity * cost_per_share
                self.balance -= total_cost
                self.open_positions[order["ticker"]] = {
                    "type": "long",
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "stop_loss": order["stop_loss"]
                }
                self.orders_filled += 1
                self.ledger.append(self._create_ledger_entry(
                    timestamp, order["ticker"], "buy", fill_price, quantity, trade_value, self.balance
                ))
        else: # short
            # Note: Proceeds from shorting are added to balance
            quantity = int((max_pos_value * (1 - current_fee)) / fill_price)
            if quantity > 0:
                trade_value = quantity * fill_price
                proceeds = trade_value * (1 - current_fee)
                self.balance += proceeds
                self.open_positions[order["ticker"]] = {
                    "type": "short",
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "stop_loss": order["stop_loss"]
                }
                self.orders_filled += 1
                self.ledger.append(self._create_ledger_entry(
                    timestamp, order["ticker"], "sell", fill_price, quantity, trade_value, self.balance
                ))

    def _liquidate(self, ticker: str, exit_price: float, timestamp: pd.Timestamp, reason: str = "manual") -> None:
        """Close an open position."""
        if ticker not in self.open_positions:
            return
            
        pos = self.open_positions.pop(ticker)
        current_fee = self.config.TARGET_TRANSACTION_FEE
        
        if pos["type"] == "long":
            exit_value = pos["quantity"] * exit_price * (1 - current_fee)
            pnl = exit_value - (pos["quantity"] * pos["entry_price"])
            pnl_pct = (exit_price * (1 - current_fee) - pos["entry_price"]) / pos["entry_price"]
            self.balance += exit_value
        else: # short
            exit_cost = pos["quantity"] * exit_price * (1 + current_fee)
            entry_proceeds = pos["quantity"] * pos["entry_price"]
            pnl = (entry_proceeds * (1 - current_fee)) - exit_cost
            pnl_pct = (pos["entry_price"] * (1 - current_fee) - exit_price * (1 + current_fee)) / pos["entry_price"]
            self.balance -= exit_cost
            
        self.ledger.append(self._create_ledger_entry(
            timestamp, ticker, reason, exit_price, pos["quantity"], pos["quantity"] * exit_price, self.balance, pnl, pnl_pct
        ))

    def _record_equity(self, timestamp: pd.Timestamp, prices_by_ticker: Dict[str, PricesDf]) -> None:
        """Record fund net worth."""
        holdings_value = 0.0
        for ticker, pos in self.open_positions.items():
            if ticker in prices_by_ticker:
                candle = prices_by_ticker[ticker][prices_by_ticker[ticker]["date"] == timestamp]
                if not candle.empty:
                    current_price = candle.iloc[0]["raw_close"]
                    if pos["type"] == "long":
                        holdings_value += pos["quantity"] * current_price
                    else: # short
                        holdings_value -= pos["quantity"] * current_price
        
        total_equity = self.balance + holdings_value
        self.equity_curve.append({
            "date": timestamp,
            "cash": self.balance,
            "holdings_value": holdings_value,
            "total_equity": total_equity,
            "num_positions": len(self.open_positions),
            "num_orders": len(self.active_limit_orders)
        })

    def run_backtest(self, start_date: str, end_date: str) -> BacktestResults:
        """Run full backtest simulation."""
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        # Load prices
        prices = self.ingest.read_prices(self.tickers)
        if prices.empty:
            raise ValueError("No price data found for backtest.")
            
        prices_by_ticker: Dict[str, PricesDf] = {}
        for ticker, df in prices.groupby("ticker"):
            prices_by_ticker[ticker] = PricesDf(df.copy()).prep_data().sort_values("date")
            
        # Get all predictions
        preds_all = self.infer.predict_timespan(
            start=start_ts, end=end_ts, tickers=self.tickers, buffer_days=30
        )
        preds_by_ts = {ts: df for ts, df in preds_all.groupby("date")} if not preds_all.empty else {}
        
        # Get unique timestamps in range
        in_range = prices[(prices["date"] >= start_ts) & (prices["date"] <= end_ts)]
        timestamps = sorted(in_range["date"].unique())
        
        for ts in tqdm(timestamps, desc="Backtesting"):
            ts_dt = pd.to_datetime(ts)
            self._run_timestamp(ts_dt, prices_by_ticker, preds_by_ts.get(ts_dt))
            
        # Close remaining positions
        if self.open_positions:
            final_ts = timestamps[-1]
            for ticker in list(self.open_positions.keys()):
                ticker_df = prices_by_ticker[ticker]
                last_price = ticker_df[ticker_df["date"] == final_ts].iloc[0]["raw_close"]
                self._liquidate(ticker, last_price, final_ts, reason="close_all")
                
        return self._calculate_results(start_date, end_date)

    def _calculate_results(self, start_date: str, end_date: str) -> BacktestResults:
        """Aggregate metrics into BacktestResults."""
        total_return = self.balance - self.initial_balance
        winning = len([l for l in self.ledger if l.get("pnl", 0) > 0])
        losing = len([l for l in self.ledger if l.get("pnl", 0) < 0])
        
        # Max Drawdown
        max_drawdown = 0.0
        if self.equity_curve:
            equities = [e["total_equity"] for e in self.equity_curve]
            peak = equities[0]
            for eq in equities:
                if eq > peak:
                    peak = eq
                drawdown = (peak - eq) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
                
        return BacktestResults(
            ledger=self.ledger,
            final_balance=self.balance,
            total_trades=winning + losing,
            winning_trades=winning,
            losing_trades=losing,
            total_return=total_return,
            orders_placed=self.orders_placed,
            orders_filled=self.orders_filled,
            initial_balance=self.initial_balance,
            max_drawdown=max_drawdown,
            equity_curve=self.equity_curve
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

