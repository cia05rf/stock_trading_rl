"""
Backtesting module for stock trading strategies.

This module provides tools for simulating trading strategies on historical
data to evaluate model performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, TypedDict, Set
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from shared.config import get_config
from shared.logging_config import get_logger
from training.inference import Infer
from training.data_ingestion import Ingestion, PricesDf

logger = get_logger(__name__)

# --- Type Definitions ---
PrioritizationType = Literal["score", "signal"]

class CandleData(TypedDict):
    """Minimal schema for cached price data."""
    open: float
    high: float
    low: float
    close: float

# Lookup structure: [Timestamp][Ticker] -> CandleData
PriceLookup = Dict[pd.Timestamp, Dict[str, CandleData]]


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
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
        prioritize_by: PrioritizationType = "score",
    ):
        self.config = get_config()
        self.infer = infer or Infer()
        self.ingest = Ingestion()
        self.balance = initial_balance or self.config.INITIAL_BALANCE
        self.initial_balance = self.balance
        self.tickers = tickers
        self.max_holding_count = max_holding_count
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prioritize_by = prioritize_by
        
        # State
        self.active_limit_orders: List[Dict] = []
        self.open_positions: Dict[str, Dict] = {} # ticker -> position
        
        # Cache for vectorized inference & O(1) price access
        self.feature_cache: Dict[pd.Timestamp, torch.Tensor] = {}
        self.price_cache: PriceLookup = {}
        self.ticker_index_map: Dict[str, int] = {}
        self.ordered_tickers: List[str] = []
        
        # Stats
        self.orders_placed = 0
        self.orders_filled = 0
        self.ledger: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        logger.info(f"MockFund (Limit Order) initialized with balance: £{self.balance:,.2f}")

    def _preload_data(self, start_date: str, end_date: str) -> None:
        """
        Pre-load and vectorize feature data for all tickers in the date range.
        Stores features in self.feature_cache[timestamp] and prices in self.price_cache.
        """
        logger.info(f"Pre-loading data for {len(self.tickers) if self.tickers else 'ALL'} tickers...")
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        # 1. Load and prepare all price data
        prices = self.ingest.read_prices(self.tickers)
        if prices.empty:
            raise ValueError("No price data found for pre-loading.")
            
        # Standardize tickers and build index map
        self.ordered_tickers = sorted(prices["ticker"].unique())
        self.ticker_index_map = {ticker: i for i, ticker in enumerate(self.ordered_tickers)}
        num_tickers = len(self.ordered_tickers)
        
        # 2. Build Price Cache (O(1) access)
        logger.info("Indexing prices for O(1) access...")
        # Filter strictly for date range to minimize memory usage
        prices_subset = prices[(prices["date"] >= start_ts) & (prices["date"] <= end_ts)]
        
        for row in tqdm(prices_subset.itertuples(index=False), total=len(prices_subset), desc="Indexing Prices"):
            ts: pd.Timestamp = row.date
            ticker: str = row.ticker
            
            if ts not in self.price_cache:
                self.price_cache[ts] = {}
            
            # Map raw_close to close if strictly needed, otherwise use close
            close_price = getattr(row, "raw_close", getattr(row, "close", 0.0))
            
            self.price_cache[ts][ticker] = {
                'open': getattr(row, "open", 0.0), 
                'high': getattr(row, "high", 0.0), 
                'low': getattr(row, "low", 0.0), 
                'close': close_price
            }
        
        # 3. Generate observations for each ticker and timestamp
        # We'll use a temporary dict to store observations: [timestamp][ticker_idx] = obs_vector
        obs_by_ts: Dict[pd.Timestamp, Dict[int, np.ndarray]] = {}
        
        # Use a temporary environment to generate observations consistently
        temp_env = self.infer.env
        
        for ticker, df_t in tqdm(prices.groupby("ticker"), desc="Vectorizing Features"):
            ticker_idx = self.ticker_index_map[ticker]
            
            # Prepare data same as environment and reset index for positional iloc access
            prepared = PricesDf(df_t.copy()).prep_data().sort_values("date").reset_index(drop=True)
            
            temp_env.df = prepared
            temp_env._precompute_features()
            
            if hasattr(temp_env, 'ticker_to_id') and ticker in temp_env.ticker_to_id:
                temp_env.current_stock_id = temp_env.ticker_to_id[ticker]
            
            # Find indices for the requested range
            mask = (prepared["date"] >= start_ts) & (prepared["date"] <= end_ts)
            target_indices = prepared.index[mask]
            
            for idx in target_indices:
                ts = prepared.iloc[idx]["date"]
                temp_env.current_step = idx
                obs = temp_env._next_observation()
                
                if ts not in obs_by_ts:
                    obs_by_ts[ts] = {}
                obs_by_ts[ts][ticker_idx] = obs

        # 4. Convert to batched PyTorch tensors
        obs_dim = temp_env.observation_space.shape[0]
        
        for ts, ticker_obs in tqdm(obs_by_ts.items(), desc="Creating Feature Tensors"):
            batch_tensor = torch.zeros((num_tickers, obs_dim), dtype=torch.float32)
            for ticker_idx, obs in ticker_obs.items():
                batch_tensor[ticker_idx] = torch.from_numpy(obs)
            self.feature_cache[ts] = batch_tensor
            
        logger.info(f"Pre-loaded {len(self.feature_cache)} timestamps for {num_tickers} tickers.")

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
        action_type: str, # buy/sell
        price: float,
        quantity: int,
        trade_value: float,
        balance_pre: float,
        balance_post: float,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
    ) -> Dict:
        """Create a ledger entry."""
        return {
            "date": pd.to_datetime(date),
            "ticker": ticker,
            "action": action,
            "action_type": action_type,
            "price": price,
            "quantity": quantity,
            "trade_value": trade_value,
            "balance_pre_trade": balance_pre,
            "balance_post_trade": balance_post,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }

    def _run_timestamp(
        self,
        timestamp: pd.Timestamp,
    ) -> None:
        """Run simulation for a single timestamp (15m candle)."""
        
        # 0. Fast Price Lookup (O(1))
        current_prices: Dict[str, CandleData] = self.price_cache.get(timestamp, {})
        if not current_prices:
            return

        # 1. Update Existing Limit Orders (TTL)
        for order in self.active_limit_orders[:]:
            order["ttl"] -= 1
            if order["ttl"] <= 0:
                self.active_limit_orders.remove(order)

        # 2. Check for Fills and Stop Losses using current candle
        active_order_tickers = {o["ticker"] for o in self.active_limit_orders}
        open_position_tickers = set(self.open_positions.keys())
        relevant_tickers: Set[str] = active_order_tickers.union(open_position_tickers)
        
        for ticker in relevant_tickers:
            if ticker not in current_prices:
                continue
            
            candle = current_prices[ticker]
            high = candle["high"]
            low = candle["low"]
            
            # 2a. Check Stop Loss for Open Positions
            if ticker in self.open_positions:
                pos = self.open_positions[ticker]
                if low <= pos["stop_loss"]:
                    exit_price = min(pos["stop_loss"], high)
                    self._liquidate(ticker, exit_price, timestamp, reason="stop_loss")
            
            # 2b. Check Fills for Active Limit Orders
            elif ticker not in self.open_positions:
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

        # 3. Process New Predictions (Batched Inference)
        feature_tensor = self.feature_cache.get(timestamp)
        if feature_tensor is not None:
            actions = self.infer.get_batch_action(feature_tensor)
            
            pending_tickers = {o["ticker"] for o in self.active_limit_orders}
            new_candidates = []
            
            for i, ticker in enumerate(self.ordered_tickers):
                # action: [signal, limit_offset, stop_loss]
                action = actions[i]
                signal = action[0]
                
                # 3a. Handle Cancellations
                if ticker in pending_tickers:
                    order = next(o for o in self.active_limit_orders if o["ticker"] == ticker)
                    cancel_buy = (order["type"] == "buy" and signal <= self.buy_threshold)
                    cancel_sell = (order["type"] == "sell" and signal >= self.sell_threshold)
                    
                    if cancel_buy or cancel_sell:
                        self.active_limit_orders.remove(order)
                        pending_tickers.remove(ticker)
                        continue
                
                # 3b. Identify New Candidates
                if ticker not in self.open_positions and ticker not in pending_tickers:
                    if signal > self.buy_threshold:
                        candle = current_prices.get(ticker)
                        if candle:
                            new_candidates.append({
                                "ticker": ticker,
                                "signal": signal,
                                "close": candle["close"],
                                "limit_offset_action": action[1],
                                "stop_loss_action": action[2]
                            })
                
                elif ticker in self.open_positions and ticker not in pending_tickers:
                    if signal < self.sell_threshold:
                        candle = current_prices.get(ticker)
                        if candle:
                            new_candidates.append({
                                "ticker": ticker,
                                "signal": signal,
                                "close": candle["close"],
                                "limit_offset_action": action[1],
                                "stop_loss_action": action[2]
                            })

            # 3c. Rank and Issue Orders
            if new_candidates:
                candidate_df = pd.DataFrame(new_candidates)
                
                # Calculate Limit Prices
                candidate_df["limit_buy"] = candidate_df.apply(
                    lambda r: self._calculate_limit_price(r.close, r.limit_offset_action, "buy"), axis=1
                )
                candidate_df["limit_sell"] = candidate_df.apply(
                    lambda r: self._calculate_limit_price(r.close, r.limit_offset_action, "sell"), axis=1
                )

                # Prioritization Logic
                if self.prioritize_by == "score":
                    # Score = Signal Strength * Predicted Discount
                    candidate_df["discount_pct"] = np.where(
                        candidate_df["signal"] > 0,
                        (candidate_df["close"] - candidate_df["limit_buy"]) / candidate_df["close"],
                        (candidate_df["limit_sell"] - candidate_df["close"]) / candidate_df["close"]
                    )
                    candidate_df["ranking_score"] = candidate_df["signal"].abs() * candidate_df["discount_pct"]
                else:
                    # Score = Pure Signal Strength
                    candidate_df["ranking_score"] = candidate_df["signal"].abs()

                # Sort by score descending
                candidate_df = candidate_df.sort_values("ranking_score", ascending=False)
                
                available_slots = (
                    self.max_holding_count - len(self.open_positions)
                )
                
                for row in candidate_df.itertuples(index=False):
                    ticker = row.ticker
                    if row.signal > 0: # Buy Signal
                        if available_slots > 0:
                            stop_loss = self._calculate_stop_price(row.limit_buy, row.stop_loss_action)
                            self.active_limit_orders.append({
                                "ticker": ticker,
                                "type": "buy",
                                "price": row.limit_buy,
                                "stop_loss": stop_loss,
                                "ttl": self.config.ORDER_TTL,
                                "timestamp": timestamp
                            })
                            self.orders_placed += 1
                            available_slots -= 1
                    else: # Sell Signal
                        self.active_limit_orders.append({
                            "ticker": ticker,
                            "type": "sell",
                            "price": row.limit_sell,
                            "ttl": self.config.ORDER_TTL,
                            "timestamp": timestamp
                        })
                        self.orders_placed += 1

        # 4. Record Equity Snapshot
        self._record_equity(timestamp)

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
                balance_pre = self.balance
                self.balance -= total_cost
                self.open_positions[order["ticker"]] = {
                    "type": "long",
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "stop_loss": order["stop_loss"]
                }
                self.orders_filled += 1
                self.ledger.append(self._create_ledger_entry(
                    timestamp, order["ticker"], "buy", "buy", fill_price, quantity, trade_value, balance_pre, self.balance
                ))
        else: # short
            # Note: Proceeds from shorting are added to balance
            quantity = int((max_pos_value * (1 - current_fee)) / fill_price)
            if quantity > 0:
                trade_value = quantity * fill_price
                proceeds = trade_value * (1 - current_fee)
                balance_pre = self.balance
                self.balance += proceeds
                self.open_positions[order["ticker"]] = {
                    "type": "short",
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "stop_loss": 0.0 # Stop loss for short? 
                }
                self.orders_filled += 1
                self.ledger.append(self._create_ledger_entry(
                    timestamp, order["ticker"], "sell", "sell", fill_price, quantity, trade_value, balance_pre, self.balance
                ))

    def _liquidate(self, ticker: str, exit_price: float, timestamp: pd.Timestamp, reason: str = "manual") -> None:
        """Close an open position."""
        if ticker not in self.open_positions:
            return
            
        pos = self.open_positions.pop(ticker)
        current_fee = self.config.TARGET_TRANSACTION_FEE
        balance_pre = self.balance
        
        if pos["type"] == "long":
            exit_value = pos["quantity"] * exit_price * (1 - current_fee)
            pnl = exit_value - (pos["quantity"] * pos["entry_price"])
            pnl_pct = (exit_price * (1 - current_fee) - pos["entry_price"]) / pos["entry_price"]
            self.balance += exit_value
            action_type = "sell"
        else: # short
            exit_cost = pos["quantity"] * exit_price * (1 + current_fee)
            entry_proceeds = pos["quantity"] * pos["entry_price"]
            pnl = (entry_proceeds * (1 - current_fee)) - exit_cost
            pnl_pct = (pos["entry_price"] * (1 - current_fee) - exit_price * (1 + current_fee)) / pos["entry_price"]
            self.balance -= exit_cost
            action_type = "buy"
            
        self.ledger.append(self._create_ledger_entry(
            timestamp, ticker, reason, action_type, exit_price, pos["quantity"], pos["quantity"] * exit_price, balance_pre, self.balance, pnl, pnl_pct
        ))

    def _record_equity(self, timestamp: pd.Timestamp) -> None:
        """Record fund net worth."""
        holdings_value = 0.0
        current_prices = self.price_cache.get(timestamp, {})
        
        for ticker, pos in self.open_positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]["close"]
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
        
        # 1. Pre-load data (Vectorized features & Price cache)
        self._preload_data(start_date, end_date)
        
        # 2. Main Event Loop
        timestamps = sorted(self.price_cache.keys())
        
        for ts in tqdm(timestamps, desc="Backtesting"):
            self._run_timestamp(ts)
            
        # 3. Close remaining positions
        if self.open_positions:
            final_ts = timestamps[-1]
            current_prices = self.price_cache.get(final_ts, {})
            
            for ticker in list(self.open_positions.keys()):
                if ticker in current_prices:
                    close_price = current_prices[ticker]["close"]
                    self._liquidate(ticker, close_price, final_ts, reason="close_all")
                
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
    buy_threshold: float = 0.3,
    sell_threshold: float = -0.3,
    prioritize_by: PrioritizationType = "score",
) -> BacktestResults:
    """
    Convenience function to run a backtest.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_balance: Starting capital
        model_path: Path to model (optional)
        output_path: Path to save results CSV (optional)
        buy_threshold: Signal strength required to buy
        sell_threshold: Signal strength required to sell
        prioritize_by: How to prioritize signals ("score" or "signal")
    
    Returns:
        BacktestResults
    """
    infer = Infer(model_path=model_path)
    fund = MockFund(
        infer, 
        initial_balance=initial_balance,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        prioritize_by=prioritize_by
    )
    results = fund.run_backtest(start_date, end_date)
    
    if output_path:
        results.to_dataframe().to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    return results

