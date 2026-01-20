"""
Optimized simulation module for backtesting with configurable thresholds.
"""
from typing import Dict, List, Optional, Literal, Union, TypedDict, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

from shared.config import get_config
from shared.logging_config import get_logger
from eval.backtest import MockFund, BacktestResults
from training.inference import Infer
from training.data_ingestion import PricesDf

logger = get_logger(__name__)

# --- Type Definitions ---
PrioritizationType = Literal["score", "signal"]
OrderSide = Literal["buy", "sell"]

class CandleData(TypedDict):
    """Minimal schema for cached price data."""
    open: float
    high: float
    low: float
    close: float

# Lookup structure: [Timestamp][Ticker] -> CandleData
PriceLookup = Dict[pd.Timestamp, Dict[str, CandleData]]


class FastMockFund(MockFund):
    """
    High-performance subclass of MockFund with configurable trading thresholds.
    
    Optimizations:
    - Pre-indexes price data into a nested dictionary for O(1) access.
    - Removes repeated DataFrame filtering inside the main loop.
    """

    def __init__(
        self,
        infer: Optional[Infer] = None,
        initial_balance: float = 10000.0,
        tickers: Optional[List[str]] = None,
        max_holding_count: int = 5,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
        prioritize_by: PrioritizationType = "score",
    ):
        super().__init__(infer, initial_balance, tickers, max_holding_count)
        self.buy_threshold: float = buy_threshold
        self.sell_threshold: float = sell_threshold
        self.prioritize_by: PrioritizationType = prioritize_by
        
        # Cache for O(1) price access
        self.price_cache: PriceLookup = {}

    def _pre_process_prices(self, prices_df: pd.DataFrame) -> None:
        """
        Convert DataFrame to a nested dictionary for O(1) access during simulation.
        
        Structure: self.price_cache[timestamp][ticker] = {open, high, low, close}
        """
        logger.info("Pre-indexing price data for performance...")
        
        # Ensure date is standardized
        if not pd.api.types.is_datetime64_any_dtype(prices_df['date']):
            prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # Reset cache
        self.price_cache = {}

        # Iterate efficiently using itertuples (faster than iterrows)
        # We assume columns: date, ticker, open, high, low, raw_close/close
        for row in tqdm(prices_df.itertuples(index=False), total=len(prices_df), desc="Indexing Prices"):
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

    def _run_timestamp(
        self,
        timestamp: pd.Timestamp,
        preds: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Overridden step function using cached price data and configurable thresholds.
        """
        # 0. Fast Price Lookup (O(1))
        current_prices: Dict[str, CandleData] = self.price_cache.get(timestamp, {})
        if not current_prices:
            return

        # 1. Update TTL on Active Orders
        # Copy list to modify safe during iteration
        for order in self.active_limit_orders[:]:
            order["ttl"] -= 1
            if order["ttl"] <= 0:
                self.active_limit_orders.remove(order)

        # 2. Check Fills/Stops
        # Identify all relevant tickers: active positions + active orders
        active_order_tickers = {o["ticker"] for o in self.active_limit_orders}
        open_position_tickers = set(self.open_positions.keys())
        relevant_tickers: Set[str] = active_order_tickers.union(open_position_tickers)
        
        for ticker in relevant_tickers:
            if ticker not in current_prices:
                continue
            
            candle = current_prices[ticker]
            high = candle["high"]
            low = candle["low"]

            # 2a. Stop Loss Logic (Existing Positions)
            if ticker in self.open_positions:
                pos = self.open_positions[ticker]
                if low <= pos["stop_loss"]:
                    # Pessimistic execution: exit at stop price or high if gap occurred
                    exit_price = min(pos["stop_loss"], high)
                    self._liquidate(ticker, exit_price, timestamp, reason="stop_loss")

            # 2b. Limit Order Fill Logic (Pending Orders)
            elif ticker not in self.open_positions:
                # Get orders for this ticker
                # Optimization: In a very high frequency loop, this list comp could be optimized 
                # by maintaining a dict of orders by ticker, but list comp is fine for <100 orders.
                ticker_orders = [o for o in self.active_limit_orders if o["ticker"] == ticker]
                
                for order in ticker_orders:
                    fill_price = order["price"]
                    filled = False
                    
                    if order["type"] == "buy":
                        if low <= fill_price:
                            filled = True
                            # If gap down, we buy at open/high? 
                            # Standard conservative fill: limit price or better.
                            # Here we assume we get filled at our limit unless gap logic is added.
                            fill_price = min(fill_price, high)
                            
                    elif order["type"] == "sell":
                        if high >= fill_price:
                            filled = True
                            fill_price = max(fill_price, low)
                    
                    if filled:
                        if order["type"] == "buy":
                            self._execute_fill(order, fill_price, timestamp)
                        else:
                            # Sell order implies short entry or manual exit logic if extended
                            self._liquidate(ticker, fill_price, timestamp, reason="limit_fill_sell")
                        
                        # Remove filled order
                        if order in self.active_limit_orders:
                            self.active_limit_orders.remove(order)

        # 3. Process Predictions (New Signal Handling)
        if preds is not None and not preds.empty:
            pending_tickers = {o["ticker"] for o in self.active_limit_orders}
            
            # 3a. Cancellations based on configurable thresholds
            for row in preds.itertuples(index=False):
                ticker: str = row.ticker
                signal: float = row.signal
                
                if ticker in pending_tickers:
                    # Find specific order to potentially cancel
                    # (Assuming 1 active order per ticker for simplicity)
                    order_iter = (o for o in self.active_limit_orders if o["ticker"] == ticker)
                    order = next(order_iter, None)
                    
                    if order:
                        cancel_buy = (order["type"] == "buy" and signal <= self.buy_threshold)
                        cancel_sell = (order["type"] == "sell" and signal >= self.sell_threshold)
                        
                        if cancel_buy or cancel_sell:
                            self.active_limit_orders.remove(order)
                            pending_tickers.remove(ticker) # Update local set

            # 3b. New Order Generation
            new_candidates = []
            
            # Filter candidates first
            for row in preds.itertuples(index=False):
                ticker = row.ticker
                signal = row.signal
                
                # Buy Logic
                if ticker not in self.open_positions and ticker not in pending_tickers:
                    if signal > self.buy_threshold:
                        new_candidates.append(row)
                
                # Sell Logic (Shorting or entering Sell Limit)
                elif ticker in self.open_positions and ticker not in pending_tickers:
                    if signal < self.sell_threshold:
                        new_candidates.append(row)

            if new_candidates:
                candidate_df = pd.DataFrame(new_candidates)
                
                # Calculate Limit Prices efficiently
                # Note: We rely on _calculate_limit_price being pure; 
                # implementing vectorized calculation here would be faster but more complex.
                candidate_df["limit_buy"] = candidate_df.apply(
                    lambda r: self._calculate_limit_price(r.close, r.limit_offset_action, "buy"), axis=1
                )
                candidate_df["limit_sell"] = candidate_df.apply(
                    lambda r: self._calculate_limit_price(r.close, r.limit_offset_action, "sell"), axis=1
                )

                # Prioritization Logic
                if self.prioritize_by == "score":
                    # Score = Signal Strength * Predicted Discount
                    # Using numpy for vectorized `where`
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

                available_slots = self.max_holding_count - len(self.open_positions)

                # Issue Orders
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

    def run_backtest(self, start_date: str, end_date: str) -> BacktestResults:
        """
        Run the optimized backtest simulation.
        """
        logger.info(f"Starting FAST backtest: {start_date} to {end_date}")
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        # 1. Ingest Prices
        prices = self.ingest.read_prices(self.tickers)
        if prices.empty:
            raise ValueError("No price data found for backtest.")
        
        # 2. Build Optimization Cache
        # Filter strictly for date range to minimize memory usage
        mask = (prices["date"] >= start_ts) & (prices["date"] <= end_ts)
        prices_subset = prices[mask].copy()
        self._pre_process_prices(prices_subset)
        
        # 3. Batch Predictions
        preds_all = self.infer.predict_timespan(
            start=start_ts, end=end_ts, tickers=self.tickers, buffer_days=30
        )
        # Group by date for easy lookup
        preds_by_ts = {ts: df for ts, df in preds_all.groupby("date")} if not preds_all.empty else {}
        
        # 4. Main Event Loop
        timestamps = sorted(self.price_cache.keys())
        
        for ts in tqdm(timestamps, desc="Simulating (Fast)"):
            # We pass empty dict for prices_lookup as we use self.price_cache internally
            self._run_timestamp(ts, preds_by_ts.get(ts))
            
        # 5. Final Cleanup (Close all positions)
        if self.open_positions:
            final_ts = timestamps[-1]
            current_prices = self.price_cache.get(final_ts, {})
            
            for ticker in list(self.open_positions.keys()):
                if ticker in current_prices:
                    close_price = current_prices[ticker]["close"]
                    self._liquidate(ticker, close_price, final_ts, reason="close_all")

        return self._calculate_results(start_date, end_date)