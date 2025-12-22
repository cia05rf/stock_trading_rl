"""
Stock trading environment for reinforcement learning.

This module provides a Gymnasium-compatible environment for training
stock trading agents using reinforcement learning algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from shared.logging_config import get_logger
from training.data_ingestion import Ingestion, PricesDf
from shared.config import get_config

logger = get_logger(__name__)


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI Gym / Gymnasium.

    This environment simulates stock trading with:
    - Multiple action types (buy/sell with different proportions, hold)
    - Transaction costs and stamp duty
    - Configurable reward functions
    - Train/test mode with data splitting

    Attributes:
        metadata: dict - Environment metadata
        ingestion: Ingestion - Data ingestion class
        mode: str - Current mode ('train' or 'test')
        tickers: list - List of tickers for current mode
        action_space: spaces.Discrete - Action space
        observation_space: spaces.Box - Observation space
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        mode: str = "train",
        test_train_split: float = 0.8,
        initial_balance: float = 10000,
        window_size: int = 10,
        transaction_cost_pct: Optional[float] = None,  # Deprecated, use config
        stamp_duty_pct: float = 0.005,
        trade_penalty: float = 0.01,
        risk_factor: float = 0.5,
        sharpe_weight: float = 1.0,
        gain_weight: float = 1.0,
        loss_weight: float = 1.0,
        hold_reward: float = 0.02,
        aritificial_decay: float = 0.0001,
        risk_aversion: float = 0.5,
        seed: Optional[int] = None,
        ticker_limit: Optional[int] = None,
        min_date: str = "2018-01-01",
        action_space_type: str = "discrete",
    ) -> None:
        """
        Initialize the stock trading environment.

        Args:
            mode: 'train' or 'test' mode
            test_train_split: Fraction of data for training
            initial_balance: Starting balance in currency units
            window_size: Number of timesteps in observation window
            transaction_cost_pct: Transaction cost percentage
            stamp_duty_pct: Stamp duty percentage
            trade_penalty: Penalty for frequent trading
            risk_factor: Risk factor for reward calculation
            sharpe_weight: Weight for Sharpe ratio in reward
            gain_weight: Weight for gains in reward
            loss_weight: Weight for losses in reward
            hold_reward: Reward for holding position
            aritificial_decay: Decay to prevent risk aversion
            risk_aversion: Risk aversion factor
            seed: Random seed for reproducibility
            ticker_limit: Maximum number of tickers to use
            min_date: Minimum date for data filtering
        """
        super().__init__()
        
        logger.info("\n" + "=" * 50)
        logger.info("INITIALIZING STOCK TRADING ENVIRONMENT")
        logger.info("=" * 50)

        # Initialize data ingestion
        self.ingestion = Ingestion(min_date=min_date)
        self.config = get_config()
        # Store parameters
        self.ticker_count = 0
        self.done = False
        self.current_step = 0
        self.mode = mode
        # Note: Refactored to always use Discrete(3) action space
        # action_space_type parameter is kept for backward compatibility but ignored
        if action_space_type and str(action_space_type).strip().lower() == "continuous":
            logger.warning(
                "Continuous action space is no longer supported. "
                "Using Discrete(3) action space (Hold/Buy/Sell)."
            )
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.obs_window_size = self.config.OBS_WINDOW_SIZE
        # Use TRANSACTION_FEE from config (transaction_cost_pct is deprecated)
        self.transaction_cost_pct = self.config.TRANSACTION_FEE
        self.stamp_duty_pct = stamp_duty_pct
        self.trade_penalty = trade_penalty
        self.risk_factor = risk_factor
        self.sharpe_weight = sharpe_weight
        self.gain_weight = gain_weight
        self.loss_weight = loss_weight
        self.hold_reward = hold_reward
        self.aritificial_decay = aritificial_decay
        self.risk_aversion = risk_aversion
        self.recent_net_worths: deque = deque(maxlen=50)
        self.recent_run_mean_return: float = 0.0
        
        # Progress tracking
        self.progress_bar = tqdm(position=0, leave=True, desc="Processing tickers")
        
        # Holdings tracking
        self.avco = None
        self.holdings: List[Tuple] = []  # (price, avco, change, shares_held)
        self.returns: List[float] = []
        self.net_worth_history: List[float] = []
        
        # Track previous values for reward calculation
        self.prev_balance = initial_balance
        self.prev_shares = 0
        self.prev_price = None

        # Split data into training and testing sets
        tickers = self.ingestion.read_tickers()
        if ticker_limit:
            tickers = tickers.iloc[:ticker_limit]

        # Create ticker-to-id mapping (using the id column from read_tickers)
        self.ticker_to_id = dict(zip(tickers["ticker"], tickers["id"]))
        self.num_stocks = len(self.ticker_to_id)
        self.current_stock_id = 0  # Will be updated when loading tickers

        train_tickers_df = tickers.sample(frac=test_train_split, random_state=seed)
        self.tickers_train = train_tickers_df["ticker"].to_list()
        self.tickers_test = tickers.drop(train_tickers_df.index)["ticker"].to_list()

        # Initialize values and set mode
        self._reset_values()
        self.set_mode(mode)
        self.timesteps = self.epoch_timesteps()

        assert len(self.tickers) > 0, "No tickers found"

        # New action space: Discrete(3) - Hold, Buy, Sell
        # Action 0: Hold (do nothing)
        # Action 1: Buy (convert 100% of cash to stock, minus fees)
        # Action 2: Sell (convert 100% of shares to cash, minus fees)
        self.action_space = spaces.Discrete(3)
        
        # Track invalid actions for reward calculation
        self.is_invalid_action = False
        
        # Define observation space
        # Features per timestep: log_return, volume_change, volatility (ATR/Price), 
        # RSI (scaled), MACD (normalized), time_sin, time_cos, position_ratio, unrealized_pnl_pct
        # Total: 9 features per timestep
        n_features_per_step = 9
        obs_shape = (self.obs_window_size * n_features_per_step,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )
        logger.info(f"Observation Space: {self.observation_space}")

        # Final initialization
        self._reset_values()
        self.full_reset = False

    def epoch_timesteps(self) -> int:
        """Returns the number of timesteps in an epoch."""
        return self.ingestion.count_timesteps(self.tickers)

    def set_mode(self, mode: str) -> None:
        """
        Set the environment mode (train or test).

        Args:
            mode: 'train' or 'test'
        """
        if mode == "train":
            self.tickers = self.tickers_train
        elif mode == "test":
            self.tickers = self.tickers_test
        else:
            raise ValueError("Mode must be either 'train' or 'test'")
        
        logger.info(f"Processing {len(self.tickers):,} tickers in {mode} mode")
        self._load_next_df(0)
        self._reset_values()
        self.progress_bar.reset()
        self.progress_bar.total = len(self.tickers)

    def _data_loader(self, ticker: str) -> PricesDf:
        """Load and prepare data for a ticker."""
        df = self.ingestion.read_prices(ticker).prep_data()
        # Handle empty dataframe (e.g., corrupted ticker data)
        if len(df) == 0:
            logger.warning(f"Empty dataframe for ticker {ticker} after prep_data(). Skipping.")
            return df
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # Check again after dropna in case all rows were dropped
        if len(df) == 0:
            logger.warning(f"Empty dataframe for ticker {ticker} after dropna(). Skipping.")
            return df
        assert not df.isnull().values.any(), f"Data for {ticker} contains NaNs!"
        return df

    def _load_next_df(self, i: int) -> None:
        """Load the next ticker's data and pre-compute all static features."""
        self.progress_bar.update(1)
        ticker = self.tickers[i]
        self.df = self._data_loader(ticker)
        self.n_steps = len(self.df)
        self.current_step = 0
        # Update current stock_id
        self.current_stock_id = self.ticker_to_id.get(ticker, 0)
        
        # Handle empty dataframe (corrupted ticker data)
        if len(self.df) == 0:
            logger.warning(f"Ticker {ticker} has no data. Will skip to next ticker.")
            return
        
        logger.debug(f"Loaded {self.n_steps:,} rows for {ticker} (stock_id={self.current_stock_id})")
        self._reset_values()
        
        # Pre-compute all static features for performance
        self._precompute_features()
        
        # Initialize previous_price to first price for proper stock return calculation
        if len(self.df) > 0:
            self.previous_price = self.df.iloc[0]["raw_close"]
    
    def _precompute_features(self) -> None:
        """
        Pre-compute all static features as NumPy arrays for fast access.
        This is called once when data is loaded, not on every step.
        Uses vectorized NumPy operations for maximum performance.
        """
        if len(self.df) == 0:
            # Initialize empty arrays
            self.log_returns = np.array([], dtype=np.float32)
            self.volume_changes = np.array([], dtype=np.float32)
            self.volatilities = np.array([], dtype=np.float32)
            self.rsi_scaled = np.array([], dtype=np.float32)
            self.macd_normalized = np.array([], dtype=np.float32)
            self.time_sin = np.array([], dtype=np.float32)
            self.time_cos = np.array([], dtype=np.float32)
            self.prices_array = np.array([], dtype=np.float32)
            return
        
        n = len(self.df)
        
        # Convert to NumPy arrays for fast access (avoid pandas overhead)
        prices = self.df['raw_close'].values.astype(np.float32)
        volumes = self.df['volume'].values.astype(np.float32)
        highs = self.df['high'].values.astype(np.float32)
        lows = self.df['low'].values.astype(np.float32)
        closes = self.df['close'].values.astype(np.float32)
        
        # 1. Pre-compute Log Returns: log(Price_t / Price_{t-1})
        # Vectorized: shift prices and compute log ratio
        prev_prices = np.roll(prices, 1)
        prev_prices[0] = prices[0]  # First element uses itself
        # Avoid log(0) and division by zero
        price_ratios = np.where(
            (prices > 0) & (prev_prices > 0),
            prices / prev_prices,
            1.0
        )
        self.log_returns = np.log(price_ratios).astype(np.float32)
        self.log_returns[0] = 0.0  # First step has no return
        
        # 2. Pre-compute Volume Change: Log change of volume
        prev_volumes = np.roll(volumes, 1)
        prev_volumes[0] = volumes[0] if len(volumes) > 0 else 1.0
        volume_ratios = np.where(
            (volumes > 0) & (prev_volumes > 0),
            volumes / prev_volumes,
            1.0
        )
        self.volume_changes = np.log(volume_ratios).astype(np.float32)
        self.volume_changes[0] = 0.0
        
        # 3. Pre-compute ATR (Average True Range) - vectorized
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        atr_period = 14
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]
        
        tr1 = highs - lows
        tr2 = np.abs(highs - prev_closes)
        tr3 = np.abs(lows - prev_closes)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using rolling mean (vectorized)
        # Use pandas rolling for simplicity, then convert to numpy
        atr_series = pd.Series(true_range).rolling(window=atr_period, min_periods=1).mean()
        atr_values = atr_series.values.astype(np.float32)
        
        # Normalize ATR by price to get volatility
        self.volatilities = np.where(
            prices > 0,
            atr_values / prices,
            0.0
        ).astype(np.float32)
        
        # 4. Pre-compute RSI and MACD (already in dataframe, just extract)
        if 'rsi_14' in self.df.columns:
            rsi_values = self.df['rsi_14'].fillna(50.0).values.astype(np.float32)
            self.rsi_scaled = (rsi_values / 100.0).astype(np.float32)
        else:
            self.rsi_scaled = np.full(n, 0.5, dtype=np.float32)
        
        # MACD normalized by price
        if 'macd_line' in self.df.columns and 'macd_signal' in self.df.columns:
            macd_line = self.df['macd_line'].fillna(0.0).values.astype(np.float32)
            macd_signal = self.df['macd_signal'].fillna(0.0).values.astype(np.float32)
            macd_diff = macd_line - macd_signal
            self.macd_normalized = np.where(
                prices > 0,
                macd_diff / prices,
                0.0
            ).astype(np.float32)
        else:
            self.macd_normalized = np.zeros(n, dtype=np.float32)
        
        # 5. Pre-compute Time Embeddings: Sine and Cosine of "Time of Day"
        if 'date' in self.df.columns:
            dates = pd.to_datetime(self.df['date'])
            # Convert to fraction of day (0-1)
            hours = dates.dt.hour.values
            minutes = dates.dt.minute.values
            time_of_day = ((hours * 60 + minutes) / (24 * 60)).astype(np.float32)
        else:
            time_of_day = np.full(n, 0.5, dtype=np.float32)
        
        self.time_sin = np.sin(2 * np.pi * time_of_day).astype(np.float32)
        self.time_cos = np.cos(2 * np.pi * time_of_day).astype(np.float32)
        
        # Store prices array for fast access
        self.prices_array = prices

    def _next_observation(self) -> np.ndarray:
        """
        Get the next observation using pre-computed features.
        Only account state is computed on-the-fly as it changes with actions.

        Returns:
            Flattened observation array of shape (obs_window_size * 9,)
        """
        # Ensure features are pre-computed
        if not hasattr(self, 'log_returns') or len(self.log_returns) == 0:
            self._precompute_features()
        
        # Get window indices
        start = max(0, self.current_step - self.obs_window_size + 1)
        end = min(self.current_step + 1, len(self.log_returns))
        
        # Get current price for account state calculation
        current_price = self.prices_array[self.current_step] if self.current_step < len(self.prices_array) else 0.0
        
        # Calculate account state (only dynamic feature)
        position_ratio = 1.0 if self.shares_held > 0 else 0.0
        
        # Unrealized_PnL_Pct: (current_value - cost_basis) / cost_basis
        if self.shares_held > 0 and hasattr(self, 'holdings') and self.holdings:
            # Calculate average cost basis
            total_cost = sum(h[0] * h[2] for h in self.holdings if h[2] > 0)
            total_shares = sum(h[2] for h in self.holdings if h[2] > 0)
            if total_shares > 0:
                avg_cost = total_cost / total_shares
                current_value = self.shares_held * current_price
                cost_basis = self.shares_held * avg_cost
                unrealized_pnl_pct = (
                    (current_value - cost_basis) / cost_basis
                    if cost_basis > 0 else 0.0
                )
            else:
                unrealized_pnl_pct = 0.0
        else:
            unrealized_pnl_pct = 0.0
        
        # Build observation window using pre-computed arrays (vectorized)
        window_size = end - start
        if window_size < self.obs_window_size:
            # Need padding at the beginning
            pad_size = self.obs_window_size - window_size
            # Create padded arrays
            log_returns_window = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                self.log_returns[start:end]
            ])
            volume_changes_window = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                self.volume_changes[start:end]
            ])
            volatilities_window = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                self.volatilities[start:end]
            ])
            rsi_scaled_window = np.concatenate([
                np.full(pad_size, 0.5, dtype=np.float32),
                self.rsi_scaled[start:end]
            ])
            macd_normalized_window = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                self.macd_normalized[start:end]
            ])
            time_sin_window = np.concatenate([
                np.zeros(pad_size, dtype=np.float32),
                self.time_sin[start:end]
            ])
            time_cos_window = np.concatenate([
                np.ones(pad_size, dtype=np.float32),
                self.time_cos[start:end]
            ])
        else:
            # No padding needed
            log_returns_window = self.log_returns[start:end]
            volume_changes_window = self.volume_changes[start:end]
            volatilities_window = self.volatilities[start:end]
            rsi_scaled_window = self.rsi_scaled[start:end]
            macd_normalized_window = self.macd_normalized[start:end]
            time_sin_window = self.time_sin[start:end]
            time_cos_window = self.time_cos[start:end]
        
        # Create account state arrays (same value for all timesteps in window)
        position_ratio_window = np.full(self.obs_window_size, position_ratio, dtype=np.float32)
        unrealized_pnl_pct_window = np.full(self.obs_window_size, unrealized_pnl_pct, dtype=np.float32)
        
        # Stack all features: shape (obs_window_size, 9)
        obs_window = np.column_stack([
            log_returns_window,
            volume_changes_window,
            volatilities_window,
            rsi_scaled_window,
            macd_normalized_window,
            time_sin_window,
            time_cos_window,
            position_ratio_window,
            unrealized_pnl_pct_window,
        ])
        
        # Flatten to 1D: (obs_window_size * 9,)
        obs = obs_window.flatten()
        
        # Handle NaNs and Infs (vectorized)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)
        
        if np.isnan(obs).any() or np.isinf(obs).any():
            raise ValueError("Observation contains NaNs or Infs!")
        
        return obs.astype(np.float32)

    def calculate_reward(self) -> float:
        """
        Calculate reward using Logarithmic Wealth Growth strategy with Conditional Cash Decay.
        
        Formula: 
        1. Log Return: log((current_val + eps) / (prev_val + eps)) * 100
        2. Conditional Cash Decay: CASH_DECAY * cash_ratio (only penalizes cash portion)
        3. Invalid Action Penalty: if applicable
        
        The cash decay is proportional to cash holdings:
        - 100% Cash -> Full decay penalty
        - 0% Cash (fully invested) -> No decay penalty
        This incentivizes the agent to invest rather than hold cash.
        
        Returns:
            Reward value based on logarithmic portfolio growth with conditional decay
        """
        # Calculate Portfolio Value (Cash + Stock_Value)
        current_val = self.balance + (
            self.shares_held * self.current_price 
            if hasattr(self, 'current_price') and self.current_price is not None 
            else 0
        )
        prev_val = self.prev_balance + (
            self.prev_shares * self.prev_price 
            if self.prev_price is not None and self.prev_price > 0 
            else 0
        )
        
        # 1. Log Return (Compound Growth)
        # Add epsilon to avoid log(0)
        reward = np.log((current_val + 1e-8) / (prev_val + 1e-8)) * 100
        
        # 2. Conditional Cash Decay (The "Tax on Idle Money")
        # Only punish the portion of the portfolio sitting in cash.
        # If CASH_DECAY is -0.001:
        # - 100% Cash -> -0.001 penalty
        # - 50% Cash -> -0.0005 penalty
        # - 0% Cash (fully invested) -> 0.0 penalty
        if current_val > 0:
            cash_ratio = self.balance / current_val
            reward += self.config.CASH_DECAY * cash_ratio
        
        # 3. Invalid Action Penalty
        if self.is_invalid_action:
            reward -= self.config.INVALID_ACTION_PENALTY
        
        # Validate and handle edge cases
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"Invalid reward: {reward}. Setting to 0.")
            reward = 0.0
        
        return float(reward)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute an action and return the result.
        
        Action space: Discrete(3)
        - 0: Hold (do nothing)
        - 1: Buy (convert 100% of cash to stock, minus fees)
        - 2: Sell (convert 100% of shares to cash, minus fees)

        Args:
            action: Action index (0=Hold, 1=Buy, 2=Sell)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        current_price = self.df.iloc[self.current_step]["raw_close"]
        self.current_price = current_price  # Store for reward calculation
        
        # Store previous state for reward calculation
        self.prev_balance = self.balance
        self.prev_shares = self.shares_held
        self.prev_price = current_price
        
        # Track invalid action
        self.is_invalid_action = False
        action_name = "hold"
        trade_executed = False
        
        # Execute action based on Discrete(3) action space
        action_idx = int(action)
        
        if action_idx == 0:  # Hold
            self.hold(current_price)
            action_name = "hold"
            trade_executed = False
            
        elif action_idx == 1:  # Buy
            # Check if we can afford at least 1 share
            min_cost = current_price * (1 + self.config.TRANSACTION_FEE)
            if self.balance < min_cost:
                # Invalid action: cannot afford even 1 share
                self.is_invalid_action = True
                self.hold(current_price)
                action_name = "hold"  # Forced hold
                trade_executed = False
            else:
                # Valid buy: convert 100% of cash to stock
                self.buy(current_price, proportion=1.0)
                action_name = "buy"
                trade_executed = True
                
        elif action_idx == 2:  # Sell
            # Check if we have shares to sell
            if self.shares_held == 0:
                # Invalid action: no shares to sell
                self.is_invalid_action = True
                self.hold(current_price)
                action_name = "hold"  # Forced hold
                trade_executed = False
            else:
                # Valid sell: convert 100% of shares to cash
                self.sell(current_price, proportion=1.0)
                action_name = "sell"
                trade_executed = True
        else:
            # Invalid action index
            logger.warning(f"Invalid action index: {action_idx}. Defaulting to hold.")
            self.hold(current_price)
            action_name = "hold"
            trade_executed = False

        # Update net worth
        self.holdings_value = self.shares_held * current_price
        self.net_worth = self.balance + self.holdings_value
        
        if np.isnan(self.net_worth) or np.isinf(self.net_worth):
            logger.error("Net worth became NaN or Inf!")
            self.done = True
            return self._next_observation(), 0.0, self.done, False, {}

        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.net_worth_history.append(self.net_worth)

        # Calculate returns
        profit_loss = self.net_worth - self.previous_net_worth
        self.returns.append(profit_loss)

        # Calculate reward using logarithmic wealth growth
        reward = self.calculate_reward()
        self.cum_reward += reward

        # Update previous values for next step
        self.previous_net_worth = self.net_worth
        self.previous_price = current_price

        # Build info dict
        info = {
            "ticker": self.tickers[self.ticker_count] if self.ticker_count < len(self.tickers) else "unknown",
            "stock_id": self.current_stock_id,
            "step": self.current_step,
            "action": action_name,
            "action_idx": action_idx,
            "reward": reward,
            "current_price": current_price,
            "shares_held": self.shares_held,
            "holdings_value": self.holdings_value,
            "balance": self.balance,
            "net_worth": self.net_worth,
            "total_shares_sold": getattr(self, 'total_shares_sold', 0),
            "total_sales_value": getattr(self, 'total_sales_value', 0),
            "profit_loss": profit_loss,
            "trade_executed": trade_executed,
            "invalid_action": self.is_invalid_action,
        }

        # Advance step
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self._load_df_loop()
            if self.recent_net_worths:
                info["recent_run_mean_return"] = self.recent_run_mean_return

        return self._next_observation(), reward, self.done, False, info

    def _load_df_loop(self) -> None:
        """Handle end of ticker data and load next."""
        final_net_worth = self.end_run()
        self.recent_net_worths.append(final_net_worth)
        self.recent_run_mean_return = float(np.mean(self.recent_net_worths))
        logger.info(f"Ticker complete: reward={self.cum_reward:.4f}, return=£{self.cum_return:.2f}")
        self.cum_rewards.append(self.cum_reward)
        self.cum_returns.append(self.cum_return)

        # Track attempts to avoid infinite loop if all tickers are corrupted
        max_attempts = len(self.tickers) * 2  # Allow going through list twice
        attempts = 0
        
        while attempts < max_attempts:
            self.ticker_count += 1
            if self.ticker_count >= len(self.tickers):
                self.ticker_count = 0
            
            if self.timesteps is not None and self.current_step >= self.timesteps:
                self.done = True
                logger.info("=" * 20 + " END OF RUN " + "=" * 20)
                self.full_reset = True
                return
            
            self._load_next_df(self.ticker_count)
            attempts += 1
            
            # If we successfully loaded data, break out of the loop
            if hasattr(self, 'df') and len(self.df) > 0:
                return
        
        # If we've exhausted all attempts and still have empty data, log error and end
        logger.error(f"Failed to load valid ticker data after {attempts} attempts. Ending episode.")
        # Set a minimal empty dataframe to prevent further errors
        self.df = PricesDf(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
        self.n_steps = 0
        self.done = True

    def _amend_holding_avco(self) -> None:
        """Recalculate average cost of holdings."""
        if not self.holdings:
            return
        
        while self.holdings and self.holdings[0][2] == 0:
            self.holdings.pop(0)
        
        if not self.holdings:
            return
            
        prices, _, changes, _ = zip(*self.holdings)
        cum_holdings = np.cumsum(changes)
        cum_values = np.cumsum(np.array(prices) * np.array(changes))
        avcos = cum_values / cum_holdings
        self.holdings = list(zip(prices, avcos, changes, cum_holdings))

    def sell(self, current_price: float, proportion: float = 1.00) -> Optional[float]:
        """
        Sell shares at current price.
        Convert proportion of shares into cash (minus transaction fees).

        Args:
            current_price: Current share price
            proportion: Proportion of holdings to sell (default 1.0 for 100%)

        Returns:
            Percentage gain/loss from trade, or None if no trade
        """
        shares_to_sell = int(self.shares_held * proportion)
        gain_loss = []
        
        if shares_to_sell > 0:
            shares_sold = 0
            while shares_sold < shares_to_sell and self.holdings:
                this_price, avco, this_shares, _ = self.holdings.pop(0)
                
                if this_shares > shares_to_sell - shares_sold:
                    remaining = this_shares - (shares_to_sell - shares_sold)
                    self.holdings.insert(0, (this_price, this_price, remaining, remaining))
                    this_shares = shares_to_sell - shares_sold
                
                shares_value = this_shares * current_price
                # Apply transaction fee: sale value = shares_value * (1 - fee)
                sale_val = shares_value * (1 - self.config.TRANSACTION_FEE)
                
                self.total_sales_value += sale_val
                self.balance += sale_val
                self.cum_return += sale_val - (avco * this_shares)
                gain_loss.append((sale_val, avco * this_shares))
                shares_sold += this_shares
            
            self.total_shares_sold += shares_sold
            self.shares_held -= shares_sold
            
            if gain_loss:
                sale_vals, cost_vals = zip(*gain_loss)
                return (sum(sale_vals) - sum(cost_vals)) / sum(cost_vals)
        
        return None

    def buy(self, current_price: float, proportion: float = 1.00) -> None:
        """
        Buy shares at current price.
        Convert proportion of available cash into stock (minus transaction fees).

        Args:
            current_price: Current share price
            proportion: Proportion of balance to use (default 1.0 for 100%)
        """
        if self.balance <= 0 or current_price is None or current_price <= 0:
            return
        
        # Calculate shares to buy with available cash (after fees)
        available_cash = self.balance * proportion
        # Account for transaction fee: cost per share = price * (1 + fee)
        cost_per_share = current_price * (1 + self.config.TRANSACTION_FEE)
        shares_to_buy = int(available_cash / cost_per_share)
        
        if shares_to_buy > 0:
            cost = shares_to_buy * cost_per_share
            self.balance -= cost
            self.shares_held += shares_to_buy
            self.holdings.append((current_price, None, shares_to_buy, self.shares_held))
            self._amend_holding_avco()

    def hold(self, current_price: float, *args, **kwargs) -> None:
        """Hold current position."""
        if self.holdings:
            _, avco, _, _ = self.holdings[-1]
            self.holdings.append((current_price, avco, 0, self.shares_held))

    def _reset_values(self) -> None:
        """Reset environment values."""
        self.done = False
        self.holdings_value = 0
        self.shares_held = 0
        self.avco = None
        self.holdings = []
        self.cum_reward = 0.0
        self.cum_return = 0.0
        
        if getattr(self, "mode", "train") == "train":
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.max_net_worth = self.initial_balance
            self.total_shares_sold = 0
            self.total_sales_value = 0
            self.current_step = 0
            self.previous_net_worth = self.initial_balance
            self.previous_price = None  # Track previous close price
            self.net_worth_history = []
            self.returns = []
            # Initialize previous values for reward calculation
            self.prev_balance = self.initial_balance
            self.prev_shares = 0
            self.prev_price = None
            self.current_price = None
            self.is_invalid_action = False
        
        if getattr(self, "full_reset", True):
            self.ticker_count = 0
            self.full_reset = False
            self.cum_rewards = []
            self.cum_returns = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed, options=options)
        self._reset_values()
        # Initialize previous_price to first price for proper stock return calculation
        if hasattr(self, 'df') and len(self.df) > 0:
            self.previous_price = self.df.iloc[0]["raw_close"]
        return self._next_observation(), {}

    def end_run(self) -> float:
        """End current run and return net worth."""
        if len(self.df):
            current_price = self.df.iloc[-1]["raw_close"]
            self.sell(current_price, proportion=1.0)
            self.holdings_value = self.shares_held * current_price
            self.net_worth = self.balance + self.holdings_value
        return self.net_worth

    def render(self, mode: str = "human") -> None:
        """Render current state."""
        profit = self.net_worth - self.initial_balance
        logger.info(f"Step: {self.current_step:,}")
        logger.info(f"Balance: {self.balance:,.2f}")
        logger.info(f"Shares held: {self.shares_held:,}")
        logger.info(f"Net worth: {self.net_worth:,.2f}")
        logger.info(f"Profit: {profit:,.2f} ({profit / self.initial_balance:.2%})")

