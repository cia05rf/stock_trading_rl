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
        logger.info("INITIALIZING STOCK TRADING ENVIRONMENT (LIMIT ORDER STYLE)")
        logger.info("=" * 50)

        # Initialize data ingestion
        self.ingestion = Ingestion(min_date=min_date)
        self.config = get_config()
        # Store parameters
        self.ticker_count = 0
        self.done = False
        self.current_step = 0
        self.mode = mode
        
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.obs_window_size = self.config.OBS_WINDOW_SIZE
        # Dynamic transaction fee for curriculum learning
        self.total_training_steps = None  # Set via set_total_training_steps()
        self.global_step_count = 0  # Track total steps across all episodes
        self.transaction_cost_pct = self.config.TRANSACTION_FEE  # Starting fee (0.0)
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
        
        # Limit Order Parameters (can be updated by curriculum)
        self.max_limit_offset = self.config.MAX_LIMIT_OFFSET
        
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

        # Limit Order State
        self.pending_order = None  # {"type": "buy"/"sell", "price": float, "ttl": int}
        self.active_position = None  # {"type": "long"/"short", "entry_price": float, "size": float, "stop_loss": float}
        self.locked_cash = 0.0

        # Load all tickers
        tickers = self.ingestion.read_tickers()
        if ticker_limit:
            tickers = tickers.iloc[:ticker_limit]

        # Apply tradability filter
        tickers = self._filter_by_tradability(tickers)
        
        unique_tickers = tickers["ticker"].unique()
        self.ticker_to_id = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
        self.num_stocks = len(self.ticker_to_id)
        self.current_stock_id = 0

        train_tickers_df = tickers.sample(frac=test_train_split, random_state=seed)
        self.tickers_train = train_tickers_df["ticker"].to_list()
        self.tickers_test = tickers.drop(train_tickers_df.index)["ticker"].to_list()

        self.np_random = None

        # Initialize values and set mode
        self._reset_values()
        self.set_mode(mode)
        self.timesteps = self.epoch_timesteps()

        assert len(self.tickers) > 0, "No tickers found"

        # Action Space: shape=(3,)
        # Action[0] (Signal): > 0.3 is BUY, < -0.3 is SELL, between is HOLD/CANCEL.
        # Action[1] (Limit Offset): Determines the entry price relative to current Close.
        # Action[2] (Stop Loss): Determines the stop loss relative to the Limit Price.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Track invalid actions for reward calculation
        self.is_invalid_action = False
        
        # Define observation space
        # Features per timestep: log_return, volume_change, volatility, RSI, MACD, time_sin, time_cos, 
        # Has_Pending_Order (0/1), Has_Active_Position (0/1), Current_Unrealized_PnL
        # Total: 10 features per timestep
        n_features_per_step = 10
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

    def _filter_by_tradability(self, tickers: pd.DataFrame) -> pd.DataFrame:
        """
        Filter tickers by volatility to ensure they are tradable.
        
        Drops stocks where mean absolute percentage change < TRANSACTION_FEE * 1.5.
        These stocks are too stable to trade profitably after fees.
        
        Args:
            tickers: DataFrame with ticker information
            seed: Random seed for reproducibility
            
        Returns:
            Filtered DataFrame with only tradable tickers
        """
        logger.info("=" * 60)
        logger.info("APPLYING TRADABILITY FILTER")
        logger.info("=" * 60)
        
        # Use TARGET_TRANSACTION_FEE for filtering (we want stocks tradable at final fee)
        min_volatility = self.config.TARGET_TRANSACTION_FEE * 1.5
        logger.info(
            f"Minimum volatility threshold: {min_volatility:.6f} "
            f"(based on TARGET_TRANSACTION_FEE={self.config.TARGET_TRANSACTION_FEE})"
        )
        
        tradable_tickers = []
        dropped_count = 0
        total_count = len(tickers)
        
        # Process each ticker
        for _, row in tickers.iterrows():
            ticker = row['ticker']
            
            try:
                # Load raw price data (before prep_data to get actual prices)
                df_raw = self.ingestion.read_prices(ticker)
                
                if len(df_raw) == 0:
                    dropped_count += 1
                    continue
                
                # Calculate volatility: mean absolute percentage change
                if 'close' in df_raw.columns and len(df_raw) > 1:
                    prices = df_raw['close'].values
                    # Calculate percentage changes
                    pct_changes = np.abs(pd.Series(prices).pct_change().dropna())
                    volatility_pct = pct_changes.mean()
                    
                    if volatility_pct >= min_volatility:
                        tradable_tickers.append(row)
                    else:
                        dropped_count += 1
                        logger.debug(
                            f"Dropped {ticker}: volatility={volatility_pct:.6f} "
                            f"< threshold={min_volatility:.6f}"
                        )
                else:
                    dropped_count += 1
                    logger.debug(f"Dropped {ticker}: insufficient data")
                    
            except Exception as e:
                dropped_count += 1
                logger.debug(f"Error processing {ticker} for volatility: {e}")
                continue
        
        # Create filtered DataFrame
        if tradable_tickers:
            filtered_df = pd.DataFrame(tradable_tickers).reset_index(drop=True)
        else:
            filtered_df = pd.DataFrame(columns=tickers.columns)
        
        # Print statistics
        logger.info("=" * 60)
        logger.info(
            f"Tradability Filter Results: "
            f"Dropped {dropped_count} boring tickers. "
            f"Training on {len(filtered_df)} volatile tickers."
        )
        logger.info(
            f"Retention rate: {len(filtered_df)/total_count*100:.1f}% "
            f"({len(filtered_df)}/{total_count})"
        )
        logger.info("=" * 60)
        
        return filtered_df

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

    def _load_next_df(self, i: int, start_index: Optional[int] = None) -> None:
        """
        Load the next ticker's data and pre-compute all static features.
        
        Args:
            i: Ticker index in self.tickers list
            start_index: Optional starting index within the ticker's history.
                        If None, starts from beginning. Used for randomization.
        """
        self.progress_bar.update(1)
        ticker = self.tickers[i]
        self.df = self._data_loader(ticker)
        self.n_steps = len(self.df)
        
        # Handle empty dataframe (corrupted ticker data)
        if len(self.df) == 0:
            logger.warning(f"Ticker {ticker} has no data. Will skip to next ticker.")
            return
        
        # Update current stock_id
        self.current_stock_id = self.ticker_to_id.get(ticker, 0)
        
        # If start_index is provided and valid, slice the dataframe
        # This allows randomization while preserving time-series order
        if start_index is not None and start_index > 0:
            if start_index < self.n_steps:
                # Slice from start_index to end (preserves time order)
                self.df = self.df.iloc[start_index:].reset_index(drop=True)
                self.n_steps = len(self.df)
            else:
                # Invalid start_index, use from beginning
                start_index = 0
        
        self.current_step = 0
        
        logger.debug(
            f"Loaded {self.n_steps:,} rows for {ticker} "
            f"(stock_id={self.current_stock_id}, start_idx={start_index or 0})"
        )
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
        opens = self.df['open'].values.astype(np.float32) if 'open' in self.df.columns else prices
        
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
        
        # Use np.divide with 'where' to avoid RuntimeWarning: divide by zero
        volume_ratios = np.divide(
            volumes, 
            prev_volumes, 
            out=np.ones_like(volumes), 
            where=(volumes > 0) & (prev_volumes > 0)
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
        self.opens_array = opens

    def _next_observation(self) -> np.ndarray:
        """
        Get the next observation using pre-computed features.
        Only account state is computed on-the-fly as it changes with actions.

        Returns:
            Flattened observation array of shape (obs_window_size * 10,)
        """
        # Ensure features are pre-computed
        if not hasattr(self, 'log_returns') or len(self.log_returns) == 0:
            self._precompute_features()
        
        # Get window indices
        start = max(0, self.current_step - self.obs_window_size + 1)
        end = min(self.current_step + 1, len(self.log_returns))
        
        # Get current price for account state calculation
        current_price = self.prices_array[self.current_step] if self.current_step < len(self.prices_array) else 0.0
        
        # Calculate account state (dynamic features)
        if self.pending_order:
            pending_price = self.pending_order["price"]
            # pending_dist = (pending_price - current_price) / current_price
            # Clip between -0.05 and 0.05 and scale to [-1, 1]
            dist = (pending_price - current_price) / current_price if current_price > 0 else 0
            pending_dist = np.clip(dist, -0.05, 0.05) / 0.05
        else:
            pending_dist = 0.0

        has_active = 1.0 if self.active_position is not None else 0.0
        
        # Current_Unrealized_PnL: (current_price - entry_price) / entry_price for long
        # (entry_price - current_price) / entry_price for short
        unrealized_pnl = 0.0
        if self.active_position:
            entry_price = self.active_position["entry_price"]
            if self.active_position["type"] == "long":
                unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            else:  # short
                unrealized_pnl = (entry_price - current_price) / entry_price if entry_price > 0 else 0.0
        
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
        pending_dist_window = np.full(self.obs_window_size, pending_dist, dtype=np.float32)
        has_active_window = np.full(self.obs_window_size, has_active, dtype=np.float32)
        unrealized_pnl_window = np.full(self.obs_window_size, unrealized_pnl, dtype=np.float32)
        
        # Stack all features: shape (obs_window_size, 10)
        obs_window = np.column_stack([
            log_returns_window,
            volume_changes_window,
            volatilities_window,
            rsi_scaled_window,
            macd_normalized_window,
            time_sin_window,
            time_cos_window,
            pending_dist_window,
            has_active_window,
            unrealized_pnl_window,
        ])
        
        # Flatten to 1D: (obs_window_size * 10,)
        obs = obs_window.flatten()
        
        # Handle NaNs and Infs (vectorized)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)
        
        if np.isnan(obs).any() or np.isinf(obs).any():
            raise ValueError("Observation contains NaNs or Infs!")
        
        return obs.astype(np.float32)

    def _liquidate_position(self, current_price: float, reason: str = "stop_loss") -> Tuple[float, float]:
        """Liquidate active position and return (cash_pnl, pct_pnl)."""
        if not self.active_position:
            return 0.0, 0.0
        
        entry_price = self.active_position["entry_price"]
        size = self.active_position["size"]
        pos_type = self.active_position["type"]
        
        current_fee = self._get_current_transaction_fee()
        
        if pos_type == "long":
            # Sell long position
            exit_value = size * current_price * (1 - current_fee)
            cash_pnl = exit_value - (size * entry_price)
            pct_pnl = (current_price * (1 - current_fee) - entry_price) / entry_price
            self.balance += exit_value
        else:
            # Buy back short position
            exit_cost = size * current_price * (1 + current_fee)
            entry_proceeds = size * entry_price # Already in balance from shorting
            cash_pnl = entry_proceeds - exit_cost
            pct_pnl = (entry_price - current_price * (1 + current_fee)) / entry_price
            self.balance -= exit_cost # Subtract the cost to buy back
            
        self.active_position = None
        self.shares_held = 0
        return cash_pnl, pct_pnl

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute an action and return the result with Order Latency and Encumbered Cash.
        
        Action space: Box(low=-1, high=1, shape=(3,))
        - Action[0]: Signal Strength
        - Action[1]: Limit Offset (Agressiveness)
        - Action[2]: Stop Loss Offset (Entry only)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # 1. Get current market data
        current_candle = self.df.iloc[self.current_step]
        close_price = current_candle["raw_close"]
        high_price = current_candle["high"]
        low_price = current_candle["low"]
        open_price = self.opens_array[self.current_step]
        
        self.current_price = close_price
        self.prev_balance = self.balance
        self.prev_shares = self.shares_held
        self.prev_price = close_price
        self.previous_net_worth = self.net_worth
        
        self.is_invalid_action = False
        fill_event = False
        execution_type = None
        cash_pnl = 0.0
        realized_pnl_pct = 0.0
        
        current_fee = self._get_current_transaction_fee()

        # --- PHASE A: CHECK FILLS (First) ---
        # Process the PREVIOUS step's pending_order against CURRENT candle
        if self.pending_order:
            order = self.pending_order
            fill_price = order["price"]
            
            if order["type"] == "buy" and not self.active_position:
                # Buy fill: current low must be below limit price
                # We use a small buffer (0.9995) to be realistic about fills
                if low_price < fill_price * 0.9995:
                    # If market opened below limit, we fill at open (better price)
                    fill_price = min(fill_price, open_price)
                    
                    self.shares_held = order["size"]
                    
                    # Cash was ALREADY deducted at placement.
                    # Refund difference if fill price < limit price
                    actual_cost = self.shares_held * fill_price * (1 + current_fee)
                    refund = order["locked_cash"] - actual_cost
                    self.balance += max(0, refund)
                    
                    self.active_position = {
                        "entry_price": fill_price,
                        "size": self.shares_held,
                        "stop_loss": order["stop_loss"],
                        "type": "long"
                    }
                    fill_event = True
                    self.orders_filled += 1
                    execution_type = "limit_fill_buy"
                    self.pending_order = None
                    self.locked_cash = 0.0
                    
            elif order["type"] == "sell" and self.active_position:
                # Sell fill: current high must be above limit price
                if high_price >= fill_price:
                    # If market opened above limit, we fill at open (better price)
                    fill_price = max(fill_price, open_price)
                    
                    cash_pnl, realized_pnl_pct = self._liquidate_position(
                        fill_price, reason="limit_fill_sell"
                    )
                    fill_event = True
                    self.orders_filled += 1
                    execution_type = "limit_fill_sell"
                    self.pending_order = None

        # A2. Check Active Position for Stop Loss (Pessimistic)
        if self.active_position and not execution_type:
            pos = self.active_position
            if low_price <= pos["stop_loss"]:
                # Stop Loss hit. Fill at SL price (or open if opened below SL)
                exit_price = min(pos["stop_loss"], open_price)
                cash_pnl, realized_pnl_pct = self._liquidate_position(
                    exit_price, reason="stop_loss"
                )
                execution_type = "stop_loss"

        # A3. TTL Check for remaining pending order
        if self.pending_order:
            self.pending_order["ttl"] -= 1
            if self.pending_order["ttl"] <= 0:
                if self.pending_order["type"] == "buy":
                    # REFUND LOCKED CASH on TTL expiry
                    self.balance += self.pending_order["locked_cash"]
                    self.locked_cash = 0.0
                self.pending_order = None
                if not execution_type:
                    execution_type = "ttl_expiry"

        # --- PHASE B: NEW ACTIONS (Second) ---
        signal = action[0]

        # 1. Handle Cancellations (Refunds)
        # If signal < 0.3 and we have a pending buy, cancel and refund
        if not self.active_position and self.pending_order and self.pending_order["type"] == "buy":
            if signal < 0.3:
                self.balance += self.pending_order["locked_cash"]
                self.locked_cash = 0.0
                self.pending_order = None
                execution_type = "cancel_pending_buy" if not execution_type else execution_type

        # 2. Handle New Orders (Deductions)
        if not self.active_position and not self.pending_order:
            if signal > 0.3: # New Buy
                limit_price = self._calculate_limit_price(close_price, action[1], order_type="buy")
                stop_loss = self._calculate_stop_price(limit_price, action[2])
                
                cost_per_share = limit_price * (1 + current_fee)
                size = int(self.balance / cost_per_share) if cost_per_share > 0 else 0
                
                if size > 0:
                    total_cost = size * cost_per_share
                    self.balance -= total_cost
                    self.locked_cash = total_cost
                    self.pending_order = {
                        "type": "buy",
                        "size": size,
                        "price": limit_price,
                        "locked_cash": total_cost,
                        "stop_loss": stop_loss,
                        "ttl": self.config.ORDER_TTL
                    }
                    self.orders_placed += 1
                    execution_type = "new_pending_buy" if not execution_type else execution_type
                else:
                    self.is_invalid_action = True
        
        elif self.active_position:
            # If Active Position: < -0.3 = Place SELL LIMIT (Exit)
            if signal < -0.3:
                if not self.pending_order:
                    limit_price = self._calculate_limit_price(close_price, action[1], order_type="sell")
                    self.pending_order = {
                        "type": "sell",
                        "price": limit_price,
                        "ttl": self.config.ORDER_TTL
                    }
                    self.orders_placed += 1
                    execution_type = "new_pending_sell" if not execution_type else execution_type
            elif signal > -0.3:
                # Cancel pending Sell orders (Hold strategy)
                if self.pending_order and self.pending_order["type"] == "sell":
                    self.pending_order = None
                    execution_type = "cancel_pending_sell" if not execution_type else execution_type
        
        # Update Net Worth
        if self.active_position:
            self.holdings_value = self.shares_held * close_price
        else:
            self.holdings_value = 0
            
        self.net_worth = self.balance + self.holdings_value + self.locked_cash
        
        # Update training step count
        self.global_step_count += 1
        
        # Check for bankruptcy
        if self.net_worth <= 0:
            logger.warning(f"Bankruptcy! Net worth: {self.net_worth:.2f}")
            self.done = True
            return self._next_observation(), -100.0, self.done, False, {"bankruptcy": True}

        # Calculate log return for reward scaling
        if self.previous_net_worth > 0 and self.net_worth > 0:
            log_return = float(np.log(self.net_worth / self.previous_net_worth))
        else:
            log_return = 0.0

        # Calculate reward
        reward = self.calculate_reward(fill_event, realized_pnl_pct, log_return)
        self.cum_reward += reward
        
        # Track history
        self.net_worth_history.append(self.net_worth)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        profit_loss = self.net_worth - self.previous_net_worth
        self.returns.append(profit_loss)
        self.previous_net_worth = self.net_worth

        info = {
            "ticker": self.tickers[self.ticker_count] if self.ticker_count < len(self.tickers) else "unknown",
            "step": self.current_step,
            "execution_type": execution_type,
            "net_worth": self.net_worth,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "locked_cash": self.locked_cash,
            "has_active": self.active_position is not None,
            "has_pending": self.pending_order is not None,
            "reward": reward,
            "cash_pnl": cash_pnl,
            "realized_pnl_pct": realized_pnl_pct,
            "fill_event": fill_event,
            "invalid_action": self.is_invalid_action,
            "fill_rate": self.orders_filled / self.orders_placed if self.orders_placed > 0 else 0.0,
            "avg_entry_discount": self.total_entry_discount / self.orders_placed if self.orders_placed > 0 else 0.0,
            "trade_executed": realized_pnl_pct != 0,
            "profit_loss": profit_loss,
        }

        # Advance step
        self.current_step += 1
        if self.current_step >= self.n_steps:
            self.done = True
        
        return self._next_observation(), reward, self.done, False, info

    def _get_current_transaction_fee(self) -> float:
        """Calculate current transaction fee based on training progress."""
        if self.total_training_steps is None or self.total_training_steps == 0:
            return self.config.TRANSACTION_FEE
        progress = min(self.global_step_count / self.total_training_steps, 1.0)
        if progress < 0.1:
            return 0.0
        adjusted_progress = (progress - 0.1) / 0.9
        return self.config.TARGET_TRANSACTION_FEE * adjusted_progress

    def calculate_reward(self, fill_event: bool, realized_pnl: float, log_return: float) -> float:
        """
        Calculate reward with multiple components:
        1. Step Penalty: Efficiency incentive.
        2. Unrealized PnL: Logarithmic return of net worth.
        3. Realized PnL: On Exit, reward (Exit_Price - Entry_Price) / Entry_Price.
        4. Order Rent: Penalty for keeping a pending order open.
        """
        reward = 0.0
        
        # 1. Step Penalty (encourage efficiency)
        reward += self.config.STEP_PENALTY
        
        # 2. Unrealized PnL (While holding)
        if self.active_position:
            # Use logarithmic return as requested
            reward += log_return
        
        # 3. Realized PnL (on close)
        # realized_pnl passed in is already (Exit - Entry) / Entry
        reward += realized_pnl
        
        # 4. Order Rent (The "No Free Lunch" Fix)
        if self.pending_order is not None:
            reward -= self.config.PENDING_ORDER_PENALTY
            
        # 5. Fill Reward (Now 0.0 by default in config)
        if fill_event:
            reward += self.config.FILL_REWARD
            
        # Invalid Action Penalty (Legacy support if needed)
        if getattr(self, "is_invalid_action", False):
            reward -= self.config.INVALID_ACTION_PENALTY
            
        return float(reward)

    def _calculate_limit_price(self, close_price: float, offset_action: float, order_type: str = "buy") -> float:
        """
        Calculate limit price based on offset action.
        Buy Limit Price = Current_Close * (1 - (abs(Action[1]) * MAX_LIMIT_OFFSET))
        Sell Limit Price = Current_Close * (1 + (abs(Action[1]) * MAX_LIMIT_OFFSET))
        """
        offset = abs(offset_action) * self.max_limit_offset
        if order_type == "buy":
            return close_price * (1 - offset)
        else: # sell
            return close_price * (1 + offset)

    def _calculate_stop_price(self, entry_price: float, sl_action: float) -> float:
        """
        Calculate stop price based on SL action.
        Stop Price = Entry_Price * (1 - (abs(sl_action) * self.config.MAX_STOP_LOSS))
        """
        return entry_price * (1 - (abs(sl_action) * self.config.MAX_STOP_LOSS))


    def _reset_values(self) -> None:
        """Reset environment values."""
        self.done = False
        self.holdings_value = 0
        self.shares_held = 0
        self.avco = None
        self.holdings = []
        self.cum_reward = 0.0
        self.cum_return = 0.0
        
        # Limit Order State Reset
        self.pending_order = None
        self.active_position = None
        self.locked_cash = 0.0
        
        # Statistics Tracking
        self.orders_placed = 0
        self.orders_filled = 0
        self.total_entry_discount = 0.0
        
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
        Reset the environment with randomized stock selection and start position.
        
        Randomizes:
        - Which ticker to use (from filtered tradable list)
        - Starting index within that ticker's history
        
        Does NOT shuffle time-series data (preserves Buy->Hold->Sell sequence).

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed, options=options)
        
        # Ensure np_random is available (initialized by super().reset())
        # Gymnasium's reset() should set self.np_random, but we add a fallback
        if not hasattr(self, 'np_random') or self.np_random is None:
            # Fallback: use numpy random if gymnasium didn't initialize it
            rng = np.random.RandomState(seed)
            self.np_random = rng
        
        # Randomize ticker selection and start position
        if len(self.tickers) > 0:
            # Randomly select a ticker
            ticker_idx = self.np_random.integers(0, len(self.tickers))
            
            # Load the ticker's data to get its length
            ticker = self.tickers[ticker_idx]
            df_temp = self._data_loader(ticker)
            
            if len(df_temp) > 0:
                # Randomly select a start index (leave room for at least obs_window_size steps)
                max_start = max(0, len(df_temp) - self.obs_window_size - 1)
                start_index = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0
                
                # Load with randomized start position
                self._load_next_df(ticker_idx, start_index=start_index)
            else:
                # Fallback: load from beginning if data is empty
                self._load_next_df(ticker_idx, start_index=None)
        else:
            # Fallback: use first ticker if list is empty
            self._load_next_df(0, start_index=None)
        
        self._reset_values()
        
        # Initialize previous_price to first price for proper stock return calculation
        if hasattr(self, 'df') and len(self.df) > 0:
            self.previous_price = self.df.iloc[0]["raw_close"]
        
        return self._next_observation(), {}

    def end_run(self) -> float:
        """End current run and return net worth."""
        if len(self.df) and self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]["raw_close"]
            # Close any open position (long or short)
            if self.active_position:
                self._liquidate_position(current_price, reason="end_of_run")
            
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

