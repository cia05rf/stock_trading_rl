"""
Inference module for stock trading model.

This module provides utilities for running trained models on new data
to generate trading predictions.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from tqdm import tqdm

from shared.config import get_config
from shared.logging_config import get_logger
from training.environment import StockTradingEnv
from training.data_ingestion import Ingestion, PricesDf

logger = get_logger(__name__)


class PredsDf(pd.DataFrame):
    """DataFrame subclass for prediction results."""

    _metadata = ['action_type', 'action_prob']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        if "action_type" not in self.columns:
            raise ValueError("action_type column required")
        if "action_prob" not in self.columns:
            raise ValueError("action_prob column required")

    @property
    def _constructor(self):
        return PredsDf

    def top_buys(self, n: int = 10) -> pd.DataFrame:
        """Get top buy recommendations."""
        return self[self["action_type"] == "buy"].nlargest(n, "action_prob")

    def top_sells(self, n: int = 10) -> pd.DataFrame:
        """Get top sell recommendations."""
        return self[self["action_type"] == "sell"].nlargest(n, "action_prob")


class MarketSummaryDf(pd.DataFrame):
    """DataFrame subclass for market summary."""

    @property
    def _constructor(self):
        return MarketSummaryDf

    @classmethod
    def from_preds_df(cls, preds_df: pd.DataFrame) -> 'MarketSummaryDf':
        """Create summary from predictions DataFrame."""
        required_columns = ["market", "action_type", "ticker", "action_prob"]
        for col in required_columns:
            if col not in preds_df.columns:
                raise ValueError(f"{col} column required")

        preds_df = preds_df.copy()
        preds_df["market"] = preds_df["market"].astype(str)
        preds_df["action_type"] = preds_df["action_type"].astype(str)

        grouped = preds_df.groupby(["market", "action_type"]).agg(
            ticker_count=pd.NamedAgg(column="ticker", aggfunc="count"),
            action_prob_sum=pd.NamedAgg(column="action_prob", aggfunc="sum"),
        )

        grouped["action_type_per_count"] = grouped.groupby("market")[
            "ticker_count"
        ].transform(lambda x: x / x.sum())
        grouped["action_prob_per_count"] = grouped.groupby("market")[
            "action_prob_sum"
        ].transform(lambda x: x / x.sum())

        return cls(grouped)


def load_latest_model(models_dir: Optional[Path] = None) -> PPO:
    """
    Load the most recent model from the models directory.

    Args:
        models_dir: Directory containing saved models

    Returns:
        Loaded PPO model
    """
    config = get_config()
    models_dir = models_dir or config.MODELS_DIR

    # Find the latest .zip file
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        logger.error(f"No models found in {models_dir}")
        raise FileNotFoundError(f"No models found in {models_dir}")

    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model: {latest_model}")

    return PPO.load(str(latest_model))


class Infer:
    """
    Inference class for generating trading predictions.

    Args:
        model_path: Path to saved model (optional, uses latest if not provided)
    """

    def __init__(self, model_path: Optional[str] = None):
        self.config = get_config()
        self.window_size = self.config.OBS_WINDOW_SIZE  # Use observation window size (e.g. 50)

        # Create environment for inference
        self.env = StockTradingEnv(
            mode="test",
            test_train_split=self.config.TEST_TRAIN_SPLIT,
            initial_balance=self.config.INITIAL_BALANCE,
            window_size=self.config.WINDOW_SIZE,
            seed=self.config.SEED,
        )

        # Load model
        if model_path:
            logger.info(f"Loading model: {model_path}")
            self.model = PPO.load(model_path)
        else:
            logger.info("Loading latest model")
            self.model = load_latest_model()
        self.device = self.model.device

        self.preds_df: Optional[PredsDf] = None
        self.market_action_counts_df: Optional[MarketSummaryDf] = None

    def _decode_action(
        self,
        action,
        obs_tensor: torch.Tensor,
    ) -> Dict:
        """
        Decode an action into a consistent schema across discrete/continuous modes.

        Returns at least: action, action_type, action_prob
        """
        # 1. Get the distribution for the current observation
        # We fetch this first to check what the model is actually outputting (Normal vs Categorical)
        action_dist = self.model.policy.get_distribution(obs_tensor)

        # 2. Check for Discrete Mode
        # We only run discrete logic if the Env is Discrete AND the Model has probabilities
        if (isinstance(self.env.action_space, gym.spaces.Discrete)
                and hasattr(action_dist.distribution, 'probs')):

            action_keys = list(self.env.actions.keys())
            action_types = {k: self.env.actions[k][2] for k in action_keys}

            action_idx = int(action)

            # Safe to access .probs here
            action_probabilities = action_dist.distribution.probs

            action_name = action_keys[action_idx]
            decoded = {
                "action": action_name,
                "action_type": action_types[action_name],
                "action_prob": action_probabilities[0][action_idx].item(),
            }
            # Add individual action probabilities
            for k, v in zip(action_keys, action_probabilities[0]):
                decoded[k] = v.item()
            return decoded

        # 3. Continuous Mode (Fallback)
        # Handles gym.spaces.Box OR the case where Env is Discrete but Model is Continuous

        # Get all 3 continuous actions
        actions = np.asarray(action).reshape(-1)
        signal = float(actions[0])
        limit_offset_action = float(actions[1]) if len(actions) > 1 else 0.0
        stop_loss_action = float(actions[2]) if len(actions) > 2 else 0.0

        # Calculate Probability Density (Confidence)
        # Normal distributions use log_prob, not probs
        log_prob = action_dist.log_prob(
            torch.tensor(action).to(obs_tensor.device))

        # Convert log likelihood to density (approximate confidence)
        action_prob = torch.exp(log_prob).item() if log_prob.numel(
        ) == 1 else torch.exp(log_prob).mean().item()

        # Domain Logic: Map continuous value to Buy/Sell/Hold
        threshold = self.config.TRADING_THRESHOLD
        if signal > threshold:
            action_type = "buy"
        elif signal < -threshold:
            action_type = "sell"
        else:
            action_type = "hold"

        return {
            "action": "continuous",
            "action_type": action_type,
            "action_prob": action_prob,  # Real probability density from the model
            "signal": signal,
            "limit_offset_action": limit_offset_action,
            "stop_loss_action": stop_loss_action,
        }

    def infer_date(
        self,
        date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> Tuple[PredsDf, MarketSummaryDf]:
        """
        Generate predictions for a specific date.

        Args:
            date: Date string (YYYY-MM-DD format)
            tickers: List of tickers to predict (None for all)

        Returns:
            Tuple of (predictions DataFrame, market summary DataFrame)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Calculate date range for data loading
        en = datetime.strptime(date, "%Y-%m-%d")
        st = en - timedelta(days=self.config.WINDOW_SIZE - 1 + 20)

        min_date_str = st.strftime("%Y-%m-%d")
        max_date_str = en.strftime("%Y-%m-%d")

        # Load and prepare data
        ingest = Ingestion(min_date=min_date_str, max_date=max_date_str)
        prices = ingest.read_prices(tickers, _ticker_fields=["company"])
        prices = prices.prep_data()
        # prices = prices.groupby(["ticker"]).tail(self.config.WINDOW_SIZE)

        # Generate predictions
        groups = prices.groupby(["ticker"])
        preds = []

        for ticker, ticker_prices in tqdm(
            groups, total=len(groups), desc="Predicting"
        ):
            self.env.reset()
            self.env.current_step = self.config.WINDOW_SIZE - 1
            self.env.df = ticker_prices
            
            # Update current_stock_id if ticker_to_id mapping exists
            if hasattr(self.env, 'ticker_to_id') and ticker in self.env.ticker_to_id:
                self.env.current_stock_id = self.env.ticker_to_id[ticker]

            obs = self.env._next_observation()
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Set stock_id in policy if available
            if hasattr(self.model.policy, 'set_stock_id') and hasattr(self.env, 'current_stock_id'):
                self.model.policy.set_stock_id(self.env.current_stock_id)

            action, _ = self.model.predict(obs, deterministic=True)
            decoded = self._decode_action(action, obs_tensor)
            pred = {
                "ticker": ticker,
                **decoded,
            }

            preds.append(pred)

        self.preds_df = PredsDf(preds)
        self.market_action_counts_df = MarketSummaryDf.from_preds_df(
            self.preds_df)

        return self.preds_df, self.market_action_counts_df

    def predict_single(
        self,
        ticker: str,
        date: Optional[str] = None,
    ) -> Dict:
        """
        Get prediction for a single ticker.

        Args:
            ticker: Stock ticker symbol
            date: Date string (YYYY-MM-DD format)

        Returns:
            Dictionary with prediction details
        """
        preds_df, _ = self.infer_date(date, [ticker])

        if len(preds_df) == 0:
            return {"error": f"No data found for {ticker}"}

        return preds_df.iloc[0].to_dict()

    def predict_for_timestamp(
        self,
        prices_by_ticker: Dict[str, "PricesDf"],
        timestamp: pd.Timestamp,
        deterministic: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions for all tickers at a specific timestamp.
        Uses the environment's observation logic for consistency.
        """
        preds = []

        for ticker, df in prices_by_ticker.items():
            # Find the index for the timestamp
            # We assume df is sorted by date
            ts_indices = df.index[df["date"] == timestamp]
            if ts_indices.empty:
                continue
            
            idx = ts_indices[0]
            
            # Use environment to get observation
            self.env.df = df
            self.env.current_step = idx
            
            # Update current_stock_id
            if hasattr(self.env, 'ticker_to_id') and ticker in self.env.ticker_to_id:
                self.env.current_stock_id = self.env.ticker_to_id[ticker]
            
            # Precompute features if needed (usually done in _load_next_df)
            # Since we're manually setting df, we call it here
            self.env._precompute_features()
            
            obs = self.env._next_observation()
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Set stock_id in policy if available
            if hasattr(self.model.policy, 'set_stock_id') and hasattr(self.env, 'current_stock_id'):
                self.model.policy.set_stock_id(self.env.current_stock_id)

            action, _ = self.model.predict(obs, deterministic=deterministic)
            decoded = self._decode_action(action, obs_tensor)
            preds.append({
                "date": timestamp,
                "ticker": ticker,
                **decoded,
                "close": df.iloc[idx]["close"] if "close" in df.columns else None,
            })

        return pd.DataFrame(preds)

    def predict_for_timestamp_with_data(
        self,
        timestamp: pd.Timestamp,
        tickers: Optional[List[str]] = None,
        buffer_days: int = 20,
        deterministic: bool = False,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: fetch data, prepare windows, and predict for a single timestamp.
        """
        ts = pd.to_datetime(timestamp)
        start_ts = ts - timedelta(days=buffer_days)
        ingest = Ingestion(
            min_date=start_ts.strftime("%Y-%m-%d"),
            max_date=ts.strftime("%Y-%m-%d"),
        )
        prices = ingest.read_prices(tickers)
        if prices.empty:
            return pd.DataFrame()
        prices = prices.sort_values(["ticker", "date"])

        prices_by_ticker: Dict[str, PricesDf] = {}
        for t, df_t in prices.groupby("ticker"):
            # Note: We use the same data loader as the environment for consistency
            prepared = PricesDf(df_t.copy()).prep_data().sort_values("date")
            prices_by_ticker[t] = prepared

        logger.info(f"Predicting for timestamp: {ts}")
        return self.predict_for_timestamp(
            prices_by_ticker=prices_by_ticker,
            timestamp=ts,
            deterministic=deterministic,
        )

    def predict_over_data(
        self,
        data: pd.DataFrame,
        ticker: Optional[str] = None,
        deterministic: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions for every step in a single ticker DataFrame.
        """
        if len(data) == 0:
            return pd.DataFrame()

        preds = []
        ticker_label = ticker or (
            data["ticker"].iloc[0] if "ticker" in data else "unknown")

        # Set up environment
        self.env.df = data
        self.env._precompute_features()
        
        # Update current_stock_id
        if hasattr(self.env, 'ticker_to_id') and ticker_label in self.env.ticker_to_id:
            self.env.current_stock_id = self.env.ticker_to_id[ticker_label]
        
        if hasattr(self.model.policy, 'set_stock_id') and hasattr(self.env, 'current_stock_id'):
            self.model.policy.set_stock_id(self.env.current_stock_id)

        for idx in range(len(data)):
            self.env.current_step = idx
            obs = self.env._next_observation()
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            action, _ = self.model.predict(obs, deterministic=deterministic)
            decoded = self._decode_action(action, obs_tensor)
            preds.append({
                "ticker": ticker_label,
                "date": data["date"].iloc[idx],
                "step": idx,
                **decoded,
                "close": data["close"].iloc[idx] if "close" in data.columns else None,
            })

        return pd.DataFrame(preds)

    def predict_timespan(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[List[str]] = None,
        buffer_days: int = 30,
        deterministic: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions for all timestamps in a window [start, end].
        Ticker-major implementation for better performance with environment-based observations.
        """
        logger.info(
            f"Predicting for {start} to {end} with tickers {'ALL' if tickers is None else tickers}")
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        load_start = start_ts - timedelta(days=buffer_days)

        ingest = Ingestion(
            min_date=load_start.strftime("%Y-%m-%d"),
            max_date=end_ts.strftime("%Y-%m-%d"),
        )
        prices = ingest.read_prices(tickers)
        if prices.empty:
            return pd.DataFrame()
        
        prices = prices.sort_values(["ticker", "date"])

        all_preds = []
        
        # Group by ticker for ticker-major processing
        groups = prices.groupby("ticker")
        for ticker, df_t in tqdm(groups, desc="Predicting per ticker"):
            # Prepare data same as environment
            prepared = PricesDf(df_t.copy()).prep_data().sort_values("date")
            
            # Filter for requested timespan
            df_filtered = prepared[(prepared["date"] >= start_ts) & (prepared["date"] <= end_ts)]
            if df_filtered.empty:
                continue
                
            # Use predict_over_data for this ticker
            ticker_preds = self.predict_over_data(
                data=prepared,
                ticker=ticker,
                deterministic=deterministic
            )
            
            # Filter predictions to requested timespan
            ticker_preds = ticker_preds[(ticker_preds["date"] >= start_ts) & (ticker_preds["date"] <= end_ts)]
            if not ticker_preds.empty:
                all_preds.append(ticker_preds)

        if not all_preds:
            return pd.DataFrame()

        return pd.concat(all_preds, ignore_index=True)
