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
        self.window_size = self.config.WINDOW_SIZE

        # Create environment for inference
        self.env = StockTradingEnv(
            mode="test",
            test_train_split=self.config.TEST_TRAIN_SPLIT,
            initial_balance=self.config.INITIAL_BALANCE,
            window_size=self.config.WINDOW_SIZE,
            seed=self.config.SEED,
            action_space_type=self.config.ACTION_SPACE_TYPE,
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

        # Get the scalar value of the action
        val = float(np.asarray(action).reshape(-1)[0])

        # Calculate Probability Density (Confidence)
        # Normal distributions use log_prob, not probs
        log_prob = action_dist.log_prob(
            torch.tensor(action).to(obs_tensor.device))

        # Convert log likelihood to density (approximate confidence)
        action_prob = torch.exp(log_prob).item() if log_prob.numel(
        ) == 1 else torch.exp(log_prob).mean().item()

        # Domain Logic: Map continuous value to Buy/Sell/Hold
        threshold = self.config.TRADING_THRESHOLD
        if val > threshold:
            action_type = "buy"
            proportion = float(np.clip(val, 0.0, 1.0))
        elif val < -threshold:
            action_type = "sell"
            proportion = float(np.clip(abs(val), 0.0, 1.0))
        else:
            action_type = "hold"
            proportion = 0.0

        return {
            "action": "continuous",
            "action_type": action_type,
            "action_prob": action_prob,  # Real probability density from the model
            "action_value": float(val),
            "proportion": float(proportion),
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
        Generate predictions for all tickers at a specific timestamp using a sliding window.
        Expects `prices_by_ticker` to be prepared (PricesDf with model_columns) and sorted.
        """
        preds = []

        for ticker, df in prices_by_ticker.items():
            df_up_to = df[df["date"] <= timestamp]
            if len(df_up_to) == 0:
                continue

            window = df_up_to.tail(self.window_size)
            obs = window[window.model_columns].values
            if len(obs) < self.window_size:
                pad = np.zeros((self.window_size - len(obs), obs.shape[1]))
                obs = np.vstack((pad, obs))

            obs = np.nan_to_num(obs.astype(np.float32),
                                nan=0.0, posinf=10.0, neginf=-10.0)
            obs = np.clip(obs, -10.0, 10.0)
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Set stock_id in policy if available
            stock_id = None
            if hasattr(self.model.policy, 'set_stock_id'):
                if hasattr(self.env, 'ticker_to_id') and ticker in self.env.ticker_to_id:
                    stock_id = self.env.ticker_to_id[ticker]
                elif hasattr(self.env, 'current_stock_id'):
                    stock_id = self.env.current_stock_id
                if stock_id is not None:
                    self.model.policy.set_stock_id(stock_id)

            action, _ = self.model.predict(obs, deterministic=deterministic)
            decoded = self._decode_action(action, obs_tensor)
            preds.append({
                "date": timestamp,
                "ticker": ticker,
                **decoded,
                "close": window["close"].iloc[-1] if "close" in window else None,
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
        prices = prices.sort_values(["ticker", "date"])

        prices_by_ticker: Dict[str, PricesDf] = {}
        for t, df_t in prices.groupby("ticker"):
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

        Args:
            data: PricesDf (sorted by date) containing model columns.
            ticker: Optional ticker label to attach.
            deterministic: Whether to use deterministic policy.

        Returns:
            DataFrame of predictions for each step >= window_size-1.
        """
        if len(data) == 0:
            return pd.DataFrame()

        preds = []
        cols = data.model_columns
        values = data[cols].values
        ticker_label = ticker or (
            data["ticker"].iloc[0] if "ticker" in data else "")

        for idx in range(self.window_size - 1, len(values)):
            start = idx - self.window_size + 1
            window = values[start: idx + 1]
            obs = window
            if len(window) < self.window_size:
                pad = np.zeros(
                    (self.window_size - len(window), window.shape[1]))
                obs = np.vstack((pad, window))

            obs = np.nan_to_num(obs.astype(np.float32),
                                nan=0.0, posinf=10.0, neginf=-10.0)
            obs = np.clip(obs, -10.0, 10.0)
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Set stock_id in policy if available
            stock_id = None
            if hasattr(self.model.policy, 'set_stock_id'):
                if hasattr(self.env, 'ticker_to_id') and ticker_label in self.env.ticker_to_id:
                    stock_id = self.env.ticker_to_id[ticker_label]
                elif hasattr(self.env, 'current_stock_id'):
                    stock_id = self.env.current_stock_id
                if stock_id is not None:
                    self.model.policy.set_stock_id(stock_id)

            action, _ = self.model.predict(obs, deterministic=deterministic)
            decoded = self._decode_action(action, obs_tensor)
            preds.append({
                "ticker": ticker_label,
                "date": data["date"].iloc[idx],
                "step": idx,
                **decoded,
                "close": data["close"].iloc[idx] if "close" in data else None,
            })

        return pd.DataFrame(preds)

    def predict_timespan(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: Optional[List[str]] = None,
        buffer_days: int = 20,
        deterministic: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions for all timestamps in a window [start, end].

        Fetches data once, prepares per-ticker windows, and iterates across all
        timestamps (e.g., intraday bars). Returns a concatenated DataFrame of
        predictions with one row per ticker per timestamp.
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
        prices = prices.sort_values(["ticker", "date"])

        if len(prices) == 0:
            return pd.DataFrame()

        # Prepare per-ticker data
        prices_by_ticker: Dict[str, PricesDf] = {}
        for ticker, df_t in prices.groupby("ticker"):
            prepared = PricesDf(df_t.copy()).prep_data().sort_values("date")
            prices_by_ticker[ticker] = prepared

        # Unique timestamps within [start, end]
        ts_filtered = prices[(prices["date"] >= start_ts)
                             & (prices["date"] <= end_ts)]
        timestamps = sorted(ts_filtered["date"].unique())
        if not timestamps:
            return pd.DataFrame()

        all_preds = []
        for ts in tqdm(timestamps, desc="Predicting over time periods"):
            preds = self.predict_for_timestamp(
                prices_by_ticker=prices_by_ticker,
                timestamp=pd.to_datetime(ts),
                deterministic=deterministic,
            )
            if not preds.empty:
                all_preds.append(preds)

        if not all_preds:
            return pd.DataFrame()

        return pd.concat(all_preds, ignore_index=True)
