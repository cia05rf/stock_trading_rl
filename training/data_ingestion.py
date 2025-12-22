"""
Data ingestion module for training.

Handles loading price data from HDF5 file and preparing it for
the trading environment.
"""

from pathlib import Path
from typing import List, Optional, Union

import h5py
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler

from shared.logging_config import get_logger
from shared.config import get_config

logger = get_logger(__name__)

# Get config for data paths
config = get_config()


class PricesDf(pd.DataFrame):
    """
    Extended DataFrame for price data with additional methods for
    technical analysis and model preparation.
    """
    
    _metadata = ['model_columns', 'days']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.days = [20, 30, 50, 100]
        self.model_columns = [
            "log_return_open",
            "log_return_high",
            "log_return_low",
            "log_return_close",
            "volume_change",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "rsi_14",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "ema_5_rel",
            "ema_30_rel",
            *[f"close_lag_{d}" for d in self.days],
        ]

    @property
    def _constructor(self):
        return PricesDf

    @property
    def model_data(self):
        """Get only the columns used for modeling."""
        return self[self.model_columns]

    def add_column(self, column_name: str, data) -> 'PricesDf':
        """Add a new column and return self for chaining."""
        self[column_name] = data
        return self

    def augment_prices(self) -> 'PricesDf':
        """Add technical indicators to the dataframe."""
        # Keep an untouched copy of close for trading logic
        self["raw_close"] = self["close"]

        # Stationary log-returns for price levels
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            log_prices = np.log(self[col].clip(lower=1e-8))
            self[f"log_return_{col}"] = log_prices.diff().fillna(0.0)

        # Volume change (percentage)
        self["volume_change"] = (
            self["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        )

        # MACD (12,26,9)
        macd = ta.trend.MACD(
            close=self["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True
        )
        self["macd_line"] = macd.macd()
        self["macd_signal"] = macd.macd_signal()
        self["macd_hist"] = macd.macd_diff()

        # RSI (14)
        self["rsi_14"] = ta.momentum.rsi(self["close"], window=14, fillna=True)

        # Bollinger Bands (volatility proxy)
        bb = ta.volatility.BollingerBands(
            close=self["close"], window=20, window_dev=2, fillna=True
        )
        self["bb_upper"] = bb.bollinger_hband()
        self["bb_lower"] = bb.bollinger_lband()
        self["bb_width"] = bb.bollinger_wband()

        # Relative EMAs to enforce stationarity
        ema_5 = ta.trend.ema_indicator(self["close"], window=5, fillna=True)
        ema_30 = ta.trend.ema_indicator(self["close"], window=30, fillna=True)
        self["ema_5_rel"] = (ema_5 - self["close"]) / self["close"]
        self["ema_30_rel"] = (ema_30 - self["close"]) / self["close"]

        # Lagged close returns
        for d in self.days:
            self[f"close_lag_{d}"] = (self["close"].shift(d) / self["close"]) - 1.0

        # Update model columns with new technical set
        self.model_columns = [
            "log_return_open",
            "log_return_high",
            "log_return_low",
            "log_return_close",
            "volume_change",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "rsi_14",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "ema_5_rel",
            "ema_30_rel",
            *[f"close_lag_{d}" for d in self.days],
        ]

        # Clean up any numerical artifacts before normalization
        self.replace([np.inf, -np.inf], np.nan, inplace=True)
        self[self.model_columns] = self[self.model_columns].fillna(0.0)
        
        return self

    def norm_prices(self) -> 'PricesDf':
        """Normalize price columns using StandardScaler."""
        # Handle empty dataframe - return early to avoid StandardScaler error
        if len(self) == 0:
            return self
        
        scaler = StandardScaler()
        # Ensure raw_close is always available for trading logic
        self["raw_close"] = self.get("raw_close", self["close"])

        for col in self.model_columns:
            self[f"raw_{col}"] = self[col]
            scaled = scaler.fit_transform(self[col].values.reshape(-1, 1)).flatten()
            # Clip to avoid extreme outliers destabilizing the LSTM
            self[col] = np.clip(scaled, -5.0, 5.0).astype(np.float32)
        return self

    def prep_data(self) -> 'PricesDf':
        """Prepare data for modeling: augment and normalize."""
        self = self.augment_prices()
        self = self.norm_prices()
        return self


class Ingestion:
    """
    Data ingestion class for loading price data from HDF5 file.
    """
    
    def __init__(
        self,
        hdf5_path: Optional[Union[str, Path]] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ):
        """
        Initialize the ingestion class.
        
        Args:
            hdf5_path: Path to HDF5 file (defaults to config.HDF5_FILE)
            min_date: Minimum date filter (YYYY-MM-DD format)
            max_date: Maximum date filter (YYYY-MM-DD format)
        """
        self.hdf5_path = Path(hdf5_path) if hdf5_path else config.HDF5_FILE
        self.min_date = min_date
        self.max_date = max_date
        
        # Initialize _file to None first to prevent AttributeError in __del__
        self._file = None
        self._ticker_names = []
        self._ticker_boundaries = np.array([])
        
        if not self.hdf5_path.exists():
            logger.warning(f"HDF5 file not found at {self.hdf5_path}")
        else:
            try:
                logger.info(f"Loading HDF5 file from {self.hdf5_path}")
                self._file = h5py.File(str(self.hdf5_path), 'r')
                self._ticker_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                                      for name in self._file['ticker_names'][:]]
                self._ticker_boundaries = self._file['ticker_boundaries'][:]
                logger.debug(f"Loaded HDF5 file with {len(self._ticker_names)} tickers")
            except (OSError, IOError) as e:
                logger.error(f"Failed to open HDF5 file {self.hdf5_path}: {e}")
                logger.error("The file may be corrupted or in use. Please check the file.")
                self._file = None
                self._ticker_names = []
                self._ticker_boundaries = np.array([])

    def read_tickers(self) -> pd.DataFrame:
        """Read all tickers from the HDF5 file."""
        if self._file is None:
            return pd.DataFrame(columns=['ticker', 'id'])
        
        tickers_df = pd.DataFrame({
            'ticker': self._ticker_names,
            'id': range(len(self._ticker_names))
        })
        return tickers_df

    def read_prices(
        self,
        tickers: Optional[Union[List[str], str]] = None,
        _ticker_fields: Optional[List[str]] = None,
    ) -> PricesDf:
        """
        Read price data for specified tickers.
        
        Args:
            tickers: Single ticker or list of tickers (None for all)
            _ticker_fields: Additional ticker fields to include (ignored for HDF5)
        
        Returns:
            PricesDf with price data
        """
        if self._file is None:
            return PricesDf(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Handle single ticker
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Get ticker indices
        if tickers is not None:
            ticker_indices = [i for i, name in enumerate(self._ticker_names) if name in tickers]
        else:
            ticker_indices = list(range(len(self._ticker_names)))
        if not ticker_indices:
            return PricesDf(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Collect data for each ticker
        all_data = []
        ohlc = self._file['ohlc']
        volume = self._file['volume']
        timestamps = self._file['timestamps']
        
        for ticker_idx in ticker_indices:
            start_idx, end_idx = self._ticker_boundaries[ticker_idx]
            ticker_name = self._ticker_names[ticker_idx]
            
            # Get data slice; guard against corrupted/unsupported HDF5 chunks
            try:
                ticker_ohlc = ohlc[start_idx:end_idx]
                ticker_volume = volume[start_idx:end_idx]
                ticker_timestamps = timestamps[start_idx:end_idx]
            except OSError as exc:
                logger.error(
                    "Failed to read HDF5 data for ticker %s (indexes %s-%s) from %s: %s",
                    ticker_name,
                    start_idx,
                    end_idx,
                    self.hdf5_path,
                    exc,
                )
                logger.error("Skipping ticker %s. Consider rebuilding the HDF5 file.", ticker_name)
                continue
            
            # Convert timestamps to datetime
            dates = pd.to_datetime(ticker_timestamps, unit='s')
            
            # Create DataFrame
            ticker_df = pd.DataFrame({
                'ticker': ticker_name,
                'date': dates,
                'open': ticker_ohlc[:, 0],
                'high': ticker_ohlc[:, 1],
                'low': ticker_ohlc[:, 2],
                'close': ticker_ohlc[:, 3],
                'volume': ticker_volume,
            })
            
            # Apply date filters
            if self.min_date:
                min_dt = pd.to_datetime(self.min_date)
                ticker_df = ticker_df[ticker_df['date'] >= min_dt]
            if self.max_date:
                max_dt = pd.to_datetime(self.max_date)
                ticker_df = ticker_df[ticker_df['date'] <= max_dt]
            
            if len(ticker_df) > 0:
                all_data.append(ticker_df)
        if not all_data:
            return PricesDf(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Combine and sort
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('date')
        
        return PricesDf(result)

    def count_timesteps(self, tickers: Optional[Union[List[str], str]] = None) -> int:
        """
        Count total timesteps for given tickers.
        
        Args:
            tickers: Single ticker or list of tickers
        
        Returns:
            Total number of timesteps
        """
        if self._file is None:
            return 0
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        if tickers is not None:
            ticker_indices = [i for i, name in enumerate(self._ticker_names) if name in tickers]
        else:
            ticker_indices = list(range(len(self._ticker_names)))
        
        total = 0
        for ticker_idx in ticker_indices:
            start_idx, end_idx = self._ticker_boundaries[ticker_idx]
            total += end_idx - start_idx
        
        return total

    def close(self):
        """Close HDF5 file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
