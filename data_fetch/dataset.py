"""
Custom PyTorch Dataset for OHLCV data stored in HDF5 format.
Supports efficient sliding window access and multiprocessing DataLoader.
"""

from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class OHLCVDataset(Dataset):
    """
    PyTorch Dataset for OHLCV (Open, High, Low, Close, Volume) time series data.
    
    Features:
    - Lazy HDF5 file opening for multiprocessing compatibility
    - Sliding window sequences for time series models
    - Optional normalization
    - Memory-efficient: only loads requested windows
    
    Args:
        hdf5_path: Path to the HDF5 file created by convert_to_hdf5.py
        indices: Array of valid window start indices for this split
        sequence_length: Number of timesteps in each input sequence
        normalize: Whether to apply normalization
        norm_stats: Tuple of (mean, std) arrays for normalization. If None and
                    normalize=True, will compute from data (not recommended for val/test)
        include_volume: Whether to include volume as a feature
        include_value: Whether to include value (midpoint * volume) as a feature
        target_type: Type of prediction target:
            - 'next_close': Predict the next close price
            - 'returns': Predict percentage return of next close
            - 'direction': Binary classification (1 if price goes up, 0 otherwise)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        indices: np.ndarray,
        sequence_length: int = 60,
        normalize: bool = True,
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        include_volume: bool = True,
        include_value: bool = False,
        target_type: str = 'returns',
    ):
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.seq_len = sequence_length
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.include_volume = include_volume
        self.include_value = include_value
        self.target_type = target_type
        
        # File handle - opened lazily for multiprocessing compatibility
        self._file: Optional[h5py.File] = None
        
        # Validate target type
        valid_targets = ('next_close', 'returns', 'direction')
        if target_type not in valid_targets:
            raise ValueError(f"target_type must be one of {valid_targets}")
    
    def _open_file(self) -> h5py.File:
        """Open HDF5 file lazily (for multiprocessing compatibility)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
        return self._file
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            x: Input tensor of shape (seq_len, n_features)
               Features are [open, high, low, close] + optional [volume, value]
            y: Target tensor (scalar for regression, or class index for classification)
        """
        f = self._open_file()
        
        start = self.indices[idx]
        end = start + self.seq_len
        
        # Load OHLC data
        ohlc = f['ohlc'][start:end + 1]  # +1 for target
        
        # Build feature list: OHLC + optional volume + optional value
        feature_arrays = [ohlc[:-1]]
        
        if self.include_volume:
            volume = f['volume'][start:end + 1]
            feature_arrays.append(volume[:-1].reshape(-1, 1))
        
        if self.include_value:
            value = f['value'][start:end + 1]
            feature_arrays.append(value[:-1].reshape(-1, 1))
        
        features = np.hstack(feature_arrays)
        
        # Compute target
        if self.target_type == 'next_close':
            target = ohlc[-1, 3]  # Next close price
        elif self.target_type == 'returns':
            # Percentage return: (next_close - current_close) / current_close
            current_close = ohlc[-2, 3]
            next_close = ohlc[-1, 3]
            if current_close != 0:
                target = (next_close - current_close) / current_close
            else:
                target = 0.0
        elif self.target_type == 'direction':
            # Binary: 1 if price goes up, 0 otherwise
            current_close = ohlc[-2, 3]
            next_close = ohlc[-1, 3]
            target = 1.0 if next_close > current_close else 0.0
        
        # Apply normalization
        if self.normalize and self.norm_stats is not None:
            mean, std = self.norm_stats
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            features = (features - mean) / std
        
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        
        return x, y
    
    def close(self):
        """Close the HDF5 file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def __del__(self):
        self.close()


class OHLCVSequenceDataset(Dataset):
    """
    Alternative dataset that returns sequence-to-sequence predictions.
    Useful for models like Transformers that predict at every timestep.
    
    Returns both input sequences and target sequences of the same length.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        indices: np.ndarray,
        sequence_length: int = 60,
        normalize: bool = True,
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        include_volume: bool = True,
        include_value: bool = False,
    ):
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.seq_len = sequence_length
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.include_volume = include_volume
        self.include_value = include_value
        self._file: Optional[h5py.File] = None
    
    def _open_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
        return self._file
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input tensor of shape (seq_len, n_features)
            y: Target tensor of shape (seq_len,) - next close at each step
        """
        f = self._open_file()
        
        start = self.indices[idx]
        end = start + self.seq_len
        
        # Load OHLC data with one extra for final target
        ohlc = f['ohlc'][start:end + 1]
        
        # Build feature list: OHLC + optional volume + optional value
        feature_arrays = [ohlc[:-1]]
        
        if self.include_volume:
            volume = f['volume'][start:end + 1]
            feature_arrays.append(volume[:-1].reshape(-1, 1))
        
        if self.include_value:
            value = f['value'][start:end + 1]
            feature_arrays.append(value[:-1].reshape(-1, 1))
        
        features = np.hstack(feature_arrays)
        
        # Target is the next close price at each timestep
        targets = ohlc[1:, 3]  # Shape: (seq_len,)
        
        if self.normalize and self.norm_stats is not None:
            mean, std = self.norm_stats
            std = np.where(std == 0, 1, std)
            features = (features - mean) / std
        
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        
        return x, y
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def __del__(self):
        self.close()

