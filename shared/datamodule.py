"""
PyTorch Lightning DataModule for OHLCV data.
Handles train/val/test splits, normalization, and DataLoader creation.
"""

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from shared.config import get_config
from shared.dataset import OHLCVDataset
from shared.logging_config import get_logger

logger = get_logger(__name__)


class OHLCVDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for OHLCV time series data.
    
    Features:
    - Chronological train/val/test split (prevents data leakage)
    - Automatic normalization statistics computation
    - Configurable sequence length and batch size
    - Multi-worker data loading
    
    Args:
        hdf5_path: Path to the HDF5 file created by convert_to_hdf5.py
        sequence_length: Number of timesteps in each input sequence
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data for training (chronologically first)
        val_split: Fraction of data for validation (after training data)
        test_split: Remaining data for testing (chronologically last)
        split_by_date: If True, split by date. If False, split by fraction.
        train_end_date: End date for training data (format: 'YYYY-MM-DD')
        val_end_date: End date for validation data (format: 'YYYY-MM-DD')
        normalize: Whether to normalize features
        include_volume: Whether to include volume as a feature
        include_value: Whether to include value (midpoint * volume) as a feature
        target_type: Type of prediction target ('next_close', 'returns', 'direction')
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory for GPU transfer
        respect_ticker_boundaries: If True, don't create windows that span multiple tickers
    """
    
    def __init__(
        self,
        hdf5_path: str = None,
        sequence_length: int = None,
        batch_size: int = None,
        train_split: float = None,
        val_split: float = None,
        split_by_date: bool = None,
        train_end_date: str = None,
        val_end_date: str = None,
        normalize: bool = None,
        include_volume: bool = None,
        include_value: bool = None,
        target_type: str = None,
        num_workers: int = None,
        pin_memory: bool = None,
        respect_ticker_boundaries: bool = None,
    ):
        super().__init__()
        
        # Get config for defaults
        config = get_config()
        
        # Use config defaults if not specified
        self.hdf5_path = str(hdf5_path) if hdf5_path is not None else str(config.HDF5_FILE)
        self.seq_len = sequence_length if sequence_length is not None else config.SEQUENCE_LENGTH
        self.batch_size = batch_size if batch_size is not None else config.DM_BATCH_SIZE
        self.train_split = train_split if train_split is not None else config.TRAIN_SPLIT
        self.val_split = val_split if val_split is not None else config.VAL_SPLIT
        self.split_by_date = split_by_date if split_by_date is not None else config.SPLIT_BY_DATE
        self.train_end_date = train_end_date if train_end_date is not None else config.TRAIN_END_DATE
        self.val_end_date = val_end_date if val_end_date is not None else config.VAL_END_DATE
        self.normalize = normalize if normalize is not None else config.NORMALIZE
        self.include_volume = include_volume if include_volume is not None else config.INCLUDE_VOLUME
        self.include_value = include_value if include_value is not None else config.INCLUDE_VALUE
        self.target_type = target_type if target_type is not None else config.TARGET_TYPE
        self.num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
        self.pin_memory = pin_memory if pin_memory is not None else config.PIN_MEMORY
        self.respect_ticker_boundaries = (
            respect_ticker_boundaries if respect_ticker_boundaries is not None 
            else config.RESPECT_TICKER_BOUNDARIES
        )
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Will be set in setup()
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.test_indices: Optional[np.ndarray] = None
        self.norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        self.train_dataset: Optional[OHLCVDataset] = None
        self.val_dataset: Optional[OHLCVDataset] = None
        self.test_dataset: Optional[OHLCVDataset] = None
        
        logger.info(f"Initialized OHLCVDataModule with HDF5 file: {self.hdf5_path}")
    
    def _get_valid_window_indices(
        self,
        f: h5py.File,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """
        Get valid window start indices within a range.
        A window is valid if it doesn't cross ticker boundaries.
        """
        if not self.respect_ticker_boundaries:
            # Simple case: all indices that fit a window are valid
            max_start = end_idx - self.seq_len - 1  # -1 for target
            if max_start < start_idx:
                return np.array([], dtype=np.int64)
            return np.arange(start_idx, max_start + 1, dtype=np.int64)
        
        # Complex case: respect ticker boundaries
        ticker_boundaries = f['ticker_boundaries'][:]
        
        valid_indices = []
        
        for ticker_start, ticker_end in ticker_boundaries:
            # Find overlap with requested range
            overlap_start = max(start_idx, ticker_start)
            overlap_end = min(end_idx, ticker_end)
            
            # Need at least seq_len + 1 rows for a valid window
            if overlap_end - overlap_start < self.seq_len + 1:
                continue
            
            # Valid window starts within this ticker
            max_window_start = overlap_end - self.seq_len - 1
            if max_window_start >= overlap_start:
                valid_indices.append(
                    np.arange(overlap_start, max_window_start + 1, dtype=np.int64)
                )
        
        if valid_indices:
            return np.concatenate(valid_indices)
        return np.array([], dtype=np.int64)
    
    def _split_by_date(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data chronologically by date."""
        timestamps = f['timestamps'][:]
        
        # Convert dates to timestamps
        train_end_ts = np.datetime64(self.train_end_date).astype('datetime64[s]').astype(np.int64)
        val_end_ts = np.datetime64(self.val_end_date).astype('datetime64[s]').astype(np.int64)
        
        # Find split points
        train_mask = timestamps < train_end_ts
        val_mask = (timestamps >= train_end_ts) & (timestamps < val_end_ts)
        test_mask = timestamps >= val_end_ts
        
        # Get indices for each split
        train_range = np.where(train_mask)[0]
        val_range = np.where(val_mask)[0]
        test_range = np.where(test_mask)[0]
        
        # Get valid window indices for each split
        if len(train_range) > 0:
            train_indices = self._get_valid_window_indices(
                f, train_range[0], train_range[-1] + 1
            )
        else:
            train_indices = np.array([], dtype=np.int64)
        
        if len(val_range) > 0:
            val_indices = self._get_valid_window_indices(
                f, val_range[0], val_range[-1] + 1
            )
        else:
            val_indices = np.array([], dtype=np.int64)
        
        if len(test_range) > 0:
            test_indices = self._get_valid_window_indices(
                f, test_range[0], test_range[-1] + 1
            )
        else:
            test_indices = np.array([], dtype=np.int64)
        
        return train_indices, val_indices, test_indices
    
    def _split_by_fraction(self, f: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data by fraction (within each ticker to maintain temporal order)."""
        ticker_boundaries = f['ticker_boundaries'][:]
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for ticker_start, ticker_end in ticker_boundaries:
            n_ticker_rows = ticker_end - ticker_start
            if n_ticker_rows < self.seq_len + 1:
                continue
            
            # Calculate split points for this ticker
            train_end = ticker_start + int(n_ticker_rows * self.train_split)
            val_end = ticker_start + int(n_ticker_rows * (self.train_split + self.val_split))
            
            # Get valid indices for each split
            train_idx = self._get_valid_window_indices(f, ticker_start, train_end)
            val_idx = self._get_valid_window_indices(f, train_end, val_end)
            test_idx = self._get_valid_window_indices(f, val_end, ticker_end)
            
            if len(train_idx) > 0:
                train_indices.append(train_idx)
            if len(val_idx) > 0:
                val_indices.append(val_idx)
            if len(test_idx) > 0:
                test_indices.append(test_idx)
        
        train_indices = np.concatenate(train_indices) if train_indices else np.array([], dtype=np.int64)
        val_indices = np.concatenate(val_indices) if val_indices else np.array([], dtype=np.int64)
        test_indices = np.concatenate(test_indices) if test_indices else np.array([], dtype=np.int64)
        
        return train_indices, val_indices, test_indices
    
    def _compute_normalization_stats(
        self, f: h5py.File, indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for normalization from training data."""
        if len(indices) == 0:
            # Return neutral stats if no training data
            n_features = 4  # OHLC
            if self.include_volume:
                n_features += 1
            if self.include_value:
                n_features += 1
            return np.zeros(n_features), np.ones(n_features)
        
        # Sample indices for efficiency (don't load all data)
        sample_size = min(10000, len(indices))
        sample_indices = np.random.choice(indices, size=sample_size, replace=False)
        
        # Collect samples
        samples = []
        ohlc = f['ohlc']
        volume = f['volume'] if self.include_volume else None
        value = f['value'] if self.include_value else None
        
        for idx in sample_indices:
            feature_arrays = [ohlc[idx:idx + self.seq_len]]
            if self.include_volume:
                feature_arrays.append(volume[idx:idx + self.seq_len].reshape(-1, 1))
            if self.include_value:
                feature_arrays.append(value[idx:idx + self.seq_len].reshape(-1, 1))
            sample = np.hstack(feature_arrays)
            samples.append(sample)
        
        samples = np.concatenate(samples, axis=0)
        
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        return mean.astype(np.float32), std.astype(np.float32)
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, validate, test, predict).
        Called by Lightning before training/validation/testing.
        """
        logger.info(f"Setting up DataModule for stage: {stage}")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Split data
            if self.split_by_date:
                self.train_indices, self.val_indices, self.test_indices = self._split_by_date(f)
            else:
                self.train_indices, self.val_indices, self.test_indices = self._split_by_fraction(f)
            
            # Compute normalization stats from training data
            if self.normalize:
                self.norm_stats = self._compute_normalization_stats(f, self.train_indices)
            
            logger.info(f"Dataset splits:")
            logger.info(f"  Train samples: {len(self.train_indices):,}")
            logger.info(f"  Val samples:   {len(self.val_indices):,}")
            logger.info(f"  Test samples:  {len(self.test_indices):,}")
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = OHLCVDataset(
                hdf5_path=self.hdf5_path,
                indices=self.train_indices,
                sequence_length=self.seq_len,
                normalize=self.normalize,
                norm_stats=self.norm_stats,
                include_volume=self.include_volume,
                include_value=self.include_value,
                target_type=self.target_type,
            )
            self.val_dataset = OHLCVDataset(
                hdf5_path=self.hdf5_path,
                indices=self.val_indices,
                sequence_length=self.seq_len,
                normalize=self.normalize,
                norm_stats=self.norm_stats,
                include_volume=self.include_volume,
                include_value=self.include_value,
                target_type=self.target_type,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = OHLCVDataset(
                hdf5_path=self.hdf5_path,
                indices=self.test_indices,
                sequence_length=self.seq_len,
                normalize=self.normalize,
                norm_stats=self.norm_stats,
                include_volume=self.include_volume,
                include_value=self.include_value,
                target_type=self.target_type,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with shuffling."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader (no shuffling)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader (no shuffling)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        if self.train_dataset is not None:
            self.train_dataset.close()
        if self.val_dataset is not None:
            self.val_dataset.close()
        if self.test_dataset is not None:
            self.test_dataset.close()


def create_datamodule(
    hdf5_path: str = None,
    sequence_length: int = None,
    batch_size: int = None,
    **kwargs,
) -> OHLCVDataModule:
    """
    Create and set up an OHLCVDataModule with sensible defaults.
    
    Example:
        datamodule = create_datamodule('./data/prices_highly_liquid.h5', batch_size=128)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model, datamodule)
    """
    dm = OHLCVDataModule(
        hdf5_path=hdf5_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        **kwargs,
    )
    return dm

