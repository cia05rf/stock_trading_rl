"""
Tests for shared module.

Tests configuration loading, logging setup, and data modules.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import Config, get_config
from shared.logging_config import setup_logging, get_logger


class TestConfig:
    """Tests for Config class."""
    
    def test_config_initialization(self):
        """Test that Config initializes without errors."""
        config = Config()
        assert config is not None
    
    def test_config_has_required_attributes(self):
        """Test that config has all required attributes."""
        config = get_config()
        
        # Check path attributes
        assert hasattr(config, 'PROJECT_ROOT')
        assert hasattr(config, 'DATA_DIR')
        assert hasattr(config, 'HDF5_FILE')
        
        # Check training attributes
        assert hasattr(config, 'INITIAL_BALANCE')
        assert hasattr(config, 'WINDOW_SIZE')
        assert hasattr(config, 'TOTAL_TIMESTEPS')
        
        # Check data module attributes
        assert hasattr(config, 'SEQUENCE_LENGTH')
        assert hasattr(config, 'DM_BATCH_SIZE')
        assert hasattr(config, 'TRAIN_SPLIT')
    
    def test_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_config_types(self):
        """Test that config values have correct types."""
        config = get_config()
        
        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.INITIAL_BALANCE, float)
        assert isinstance(config.WINDOW_SIZE, int)
        assert isinstance(config.TRAIN_SPLIT, float)


class TestLogging:
    """Tests for logging configuration."""
    
    def test_setup_logging(self):
        """Test that logging setup works."""
        logger = setup_logging(level="DEBUG", log_to_file=False)
        assert logger is not None
    
    def test_get_logger(self):
        """Test that get_logger returns a logger."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"
    
    def test_logger_levels(self):
        """Test that logger respects log levels."""
        setup_logging(level="WARNING", log_to_file=False)
        logger = get_logger("test_levels")
        
        # Should not raise
        logger.warning("Test warning")
        logger.error("Test error")


class TestDataset:
    """Tests for OHLCVDataset."""
    
    @pytest.fixture
    def mock_hdf5_file(self, tmp_path):
        """Create a mock HDF5 file for testing."""
        import h5py
        
        hdf5_path = tmp_path / "test_prices_highly_liquid.h5"
        
        n_rows = 1000
        with h5py.File(hdf5_path, 'w') as f:
            # Create OHLC data
            f.create_dataset('ohlc', data=np.random.randn(n_rows, 4).astype(np.float32))
            f.create_dataset('volume', data=np.random.randn(n_rows).astype(np.float32))
            f.create_dataset('value', data=np.random.randn(n_rows).astype(np.float32))
            f.create_dataset('timestamps', data=np.arange(n_rows).astype(np.int64))
            f.create_dataset('ticker_ids', data=np.zeros(n_rows).astype(np.int32))
            f.create_dataset('ticker_boundaries', data=np.array([[0, n_rows]]).astype(np.int64))
        
        return hdf5_path
    
    def test_dataset_creation(self, mock_hdf5_file):
        """Test dataset creation."""
        from shared.dataset import OHLCVDataset
        
        indices = np.arange(100)
        dataset = OHLCVDataset(
            hdf5_path=str(mock_hdf5_file),
            indices=indices,
            sequence_length=60,
        )
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self, mock_hdf5_file):
        """Test dataset __getitem__."""
        from shared.dataset import OHLCVDataset
        
        indices = np.arange(100)
        dataset = OHLCVDataset(
            hdf5_path=str(mock_hdf5_file),
            indices=indices,
            sequence_length=60,
        )
        
        x, y = dataset[0]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape[0] == 60  # sequence length
    
    def test_dataset_target_types(self, mock_hdf5_file):
        """Test different target types."""
        from shared.dataset import OHLCVDataset
        
        indices = np.arange(100)
        
        for target_type in ['next_close', 'returns', 'direction']:
            dataset = OHLCVDataset(
                hdf5_path=str(mock_hdf5_file),
                indices=indices,
                sequence_length=60,
                target_type=target_type,
            )
            
            x, y = dataset[0]
            assert y.shape == ()  # scalar target


class TestDataModule:
    """Tests for OHLCVDataModule."""
    
    @pytest.fixture
    def mock_hdf5_file(self, tmp_path):
        """Create a mock HDF5 file for testing."""
        import h5py
        
        hdf5_path = tmp_path / "test_prices_highly_liquid.h5"
        
        n_rows = 1000
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('ohlc', data=np.random.randn(n_rows, 4).astype(np.float32))
            f.create_dataset('volume', data=np.random.randn(n_rows).astype(np.float32))
            f.create_dataset('value', data=np.random.randn(n_rows).astype(np.float32))
            f.create_dataset('timestamps', data=np.arange(n_rows).astype(np.int64))
            f.create_dataset('ticker_ids', data=np.zeros(n_rows).astype(np.int32))
            f.create_dataset('ticker_boundaries', data=np.array([[0, n_rows]]).astype(np.int64))
        
        return hdf5_path
    
    def test_datamodule_creation(self, mock_hdf5_file):
        """Test DataModule creation."""
        from shared.datamodule import OHLCVDataModule
        
        dm = OHLCVDataModule(
            hdf5_path=str(mock_hdf5_file),
            sequence_length=60,
            batch_size=32,
            split_by_date=False,
        )
        
        assert dm is not None
    
    def test_datamodule_setup(self, mock_hdf5_file):
        """Test DataModule setup."""
        from shared.datamodule import OHLCVDataModule
        
        dm = OHLCVDataModule(
            hdf5_path=str(mock_hdf5_file),
            sequence_length=60,
            batch_size=32,
            split_by_date=False,
        )
        
        dm.setup()
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
    
    def test_datamodule_dataloaders(self, mock_hdf5_file):
        """Test DataModule dataloaders."""
        from shared.datamodule import OHLCVDataModule
        
        dm = OHLCVDataModule(
            hdf5_path=str(mock_hdf5_file),
            sequence_length=60,
            batch_size=32,
            split_by_date=False,
        )
        
        dm.setup()
        
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
        
        # Get one batch
        batch = next(iter(train_dl))
        assert len(batch) == 2  # x, y

