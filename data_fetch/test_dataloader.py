"""
Test script to verify the data loader works with the HDF5 file.
Run: python test_dataloader.py
"""

import h5py
from datamodule import OHLCVDataModule
import config


def inspect_hdf5(path: str = None):
    """Inspect the HDF5 file structure."""
    if path is None:
        path = config.HDF5_FILE
    
    print("=" * 60)
    print("HDF5 File Structure")
    print("=" * 60)
    
    with h5py.File(path, 'r') as f:
        print(f"\nDatasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        print(f"\nAttributes:")
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")
        
        # Sample some data
        print(f"\nSample OHLC (first 3 rows):")
        print(f"  {f['ohlc'][:3]}")
        
        print(f"\nSample volume (first 3):")
        print(f"  {f['volume'][:3]}")
        
        print(f"\nSample value (first 3):")
        print(f"  {f['value'][:3]}")
        
        print(f"\nSample timestamps (first 3):")
        print(f"  {f['timestamps'][:3]}")
        
        # Get ticker names
        ticker_names = [name.decode('utf-8') for name in f['ticker_names'][:5]]
        print(f"\nFirst 5 ticker names: {ticker_names}")


def test_datamodule():
    """Test the DataModule with real data."""
    print("\n" + "=" * 60)
    print("Testing OHLCVDataModule")
    print("=" * 60)
    
    # Create datamodule with fraction-based split (uses config defaults)
    dm = OHLCVDataModule(
        split_by_date=False,  # Use fraction split for testing
        batch_size=32,
        include_value=True,  # Override for testing
    )
    
    # Setup the data
    print("\nSetting up datamodule...")
    dm.setup()
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = dm.train_dataloader()
    
    # Get one batch
    batch = next(iter(train_loader))
    x, y = batch
    
    print(f"\nBatch shapes:")
    print(f"  X (input):  {x.shape}")  # Should be (batch_size, seq_len, n_features)
    print(f"  Y (target): {y.shape}")  # Should be (batch_size,)
    
    print(f"\nFeature stats (first sample):")
    print(f"  X min: {x[0].min().item():.4f}")
    print(f"  X max: {x[0].max().item():.4f}")
    print(f"  X mean: {x[0].mean().item():.4f}")
    
    print(f"\nTarget stats:")
    print(f"  Y min: {y.min().item():.6f}")
    print(f"  Y max: {y.max().item():.6f}")
    print(f"  Y mean: {y.mean().item():.6f}")
    
    # Test val and test loaders
    print("\nTesting val dataloader...")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    print(f"  Val batch X shape: {val_batch[0].shape}")
    
    print("\nTesting test dataloader...")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    print(f"  Test batch X shape: {test_batch[0].shape}")
    
    # Cleanup
    dm.teardown()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return dm


if __name__ == '__main__':
    inspect_hdf5()
    test_datamodule()

