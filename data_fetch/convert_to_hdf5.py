"""
One-time script to convert all CSV files in ../data/ to a single HDF5 file.
This provides 10-100x faster loading for deep learning training.

Usage:
    python convert_to_hdf5.py
    python convert_to_hdf5.py --output custom_path.h5
    python convert_to_hdf5.py --min-avg-volume 100000 --min-avg-value 500000
    python convert_to_hdf5.py --exchange CC  # Crypto only
    python convert_to_hdf5.py --asset-type crypto  # Crypto only
"""

import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import config


def get_csv_files(data_dir: str) -> list[str]:
    """Get all CSV files in the data directory."""
    pattern = os.path.join(data_dir, f"*_{config.INTERVAL}.csv")
    files = glob.glob(pattern)
    return sorted(files)


def parse_csv(filepath: str, nrows: int = None) -> pd.DataFrame:
    """Parse a single CSV file with proper dtypes."""
    df = pd.read_csv(
        filepath,
        dtype={
            'timestamp': np.int64,
            'gmtoffset': np.int32,
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'volume': np.float32,
            'Ticker': str,
            'Exchange': str,
        },
        parse_dates=['datetime'],
        nrows=nrows,
    )
    return df


def get_exchange_from_csv(filepath: str) -> str:
    """Get the exchange code from a CSV file (reads first row only)."""
    try:
        df = parse_csv(filepath, nrows=1)
        if len(df) > 0 and 'Exchange' in df.columns:
            return str(df['Exchange'].iloc[0])
    except Exception:
        pass
    return None


def calculate_liquidity_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate liquidity metrics for a ticker's data.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        dict with liquidity metrics
    """
    volume = df['volume'].fillna(0).values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Average daily volume (handle NaN)
    avg_volume = float(np.nanmean(volume)) if len(volume) > 0 else 0.0

    # Total volume
    total_volume = float(np.nansum(volume))

    # Average daily traded value (midpoint * volume)
    midpoint = (high + low) / 2
    value_arr = midpoint * volume
    # Handle NaN values
    avg_value = float(np.nanmean(value_arr)) if len(value_arr) > 0 else 0.0

    # Total value traded
    total_value = float(np.nansum(value_arr))

    # Median volume (more robust to outliers)
    median_volume = float(np.median(volume))

    # Trading activity: percentage of periods with volume > 0
    trading_pct = float(np.sum(volume > 0) / len(volume)
                        ) if len(volume) > 0 else 0.0

    # Average price (for context)
    avg_price = float(np.mean(close))

    return {
        'avg_volume': avg_volume,
        'total_volume': total_volume,
        'avg_value': avg_value,
        'total_value': total_value,
        'median_volume': median_volume,
        'trading_pct': trading_pct,
        'avg_price': avg_price,
        'n_rows': len(df),
    }


def filter_by_exchange_or_type(
    ticker_info: dict,
    exchange: str = None,
    asset_type: str = None,
) -> dict:
    """
    Filter tickers by exchange code or asset type.

    Args:
        ticker_info: dict of ticker_key -> info dict
        exchange: Exchange code to filter by (e.g., "CC", "FOREX", "US")
        asset_type: Asset type to filter by (e.g., "crypto", "forex", "stock")

    Returns:
        Filtered ticker_info dict
    """
    if not exchange and not asset_type:
        return ticker_info

    # Map asset types to exchange codes
    asset_type_to_exchange = {
        'crypto': 'CC',
        'forex': 'FOREX',
        'stock': None,  # Stocks have various exchanges (US, LSE, etc.)
    }

    # Determine target exchange(s)
    target_exchanges = set()
    if exchange:
        target_exchanges.add(exchange)
    if asset_type:
        if asset_type.lower() in asset_type_to_exchange:
            mapped_exchange = asset_type_to_exchange[asset_type.lower()]
            if mapped_exchange:
                target_exchanges.add(mapped_exchange)
            elif asset_type.lower() == 'stock':
                # For stocks, we need to check if exchange is NOT CC or FOREX
                # We'll filter by checking if exchange is not in crypto/forex
                pass

    filtered = {}
    rejected_count = 0

    for ticker_key, info in ticker_info.items():
        filepath = info['filepath']
        ticker_exchange = get_exchange_from_csv(filepath)

        if ticker_exchange is None:
            rejected_count += 1
            continue

        # Filter by specific exchange
        if exchange:
            if ticker_exchange != exchange:
                rejected_count += 1
                continue

        # Filter by asset type
        if asset_type:
            asset_type_lower = asset_type.lower()
            if asset_type_lower == 'crypto':
                if ticker_exchange != 'CC':
                    rejected_count += 1
                    continue
            elif asset_type_lower == 'forex':
                if ticker_exchange != 'FOREX':
                    rejected_count += 1
                    continue
            elif asset_type_lower == 'stock':
                # Stocks are anything that's not crypto or forex
                if ticker_exchange in ('CC', 'FOREX'):
                    rejected_count += 1
                    continue

        filtered[ticker_key] = info

    # Print filtering summary
    if rejected_count > 0:
        filter_desc = []
        if exchange:
            filter_desc.append(f"exchange={exchange}")
        if asset_type:
            filter_desc.append(f"asset_type={asset_type}")
        print(f"\nExchange/Type filtering results ({', '.join(filter_desc)}):")
        print(f"  Original tickers:  {len(ticker_info)}")
        print(f"  Passed filters:    {len(filtered)}")
        print(f"  Rejected:          {rejected_count}")

    return filtered


def filter_by_liquidity(
    ticker_info: dict,
    min_avg_volume: float = 0,
    min_avg_value: float = 0,
    min_data_points: int = 0,
    min_trading_pct: float = 0,
) -> dict:
    """
    Filter tickers by liquidity criteria.

    Args:
        ticker_info: dict of ticker_key -> info dict (with liquidity metrics)
        min_avg_volume: Minimum average volume per period
        min_avg_value: Minimum average traded value per period
        min_data_points: Minimum number of data points
        min_trading_pct: Minimum percentage of periods with trading activity

    Returns:
        Filtered ticker_info dict
    """
    filtered = {}
    rejected_reasons = {
        'volume': 0,
        'value': 0,
        'data_points': 0,
        'trading_pct': 0,
    }

    for ticker_key, info in ticker_info.items():
        # Check minimum data points
        if info['n_rows'] < min_data_points:
            rejected_reasons['data_points'] += 1
            continue

        # Check minimum average volume
        if info.get('avg_volume', 0) < min_avg_volume:
            rejected_reasons['volume'] += 1
            continue

        # Check minimum average value
        if info.get('avg_value', 0) < min_avg_value:
            rejected_reasons['value'] += 1
            continue

        # Check minimum trading percentage
        if info.get('trading_pct', 0) < min_trading_pct:
            rejected_reasons['trading_pct'] += 1
            continue

        filtered[ticker_key] = info

    # Print filtering summary
    total_rejected = len(ticker_info) - len(filtered)
    if total_rejected > 0:
        print(f"\nLiquidity filtering results:")
        print(f"  Original tickers:  {len(ticker_info)}")
        print(f"  Passed filters:    {len(filtered)}")
        print(f"  Rejected:          {total_rejected}")
        print(f"    - Low volume:      {rejected_reasons['volume']}")
        print(f"    - Low value:       {rejected_reasons['value']}")
        print(f"    - Few data points: {rejected_reasons['data_points']}")
        print(f"    - Low activity:    {rejected_reasons['trading_pct']}")

    return filtered


def convert_csvs_to_hdf5(
    data_dir: str,
    output_path: str,
    min_avg_volume: float = 0,
    min_avg_value: float = 0,
    min_data_points: int = 0,
    min_trading_pct: float = 0,
    exchange: str = None,
    asset_type: str = None,
) -> dict:
    """
    Convert all CSV files to a single HDF5 file.

    Args:
        data_dir: Directory containing CSV files
        output_path: Output HDF5 file path
        min_avg_volume: Minimum average volume for liquidity filter
        min_avg_value: Minimum average traded value for liquidity filter
        min_data_points: Minimum data points required
        min_trading_pct: Minimum trading activity percentage (0-1)
        exchange: Filter by exchange code (e.g., "CC", "FOREX", "US")
        asset_type: Filter by asset type (e.g., "crypto", "forex", "stock")

    Returns:
        dict with statistics about the conversion
    """
    csv_files = get_csv_files(data_dir)

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files to process")

    # First pass: count total rows, collect metadata, and calculate liquidity
    print("Pass 1/2: Scanning files and calculating liquidity metrics...")
    ticker_info = {}

    for filepath in tqdm(csv_files, desc="Scanning"):
        df = parse_csv(filepath)

        # Extract ticker from filename (e.g., "AAPL_US_15m.csv" -> "AAPL_US")
        filename = os.path.basename(filepath)
        ticker_key = filename.replace(f"_{config.INTERVAL}.csv", "")

        # Calculate liquidity metrics
        liquidity = calculate_liquidity_metrics(df)

        ticker_info[ticker_key] = {
            'start_idx': 0,  # Will be set in second pass
            'filepath': filepath,
            **liquidity,  # Include all liquidity metrics
        }

    print(f"Scanned {len(ticker_info)} tickers")

    # Apply exchange/asset type filters first
    if exchange or asset_type:
        ticker_info = filter_by_exchange_or_type(
            ticker_info,
            exchange=exchange,
            asset_type=asset_type,
        )

    if not ticker_info:
        raise ValueError(
            "No tickers passed exchange/asset type filters! Check your filter criteria.")

    # Apply liquidity filters
    applying_filters = any([min_avg_volume > 0, min_avg_value > 0,
                           min_data_points > 0, min_trading_pct > 0])

    if applying_filters:
        print(f"\nApplying liquidity filters:")
        print(f"  min_avg_volume:   {min_avg_volume:,.0f}")
        print(f"  min_avg_value:    {min_avg_value:,.0f}")
        print(f"  min_data_points:  {min_data_points:,}")
        print(f"  min_trading_pct:  {min_trading_pct:.1%}")

        ticker_info = filter_by_liquidity(
            ticker_info,
            min_avg_volume=min_avg_volume,
            min_avg_value=min_avg_value,
            min_data_points=min_data_points,
            min_trading_pct=min_trading_pct,
        )

    if not ticker_info:
        raise ValueError(
            "No tickers passed liquidity filters! Try relaxing the criteria.")

    # Calculate total rows and aggregate stats after filtering
    total_rows = sum(info['n_rows'] for info in ticker_info.values())
    total_volume = sum(info['total_volume'] for info in ticker_info.values())
    total_value = sum(info['total_value'] for info in ticker_info.values())

    print(f"\nTotal rows after filtering: {total_rows:,}")
    print(f"Total volume: {total_volume:,.0f}")
    print(f"Total value: ${total_value:,.0f}")

    # Create HDF5 file with pre-allocated arrays
    print(f"\nPass 2/2: Writing to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Create datasets with chunking for efficient access
        chunk_size = min(10000, total_rows)

        ohlc = f.create_dataset(
            'ohlc',
            shape=(total_rows, 4),
            dtype=np.float32,
            chunks=(chunk_size, 4),
            compression='gzip',
            compression_opts=4,
        )

        volume = f.create_dataset(
            'volume',
            shape=(total_rows,),
            dtype=np.float32,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=4,
        )

        timestamps = f.create_dataset(
            'timestamps',
            shape=(total_rows,),
            dtype=np.int64,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=4,
        )

        ticker_ids = f.create_dataset(
            'ticker_ids',
            shape=(total_rows,),
            dtype=np.int32,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=4,
        )

        value_ds = f.create_dataset(
            'value',
            shape=(total_rows,),
            dtype=np.float32,
            chunks=(chunk_size,),
            compression='gzip',
            compression_opts=4,
        )

        # Create ticker mapping (only for filtered tickers)
        ticker_names = sorted(ticker_info.keys())
        ticker_to_id = {name: idx for idx, name in enumerate(ticker_names)}

        # Store ticker mapping (as dataset, not attribute - too large for attr)
        f.create_dataset(
            'ticker_names',
            data=np.array(ticker_names, dtype='S64'),
            compression='gzip',
        )
        f.attrs['n_tickers'] = len(ticker_names)
        f.attrs['n_rows'] = total_rows
        f.attrs['total_volume'] = total_volume
        f.attrs['total_value'] = total_value

        # Store filter parameters as attributes for reproducibility
        f.attrs['filter_min_avg_volume'] = min_avg_volume
        f.attrs['filter_min_avg_value'] = min_avg_value
        f.attrs['filter_min_data_points'] = min_data_points
        f.attrs['filter_min_trading_pct'] = min_trading_pct
        if exchange:
            f.attrs['filter_exchange'] = exchange
        if asset_type:
            f.attrs['filter_asset_type'] = asset_type

        # Second pass: write data (only filtered tickers)
        current_idx = 0

        # Get file paths for filtered tickers
        filtered_filepaths = [(info['filepath'], ticker_key)
                              for ticker_key, info in ticker_info.items()]

        for filepath, ticker_key in tqdm(filtered_filepaths, desc="Writing"):
            df = parse_csv(filepath)
            n_rows = len(df)

            # Get ticker ID
            ticker_id = ticker_to_id[ticker_key]

            # Update ticker info with actual start index
            ticker_info[ticker_key]['start_idx'] = current_idx

            # Write data
            end_idx = current_idx + n_rows

            ohlc[current_idx:end_idx, 0] = df['open'].values
            ohlc[current_idx:end_idx, 1] = df['high'].values
            ohlc[current_idx:end_idx, 2] = df['low'].values
            ohlc[current_idx:end_idx, 3] = df['close'].values

            vol_data = df['volume'].fillna(0).values
            volume[current_idx:end_idx] = vol_data
            timestamps[current_idx:end_idx] = df['timestamp'].values
            ticker_ids[current_idx:end_idx] = ticker_id

            # Calculate value: midpoint of high/low * volume (0 if vol missing)
            midpoint = (df['high'].values + df['low'].values) / 2
            value_ds[current_idx:end_idx] = midpoint * vol_data

            current_idx = end_idx

        # Store ticker boundaries for efficient per-ticker access
        boundaries = np.zeros((len(ticker_names), 2), dtype=np.int64)
        for ticker_name, info in ticker_info.items():
            tid = ticker_to_id[ticker_name]
            boundaries[tid, 0] = info['start_idx']
            boundaries[tid, 1] = info['start_idx'] + info['n_rows']

        f.create_dataset('ticker_boundaries', data=boundaries)

        # Store liquidity metrics for each ticker (useful for analysis)
        liquidity_metrics = np.zeros((len(ticker_names), 6), dtype=np.float64)
        for ticker_name, info in ticker_info.items():
            tid = ticker_to_id[ticker_name]
            liquidity_metrics[tid, 0] = info.get('avg_volume', 0)
            liquidity_metrics[tid, 1] = info.get('avg_value', 0)
            liquidity_metrics[tid, 2] = info.get('median_volume', 0)
            liquidity_metrics[tid, 3] = info.get('trading_pct', 0)
            liquidity_metrics[tid, 4] = info.get('avg_price', 0)
            liquidity_metrics[tid, 5] = info.get('total_value', 0)

        f.create_dataset(
            'liquidity_metrics',
            data=liquidity_metrics,
            compression='gzip',
        )
        f.attrs['liquidity_metrics_columns'] = 'avg_volume,avg_value,median_volume,trading_pct,avg_price,total_value'

    stats = {
        'total_rows': total_rows,
        'n_tickers': len(ticker_names),
        'total_volume': total_volume,
        'total_value': total_value,
        'output_path': output_path,
        'output_size_mb': os.path.getsize(output_path) / (1024 * 1024),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV price data to HDF5 format for efficient deep learning'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing CSV files (default: ../data)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/prices_highly_liquid.h5',
        help='Output HDF5 file path (default: ../data/prices_highly_liquid.h5)',
    )
    # Liquidity filter arguments
    parser.add_argument(
        '--min-avg-volume',
        type=float,
        default=0,
        help='Minimum average volume per period (default: 0 = no filter)',
    )
    parser.add_argument(
        '--min-avg-value',
        type=float,
        default=0,
        help='Minimum average traded value per period (default: 0 = no filter)',
    )
    parser.add_argument(
        '--min-data-points',
        type=int,
        default=0,
        help='Minimum number of data points required (default: 0 = no filter)',
    )
    parser.add_argument(
        '--min-trading-pct',
        type=float,
        default=0,
        help='Minimum trading activity percentage 0-1 (default: 0 = no filter)',
    )
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Preset to use for the conversion (default: None)',
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default=None,
        help='Filter by exchange code (e.g., CC, FOREX, US, LSE). Mutually exclusive with --asset-type.',
    )
    parser.add_argument(
        '--asset-type',
        type=str,
        default=None,
        choices=['crypto', 'forex', 'stock'],
        help='Filter by asset type: crypto, forex, or stock. Mutually exclusive with --exchange.',
    )

    args = parser.parse_args()

    # Validate that exchange and asset_type are not both specified
    if args.exchange and args.asset_type:
        parser.error("--exchange and --asset-type are mutually exclusive. Use only one.")

    # Presets use other arguments if given otherwise use presets
    if args.preset:
        if args.preset == 'all':
            args.min_avg_volume = args.min_avg_volume if args.min_avg_volume != parser.get_default(
                "min_avg_volume") else 0
            args.min_avg_value = args.min_avg_value if args.min_avg_value != parser.get_default(
                "min_avg_value") else 0
            args.min_data_points = args.min_data_points if args.min_data_points != parser.get_default(
                "min_data_points") else 0
            args.min_trading_pct = args.min_trading_pct if args.min_trading_pct != parser.get_default(
                "min_trading_pct") else 0
        elif args.preset == 'liquid':
            args.min_avg_volume = args.min_avg_volume if args.min_avg_volume != parser.get_default(
                "min_avg_volume") else 100000
            args.min_avg_value = args.min_avg_value if args.min_avg_value != parser.get_default(
                "min_avg_value") else 500000
            args.min_data_points = args.min_data_points if args.min_data_points != parser.get_default(
                "min_data_points") else 500
            args.min_trading_pct = args.min_trading_pct if args.min_trading_pct != parser.get_default(
                "min_trading_pct") else 0.5
        elif args.preset == 'highly_liquid':
            args.min_avg_volume = args.min_avg_volume if args.min_avg_volume != parser.get_default(
                "min_avg_volume") else 500000
            args.min_avg_value = args.min_avg_value if args.min_avg_value != parser.get_default(
                "min_avg_value") else 5000000
            args.min_data_points = args.min_data_points if args.min_data_points != parser.get_default(
                "min_data_points") else 1000
            args.min_trading_pct = args.min_trading_pct if args.min_trading_pct != parser.get_default(
                "min_trading_pct") else 0.9

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = convert_csvs_to_hdf5(
        args.data_dir,
        args.output,
        min_avg_volume=args.min_avg_volume,
        min_avg_value=args.min_avg_value,
        min_data_points=args.min_data_points,
        min_trading_pct=args.min_trading_pct,
        exchange=args.exchange,
        asset_type=args.asset_type,
    )

    print("\n" + "=" * 50)
    print("Conversion complete!")
    print(f"  Total rows:    {stats['total_rows']:,}")
    print(f"  Total tickers: {stats['n_tickers']}")
    print(f"  Total volume:  {stats['total_volume']:,.0f}")
    print(f"  Total value:   ${stats['total_value']:,.0f}")
    print(f"  Output file:   {stats['output_path']}")
    print(f"  File size:     {stats['output_size_mb']:.2f} MB")
    print("=" * 50)


if __name__ == '__main__':
    main()
