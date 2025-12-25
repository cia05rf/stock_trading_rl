"""
Visualize stock prices as candlestick charts.

Reads price data from HDF5 files and generates color-coded candlestick charts
for each ticker, ordered by timestamp.

Uses mplfinance for professional-quality financial charts.

Usage:
    python charting/visualize_prices.py
    python charting/visualize_prices.py --hdf5-file data/prices_highly_liquid.h5
    python charting/visualize_prices.py --tickers AAPL_US MSFT_US --output-dir charts
"""

import argparse
import h5py
import pandas as pd
import mplfinance as mpf
from pathlib import Path
from typing import List, Optional

from shared.config import get_config
from shared.logging_config import get_logger

logger = get_logger(__name__)


def load_data_from_hdf5(
    hdf5_path: Path,
    tickers: Optional[List[str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load price data from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        tickers: Optional list of tickers to load (None for all)
        min_date: Optional minimum date filter (YYYY-MM-DD)
        max_date: Optional maximum date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    logger.info(f"Loading data from {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get ticker information
        ticker_names = [name.decode('utf-8') for name in f['ticker_names'][:]]
        ticker_boundaries = f['ticker_boundaries'][:]
        
        # Filter tickers if specified
        if tickers is not None:
            ticker_indices = [i for i, name in enumerate(ticker_names) if name in tickers]
            if not ticker_indices:
                logger.warning(f"No matching tickers found. Available tickers: {ticker_names[:10]}...")
                return pd.DataFrame(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
        else:
            ticker_indices = list(range(len(ticker_names)))
        
        # Load data arrays
        ohlc = f['ohlc']
        volume = f['volume']
        timestamps = f['timestamps']
        
        all_data = []
        
        for ticker_idx in ticker_indices:
            start_idx, end_idx = ticker_boundaries[ticker_idx]
            ticker_name = ticker_names[ticker_idx]
            
            try:
                # Extract data for this ticker
                ticker_ohlc = ohlc[start_idx:end_idx]
                ticker_volume = volume[start_idx:end_idx]
                ticker_timestamps = timestamps[start_idx:end_idx]
                
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
                if min_date:
                    min_dt = pd.to_datetime(min_date)
                    ticker_df = ticker_df[ticker_df['date'] >= min_dt]
                if max_date:
                    max_dt = pd.to_datetime(max_date)
                    ticker_df = ticker_df[ticker_df['date'] <= max_dt]
                
                if len(ticker_df) > 0:
                    all_data.append(ticker_df)
                    
            except Exception as e:
                logger.error(f"Error loading data for {ticker_name}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame(columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Combine and sort by ticker and timestamp
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['ticker', 'date'])
        
        logger.info(f"Loaded {len(result)} rows for {result['ticker'].nunique()} tickers")
        return result


def prepare_data_for_mplfinance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for mplfinance.
    
    mplfinance expects:
    - Index: DatetimeIndex
    - Columns: Open, High, Low, Close, Volume (case-sensitive)
    
    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
    
    Returns:
        DataFrame formatted for mplfinance
    """
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Ensure date column is datetime
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Set date as index
    df_copy.set_index('date', inplace=True)
    
    # Rename columns to match mplfinance expectations (case-sensitive)
    df_copy = df_copy.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if 'Volume' not in df_copy.columns:
        df_copy['Volume'] = 0
    
    # Sort by index (date)
    df_copy = df_copy.sort_index()
    
    # Remove any duplicate indices (keep last)
    df_copy = df_copy[~df_copy.index.duplicated(keep='last')]
    
    return df_copy[required_cols + ['Volume']]


def create_charts(
    df: pd.DataFrame,
    output_dir: Path,
    max_candles_per_chart: int = 1000,
    max_tickers: Optional[int] = None,
    show_volume: bool = True,
):
    """
    Create candlestick charts for each ticker using mplfinance.
    
    Args:
        df: DataFrame with price data
        output_dir: Directory to save charts
        max_candles_per_chart: Maximum candles per chart (for performance)
        max_tickers: Maximum number of tickers to chart (None for all)
        show_volume: Whether to show volume bars
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique tickers
    tickers = df['ticker'].unique()
    if max_tickers:
        tickers = tickers[:max_tickers]
        logger.info(f"Limiting to {max_tickers} tickers")
    
    logger.info(f"Creating charts for {len(tickers)} tickers")
    
    # Define style for candlesticks
    # Use built-in 'charles' style which has green/red colors
    # This is simpler and more reliable than custom marketcolors
    style = 'charles'
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        if len(ticker_data) == 0:
            logger.warning(f"No data for {ticker}")
            continue
        
        # Limit candles for performance if needed
        if len(ticker_data) > max_candles_per_chart:
            logger.info(f"Limiting {ticker} to {max_candles_per_chart} candles (from {len(ticker_data)})")
            # Take evenly spaced samples
            step = len(ticker_data) // max_candles_per_chart
            ticker_data = ticker_data.iloc[::step].copy()
        
        # Prepare data for mplfinance
        try:
            mplf_data = prepare_data_for_mplfinance(ticker_data)
        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {e}")
            continue
        
        if len(mplf_data) == 0:
            logger.warning(f"No valid data for {ticker} after preparation")
            continue
        
        # Create output path
        safe_ticker = ticker.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"{safe_ticker}_candlestick.png"
        
        try:
            # Create candlestick chart
            # volume parameter expects bool or Axes, not string
            mpf.plot(
                mplf_data,
                type='candle',
                style=style,
                volume=show_volume,  # Boolean, not string
                title=f"{ticker} - Price Chart ({len(mplf_data)} candles)",
                ylabel='Price',
                ylabel_lower='Volume' if show_volume else '',
                figsize=(14, 8),
                savefig=dict(
                    fname=str(output_path),
                    dpi=150,
                    bbox_inches='tight',
                ),
                show_nontrading=False,
                tight_layout=True,
            )
            
            logger.info(f"Saved chart for {ticker} to {output_path}")
        except Exception as e:
            logger.error(f"Error creating chart for {ticker}: {e}")
            continue
    
    logger.info(f"All charts saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate candlestick charts from HDF5 price data"
    )
    parser.add_argument(
        '--hdf5-file',
        type=str,
        default=None,
        help='Path to HDF5 file (default: from config)',
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=None,
        help='Specific tickers to chart (default: all)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for charts (default: charting/output)',
    )
    parser.add_argument(
        '--min-date',
        type=str,
        default=None,
        help='Minimum date filter (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--max-date',
        type=str,
        default=None,
        help='Maximum date filter (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--max-candles',
        type=int,
        default=1000,
        help='Maximum candles per chart (default: 1000)',
    )
    parser.add_argument(
        '--max-tickers',
        type=int,
        default=None,
        help='Maximum number of tickers to chart (default: all)',
    )
    parser.add_argument(
        '--no-volume',
        action='store_true',
        help='Hide volume bars from charts',
    )
    
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    
    # Determine HDF5 file path
    if args.hdf5_file:
        hdf5_path = Path(args.hdf5_file)
    else:
        hdf5_path = config.HDF5_FILE
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'output'
    
    # Load data
    df = load_data_from_hdf5(
        hdf5_path,
        tickers=args.tickers,
        min_date=args.min_date,
        max_date=args.max_date,
    )
    
    if len(df) == 0:
        logger.error("No data loaded. Check HDF5 file path and filters.")
        return
    
    # Create charts
    create_charts(
        df,
        output_dir,
        max_candles_per_chart=args.max_candles,
        max_tickers=args.max_tickers,
        show_volume=not args.no_volume,
    )
    
    logger.info("Chart generation complete!")


if __name__ == '__main__':
    main()


