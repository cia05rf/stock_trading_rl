"""
Fetch price data using either yfinance or EODHD API.

This script supports both data sources and can resume from the latest
timestamp in existing CSV files.

Note: yfinance has a 60-day limit for 15-minute interval data. For historical
data beyond 60 days, use the EODHD API source.

Usage:
    # Use yfinance (default) - good for recent data (within 60 days)
    python fetch_prices.py
    
    # Use yfinance explicitly
    python fetch_prices.py --source yfinance
    
    # Use EODHD API - for historical data beyond 60 days
    python fetch_prices.py --source eodhd
"""
import argparse
import asyncio
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config

# Import EODHD functions if needed
try:
    from price_scraper import (
        get_all_symbols as get_all_symbols_eodhd,
        process_symbol as process_symbol_eodhd,
        main as main_eodhd,
    )
    EODHD_AVAILABLE = True
except ImportError:
    EODHD_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_last_timestamp(file_path: Path) -> Optional[int]:
    """
    Get the last timestamp from an existing CSV file.
    Efficiently reads the last line to extract the timestamp.
    """
    if not file_path.exists():
        return None
    
    try:
        # Use simple file reading to get last line
        with open(file_path, 'rb') as f:
            try:
                # Go to the end of file
                f.seek(-2, os.SEEK_END)
                # Read backwards until newline
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                # File is too small (e.g. only header or one line)
                f.seek(0)
            
            last_line = f.readline().decode().strip()
            
            # Check if it's the header or empty
            if not last_line or "timestamp" in last_line.lower():
                return None
            
            # Parse timestamp (first column)
            try:
                return int(last_line.split(',')[0])
            except ValueError:
                return None
                
    except Exception as e:
        logger.warning("Error reading last timestamp from %s: %s", file_path, e)
        return None


def get_yfinance_symbol(ticker: str, exchange_code: str) -> str:
    """
    Convert (ticker, exchange) to yfinance symbol format.
    
    For US stocks, just use the ticker.
    For other exchanges, yfinance uses exchange suffixes.
    """
    if exchange_code == "US":
        return ticker
    elif exchange_code == "LSE":
        # London Stock Exchange
        return f"{ticker}.L"
    elif exchange_code == "FOREX":
        # Forex pairs - yfinance uses format like EURUSD=X
        # EODHD format might be like "EURUSD" or "EUR/USD"
        clean_ticker = ticker.replace("/", "").replace("-", "")
        if len(clean_ticker) == 6:  # e.g., EURUSD
            return f"{clean_ticker}=X"
        elif "=" in ticker:
            # Already in yfinance format
            return ticker
        return f"{clean_ticker}=X"
    elif exchange_code == "CC":
        # Cryptocurrency - yfinance uses format like BTC-USD
        # EODHD format might be like "BTC" or "BTCUSD"
        if "-" in ticker:
            # Already in yfinance format
            return ticker
        elif ticker.endswith("USD") and len(ticker) > 3:
            # e.g., BTCUSD -> BTC-USD
            base = ticker[:-3]
            return f"{base}-USD"
        else:
            # Assume USD quote
            return f"{ticker}-USD"
    else:
        # Default: try ticker as-is
        logger.debug(
            "Unknown exchange code %s for ticker %s, using ticker as-is",
            exchange_code, ticker
        )
        return ticker


def fetch_yfinance_data(
    ticker: str,
    exchange_code: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: str = '15m'
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday data using yfinance.
    
    Note: yfinance has a 60-day limit for 15-minute interval data.
    For intervals beyond 60 days, this function will only return
    the most recent 60 days of data.
    
    Args:
        ticker: Stock ticker symbol
        exchange_code: Exchange code (US, LSE, etc.)
        start_date: Start datetime
        end_date: End datetime
        interval: Data interval (15m, 1h, 1d, etc.)
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    yf_symbol = get_yfinance_symbol(ticker, exchange_code)
    
    try:
        # yfinance uses different interval strings
        yf_interval = interval
        if interval == '15m':
            yf_interval = '15m'
        elif interval == '1h':
            yf_interval = '1h'
        elif interval == '1d':
            yf_interval = '1d'
        
        # For 15m intervals, yfinance only provides last 60 days
        # Adjust start_date if needed
        if interval == '15m':
            max_start = end_date - datetime.timedelta(days=60)
            if start_date < max_start:
                logger.debug(
                    "yfinance 15m limit: adjusting start_date from %s to %s for %s",
                    start_date, max_start, yf_symbol
                )
                start_date = max_start
        
        # Download data
        stock = yf.Ticker(yf_symbol)
        df = stock.history(
            start=start_date,
            end=end_date,
            interval=yf_interval,
            prepost=False,  # Don't include pre/post market data
            repair=True,  # Auto-repair common issues
        )
        
        if df.empty:
            logger.debug("No data returned for %s (%s)", yf_symbol, ticker)
            return None
        
        # Convert to standard format
        # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        # Index is DatetimeIndex
        
        # Reset index to get datetime as column
        df = df.reset_index()
        
        # Handle datetime column name (could be 'Date' or 'Datetime')
        datetime_col = None
        for col in ['Datetime', 'Date', 'datetime']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            # If no datetime column found, use index
            df['datetime'] = df.index
            datetime_col = 'datetime'
        
        # Rename datetime column to 'datetime'
        if datetime_col != 'datetime':
            df = df.rename(columns={datetime_col: 'datetime'})
        
        # Rename other columns to lowercase
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        })
        
        # Convert datetime to timestamp
        df['timestamp'] = df['datetime'].apply(
            lambda x: int(pd.Timestamp(x).timestamp())
        )
        # Calculate GMT offset (simplified - assumes UTC)
        df['gmtoffset'] = 0
        
        # Ensure datetime is in the right format (string for CSV)
        df['datetime'] = df['datetime'].apply(
            lambda x: pd.Timestamp(x).strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Add ticker and exchange columns
        df['Ticker'] = ticker
        df['Exchange'] = exchange_code
        
        # Select and order columns to match expected format
        columns = [
            'timestamp', 'gmtoffset', 'datetime', 'open', 'high', 'low',
            'close', 'volume', 'Ticker', 'Exchange'
        ]
        df = df[[col for col in columns if col in df.columns]]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        logger.warning(
            "Error fetching %s (%s) from yfinance: %s",
            yf_symbol, ticker, e
        )
        return None


def process_symbol_yfinance(
    ticker: str,
    exchange_code: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: str = '15m'
) -> Tuple[str, str, int]:
    """
    Process a single symbol using yfinance and return ticker, exchange, and row count.
    All outputs are saved to the data directory.
    """
    output_file = config.OUTPUT_DIR / f"{ticker}_{exchange_code}_{interval}.csv"
    logger.debug(
        "Output file for %s.%s: %s", ticker, exchange_code, output_file
    )
    
    # Check for existing data to resume
    current_start = start_date
    resume = False
    
    last_ts = get_last_timestamp(output_file)
    if last_ts:
        # Convert timestamp to datetime
        last_dt = datetime.datetime.fromtimestamp(last_ts)
        logger.debug(
            "Found existing data for %s.%s up to %s. Resuming...",
            ticker, exchange_code, last_dt
        )
        # Start from the next interval to avoid overlap
        # For 15m intervals, add 15 minutes
        if interval == '15m':
            current_start = last_dt + datetime.timedelta(minutes=15)
        elif interval == '1h':
            current_start = last_dt + datetime.timedelta(hours=1)
        else:
            current_start = last_dt + datetime.timedelta(days=1)
        
        # Ensure we don't start after end_date
        if current_start >= end_date:
            logger.debug(
                "%s.%s is already up to date.", ticker, exchange_code
            )
            return (ticker, exchange_code, 0)
        
        resume = True
    
    # Fetch data
    df = fetch_yfinance_data(
        ticker, exchange_code, current_start, end_date, interval
    )
    
    if df is None or df.empty:
        return (ticker, exchange_code, 0)
    
    # Write to CSV
    mode = 'a' if resume else 'w'
    header = not resume
    
    df.to_csv(output_file, mode=mode, header=header, index=False)
    
    return (ticker, exchange_code, len(df))


def get_all_symbols() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get all available symbol lists from symbol_lists.json.
    Returns dict mapping list_type to list of (ticker, exchange_code) tuples.
    """
    json_file = config.SYMBOL_LISTS_FILE
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                result = {}
                for list_type, items in data.items():
                    # Convert lists back to tuples
                    result[list_type] = [tuple(item) for item in items]
                return result
        except Exception as e:
            logger.warning("Error loading from JSON: %s", e)
    
    logger.warning(
        "No symbols found. Run fetch_symbols.py to fetch full lists."
    )
    return {}


def main_yfinance():
    """
    Main function to fetch data for all symbols using yfinance.
    """
    # Adjust date range for test mode
    if config.TEST_MODE:
        END = datetime.datetime.now()
        START = END - datetime.timedelta(days=7)
        logger.info(
            "TEST MODE ENABLED - Limited to %s tickers, %s records each",
            config.TEST_TICKERS_LIMIT, config.TEST_RECORDS_LIMIT
        )
    else:
        START = config.START_DATE
        # Ensure END_DATE is set to now if it's None/null
        END = config.END_DATE if config.END_DATE is not None else datetime.datetime.now()
    
    # Get all symbols to process
    all_symbols = []
    
    logger.info("Collecting symbol lists...")
    logger.info("All output files will be saved to: %s", config.OUTPUT_DIR)
    
    symbol_groups = get_all_symbols()
    
    for list_type, symbols in symbol_groups.items():
        all_symbols.extend(symbols)
        logger.info("Added %s symbols from %s", len(symbols), list_type)
    
    if not all_symbols:
        logger.warning("No symbols found in symbol_lists.json")
        return
    
    # Limit symbols in test mode
    if config.TEST_MODE:
        all_symbols = all_symbols[:config.TEST_TICKERS_LIMIT]
        logger.info(
            "TEST MODE: Limited to first %s symbols", config.TEST_TICKERS_LIMIT
        )
    
    logger.info("Processing %d symbols with yfinance...", len(all_symbols))
    logger.info("Interval: %s", config.INTERVAL)
    logger.info("Date range: %s to %s", START, END)
    
    start_time = time.time()
    successful = 0
    total_rows = 0
    failed = []
    
    # Process symbols with progress bar
    pbar = tqdm(
        total=len(all_symbols),
        desc="📈 Fetching prices (yfinance)",
        unit="sym",
        ncols=100
    )
    
    for ticker, exchange in all_symbols:
        try:
            ticker, exchange, rows = process_symbol_yfinance(
                ticker, exchange, START, END, config.INTERVAL
            )
            if rows > 0:
                successful += 1
                total_rows += rows
        except Exception as e:
            logger.error("Error processing %s.%s: %s", ticker, exchange, e)
            failed.append((ticker, exchange, str(e)))
        finally:
            pbar.update(1)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
    
    pbar.close()
    
    elapsed = time.time() - start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info("✅ Completed in %.1fs", elapsed)
    logger.info("   Successful: %d/%d symbols", successful, len(all_symbols))
    logger.info("   Total rows: %s", f"{total_rows:,}")
    logger.info("   Failed: %d", len(failed))
    logger.info("   Output: %s", config.OUTPUT_DIR.absolute())
    if failed:
        logger.info("   Failed symbols:")
        for ticker, exchange, error in failed[:10]:  # Show first 10
            logger.info("     - %s.%s: %s", ticker, exchange, error)
    logger.info("=" * 60)


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Fetch price data using yfinance or EODHD API"
    )
    parser.add_argument(
        '--source',
        choices=['yfinance', 'eodhd'],
        default='yfinance',
        help='Data source to use (default: yfinance)'
    )
    
    args = parser.parse_args()
    
    if args.source == 'yfinance':
        logger.info("Using yfinance as data source")
        main_yfinance()
    elif args.source == 'eodhd':
        if not EODHD_AVAILABLE:
            logger.error(
                "EODHD support not available. "
                "Make sure price_scraper.py is accessible."
            )
            return
        logger.info("Using EODHD API as data source")
        # Run the async EODHD main function
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main_eodhd())
    else:
        logger.error("Unknown data source: %s", args.source)


if __name__ == "__main__":
    main()

