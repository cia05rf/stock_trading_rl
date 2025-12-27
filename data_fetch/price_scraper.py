import asyncio
import aiohttp
import pandas as pd
import datetime
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import os
from tqdm import tqdm
from rate_limiter import RateLimiter
from http_client import HttpClient
import config


class QuotaExhaustedException(Exception):
    """Raised when API quota is exhausted (402 Payment Required)."""
    pass


# Global flag to signal all tasks to stop
_quota_exhausted = False

# Configure logging - INFO level so tqdm progress bar renders cleanly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers so tqdm bar renders cleanly
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

# Ensure output directory exists
config.OUTPUT_DIR.mkdir(exist_ok=True)
logger.info("Output directory: %s", config.OUTPUT_DIR)

# Semaphore for concurrent request limiting
request_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

# Global aiohttp session (created in main, closed at end)
http_session: Optional[aiohttp.ClientSession] = None


# Global rate limiter instance
rate_limiter = RateLimiter(
    config.API_CALLS_PER_MINUTE,
    config.RATE_LIMIT_WINDOW,
    config.RATE_LIMIT_BUFFER
)


async def fetch_intraday_chunk(
    ticker: str,
    exchange_code: str,
    ts_from: int,
    ts_to: int,
    interval: str = '15m'
) -> Optional[pd.DataFrame]:
    """
    Fetch a single chunk of intraday data using aiohttp (true async).
    Respects rate limiting (calls per minute).
    
    Raises QuotaExhaustedException if API quota is exhausted (402 error).
    """
    global http_session, _quota_exhausted
    
    # Check if quota was already exhausted by another task
    if _quota_exhausted:
        return None
    
    if http_session is None:
        raise RuntimeError("HTTP session not initialized")
    
    full_symbol = f"{ticker}.{exchange_code}"

    # First, acquire rate limit permission
    await rate_limiter.acquire()

    # Then, acquire semaphore for concurrent request limiting
    async with request_semaphore:
        for attempt in range(config.RETRY_ATTEMPTS):
            # Re-check quota flag before each attempt
            if _quota_exhausted:
                return None
            
            try:
                # Build API URL
                url = f"{config.EODHD_BASE_URL}/intraday/{full_symbol}"
                params = {
                    "api_token": config.API_KEY,
                    "interval": interval,
                    "from": ts_from,
                    "to": ts_to,
                    "fmt": "json"
                }

                # Make async HTTP request with timeout
                timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
                async with http_session.get(
                    url, params=params, timeout=timeout
                ) as response:
                    # Check for quota exhaustion (402 Payment Required)
                    if response.status == 402:
                        _quota_exhausted = True
                        logger.error(
                            "🛑 API quota exhausted! (402 Payment Required)"
                        )
                        raise QuotaExhaustedException(
                            "API quota exhausted - 402 Payment Required"
                        )
                    
                    response.raise_for_status()
                    data = await response.json()
                
                if data and len(data) > 0:
                    # EODHD returns data in format: {"datetime": [...], "open": [...], ...}
                    # or as list of dicts
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        # Handle dict format if needed
                        if "datetime" in data:
                            df = pd.DataFrame(data)
                        else:
                            # Might be wrapped in another key
                            df = pd.DataFrame(data.get("data", []))
                    else:
                        return None
                    
                    if not df.empty:
                        return df
                return None

            except QuotaExhaustedException:
                # Re-raise quota exceptions - don't retry
                raise

            except asyncio.TimeoutError:
                logger.debug(
                    "Timeout fetching %s chunk (%s-%s), "
                    "attempt %s/%s",
                    full_symbol, ts_from, ts_to,
                    attempt + 1, config.RETRY_ATTEMPTS
                )
                if attempt < config.RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    return None

            except aiohttp.ClientResponseError as e:
                # Check for 402 in ClientResponseError as well
                if e.status == 402:
                    _quota_exhausted = True
                    logger.error(
                        "🛑 API quota exhausted! (402 Payment Required)"
                    )
                    raise QuotaExhaustedException(
                        "API quota exhausted - 402 Payment Required"
                    )
                logger.debug(
                    "HTTP error fetching %s chunk (%s-%s): %s, "
                    "attempt %s/%s",
                    full_symbol, ts_from, ts_to, e,
                    attempt + 1, config.RETRY_ATTEMPTS
                )
                if attempt < config.RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    return None

            except aiohttp.ClientError as e:
                logger.debug(
                    "HTTP error fetching %s chunk (%s-%s): %s, "
                    "attempt %s/%s",
                    full_symbol, ts_from, ts_to, e,
                    attempt + 1, config.RETRY_ATTEMPTS
                )
                if attempt < config.RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    return None

            except Exception as e:
                logger.debug(
                    "Error fetching %s chunk (%s-%s): %s, "
                    "attempt %s/%s",
                    full_symbol, ts_from, ts_to, e,
                    attempt + 1, config.RETRY_ATTEMPTS
                )
                if attempt < config.RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
                else:
                    return None
        
        return None



def get_last_timestamp(file_path: Path) -> Optional[int]:
    """
    Get the last timestamp from an existing CSV file.
    Efficiently reads the last line to extract the timestamp.
    """
    if not file_path.exists():
        return None
    
    try:
        # Use simple file reading to get last line
        # This is more efficient than pandas for large files
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


def check_symbol_needs_update(
    ticker: str,
    exchange_code: str,
    end_date: datetime.datetime,
    interval: str = '15m'
) -> Tuple[bool, Optional[datetime.datetime]]:
    """
    Check if a symbol needs to be updated (without making any API calls).
    Returns (needs_update, last_data_datetime).
    
    A symbol is considered up-to-date if its last timestamp is within
    the same trading day or very close to end_date.
    """
    output_file = config.OUTPUT_DIR / f"{ticker}_{exchange_code}_{interval}.csv"
    last_ts = get_last_timestamp(output_file)
    
    if last_ts is None:
        # No existing data - needs full fetch
        return (True, None)
    
    last_dt = datetime.datetime.fromtimestamp(last_ts)
    
    # Check if data is already up to date
    # We consider it up-to-date if the last data point is after end_date
    # or within the same day (accounting for market hours)
    if last_dt >= end_date:
        return (False, last_dt)
    
    # Also check if it's close enough (within 1 day) - might be up to date
    # depending on when markets closed
    time_diff = end_date - last_dt
    if time_diff < datetime.timedelta(days=1):
        # Close enough - consider up to date for now
        # (the actual process_symbol will double-check)
        return (False, last_dt)
    
    return (True, last_dt)


async def get_full_intraday_history(
    ticker: str,
    exchange_code: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: str = '15m',
    output_file: Optional[Path] = None,
    max_records: Optional[int] = None,
    resume: bool = False
) -> int:
    """
    Downloads intraday data in chunks asynchronously and writes incrementally.
    Returns the number of rows fetched.
    
    Args:
        max_records: If set, limit the number of records fetched per ticker
        resume: If True, append to existing file without writing header
    """
    full_symbol = f"{ticker}.{exchange_code}"
    logger.debug("Fetching data for %s... (Start: %s)", full_symbol, start_date)

    # Generate all chunk tasks
    chunk_tasks = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + datetime.timedelta(
            days=config.CHUNK_SIZE_DAYS
        )
        if current_end > end_date:
            current_end = end_date

        ts_from = int(pd.Timestamp(current_start).timestamp())
        ts_to = int(pd.Timestamp(current_end).timestamp())

        chunk_tasks.append(
            fetch_intraday_chunk(
                ticker, exchange_code, ts_from, ts_to, interval
            )
        )
        
        current_start = current_end
    
    # Execute all chunks concurrently (with semaphore limiting)
    chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Process results incrementally to minimize memory
    total_rows = 0
    batch_buffer = []
    first_write = True
    records_collected = 0

    for i, result in enumerate(chunk_results):
        if isinstance(result, Exception):
            logger.debug(
                "Exception in chunk %s for %s: %s",
                i, full_symbol, result
            )
            continue

        if result is not None and not result.empty:
            # Limit records if max_records is set (for test mode)
            if max_records is not None:
                remaining = max_records - records_collected
                if remaining <= 0:
                    break
                if len(result) > remaining:
                    result = result.head(remaining).copy()
            
            # Add ticker column
            result['Ticker'] = ticker
            result['Exchange'] = exchange_code
            
            batch_buffer.append(result)
            total_rows += len(result)
            records_collected += len(result)
            
            # Stop if we've reached the limit
            if max_records is not None and records_collected >= max_records:
                break

            # Write in batches to balance memory and IO
            buffer_size = sum(len(df) for df in batch_buffer)
            if len(batch_buffer) >= 5 or buffer_size >= config.BATCH_WRITE_SIZE:
                combined = pd.concat(batch_buffer, ignore_index=True)
                
                if output_file:
                    # Append mode for incremental writes
                    # If resuming, always append and never write header
                    # If not resuming (fresh start), first write is 'w' with header, subsequent 'a' without
                    if resume:
                        mode = 'a'
                        header = False
                    else:
                        mode = 'a' if not first_write else 'w'
                        header = first_write

                    combined.to_csv(
                        output_file,
                        mode=mode,
                        header=header,
                        index=False
                    )
                    first_write = False

                batch_buffer.clear()
                logger.debug(
                    "Wrote batch for %s, total rows so far: %s",
                    full_symbol, total_rows
                )
    
    # Write remaining buffer
    if batch_buffer:
        combined = pd.concat(batch_buffer, ignore_index=True)
        if output_file:
            if resume:
                mode = 'a'
                header = False
            else:
                mode = 'a' if not first_write else 'w'
                header = first_write
            combined.to_csv(output_file, mode=mode, header=header, index=False)
    
    logger.debug("Completed %s: %s rows fetched", full_symbol, total_rows)
    return total_rows


async def process_symbol(
    ticker: str,
    exchange_code: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: str = '15m',
    max_records: Optional[int] = None
) -> Tuple[str, str, int]:
    """
    Process a single symbol and return ticker, exchange, and row count.
    All outputs are saved to the data directory.
    """
    # Ensure output file is in the data directory
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
        # Start from the next second to avoid overlap (or same second if granularity is high)
        # EODHD 'from' is inclusive, so we might get one duplicate if we use exact timestamp
        # But pandas to_csv usually handles writes, duplicates in data processing might happen
        # Ideally we add 1 second or 1 interval.
        current_start = last_dt + datetime.timedelta(seconds=1)
        
        # Ensure we don't start after end_date
        if current_start >= end_date:
            logger.debug(
                "%s.%s is already up to date.", ticker, exchange_code
            )
            return (ticker, exchange_code, 0)
        
        resume = True
    
    rows = await get_full_intraday_history(
        ticker, exchange_code, current_start, end_date, interval, output_file,
        max_records=max_records, resume=resume
    )
    return (ticker, exchange_code, rows)


async def get_all_symbols() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get all available symbol lists.
    Prioritizes loading from symbol_lists.json (freshly generated).
    Returns dict mapping list_type to list of (ticker, exchange_code) tuples.
    """
    # Try loading from JSON file (use config path to work from any directory)
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
            
    # Fallback to example symbols
    logger.warning(
        "No symbols found. Run fetch_symbols.py to fetch full lists."
    )
    return []


async def main():
    """
    Main async function to fetch data for all symbols.
    Uses aiohttp for true async HTTP requests.
    """
    global http_session
    
    # Create HTTP client with connection pooling and robust SSL
    http_client = HttpClient(
        connection_limit=100,
        connection_limit_per_host=20,
        timeout_total=config.REQUEST_TIMEOUT,
        timeout_connect=30
    )
    await http_client.start()
    http_session = http_client.session
    
    try:
        # Adjust date range for test mode (shorter period = faster)
        if config.TEST_MODE:
            # For test mode, use last 7 days to ensure we get some data
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
        
        symbol_groups = await get_all_symbols()
        
        for list_type, symbols in symbol_groups.items():
            all_symbols.extend(symbols)
            logger.info("Added %s symbols from %s", len(symbols), list_type)

        if not all_symbols:
            logger.warning(
                "No symbols found. Please populate get_symbol_list() "
                "with actual symbol lists."
            )
            # Fallback to example symbols
            all_symbols = [
                ('RR', 'LSE'),
                ('AAPL', 'US')
            ]
            logger.info("Using fallback symbols: %s", all_symbols)

        # Limit symbols in test mode
        if config.TEST_MODE:
            all_symbols = all_symbols[:config.TEST_TICKERS_LIMIT]
            logger.info(
                "TEST MODE: Limited to first %s symbols", config.TEST_TICKERS_LIMIT
            )

        logger.info("Checking %d symbols for updates...", len(all_symbols))
        
        # Pre-check which symbols need updates (no API calls, just file checks)
        symbols_to_update = []
        symbols_already_done = []
        
        for ticker, exchange in all_symbols:
            needs_update, last_dt = check_symbol_needs_update(
                ticker, exchange, END, config.INTERVAL
            )
            if needs_update:
                symbols_to_update.append((ticker, exchange))
            else:
                symbols_already_done.append((ticker, exchange, last_dt))
        
        logger.info(
            "✓ %d symbols already up-to-date, %d need updates",
            len(symbols_already_done), len(symbols_to_update)
        )
        logger.info(
            "Rate limit: %d calls/%ds | Concurrent: %d",
            config.API_CALLS_PER_MINUTE, config.RATE_LIMIT_WINDOW, config.MAX_CONCURRENT_REQUESTS
        )
        
        start_time = time.time()

        # Reset global quota flag at start
        global _quota_exhausted
        _quota_exhausted = False
        
        # Tally results - start with already done symbols
        successful = len(symbols_already_done)  # These are already successful
        total_rows = 0
        failed = []
        quota_hit = False
        
        # Batch size for processing - small enough to stop quickly on quota hit
        
        # Only process symbols that need updates
        if symbols_to_update:
            max_records = config.TEST_RECORDS_LIMIT if config.TEST_MODE else None
            
            # Run with tqdm progress bar - show total including already done
            bar_fmt = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            )
            pbar = tqdm(
                total=len(all_symbols),
                initial=len(symbols_already_done),
                desc="📈 Fetching prices",
                unit="sym",
                ncols=100,
                bar_format=bar_fmt
            )
            
            try:
                # Process in batches for quick stopping on quota hit
                for batch_start in range(0, len(symbols_to_update), config.BATCH_SIZE):
                    # Check quota flag before starting new batch
                    if _quota_exhausted:
                        quota_hit = True
                        pbar.set_description("🛑 Quota hit - stopping")
                        logger.warning("Quota exhausted. Stopping...")
                        break
                    
                    batch_end = min(
                        batch_start + config.BATCH_SIZE, len(symbols_to_update)
                    )
                    batch_symbols = symbols_to_update[batch_start:batch_end]
                    
                    # Create tasks only for this batch
                    batch_tasks = [
                        asyncio.create_task(
                            process_symbol(
                                ticker, exchange,
                                START, END, config.INTERVAL, max_records
                            )
                        )
                        for ticker, exchange in batch_symbols
                    ]
                    
                    # Wait for batch to complete
                    batch_results = await asyncio.gather(
                        *batch_tasks, return_exceptions=True
                    )
                    
                    # Process batch results
                    for result in batch_results:
                        pbar.update(1)
                        
                        if isinstance(result, QuotaExhaustedException):
                            quota_hit = True
                            pbar.set_description("🛑 Quota hit - stopping")
                            break
                        elif isinstance(result, Exception):
                            failed.append(result)
                        elif result is not None:
                            ticker, exchange, rows = result
                            if rows > 0:
                                successful += 1
                                total_rows += rows
                    
                    # Stop processing more batches if quota hit
                    if quota_hit or _quota_exhausted:
                        quota_hit = True
                        pbar.set_description("🛑 Quota hit - stopping")
                        logger.warning(
                            "API quota exhausted. Stopped after batch %d/%d",
                            batch_start // config.BATCH_SIZE + 1,
                            (len(symbols_to_update) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
                        )
                        break
            finally:
                pbar.close()
        else:
            logger.info("All symbols already up-to-date, nothing to fetch!")

        elapsed = time.time() - start_time
        
        # Summary
        logger.info("=" * 60)
        if quota_hit:
            logger.error("🛑 STOPPED: API quota exhausted (402 Payment Required)")
            logger.info("   Run again later to continue from where you left off")
        else:
            logger.info("✅ Completed in %.1fs", elapsed)
        logger.info("   Successful: %d/%d symbols", successful, len(all_symbols))
        logger.info("   Total rows: %s", f"{total_rows:,}")
        logger.info("   Failed: %d", len(failed))
        logger.info("   Output: %s", config.OUTPUT_DIR.absolute())
        logger.info("=" * 60)
    
    finally:
        # Close HTTP client
        await http_client.close()
        logger.info("Closed HTTP client")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()  # Allow nested event loops
    asyncio.run(main())
