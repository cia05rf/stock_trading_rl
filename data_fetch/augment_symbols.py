"""
Script to augment symbol lists with start and end dates.
Reads ../data/symbol_lists.json and creates ../data/symbol_lists_st_en.json.
Uses EODHD API "First Record" and "Last Record" tricks.
"""
import asyncio
import aiohttp
import nest_asyncio
import json
import logging
import time
from typing import Dict, Optional, Any
from tqdm.asyncio import tqdm_asyncio
from rate_limiter import RateLimiter
from http_client import HttpClient
import config

# Configure logging - reduce verbosity so tqdm progress bar is visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers so tqdm bar renders cleanly
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

# Global semaphore
request_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS_AUGMENT)

rate_limiter = RateLimiter(
    config.AUGMENT_CALLS_PER_MINUTE,
    config.AUGMENT_RATE_LIMIT_WINDOW,
    config.AUGMENT_RATE_LIMIT_BUFFER
)


async def get_date_limit(
    session: aiohttp.ClientSession,
    ticker: str,
    exchange: str,
    order: str,
    max_retries: int = 3
) -> Optional[str]:
    """
    Fetch the first (order='a') or last (order='d') date for a symbol.
    Includes retry logic with exponential backoff.
    """
    full_symbol = f"{ticker}.{exchange}"
    url = f"{config.EODHD_BASE_URL}/eod/{full_symbol}"
    params = {
        "api_token": config.API_KEY,
        "period": "d",
        "order": order,
        "limit": "1",
        "fmt": "json"
    }

    order_name = "start" if order == "a" else "end"
    
    for attempt in range(max_retries):
        attempt_str = f"attempt {attempt + 1}/{max_retries}"
        logger.debug(
            f"[{full_symbol}] Waiting for rate limiter ({order_name} date, {attempt_str})..."
        )
        await rate_limiter.acquire()
        logger.debug(f"[{full_symbol}] Rate limiter acquired, waiting for semaphore...")
        
        async with request_semaphore:
            logger.debug(
                f"[{full_symbol}] Semaphore acquired, making request for {order_name} date..."
            )
            request_start = time.time()
            try:
                async with session.get(url, params=params) as response:
                    elapsed = time.time() - request_start
                    logger.debug(
                        f"[{full_symbol}] Response received in {elapsed:.2f}s, "
                        f"status={response.status}"
                    )
                    
                    # 404 = symbol not found, no point retrying
                    if response.status == 404:
                        logger.debug(f"[{full_symbol}] 404 - symbol not found")
                        return None
                    
                    # 429 = rate limited by API, retry with backoff
                    if response.status == 429:
                        wait_time = 5 * (attempt + 1)  # Exponential backoff
                        logger.debug(
                            "[%s] Rate limit 429 after %.2fs, %s. Waiting %ds...",
                            full_symbol, elapsed, attempt_str, wait_time
                        )
                        await asyncio.sleep(wait_time)
                        continue  # Retry
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        result = data[0].get("date")
                        logger.debug(f"[{full_symbol}] Got {order_name} date: {result}")
                        return result
                    logger.debug(f"[{full_symbol}] Empty response for {order_name} date")
                    return None
                    
            except asyncio.TimeoutError:
                elapsed = time.time() - request_start
                logger.debug(
                    "[%s] TIMEOUT after %.2fs fetching %s date, %s",
                    full_symbol, elapsed, order_name, attempt_str
                )
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.debug(f"[{full_symbol}] Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue  # Retry
                    
            except aiohttp.ClientResponseError as e:
                elapsed = time.time() - request_start
                logger.debug(
                    "[%s] HTTP error %s after %.2fs fetching %s date, %s",
                    full_symbol, e.status, elapsed, order_name, attempt_str
                )
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                    continue  # Retry
                    
            except Exception as e:
                elapsed = time.time() - request_start
                logger.debug(
                    "[%s] Error after %.2fs fetching %s date: %s",
                    full_symbol, elapsed, order_name, e
                )
                # Unknown errors - don't retry
                return None
    
    # All retries exhausted
    logger.debug(
        "[%s] Failed to fetch %s date after %d attempts",
        full_symbol, order_name, max_retries
    )
    return None


async def process_symbol(
    session: aiohttp.ClientSession,
    ticker: str,
    exchange: str
) -> Optional[Dict[str, Any]]:
    """
    Fetch start and end dates for a single symbol.
    Returns dict with symbol info and dates if found, else None.
    """
    # Fetch start date
    start_date = await get_date_limit(session, ticker, exchange, "a")
    if not start_date:
        return None

    # Fetch end date
    end_date = await get_date_limit(session, ticker, exchange, "d")
    if not end_date:
        # Use start as end if only 1 record
        end_date = start_date
    
    return {
        "ticker": ticker,
        "exchange": exchange,
        "start_date": start_date,
        "end_date": end_date
    }


async def main():
    logger.info("=" * 60)
    logger.info("Starting augment_symbols script")
    logger.info("=" * 60)
    
    if not config.SYMBOL_LISTS_FILE.exists():
        logger.error("Input file %s not found.", config.SYMBOL_LISTS_FILE)
        return

    if not config.API_KEY:
        logger.error("API_KEY not found in environment variables.")
        return

    # Load symbols
    logger.info("Loading symbols from %s...", config.SYMBOL_LISTS_FILE)
    with open(config.SYMBOL_LISTS_FILE, 'r', encoding='utf-8') as f:
        symbols_data = json.load(f)
    
    # Log what we loaded
    for list_type, symbols_list in symbols_data.items():
        logger.info("  - %s: %d symbols", list_type, len(symbols_list))

    # Prepare output structure
    output_data = {}
    start_time = time.time()
    
    # Setup connection using shared HTTP client
    async with HttpClient(
        connection_limit=config.MAX_CONCURRENT_REQUESTS_AUGMENT,
        timeout_total=120,
        timeout_connect=30
    ) as client:
        session = client.session
        
        # Collect all symbols first
        all_symbols = []
        for list_type, symbols_list in symbols_data.items():
            for item in symbols_list:
                # Handle both list [ticker, exchange] and dict formats
                if isinstance(item, list):
                    ticker, exchange = item
                else:
                    ticker = item["ticker"]
                    exchange = item["exchange"]
                
                all_symbols.append((list_type, ticker, exchange))
        
        # Limit to test mode if enabled
        if config.TEST_MODE:
            all_symbols = all_symbols[:config.TEST_TICKERS_LIMIT]
            logger.info(
                "TEST MODE: Processing %d of %d tickers",
                len(all_symbols), config.TEST_TICKERS_LIMIT
            )
        
        total_symbols = len(all_symbols)
        logger.info("Will process %d symbols", total_symbols)
        logger.info(
            "Rate limit: %d calls/%ds | Concurrent: %d",
            config.AUGMENT_CALLS_PER_MINUTE, config.AUGMENT_RATE_LIMIT_WINDOW, 
            config.MAX_CONCURRENT_REQUESTS_AUGMENT
        )
        
        # Process symbols by list_type with tqdm progress bar
        for list_type, symbols_list in symbols_data.items():
            # Filter symbols for this list_type
            list_symbols = [
                (ticker, exchange)
                for lt, ticker, exchange in all_symbols
                if lt == list_type
            ]
            
            if not list_symbols:
                continue
            
            # Create tasks
            tasks = [
                process_symbol(session, ticker, exchange)
                for ticker, exchange in list_symbols
            ]
            
            # Run with tqdm progress bar
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"📊 {list_type}",
                unit="sym",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            # Filter None results
            valid_symbols = [r for r in results if r is not None]
            output_data[list_type] = valid_symbols
            
            logger.info(
                "✓ %s: %d/%d valid symbols",
                list_type, len(valid_symbols), len(list_symbols)
            )

    # Save output
    with open(config.SYMBOL_LISTS_AUGMENTED_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    total_elapsed = time.time() - start_time
    
    # Summary
    total_valid = sum(len(syms) for syms in output_data.values())
    logger.info("=" * 60)
    logger.info("✅ Completed in %.1fs", total_elapsed)
    logger.info("   Total valid symbols: %d", total_valid)
    logger.info("   Output: %s", config.SYMBOL_LISTS_AUGMENTED_FILE)
    logger.info("=" * 60)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
