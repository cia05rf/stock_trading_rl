"""
Fetch symbol lists from EODHD API and save to file.
Run this ahead of time to populate symbol_lists.json.
Filters to Common Stock on major exchanges (NASDAQ, NYSE, LSE)
and includes major currencies and cryptocurrencies.
"""
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set
import logging
from http_client import HttpClient
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_exchange_symbols(
    session: aiohttp.ClientSession,
    exchange_code: str
) -> List[Dict]:
    """
    Fetch all symbols from an exchange.
    """
    url = f"{config.EODHD_BASE_URL}/exchange-symbol-list/{exchange_code}"
    params = {
        "api_token": config.API_KEY,
        "fmt": "json",
        "delisted": 0,  # Only active symbols
    }

    try:
        timeout = aiohttp.ClientTimeout(total=120)
        logger.info("Fetching symbols from %s exchange...", exchange_code)
        async with session.get(url, params=params, timeout=timeout) as response:
            response.raise_for_status()
            data = await response.json()

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            return []

    except Exception as e:
        logger.error("Error fetching symbols for %s: %s", exchange_code, e)
        return []


def filter_stocks(
    symbols: List[Dict],
    output_exchange: str,
    allowed_types: Set[str],
    allowed_exchanges: Set[str] = None
) -> List[Tuple[str, str]]:
    """
    Filter symbols by type and exchange.
    Returns list of (ticker, exchange) tuples.
    """
    result = []
    type_counts = {}
    exchange_counts = {}
    
    for symbol in symbols:
        symbol_type = symbol.get("Type", "Unknown")
        code = symbol.get("Code", "")
        symbol_exchange = symbol.get("Exchange", "Unknown")
        
        # Track counts
        type_counts[symbol_type] = type_counts.get(symbol_type, 0) + 1
        exchange_counts[symbol_exchange] = exchange_counts.get(
            symbol_exchange, 0
        ) + 1
        
        # Filter by type
        if symbol_type not in allowed_types:
            continue
            
        # Filter by exchange if specified
        if allowed_exchanges and symbol_exchange not in allowed_exchanges:
            continue
            
        if code:
            result.append((code, output_exchange))
    
    # Log type breakdown
    logger.info("Symbol type breakdown:")
    for stype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:8]:
        marker = "✓" if stype in allowed_types else "✗"
        logger.info("  %s %s: %s", marker, stype, count)
    
    # Log exchange breakdown (for US)
    if allowed_exchanges:
        logger.info("Exchange breakdown (filtered):")
        for exch, count in sorted(
            exchange_counts.items(), key=lambda x: -x[1]
        )[:10]:
            marker = "✓" if exch in allowed_exchanges else "✗"
            logger.info("  %s %s: %s", marker, exch, count)
    
    return result


async def fetch_all_symbols() -> Dict[str, List[Tuple[str, str]]]:
    """
    Fetch symbols from configured exchanges.
    Returns filtered Common Stock + major currencies/cryptos.
    """
    async with HttpClient(
        connection_limit=10,
        timeout_total=120,
        timeout_connect=30,
        use_ipv4_only=True
    ) as client:
        session = client.session
        results = {}

        # Fetch stock exchanges
        for list_name, exch_config in config.STOCK_EXCHANGE_CONFIGS.items():
            exchange_code = exch_config["exchange_code"]
            output_exchange = exch_config["output_exchange"]
            description = exch_config["description"]
            allowed_types = exch_config["allowed_types"]
            allowed_exchanges = exch_config.get("allowed_exchanges")

            logger.info("Fetching %s...", description)

            all_symbols = await fetch_exchange_symbols(session, exchange_code)
            
            if all_symbols:
                logger.info(
                    "Retrieved %s total symbols from %s",
                    len(all_symbols), exchange_code
                )
                
                filtered = filter_stocks(
                    all_symbols,
                    output_exchange,
                    allowed_types,
                    allowed_exchanges
                )
                results[list_name] = filtered
                
                logger.info(
                    "Filtered to %s symbols for %s",
                    len(filtered), list_name
                )
            else:
                logger.warning("No symbols found for %s", exchange_code)

        # Add major currencies
        logger.info("Adding %s major currency pairs...", len(config.MAJOR_CURRENCIES))
        results["CURRENCY"] = config.MAJOR_CURRENCIES

        # Add major cryptocurrencies
        logger.info("Adding %s major cryptocurrencies...", len(config.MAJOR_CRYPTOS))
        results["CRYPTO"] = config.MAJOR_CRYPTOS

        return results


def save_to_json(symbols: Dict[str, List[Tuple[str, str]]], filepath: Path):
    """Save symbols to JSON file."""
    json_data = {
        key: [list(tup) for tup in value]
        for key, value in symbols.items()
    }

    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)

    logger.info("Saved symbols to %s", filepath)


async def main():
    """Main function to fetch and save all symbols."""
    logger.info("Starting symbol fetch from EODHD API...")
    logger.info("Filtering to:")
    logger.info("  - Common Stock on NASDAQ/NYSE (US)")
    logger.info("  - Common Stock on LSE (UK)")
    logger.info("  - Major currency pairs")
    logger.info("  - Major cryptocurrencies")

    if not config.API_KEY:
        logger.error("API_KEY not found in environment variables")
        return

    try:
        symbols = await fetch_all_symbols()

        if not symbols:
            logger.error("No symbols found!")
            return

        # Save to JSON
        save_to_json(symbols, config.SYMBOL_LISTS_FILE)

        # Print summary
        logger.info("\n%s", "=" * 60)
        logger.info("Symbol Fetch Summary:")
        logger.info("%s", "=" * 60)
        total = 0
        for list_type, symbol_list in symbols.items():
            count = len(symbol_list)
            total += count
            logger.info("  %s: %s symbols", list_type, count)
        logger.info("  Total: %s symbols", total)
        logger.info("%s", "=" * 60)

    except Exception as e:
        logger.error("Error in main: %s", e, exc_info=True)


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
