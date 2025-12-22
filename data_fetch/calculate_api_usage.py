"""
Calculate API usage and time estimates for data fetching.
"""
import math
import config


def calculate_api_calls_per_symbol(start_date, end_date, chunk_size_days):
    """Calculate number of API calls needed per symbol."""
    days_diff = (end_date - start_date).days
    calls_per_symbol = math.ceil(days_diff / chunk_size_days)
    return calls_per_symbol


def calculate_total_requirements():
    """Calculate total API calls and days needed."""
    # Calculate calls per symbol
    calls_per_symbol = calculate_api_calls_per_symbol(
        config.START_DATE, config.END_DATE, config.CHUNK_SIZE_DAYS
    )
    
    # Calculate total symbols
    total_symbols = sum(config.SYMBOL_COUNTS.values())
    
    # Calculate total API calls (assuming no retries for base calculation)
    total_api_calls = total_symbols * calls_per_symbol
    
    # Account for retries (assume 5% failure rate requiring retries)
    # Each retry is an additional API call
    retry_overhead = 0.05
    total_with_retries = total_api_calls * (1 + retry_overhead)
    
    # Calculate days needed
    days_needed = math.ceil(total_with_retries / config.DAILY_API_LIMIT)
    
    return {
        "calls_per_symbol": calls_per_symbol,
        "total_symbols": total_symbols,
        "total_api_calls": int(total_api_calls),
        "total_with_retries": int(total_with_retries),
        "days_needed": days_needed,
        "date_range_days": (config.END_DATE - config.START_DATE).days,
    }


def print_breakdown():
    """Print detailed breakdown of API usage."""
    results = calculate_total_requirements()
    
    print("=" * 70)
    print("API USAGE CALCULATION")
    print("=" * 70)
    print(f"\nDate Range: {config.START_DATE.date()} to {config.END_DATE.date()}")
    print(f"Total Days: {results['date_range_days']} days")
    print(f"Chunk Size: {config.CHUNK_SIZE_DAYS} days per API call")
    print(f"API Calls per Symbol: {results['calls_per_symbol']}")
    print(f"\nDaily API Limit: {config.DAILY_API_LIMIT:,} calls/day")
    
    print("\n" + "-" * 70)
    print("SYMBOL BREAKDOWN:")
    print("-" * 70)
    total_calls_by_type = {}
    for list_type, count in config.SYMBOL_COUNTS.items():
        calls = count * results['calls_per_symbol']
        total_calls_by_type[list_type] = calls
        print(f"  {list_type:12s}: {count:4d} symbols × {results['calls_per_symbol']:2d} calls = {calls:6,d} calls")
    
    print("\n" + "-" * 70)
    print("TOTAL ESTIMATES:")
    print("-" * 70)
    print(f"  Total Symbols:        {results['total_symbols']:,}")
    print(f"  Base API Calls:        {results['total_api_calls']:,}")
    print(f"  With Retry Overhead:   {results['total_with_retries']:,} (5% overhead)")
    print(f"\n  Days Needed:           {results['days_needed']} days")
    print(f"  Weeks Needed:          {results['days_needed'] / 7:.1f} weeks")
    print(f"  Months Needed:         {results['days_needed'] / 30:.1f} months")
    
    print("\n" + "-" * 70)
    print("OPTIMIZATION SUGGESTIONS:")
    print("-" * 70)
    
    # Suggest increasing chunk size if possible
    if config.CHUNK_SIZE_DAYS < 120:
        larger_chunk = 120
        new_calls = math.ceil(results['date_range_days'] / larger_chunk)
        new_total = results['total_symbols'] * new_calls * 1.05
        new_days = math.ceil(new_total / config.DAILY_API_LIMIT)
        print(f"  • Increase chunk size to {larger_chunk} days:")
        print(f"    → {new_days} days needed (saves {results['days_needed'] - new_days} days)")
    
    # Suggest reducing symbol count
    if results['days_needed'] > 30:
        print(f"  • Process in batches (e.g., 100 symbols at a time)")
        print(f"    → Each batch: ~{math.ceil((results['total_symbols'] / 100) * results['calls_per_symbol'] * 1.05 / config.DAILY_API_LIMIT)} days")
    
    # Suggest reducing date range
    if results['date_range_days'] > 730:
        shorter_range = 730  # 2 years
        shorter_calls = math.ceil(shorter_range / config.CHUNK_SIZE_DAYS)
        shorter_total = results['total_symbols'] * shorter_calls * 1.05
        shorter_days = math.ceil(shorter_total / config.DAILY_API_LIMIT)
        print(f"  • Reduce date range to 2 years:")
        print(f"    → {shorter_days} days needed (saves {results['days_needed'] - shorter_days} days)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_breakdown()

