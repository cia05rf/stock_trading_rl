"""
Centralized configuration module for intraday-price-fetch project.

Loads settings from config.json and environment variables (.env).
Import this module and access variables as config.VARIABLE_NAME.

Usage:
    import config
    
    print(config.DATA_DIR)
    print(config.API_BASE_URL)
    print(config.CHUNK_SIZE_DAYS)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv

# =============================================================================
# LOAD CONFIGURATION FILES
# =============================================================================

# Load environment variables from .env file (contains API_KEY)
dotenv.load_dotenv()

# Load config.json - paths in config are relative to this file's directory
_config_path = Path(__file__).parent / "config.json"
_config_dir = _config_path.parent  # data_fetch/ directory
with open(_config_path, "r", encoding="utf-8") as _f:
    _config = json.load(_f)

# =============================================================================
# ENVIRONMENT VARIABLES (from .env file)
# =============================================================================

# API key for EODHD - stored in .env for security
API_KEY: str = os.getenv("API_KEY", "")

# Interval for price data (e.g., "15m", "1h", "1d")
INTERVAL: str = os.getenv("INTERVAL", "15m")

# Test mode flag - can be overridden via environment
TEST_MODE: bool = os.getenv("TEST_MODE", "false").lower() == "true"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Helper to resolve paths relative to config file location (not cwd)
def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the config file directory."""
    return (_config_dir / relative_path).resolve()

# Base data directory for all data files
DATA_DIR: Path = _resolve_path(_config["paths"]["data_dir"])

# Output directory for price data CSVs
OUTPUT_DIR: Path = _resolve_path(_config["paths"]["output_dir"])

# HDF5 file path for consolidated price data
HDF5_FILE: Path = _resolve_path(_config["paths"]["hdf5_file"])

# Symbol lists JSON file path
SYMBOL_LISTS_FILE: Path = _resolve_path(_config["paths"]["symbol_lists_file"])

# Augmented symbol lists (with start/end dates) JSON file path
SYMBOL_LISTS_AUGMENTED_FILE: Path = _resolve_path(_config["paths"]["symbol_lists_augmented_file"])

# =============================================================================
# API CONFIGURATION
# =============================================================================

# EODHD API base URL
API_BASE_URL: str = _config["api"]["base_url"]

# Alias for backwards compatibility
EODHD_BASE_URL: str = API_BASE_URL

# Daily API call limit
DAILY_API_LIMIT: int = _config["api"]["daily_limit"]

# =============================================================================
# DATE CONFIGURATION
# =============================================================================

# Start date for historical data fetching
START_DATE: datetime = datetime.strptime(_config["dates"]["start_date"], "%Y-%m-%d")

# End date for historical data (None/null means current date)
_end_date_str: Optional[str] = _config["dates"]["end_date"]
END_DATE: datetime = (
    datetime.strptime(_end_date_str, "%Y-%m-%d")
    if _end_date_str is not None
    else datetime.now()
)

# Training/validation split dates for datamodule
TRAIN_END_DATE: str = _config["dates"]["train_end_date"]
VAL_END_DATE: str = _config["dates"]["val_end_date"]

# =============================================================================
# RATE LIMITING - SCRAPER
# =============================================================================

# Rate limit settings for price_scraper.py
API_CALLS_PER_MINUTE: int = _config["rate_limiting"]["scraper"]["calls_per_minute"]
RATE_LIMIT_BUFFER: float = _config["rate_limiting"]["scraper"]["rate_limit_buffer"]
RATE_LIMIT_WINDOW: int = _config["rate_limiting"]["scraper"]["rate_limit_window_seconds"]

# =============================================================================
# RATE LIMITING - AUGMENT SYMBOLS
# =============================================================================

# Rate limit settings for augment_symbols.py (more conservative)
AUGMENT_CALLS_PER_MINUTE: int = _config["rate_limiting"]["augment"]["calls_per_minute"]
AUGMENT_RATE_LIMIT_BUFFER: float = _config["rate_limiting"]["augment"]["rate_limit_buffer"]
AUGMENT_RATE_LIMIT_WINDOW: int = _config["rate_limiting"]["augment"]["rate_limit_window_seconds"]

# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

# HTTP request timeout in seconds
REQUEST_TIMEOUT: int = _config["processing"]["request_timeout_seconds"]

# Maximum concurrent API requests for price scraping
MAX_CONCURRENT_REQUESTS: int = _config["processing"]["max_concurrent_requests"]

# Maximum concurrent requests for symbol augmentation (more conservative)
MAX_CONCURRENT_REQUESTS_AUGMENT: int = _config["processing"]["max_concurrent_requests_augment"]

# Number of days per API chunk request
CHUNK_SIZE_DAYS: int = _config["processing"]["chunk_size_days"]

# Rows to accumulate before writing to disk
BATCH_WRITE_SIZE: int = _config["processing"]["batch_write_size"]

# Number of retry attempts for failed requests
RETRY_ATTEMPTS: int = _config["processing"]["retry_attempts"]

# Delay between retries in seconds
RETRY_DELAY: int = _config["processing"]["retry_delay_seconds"]

# Number of symbols to process at a time
BATCH_SIZE: int = _config["processing"]["batch_size"]

# =============================================================================
# TEST MODE SETTINGS
# =============================================================================

# Number of tickers to process in test mode
TEST_TICKERS_LIMIT: int = _config["test_mode"]["tickers_limit"]

# Number of records per ticker in test mode
TEST_RECORDS_LIMIT: int = _config["test_mode"]["records_limit"]

# =============================================================================
# DATAMODULE SETTINGS (for PyTorch Lightning)
# =============================================================================

# Number of timesteps in each input sequence
SEQUENCE_LENGTH: int = _config["datamodule"]["sequence_length"]

# Batch size for DataLoaders
DM_BATCH_SIZE: int = _config["datamodule"]["batch_size"]

# Fraction of data for training (chronologically first)
TRAIN_SPLIT: float = _config["datamodule"]["train_split"]

# Fraction of data for validation (after training data)
VAL_SPLIT: float = _config["datamodule"]["val_split"]

# Number of DataLoader workers (0 for Windows compatibility)
NUM_WORKERS: int = _config["datamodule"]["num_workers"]

# Whether to pin memory for GPU transfer
PIN_MEMORY: bool = _config["datamodule"]["pin_memory"]

# Whether to normalize features
NORMALIZE: bool = _config["datamodule"]["normalize"]

# Whether to include volume as a feature
INCLUDE_VOLUME: bool = _config["datamodule"]["include_volume"]

# Whether to include value (midpoint * volume) as a feature
INCLUDE_VALUE: bool = _config["datamodule"]["include_value"]

# Type of prediction target ('next_close', 'returns', 'direction')
TARGET_TYPE: str = _config["datamodule"]["target_type"]

# If True, don't create windows that span multiple tickers
RESPECT_TICKER_BOUNDARIES: bool = _config["datamodule"]["respect_ticker_boundaries"]

# If True, split by date. If False, split by fraction.
SPLIT_BY_DATE: bool = _config["datamodule"]["split_by_date"]

# =============================================================================
# EXCHANGE CONFIGURATIONS
# =============================================================================

# Stock exchange configurations for symbol fetching
STOCK_EXCHANGE_CONFIGS: Dict = {}
for _name, _cfg in _config["exchanges"].items():
    STOCK_EXCHANGE_CONFIGS[_name] = {
        "exchange_code": _cfg["exchange_code"],
        "output_exchange": _cfg["output_exchange"],
        "description": _cfg["description"],
        "allowed_exchanges": set(_cfg["allowed_exchanges"]) if _cfg["allowed_exchanges"] else None,
        "allowed_types": set(_cfg["allowed_types"]),
    }

# =============================================================================
# CURRENCY AND CRYPTO PAIRS
# =============================================================================

# Major currency pairs to include
MAJOR_CURRENCIES: List[Tuple[str, str]] = [
    tuple(pair) for pair in _config["currencies"]
]

# Major cryptocurrencies to include
MAJOR_CRYPTOS: List[Tuple[str, str]] = [
    tuple(pair) for pair in _config["cryptos"]
]

# =============================================================================
# API USAGE ESTIMATES
# =============================================================================

# Estimated symbol counts for API usage calculations
SYMBOL_COUNTS: Dict[str, int] = _config["api_usage_estimates"]["symbol_counts"]

# =============================================================================
# CLEANUP
# =============================================================================

# Remove temporary variables from module namespace
del _config_path, _config_dir, _f, _config, _end_date_str, _resolve_path

