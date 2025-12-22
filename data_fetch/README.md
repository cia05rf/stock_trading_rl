# Async Intraday Price Fetcher

An optimized async implementation for fetching intraday price data from the EODHD API for US stocks (NASDAQ/NYSE), UK stocks (LSE), major currencies, and cryptocurrencies.

## Features

- **Async/Await**: Fully asynchronous implementation for concurrent API calls using aiohttp
- **Rate Limiting**: Sliding window rate limiter to respect API limits
- **Timeout Handling**: Configurable timeouts with automatic retries and exponential backoff
- **Incremental Writes**: Batched writes to disk to minimize memory usage
- **Resume Support**: Automatically resumes from last fetched timestamp
- **Robust SSL**: Proper SSL/TLS handling with certifi certificates
- **Test Mode**: Built-in test mode for development and debugging
- **Progress Bars**: Visual progress tracking with tqdm

## Project Structure

```
intraday-price-fetch/
├── data/                 # Output directory (at project root)
│   ├── symbol_lists.json       # Generated symbol lists
│   ├── symbol_lists_st_en.json # Symbols with start/end dates
│   └── {TICKER}_{EXCHANGE}_{INTERVAL}.csv  # Price data files
├── data_fetch/           # Data fetching module
│   ├── price_scraper.py      # Main script - fetches intraday price data
│   ├── fetch_symbols.py      # Fetches symbol lists from EODHD API
│   ├── augment_symbols.py    # Adds start/end dates to symbol lists
│   ├── convert_to_hdf5.py    # Converts CSVs to a single .h5 (fast training loads)
│   ├── datamodule.py         # PyTorch Lightning DataModule for .h5
│   ├── dataset.py            # PyTorch Dataset for .h5
│   ├── test_dataloader.py    # Sanity-check the .h5 and DataModule
│   ├── compare_hdf5.py       # Compare multiple .h5 “liquidity levels”
│   ├── http_client.py        # Shared async HTTP client with SSL handling
│   ├── rate_limiter.py       # Sliding window rate limiter
│   ├── calculate_api_usage.py # Utility to estimate API usage and time
│   ├── debug_ticker.py       # Debug utility to search/verify tickers
│   ├── pyproject.toml        # Python dependencies (uv)
│   ├── config.json           # Defaults for paths/dates/rate limits
│   ├── config.py             # Loads config.json + .env (API_KEY, INTERVAL, etc.)
│   └── .env                  # Environment variables (create this)
└── training/             # Training module
```

## Installation

1. Install dependencies using `uv`:
   ```bash
   cd data_fetch
   uv sync
   ```

2. Create a `.env` file with your EODHD API key:
   ```
   API_KEY=your_eodhd_api_key_here
   ```

### Optional Dependencies

For HDF5 conversion (`convert_to_hdf5.py`):
```bash
uv sync --extra hdf5
```

For PyTorch data loading (`datamodule.py`, `dataset.py`):
```bash
uv sync --extra torch
```

## Usage

### Step 1: Fetch Symbol Lists

```bash
uv run python fetch_symbols.py
```

This fetches all symbols from the EODHD API and saves them to `../data/symbol_lists.json`. The script:
- Fetches US common stocks from NASDAQ and NYSE
- Fetches UK common stocks from LSE
- Includes major currency pairs (EUR/USD, GBP/USD, etc.)
- Includes major cryptocurrencies (BTC, ETH, etc.)

### Step 2: (Optional) Augment with Date Ranges

```bash
uv run python augment_symbols.py
```

This queries the API for each symbol's first and last trading dates, saving to `../data/symbol_lists_st_en.json`. Useful for filtering symbols by data availability.

### Step 3: Run the Price Scraper

```bash
uv run python price_scraper.py
```

This downloads 15-minute intraday data for all symbols in `../data/symbol_lists.json`.

## Creating HDF5 datasets (`.h5`) for training / fast loading

The raw fetch pipeline produces **one CSV per ticker** in `../data/` (e.g. `AAPL_US_15m.csv`). For model training and backtesting, loading thousands of CSVs is slow, so `convert_to_hdf5.py` consolidates them into **a single HDF5 file**.

### What “levels” means

In this repo, “levels” refers to **multiple HDF5 files produced from the same CSV universe but with different liquidity filters**. The convention used by `compare_hdf5.py` is:

- `../data/prices_all.h5`: no liquidity filtering (max coverage)
- `../data/prices_liquid.h5`: moderate liquidity filter (recommended default for most work)
- `../data/prices_highly_liquid.h5`: strict liquidity filter (small set, fastest iteration)

You create these “levels” by running `convert_to_hdf5.py` multiple times with different filter flags and different `--output` paths.

### Prerequisites

- You must have CSVs for the interval you want (see **Interval notes** below).
- Install the optional HDF5 dependencies:

```bash
cd data_fetch
uv sync --extra hdf5
```

### Build the `.h5` files (ALL / LIQUID / HIGHLY_LIQUID)

All commands below assume you run them from `data_fetch/` (so `../data/` resolves correctly).

**1) ALL (no filtering):**

```bash
uv run python convert_to_hdf5.py --output ../data/prices_all.h5
```

**2) LIQUID (choose thresholds that fit your universe):**

```bash
uv run python convert_to_hdf5.py --output ../data/prices_liquid.h5 --min-avg-value 50000 --min-data-points 5000 --min-trading-pct 0.05
```

The filters are:
- `--min-avg-volume`: average volume per bar (over the whole ticker history)
- `--min-avg-value`: average traded value per bar (midpoint * volume)
- `--min-data-points`: minimum rows required per ticker
- `--min-trading-pct`: fraction of bars with `volume > 0` (0–1)

Set a flag to `0` to disable that particular filter.

**3) HIGHLY_LIQUID (stricter version of LIQUID):**

```bash
uv run python convert_to_hdf5.py --output ../data/prices_highly_liquid.h5 --min-avg-value 250000 --min-data-points 20000 --min-trading-pct 0.20
```

### Sanity check / inspect the resulting file

Inspect the datasets, shapes, and attributes:

```bash
uv run python test_dataloader.py
```

Compare the three “levels” side-by-side (expects the conventional paths above):

```bash
uv run python compare_hdf5.py
```

### Interval notes (15m vs 1m vs etc.)

Both `price_scraper.py` and `convert_to_hdf5.py` key off `INTERVAL`:
- The scraper writes files matching `*_{INTERVAL}.csv`
- The converter only reads files matching `*_{INTERVAL}.csv`

To build HDF5s for a different interval, you typically:
- Set `INTERVAL` (in `.env` or environment variables)
- Re-run the scraper (to create the matching CSVs)
- Re-run the converter, writing to an interval-specific output name (recommended)

Example (PowerShell):

```powershell
$env:INTERVAL="1m"
uv run python price_scraper.py
uv run python convert_to_hdf5.py --output ../data/prices_all_1m.h5
```

Example (bash):

```bash
export INTERVAL="1m"
uv run python price_scraper.py
uv run python convert_to_hdf5.py --output ../data/prices_all_1m.h5
```

### HDF5 layout (what’s inside)

The `.h5` file is a set of aligned arrays (concatenated across tickers) plus lookup tables:

- `ohlc`: `(n_rows, 4)` float32 = `[open, high, low, close]`
- `volume`: `(n_rows,)` float32
- `value`: `(n_rows,)` float32 = midpoint(high, low) * volume
- `timestamps`: `(n_rows,)` int64 (epoch seconds)
- `ticker_ids`: `(n_rows,)` int32 (row → ticker index)
- `ticker_names`: `(n_tickers,)` fixed-width bytes (id → ticker string like `AAPL_US`)
- `ticker_boundaries`: `(n_tickers, 2)` int64 (id → `[start_row, end_row)` slice)
- `liquidity_metrics`: `(n_tickers, 6)` float64 (summary stats; see `liquidity_metrics_columns` attr)

This layout is what `data_fetch/datamodule.py` + `data_fetch/dataset.py` (and `training/data_ingestion.py`) expect.

## Configuration

### Environment Variables

Create a `.env` file:
```
API_KEY=your_eodhd_api_key
TEST_MODE=false
```

### Price Scraper Settings (price_scraper.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `REQUEST_TIMEOUT` | 30 | Timeout per API call (seconds) |
| `MAX_CONCURRENT_REQUESTS` | 50 | Concurrent request limit |
| `CHUNK_SIZE_DAYS` | 60 | Days per API call chunk |
| `BATCH_WRITE_SIZE` | 10000 | Rows to accumulate before writing |
| `RETRY_ATTEMPTS` | 3 | Retry attempts for failed requests |
| `API_CALLS_PER_MINUTE` | 1000 | API rate limit |
| `START_DATE` | 2022-01-01 | Start of date range |
| `END_DATE` | now | End of date range |

### Test Mode

Set `TEST_MODE=true` in your `.env` file to enable test mode:
- Limits to 3 tickers
- Limits to 3 records per ticker
- Uses last 7 days instead of full date range

## Output

Data is saved incrementally to the `../data/` directory:
- Each symbol gets its own CSV file: `{TICKER}_{EXCHANGE}_{INTERVAL}.csv`
- Files contain columns: `timestamp`, `gmtoffset`, `datetime`, `open`, `high`, `low`, `close`, `volume`, `Ticker`, `Exchange`
- Resume support: If a file exists, scraper continues from the last timestamp

## Utility Scripts

### Calculate API Usage

Estimate API calls and time needed:
```bash
uv run python calculate_api_usage.py
```

### Debug Ticker

Search for tickers or verify symbol information:
```bash
# Search for a company
uv run python debug_ticker.py Siemens

# Get fundamentals for a specific ticker
uv run python debug_ticker.py AAPL US
```

## Module Details

### http_client.py

Shared HTTP client with:
- Connection pooling
- Proper SSL verification using certifi
- Configurable timeouts and connection limits
- IPv4-only mode option for network compatibility
- Async context manager support

```python
async with HttpClient(connection_limit=100, timeout_total=120) as client:
    async with client.session.get(url) as response:
        data = await response.json()
```

### rate_limiter.py

Sliding window rate limiter:
- Tracks calls within a time window
- Automatically waits when limit is reached
- Configurable safety buffer (default 95% of limit)

```python
rate_limiter = RateLimiter(
    max_calls=1000,
    time_window=60,
    buffer=0.95
)
await rate_limiter.acquire()  # Waits if at limit
```

## Memory Optimization

- Data is written to disk in batches (configurable via `BATCH_WRITE_SIZE`)
- Each symbol is processed independently and saved immediately
- Chunks are processed concurrently but results are written incrementally
- No large DataFrames are kept in memory unnecessarily

## API Optimization

- Concurrent requests with semaphore-based limiting
- Automatic retries with exponential backoff
- Timeout handling to prevent hanging requests
- Configurable chunk sizes to balance API limits and efficiency
- Resume support to avoid re-fetching existing data

## Symbol Lists

The `fetch_symbols.py` script fetches and filters:

| List Type | Description |
|-----------|-------------|
| `US_STOCKS` | Common stocks from NASDAQ, NYSE, NYSE ARCA, NYSE MKT |
| `LSE_STOCKS` | Common stocks from London Stock Exchange |
| `CURRENCY` | 10 major forex pairs (EUR/USD, GBP/USD, etc.) |
| `CRYPTO` | 15 major cryptocurrencies (BTC, ETH, etc.) |

## License

MIT
