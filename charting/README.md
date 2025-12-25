# Charting Module

Visualization tools for stock price data using candlestick charts.

## Overview

This module provides scripts to generate color-coded candlestick charts from HDF5 price data files. Charts are automatically split by ticker and ordered by timestamp, making it easy to assess if stocks are tradable by visual inspection.

## Features

- **Candlestick Charts**: Color-coded candles (green for up, red for down)
- **1-Minute Intervals**: Displays price data at 1-minute intervals (or any interval in your HDF5 file)
- **Ticker Separation**: Each ticker gets its own chart
- **Timestamp Ordering**: Data is automatically sorted by timestamp
- **Flexible Filtering**: Filter by tickers, date ranges, and more

## Usage

### Basic Usage

Generate charts for all tickers in the default HDF5 file:

```bash
python charting/visualize_prices.py
```

### Specify HDF5 File

```bash
python charting/visualize_prices.py --hdf5-file data/prices_highly_liquid.h5
```

### Chart Specific Tickers

```bash
python charting/visualize_prices.py --tickers AAPL_US MSFT_US GOOGL_US
```

### Filter by Date Range

```bash
python charting/visualize_prices.py --min-date 2024-01-01 --max-date 2024-12-31
```

### Limit Number of Tickers

```bash
python charting/visualize_prices.py --max-tickers 10
```

### Custom Output Directory

```bash
python charting/visualize_prices.py --output-dir my_charts
```

### Adjust Chart Density

```bash
python charting/visualize_prices.py --max-candles 500
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--hdf5-file` | Path to HDF5 file | From config (`HDF5_FILE`) |
| `--tickers` | Specific tickers to chart | All tickers |
| `--output-dir` | Output directory for charts | `charting/output` |
| `--min-date` | Minimum date filter (YYYY-MM-DD) | None |
| `--max-date` | Maximum date filter (YYYY-MM-DD) | None |
| `--max-candles` | Maximum candles per chart | 1000 |
| `--max-tickers` | Maximum number of tickers to chart | All |

## Output

Charts are saved as PNG files in the output directory with the naming pattern:
```
{TICKER}_candlestick.png
```

For example:
- `AAPL_US_candlestick.png`
- `MSFT_US_candlestick.png`

## Chart Features

- **Color Coding**:
  - Green (`#26a69a`): Up candles (close >= open)
  - Red (`#ef5350`): Down candles (close < open)

- **Candlestick Elements**:
  - **Wick**: Vertical line showing high-low range
  - **Body**: Rectangle showing open-close range

- **Chart Information**:
  - Ticker symbol in title
  - Number of candles displayed
  - Timestamp axis with automatic formatting
  - Grid for easier reading

## Examples

### Quick Assessment of Top Liquid Stocks

```bash
# Chart first 20 tickers from highly liquid dataset
python charting/visualize_prices.py \
  --hdf5-file data/prices_highly_liquid.h5 \
  --max-tickers 20 \
  --output-dir charts/assessment
```

### Recent Data Only

```bash
# Chart last 30 days of data
python charting/visualize_prices.py \
  --min-date 2024-11-01 \
  --output-dir charts/recent
```

### Specific Stock Analysis

```bash
# Deep dive into specific stocks
python charting/visualize_prices.py \
  --tickers AAPL_US MSFT_US GOOGL_US AMZN_US \
  --max-candles 2000 \
  --output-dir charts/analysis
```

## Requirements

- Python 3.11+
- matplotlib
- mplfinance (for professional candlestick charts)
- pandas
- h5py

All dependencies should already be installed as part of the main project. If `mplfinance` is not installed, run:

```bash
uv sync
```

Or install manually:
```bash
pip install mplfinance
```

## Integration

The charting module integrates with the project's configuration system:
- Uses `shared.config` for HDF5 file paths
- Uses `shared.logging_config` for logging
- Follows project structure conventions

## Notes

- Charts are automatically limited to 1000 candles by default for performance. Adjust with `--max-candles` if needed.
- Large datasets may take time to process. Use `--max-tickers` to limit scope during exploration.
- The script handles missing or corrupted ticker data gracefully, logging warnings and continuing.


