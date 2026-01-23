# Intraday Price Fetch & Trading System

A comprehensive stock price data pipeline, reinforcement learning training system, and backtesting framework for automated stock trading.

## 🏗️ Architecture

```
intraday-price-fetch/
├── data_fetch/          # Data fetching and processing
│   ├── price_scraper.py     # Fetch intraday prices from EODHD API
│   ├── fetch_symbols.py     # Fetch available symbols
│   ├── convert_to_hdf5.py   # Convert CSV data to HDF5 format
│   └── datamodule.py        # PyTorch Lightning DataModule
│
├── shared/              # Shared utilities
│   ├── config.py            # Centralized configuration
│   ├── logging_config.py    # Logging setup
│   ├── datamodule.py        # Shared DataModule
│   └── dataset.py           # PyTorch datasets
│
├── training/            # RL training module
│   ├── environment.py       # Gymnasium trading environment
│   ├── networks.py          # Neural network architectures
│   ├── policies.py          # Custom SB3 policies
│   ├── train_ppo.py         # PPO training script
│   └── inference.py         # Model inference
│
├── eval/                # Evaluation & backtesting
│   ├── backtest.py          # Mock fund backtesting
│   └── analysis.py          # Performance analysis
│
├── charting/            # Visualization tools
│   ├── visualize_prices.py  # Candlestick chart generation
│   └── README.md            # Charting documentation
│
├── tests/               # Test suite
├── data/                # Data directory
└── logs/                # Log files
```

## 🎯 Active Sniper Mode (Limit Position Management)

The system uses an **"Active Sniper"** architecture with continuous control over entries and exits. The agent doesn't just buy and hope; it sets traps (Buy Limits), protects them with hard Stop Losses, and actively manages exits (Sell Limits).

### 🧠 Logic & Strategy
*   **The "Trap" (Entry):** The agent places a Buy Limit order at a calculated discount. If the price never drops to trigger the trap, no capital is risked (TTL expiry).
*   **Active Management (Exit):** Unlike fixed "Take Profit" brackets, the agent must actively decide when to place a Sell Limit to exit a position based on evolving market conditions.
*   **Hard Protection:** Every entry is automatically protected by a Stop Loss, which is checked intra-candle for maximum safety.

### 📥 Inputs & 📤 Outputs
*   **Inputs (Observation Space):** 
    *   10 features per timestep: `[Log_Return, Volume_Change, Volatility, RSI, MACD, Time_Sin, Time_Cos, Has_Pending_Order, Has_Active_Position, Unrealized_PnL]`.
    *   Processed via a **Multi-Scale LSTM** capturing patterns at 16, 64, and 256 steps.
*   **Outputs (3D Continuous Action Space):**
    1.  **Signal Strength ($[-1, 1]$):** 
        *   *No Position:* $> 0.3$ = Place **BUY LIMIT**.
        *   *Active Position:* $< -0.3$ = Place **SELL LIMIT** (Exit).
        *   *Otherwise:* HOLD / CANCEL pending orders.
    2.  **Limit Offset ($[-1, 1]$):** Determines entry/exit aggressiveness.
        *   `Buy_Price = Close * (1 - (abs(Action[1]) * MAX_LIMIT_OFFSET))`
        *   `Sell_Price = Close * (1 + (abs(Action[1]) * MAX_LIMIT_OFFSET))`
    3.  **Stop Loss Offset ($[-1, 1]$):** (Entry only)
        *   `Stop_Price = Entry_Price * (1 - (abs(Action[2]) * MAX_STOP_LOSS))`

### ⚙️ Mechanics
*   **Order Execution Phase:** Every step begins by checking for Stop Loss hits (pessimistic assumption: SL always hits first) and Limit Order fills using the current candle's **High/Low**.
*   **TTL (Time To Live):** Pending orders have a default lifespan of 4 steps (1 hour). If not filled, they are auto-cancelled.
*   **Execution Ranking:** In backtesting, if multiple tickers generate signals, they are ranked by `abs(Signal) * Predicted_Discount` to prioritize the most confident and efficient trades.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for training)
- EODHD API key (for data fetching)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/intraday-price-fetch.git
cd intraday-price-fetch

# Install uv (if not installed)
pip install uv

# Create virtual environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Configuration

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` with your settings:
```env
API_KEY=your_eodhd_api_key
DEVICE=cuda
LOG_LEVEL=INFO
# Sniper Mode requires continuous action space
ACTION_SPACE_TYPE=continuous
```

## 📊 Data Pipeline

### 1. Fetch Symbols

```bash
cd data_fetch
python fetch_symbols.py
```

### 2. Scrape Price Data

```bash
python price_scraper.py
```

### 3. Convert to HDF5

From the project root, convert CSVs to HDF5 (default output now writes `data/prices_highly_liquid.h5`):
```bash
cd data_fetch
python convert_to_hdf5.py
```

Optional: create multiple datasets with different liquidity filters
```bash
# All tickers (no filtering) — largest file
python convert_to_hdf5.py --output ../data/prices_all.h5

# Liquid (recommended) — balanced quality and coverage
python convert_to_hdf5.py ^
  --output ../data/prices_liquid.h5 ^
  --min-avg-volume 100000 ^
  --min-avg-value 500000 ^
  --min-data-points 500 ^
  --min-trading-pct 0.9

# Highly liquid — fastest to iterate, top-volume names only
python convert_to_hdf5.py ^
  --output ../data/prices_highly_liquid.h5 ^
  --min-avg-volume 500000 ^
  --min-avg-value 5000000 ^
  --min-data-points 1000 ^
  --min-trading-pct 0.95
```

Then point training to the desired file (e.g., in `.env`):
```bash
HDF5_FILE=data/prices_highly_liquid.h5
```

## 🤖 Training

Train the Sniper agent using PPO with continuous actions:

```bash
# Basic training (recommended for Sniper Mode)
python training/train_ppo.py --action-space-type continuous

# With custom timesteps
python training/train_ppo.py --timesteps 2000000 --device cuda

# Monitor with TensorBoard
tensorboard --logdir training/tensorboard_logs/
```

### Fine-Tuning & Resuming

You can resume training from a saved model and its environment statistics (`VecNormalize`) using the following flags:

```bash
# Resume training with a specific model and normalization statistics
python training/train_ppo.py \
    --resume-model training/models/ppo_stock_trading_20240123_0904.zip \
    --resume-vec-norm training/models/vec_normalize_20240123_0904.pkl \
    --timesteps 500000 \
    --n-envs 4
```

**Key Fine-Tuning Behaviors:**
*   **Constant Learning Rate:** Automatically switches from a warmup schedule to a constant low learning rate (using `LEARNING_RATE_END`).
*   **Instant Difficulty:** The curriculum starting point is adjusted to maximum difficulty (`MAX_LIMIT_OFFSET`) when resuming.
*   **Continuous Stats:** Environment normalization statistics continue to update during the fine-tuning phase.

## 📈 Evaluation & Backtesting

The backtester simulates a realistic Limit Order Book environment:

```python
from eval.backtest import run_backtest
from eval.analysis import generate_report

# Run realistic Limit Order backtest
results = run_backtest(
    start_date="2024-01-01",
    end_date="2024-06-30",
    initial_balance=10000,
)

# The summary now includes Fill_Rate and Orders_Placed vs Filled
print(results.summary())

# Generate detailed PDF/HTML report
generate_report(results.to_dataframe(), output_dir="./reports")
```


## 📊 Visualization

Generate candlestick charts to visually assess stock tradability:

```bash
# Generate charts for all tickers
python charting/visualize_prices.py

# Chart specific tickers
python charting/visualize_prices.py --tickers AAPL_US MSFT_US GOOGL_US

# Filter by date range
python charting/visualize_prices.py --min-date 2024-01-01 --max-date 2024-12-31

# Limit to top N tickers
python charting/visualize_prices.py --max-tickers 20
```

Charts are saved as PNG files in `charting/output/` by default. See [charting/README.md](charting/README.md) for full documentation.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shared --cov=training --cov=eval

# Run specific module tests
pytest tests/test_training.py -v
```

## 📦 Project Structure

### Shared Module (`shared/`)

Common utilities used across the project:

- **`config.py`**: Centralized configuration loading from `config.json` and `.env`
- **`logging_config.py`**: Colored console output and rotating file logs
- **`datamodule.py`**: PyTorch Lightning DataModule for OHLCV data
- **`dataset.py`**: PyTorch datasets for time series data

### Data Fetch Module (`data_fetch/`)

Data acquisition and processing:

- Async HTTP client with rate limiting
- Support for stocks, forex, and crypto
- HDF5 conversion for efficient storage
- PyTorch Lightning integration

### Training Module (`training/`)

Reinforcement learning training:

- Custom Gymnasium environment for stock trading
- Multi-scale LSTM architecture
- PPO training with Stable-Baselines3
- Learning rate schedules
- Training callbacks and monitoring

### Evaluation Module (`eval/`)

Strategy evaluation and backtesting:

- Mock fund simulation
- Performance metrics (Sharpe, drawdown, etc.)
- Visualization tools
- Automated report generation

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | - | EODHD API key |
| `DEVICE` | cuda | PyTorch device |
| `LOG_LEVEL` | INFO | Logging level |
| `SEED` | 42 | Random seed |
| `INITIAL_BALANCE` | 10000 | Starting capital |
| `TOTAL_TIMESTEPS` | 2000000 | Training steps |

### config.json

Main configuration file with settings for:
- Data paths
- API endpoints
- Rate limiting
- Training parameters
- Data module settings

## 📝 Logging

Logs are written to:
- Console (with colors)
- `logs/app_YYYYMMDD.log` (rotating)

Configure via `LOG_LEVEL` environment variable.

## 🔧 Development

### Code Style

```bash
# Format code
ruff format .

# Check lints
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

## 📚 Documentation

- [Training Module README](training/README.md)
- [Evaluation Module README](eval/README.md)
- [Data Fetch README](data_fetch/README.md)
- [Charting Module README](charting/README.md)

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice and should not be used for actual trading without proper risk assessment. Past performance does not guarantee future results.

