# Evaluation Module

This module provides tools for evaluating and backtesting stock trading strategies trained with the training module.

## Overview

The evaluation module enables:

- **Backtesting**: Simulate trading strategies on historical data
- **Analysis**: Compute performance metrics (Sharpe ratio, drawdown, win rate)
- **Visualization**: Generate performance charts and reports
- **Comparison**: Compare multiple strategies

## Architecture

```
eval/
├── __init__.py      # Module exports
├── backtest.py      # MockFund class for backtesting
└── analysis.py      # Analysis and visualization tools
```

## Quick Start

### 1. Run a Backtest

```python
from eval.backtest import MockFund, run_backtest

# Simple backtest
results = run_backtest(
    start_date="2024-01-01",
    end_date="2024-06-30",
    initial_balance=10000,
)

# Print summary
print(results.summary())
```

### 2. Analyze Results

```python
from eval.analysis import analyze_results, plot_performance

# Get detailed metrics
metrics = analyze_results(results.to_dataframe())
print(f"Return: {metrics['return_pct']:.2f}%")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

# Generate visualization
fig = plot_performance(results.to_dataframe())
fig.savefig("performance.png")
```

### 3. Generate Report

```python
from eval.analysis import generate_report

report = generate_report(
    results.to_dataframe(),
    output_dir="./reports",
    strategy_name="PPO Trading Strategy"
)
```

## MockFund Class

The `MockFund` class simulates a trading fund:

```python
from eval.backtest import MockFund
from training.inference import Infer

# Create fund with custom settings
fund = MockFund(
    infer=Infer(),
    initial_balance=10000,
    spread=0.01,           # 1% bid-ask spread
    stamp_duty=0.005,      # 0.5% stamp duty
    max_holding_count=5,   # Max 5 positions
)

# Run backtest
results = fund.run_backtest("2024-01-01", "2024-06-30")
```

### Trading Logic

1. **Sells First**: Execute sell orders before buys to free up capital
2. **Position Sizing**: Buy orders use `balance / max_holding_count * proportion`
3. **Transaction Costs**: Realistic costs including spread and stamp duty
4. **Close Positions**: All positions closed at end of backtest

## Performance Metrics

### Computed by `analyze_results()`

| Metric | Description |
|--------|-------------|
| `total_trades` | Total number of trades |
| `buy_trades` | Number of buy orders |
| `sell_trades` | Number of sell orders |
| `initial_balance` | Starting capital |
| `final_balance` | Ending capital |
| `total_return` | Absolute P&L |
| `return_pct` | Percentage return |
| `win_rate` | Fraction of profitable trades |
| `sharpe_ratio` | Risk-adjusted return (annualized) |
| `max_drawdown` | Maximum peak-to-trough decline |
| `max_drawdown_pct` | Max drawdown as percentage |
| `trades_per_day` | Average daily trading frequency |

### Sharpe Ratio Calculation

```
Sharpe = sqrt(252) * mean(daily_returns) / std(daily_returns)
```

Where 252 is the number of trading days per year.

## Visualization

### `plot_performance()`

Generates a 6-panel figure:

1. **Portfolio Value**: Balance over time
2. **Trade P&L**: Individual trade profit/loss
3. **Cumulative Returns**: Running return percentage
4. **Drawdown**: Peak-to-trough drawdowns
5. **Action Distribution**: Pie chart of buy/sell/hold
6. **Monthly Returns**: Bar chart by month

### `compare_strategies()`

Compare multiple strategies side-by-side:

```python
from eval.analysis import compare_strategies

results = [
    ("Strategy A", df_a),
    ("Strategy B", df_b),
    ("Baseline", df_baseline),
]

fig = compare_strategies(results, save_path="comparison.png")
```

## Report Generation

Generate markdown reports with `generate_report()`:

```python
report = generate_report(
    ledger_df,
    output_dir="./reports",
    strategy_name="My Strategy"
)
```

Output files:
- `My_Strategy_report.md`: Markdown report
- `My_Strategy_performance.png`: Performance chart

## Command Line Usage

### Backtest and auto-generate report/plots

```bash
python -m eval.run_backtest \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --model training/models/ppo_stock_trading_20251211_2110.zip \
  --output-dir ./reports \
  --balance 10000 \
  --spread 0.01 \
  --stamp-duty 0.005
```

This writes a timestamped ledger CSV, optional equity curve CSV, a markdown report, and a performance chart into `./reports`.

### Analyze an existing ledger and regenerate the report

If you already have a ledger CSV (e.g., from a previous backtest):

```bash
python -m eval.run_analysis \
  --ledger ./reports/backtest_ledger_20250101_120000.csv \
  --equity ./reports/equity_curve_20250101_120000.csv \
  --strategy-name "PPO Trading Strategy" \
  --output-dir ./reports
```

Use `--no-significance-test` to skip the Monte Carlo section.

### Generate a quick comparison chart from multiple backtests

Run backtests for different models (saving to separate ledgers), then compare in one shot:

```bash
python - <<'PY'
from eval.analysis import compare_strategies
import pandas as pd

strategies = [
    ("Model A", pd.read_csv("./reports/model_a_ledger.csv")),
    ("Model B", pd.read_csv("./reports/model_b_ledger.csv")),
]

compare_strategies(strategies, save_path="./reports/model_comparison.png")
print("Saved comparison chart to ./reports/model_comparison.png")
PY
```

### Intraday inference helpers

Use the bulk inference utilities for intraday bars (e.g., 15m):

```python
from training.inference import Infer

infer = Infer(model_path="training/models/ppo_stock_trading_20251214_1636.zip")

# Predictions for all tickers between start/end (one row per ticker per bar)
preds = infer.predict_timespan(
    start="2024-01-02 09:00:00",
    end="2024-01-02 16:00:00",
)

# Single timestamp with automatic fetch + window prep
pred_at_open = infer.predict_for_timestamp_with_data(
    timestamp="2024-01-02 09:00:00",
    tickers=["AAPL", "MSFT"],
)
```

## BacktestResults Class

Container for backtest results:

```python
@dataclass
class BacktestResults:
    ledger: List[Dict]        # All trades
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    def to_dataframe(self) -> pd.DataFrame: ...
    def summary(self) -> Dict: ...
```

## Ledger Format

Each trade in the ledger contains:

```python
{
    "date": "2024-01-15",
    "ticker": "AAPL",
    "action": "buy",           # buy, sell
    "action_type": "buy25",    # From model prediction
    "action_prob": 0.85,       # Model confidence
    "price": 185.50,
    "quantity": 10,
    "stamp_duty": 9.28,
    "spread_cost": 18.55,
    "trade_value": 1855.00,
    "balance_pre_trade": 10000.00,
    "balance_post_trade": 8117.17,
}
```

## Example Workflows

### Basic Evaluation

```python
from eval.backtest import run_backtest
from eval.analysis import analyze_results, generate_report

# Run backtest
results = run_backtest(
    start_date="2023-07-01",
    end_date="2024-07-01",
    initial_balance=10000,
)

# Analyze
metrics = analyze_results(results.to_dataframe())

# Generate report
generate_report(
    results.to_dataframe(),
    output_dir="./reports",
    strategy_name="Annual Backtest"
)
```

### Compare Models

```python
from eval.backtest import MockFund
from training.inference import Infer

# Test different models
models = [
    ("Model v1", "models/ppo_v1.zip"),
    ("Model v2", "models/ppo_v2.zip"),
]

results = []
for name, path in models:
    infer = Infer(model_path=path)
    fund = MockFund(infer)
    backtest = fund.run_backtest("2024-01-01", "2024-06-30")
    results.append((name, backtest.to_dataframe()))

compare_strategies(results, save_path="model_comparison.png")
```

### Walk-Forward Analysis

```python
from datetime import datetime, timedelta

# Monthly walk-forward
start = datetime(2024, 1, 1)
monthly_results = []

for i in range(6):
    period_start = (start + timedelta(days=30*i)).strftime("%Y-%m-%d")
    period_end = (start + timedelta(days=30*(i+1))).strftime("%Y-%m-%d")
    
    results = run_backtest(period_start, period_end)
    metrics = analyze_results(results.to_dataframe())
    monthly_results.append({
        'period': period_start,
        'return': metrics['return_pct'],
        'sharpe': metrics['sharpe_ratio'],
    })
```

## Troubleshooting

### No Trades Executed
- Check that model predictions include buy/sell actions
- Verify price data exists for the date range
- Ensure initial balance is sufficient

### Poor Performance
- Review model training metrics
- Check for look-ahead bias in data
- Consider transaction costs impact

### Memory Issues
- Use date ranges instead of full history
- Process in chunks for long backtests

## Contributing

1. Run tests: `pytest tests/test_eval.py`
2. Format code: `ruff format eval/`
3. Check lints: `ruff check eval/`

