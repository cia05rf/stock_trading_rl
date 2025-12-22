# Training Module

This module provides reinforcement learning training for stock trading using PPO (Proximal Policy Optimization) with custom multi-scale LSTM architectures.

## Overview

The training module implements a complete RL pipeline for learning stock trading strategies:

- **Environment**: Custom Gymnasium environment simulating stock trading
- **Policy**: Multi-scale LSTM architecture capturing temporal patterns at different time scales
- **Training**: PPO algorithm with customizable learning rate schedules
- **Inference**: Model inference for generating trading predictions

## Architecture

```
training/
├── __init__.py           # Module exports
├── environment.py        # StockTradingEnv - Gymnasium environment
├── networks.py           # MultiScaleActorCritic neural network
├── policies.py           # Custom SB3 policies
├── callbacks.py          # Training callbacks
├── lr_schedules.py       # Learning rate schedules
├── train_ppo.py          # Main training script
├── inference.py          # Model inference utilities
└── data_ingestion.py     # Data loading from SQLite
```

## Quick Start

### 1. Setup Environment

```bash
# From project root
uv sync

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 2. Configure Training

Training parameters are configured via environment variables or `config.json`:

```bash
# Key parameters (in .env)
INITIAL_BALANCE=10000
WINDOW_SIZE=260
TOTAL_TIMESTEPS=2000000
EPOCHS=5
DEVICE=cuda
```

### 3. Run Training

```bash
# Basic training
python training/train_ppo.py

# With custom parameters
python training/train_ppo.py --timesteps 1000000 --device cuda --seed 42
```

### 4. Monitor Training

TensorBoard logs are saved to `training/tensorboard_logs/`:

```bash
tensorboard --logdir training/tensorboard_logs/
```

## Environment Details

### `StockTradingEnv`

A Gymnasium-compatible environment that simulates stock trading:

**Observation Space**: 2D array `(window_size, n_features)`
- Price data: open, high, low, close, volume
- Technical indicators: EMAs, MACD, RSI
- Lagged returns

**Action Space**: Hybrid (configurable)

- **Discrete (default)**: `Discrete(9)`
  - `buy25`, `buy50`, `buy75`, `buy100`: Buy with proportion of balance
  - `sell25`, `sell50`, `sell75`, `sell100`: Sell proportion of holdings
  - `hold`: Do nothing
- **Continuous**: `Box(low=-1, high=1, shape=(1,))`
  - \(a > 0.05\): **buy** with `proportion = a`
  - \(a < -0.05\): **sell** with `proportion = |a|`
  - Otherwise: **hold**

Select the mode via `.env` (`ACTION_SPACE_TYPE=discrete|continuous`) or CLI (`--action-space-type`).

**Reward Function**:
- Profit/loss from sales
- Artificial decay to encourage trading
- Configurable weights for gains vs losses

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_balance` | 10000 | Starting capital |
| `window_size` | 260 | Observation window (trading days) |
| `transaction_cost_pct` | 0.001 | Transaction cost (0.1%) |
| `stamp_duty_pct` | 0.005 | Stamp duty (0.5%) |
| `trade_penalty` | 0.01 | Penalty for frequent trading |

## Neural Network Architecture

### `MultiScaleActorCritic`

A custom network using multiple LSTM branches to capture patterns at different time scales:

```
Input: (batch, seq_len, features)
    │
    ├── LSTM Branch 1 (seq_len=1)   ──┐
    ├── LSTM Branch 2 (seq_len=5)   ──┼── Concat
    ├── LSTM Branch 3 (seq_len=20)  ──┤
    ├── LSTM Branch 4 (seq_len=60)  ──┤
    └── LSTM Branch 5 (seq_len=120) ──┘
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                     Actor Network           Critic Network
                           │                       │
                     FC → ReLU → FC          FC → ReLU → FC
                           │                       │
                      Softmax                   Value
                           │                       │
                    Action Probs              V(s) estimate
```

## Learning Rate Schedules

Available schedules in `lr_schedules.py`:

- **Linear**: Linear decay to final value
- **Exponential**: Exponential decay
- **Exponential with target**: Decay to specific final value
- **Cosine annealing**: Cosine curve with optional warmup
- **Warmup + exponential**: Linear warmup then exponential decay

## Training Callbacks

### `InfoLoggerCallback`
Logs environment info (net worth, actions, rewards) during training. Saves to CSV.

### `GradientMonitoringCallback`
Monitors gradient health, stops training on NaN/Inf gradients.

### `PerformanceCallback`
Tracks episode rewards and lengths for analysis.

## Inference

Use the `Infer` class for generating predictions:

```python
from training.inference import Infer

# Load latest model
infer = Infer()

# Or specify model path
infer = Infer(model_path="training/models/ppo_stock_trading_20240101.zip")

# Generate predictions for a date
predictions, summary = infer.infer_date("2024-07-01")

# Get top recommendations
top_buys = predictions.top_buys(n=10)
top_sells = predictions.top_sells(n=10)
```

## Model Files

Trained models are saved to `training/models/`:
- `ppo_stock_trading_YYYYMMDD_HHMM.zip`: Model weights
- `vec_normalize_YYYYMMDD_HHMM.pkl`: Normalization stats (if using VecNormalize)

## Performance Tips

1. **GPU Training**: Use `--device cuda` for faster training
2. **Batch Size**: Larger batches (128-256) often improve stability
3. **Learning Rate**: Start with 0.01, decay to 0.00001
4. **Window Size**: 260 (1 year) captures annual patterns
5. **Epochs**: 5-10 epochs usually sufficient

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Reduce window size
- Use gradient checkpointing

### Training Instability
- Lower learning rate
- Increase gradient clipping (max_grad_norm)
- Check for NaN values in data

### Slow Training
- Increase n_steps (buffer size)
- Use multiple environments (n_envs > 1)
- Profile with PyTorch profiler

## Contributing

1. Run tests: `pytest tests/test_training.py`
2. Format code: `ruff format training/`
3. Check lints: `ruff check training/`

