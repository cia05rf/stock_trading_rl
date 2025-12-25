"""
Centralized configuration for the intraday-price-fetch project.

Loads settings from config.json and environment variables (.env).
Provides a Config class that can be imported and used across all modules.

Usage:
    from shared.config import Config
    
    config = Config()
    print(config.DATA_DIR)
    print(config.API_KEY)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


def _find_project_root() -> Path:
    """Find the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to parent of shared directory
    return Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """
    Configuration class for the intraday-price-fetch project.
    
    Loads configuration from:
    1. config.json (default values)
    2. Environment variables (overrides)
    3. .env file (loaded automatically)
    """
    
    # Computed paths
    _project_root: Path = field(default_factory=_find_project_root)
    _config_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Load configuration after initialization."""
        # Load .env file
        env_file = self._project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Also check data_fetch for backward compatibility
        legacy_env = self._project_root / "data_fetch" / ".env"
        if legacy_env.exists():
            load_dotenv(legacy_env)
        
        # Load config.json
        config_paths = [
            self._project_root / "config.json",
            self._project_root / "data_fetch" / "config.json",
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config_data = json.load(f)
                break
    
    # =========================================================================
    # ENVIRONMENT VARIABLES
    # =========================================================================
    
    @property
    def API_KEY(self) -> str:
        """API key for EODHD - stored in .env for security."""
        return os.getenv("API_KEY", "")
    
    @property
    def INTERVAL(self) -> str:
        """Interval for price data (e.g., '15m', '1h', '1d')."""
        return os.getenv("INTERVAL", "15m")
    
    @property
    def TEST_MODE(self) -> bool:
        """Test mode flag - can be overridden via environment."""
        return os.getenv("TEST_MODE", "false").lower() == "true"
    
    @property
    def LOG_LEVEL(self) -> str:
        """Logging level."""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def DEVICE(self) -> str:
        """PyTorch device (cuda, cpu, mps)."""
        return os.getenv("DEVICE", "cuda")
    
    @property
    def SEED(self) -> int:
        """Random seed for reproducibility."""
        return int(os.getenv("SEED", "42"))
    
    # =========================================================================
    # PATH CONFIGURATION
    # =========================================================================
    
    @property
    def PROJECT_ROOT(self) -> Path:
        """Project root directory."""
        return self._project_root
    
    @property
    def DATA_DIR(self) -> Path:
        """Base data directory for all data files."""
        env_data_dir = os.getenv("DATA_DIR")
        if env_data_dir:
            return Path(env_data_dir).resolve()
        return (self._project_root / self._config_data.get("paths", {}).get("data_dir", "data")).resolve()
    
    @property
    def OUTPUT_DIR(self) -> Path:
        """Output directory for price data CSVs."""
        return (self._project_root / self._config_data.get("paths", {}).get("output_dir", "data")).resolve()
    
    @property
    def HDF5_FILE(self) -> Path:
        """HDF5 file path for consolidated price data."""
        env_hdf5 = os.getenv("HDF5_FILE")
        if env_hdf5:
            return Path(env_hdf5).resolve()
        hdf5_path = self._config_data.get("paths", {}).get("hdf5_file", "data/prices_highly_liquid.h5")
        return (self._project_root / hdf5_path).resolve()
    
    @property
    def SYMBOL_LISTS_FILE(self) -> Path:
        """Symbol lists JSON file path."""
        return (self._project_root / self._config_data.get("paths", {}).get("symbol_lists_file", "data/symbol_lists.json")).resolve()
    
    @property
    def SYMBOL_LISTS_AUGMENTED_FILE(self) -> Path:
        """Augmented symbol lists JSON file path."""
        return (self._project_root / self._config_data.get("paths", {}).get("symbol_lists_augmented_file", "data/symbol_lists_st_en.json")).resolve()
    
    @property
    def MODELS_DIR(self) -> Path:
        """Directory for saved models."""
        return self._project_root / "training" / "models"
    
    @property
    def LOGS_DIR(self) -> Path:
        """Directory for log files."""
        return self._project_root / "logs"
    
    @property
    def TB_LOG_DIR(self) -> Path:
        """TensorBoard log directory."""
        return self._project_root / "training" / "tensorboard_logs"
    
    # =========================================================================
    # API CONFIGURATION
    # =========================================================================
    
    @property
    def API_BASE_URL(self) -> str:
        """EODHD API base URL."""
        return self._config_data.get("api", {}).get("base_url", "https://eodhd.com/api")
    
    @property
    def EODHD_BASE_URL(self) -> str:
        """Alias for API_BASE_URL."""
        return self.API_BASE_URL
    
    @property
    def DAILY_API_LIMIT(self) -> int:
        """Daily API call limit."""
        return self._config_data.get("api", {}).get("daily_limit", 100000)
    
    # =========================================================================
    # DATE CONFIGURATION
    # =========================================================================
    
    @property
    def START_DATE(self) -> datetime:
        """Start date for historical data fetching."""
        date_str = self._config_data.get("dates", {}).get("start_date", "2020-10-01")
        return datetime.strptime(date_str, "%Y-%m-%d")
    
    @property
    def END_DATE(self) -> datetime:
        """End date for historical data (None means current date)."""
        end_date_str = self._config_data.get("dates", {}).get("end_date")
        if end_date_str:
            return datetime.strptime(end_date_str, "%Y-%m-%d")
        return datetime.now()
    
    @property
    def TRAIN_END_DATE(self) -> str:
        """End date for training data."""
        return self._config_data.get("dates", {}).get("train_end_date", "2024-01-01")
    
    @property
    def VAL_END_DATE(self) -> str:
        """End date for validation data."""
        return self._config_data.get("dates", {}).get("val_end_date", "2024-07-01")
    
    # =========================================================================
    # RATE LIMITING - SCRAPER
    # =========================================================================
    
    @property
    def API_CALLS_PER_MINUTE(self) -> int:
        """Rate limit: calls per minute for scraper."""
        return self._config_data.get("rate_limiting", {}).get("scraper", {}).get("calls_per_minute", 1000)
    
    @property
    def RATE_LIMIT_BUFFER(self) -> float:
        """Rate limit buffer percentage."""
        return self._config_data.get("rate_limiting", {}).get("scraper", {}).get("rate_limit_buffer", 0.95)
    
    @property
    def RATE_LIMIT_WINDOW(self) -> int:
        """Rate limit window in seconds."""
        return self._config_data.get("rate_limiting", {}).get("scraper", {}).get("rate_limit_window_seconds", 30)
    
    # =========================================================================
    # RATE LIMITING - AUGMENT SYMBOLS
    # =========================================================================
    
    @property
    def AUGMENT_CALLS_PER_MINUTE(self) -> int:
        """Rate limit: calls per minute for augment."""
        return self._config_data.get("rate_limiting", {}).get("augment", {}).get("calls_per_minute", 100)
    
    @property
    def AUGMENT_RATE_LIMIT_BUFFER(self) -> float:
        """Rate limit buffer for augment."""
        return self._config_data.get("rate_limiting", {}).get("augment", {}).get("rate_limit_buffer", 0.95)
    
    @property
    def AUGMENT_RATE_LIMIT_WINDOW(self) -> int:
        """Rate limit window for augment."""
        return self._config_data.get("rate_limiting", {}).get("augment", {}).get("rate_limit_window_seconds", 10)
    
    # =========================================================================
    # PROCESSING SETTINGS
    # =========================================================================
    
    @property
    def REQUEST_TIMEOUT(self) -> int:
        """HTTP request timeout in seconds."""
        return self._config_data.get("processing", {}).get("request_timeout_seconds", 30)
    
    @property
    def MAX_CONCURRENT_REQUESTS(self) -> int:
        """Maximum concurrent API requests for price scraping."""
        return self._config_data.get("processing", {}).get("max_concurrent_requests", 50)
    
    @property
    def MAX_CONCURRENT_REQUESTS_AUGMENT(self) -> int:
        """Maximum concurrent requests for symbol augmentation."""
        return self._config_data.get("processing", {}).get("max_concurrent_requests_augment", 5)
    
    @property
    def CHUNK_SIZE_DAYS(self) -> int:
        """Number of days per API chunk request."""
        return self._config_data.get("processing", {}).get("chunk_size_days", 120)
    
    @property
    def BATCH_WRITE_SIZE(self) -> int:
        """Rows to accumulate before writing to disk."""
        return self._config_data.get("processing", {}).get("batch_write_size", 10000)
    
    @property
    def RETRY_ATTEMPTS(self) -> int:
        """Number of retry attempts for failed requests."""
        return self._config_data.get("processing", {}).get("retry_attempts", 3)
    
    @property
    def RETRY_DELAY(self) -> int:
        """Delay between retries in seconds."""
        return self._config_data.get("processing", {}).get("retry_delay_seconds", 1)
    
    @property
    def BATCH_SIZE(self) -> int:
        """Number of symbols to process at a time."""
        return self._config_data.get("processing", {}).get("batch_size", 100)
    
    # =========================================================================
    # TEST MODE SETTINGS
    # =========================================================================
    
    @property
    def TEST_TICKERS_LIMIT(self) -> int:
        """Number of tickers to process in test mode."""
        return self._config_data.get("test_mode", {}).get("tickers_limit", 3)
    
    @property
    def TEST_RECORDS_LIMIT(self) -> int:
        """Number of records per ticker in test mode."""
        return self._config_data.get("test_mode", {}).get("records_limit", 3)
    
    # =========================================================================
    # DATAMODULE SETTINGS
    # =========================================================================
    
    @property
    def SEQUENCE_LENGTH(self) -> int:
        """Number of timesteps in each input sequence."""
        return self._config_data.get("datamodule", {}).get("sequence_length", 60)
    
    @property
    def DM_BATCH_SIZE(self) -> int:
        """Batch size for DataLoaders."""
        return self._config_data.get("datamodule", {}).get("batch_size", 64)
    
    @property
    def TRAIN_SPLIT(self) -> float:
        """Fraction of data for training."""
        return self._config_data.get("datamodule", {}).get("train_split", 0.7)
    
    @property
    def VAL_SPLIT(self) -> float:
        """Fraction of data for validation."""
        return self._config_data.get("datamodule", {}).get("val_split", 0.15)
    
    @property
    def NUM_WORKERS(self) -> int:
        """Number of DataLoader workers."""
        return self._config_data.get("datamodule", {}).get("num_workers", 0)
    
    @property
    def PIN_MEMORY(self) -> bool:
        """Whether to pin memory for GPU transfer."""
        return self._config_data.get("datamodule", {}).get("pin_memory", True)
    
    @property
    def NORMALIZE(self) -> bool:
        """Whether to normalize features."""
        return self._config_data.get("datamodule", {}).get("normalize", True)
    
    @property
    def INCLUDE_VOLUME(self) -> bool:
        """Whether to include volume as a feature."""
        return self._config_data.get("datamodule", {}).get("include_volume", True)
    
    @property
    def INCLUDE_VALUE(self) -> bool:
        """Whether to include value as a feature."""
        return self._config_data.get("datamodule", {}).get("include_value", False)
    
    @property
    def TARGET_TYPE(self) -> str:
        """Type of prediction target."""
        return self._config_data.get("datamodule", {}).get("target_type", "returns")
    
    @property
    def RESPECT_TICKER_BOUNDARIES(self) -> bool:
        """If True, don't create windows that span multiple tickers."""
        return self._config_data.get("datamodule", {}).get("respect_ticker_boundaries", True)
    
    @property
    def SPLIT_BY_DATE(self) -> bool:
        """If True, split by date. If False, split by fraction."""
        return self._config_data.get("datamodule", {}).get("split_by_date", True)
    
    # =========================================================================
    # TRAINING SETTINGS
    # =========================================================================
    
    @property
    def ACTION_SPACE_TYPE(self) -> str:
        """
        Action space type for the trading environment.

        Supported values:
        - "discrete": Discrete(9) action space (buy/sell at fixed proportions + hold)
        - "continuous": Box([-1, 1], shape=(1,)) action space (signed trade intensity)
        """
        default = self._config_data.get("training", {}).get("action_space_type", "discrete")
        val = str(os.getenv("ACTION_SPACE_TYPE", default)).strip().lower()
        if val not in {"discrete", "continuous"}:
            # Fall back to discrete for robustness if misconfigured.
            return "discrete"
        return val

    @property
    def INITIAL_BALANCE(self) -> float:
        """Initial balance for trading environment."""
        return float(os.getenv("INITIAL_BALANCE", "10000"))
    
    @property
    def TICKER_LIMIT(self) -> Optional[int]:
        """Limit on number of tickers for training."""
        limit = os.getenv("TICKER_LIMIT")
        return int(limit) if limit else None
    
    @property
    def TEST_TRAIN_SPLIT(self) -> float:
        """Train/test split ratio."""
        return float(os.getenv("TEST_TRAIN_SPLIT", "0.9"))
    
    @property
    def BUFFER_SIZE(self) -> int:
        """PPO buffer size (n_steps)."""
        return int(os.getenv("BUFFER_SIZE", "2048"))
    
    @property
    def TRAINING_BATCH_SIZE(self) -> int:
        """Training batch size for PPO."""
        return int(os.getenv("TRAINING_BATCH_SIZE", "64"))
    
    @property
    def WINDOW_SIZE(self) -> int:
        """Window size for observations."""
        return int(os.getenv("WINDOW_SIZE", str(52 * 5)))
    
    @property
    def OBS_WINDOW_SIZE(self) -> int:
        """Observation window size (steps of history to observe)."""
        return int(os.getenv("OBS_WINDOW_SIZE", "50"))
    
    @property
    def TOTAL_TIMESTEPS(self) -> int:
        """Total training timesteps."""
        return int(os.getenv("TOTAL_TIMESTEPS", "2000000"))
    
    @property
    def N_ENVS(self) -> int:
        """Number of parallel environments."""
        return int(os.getenv("N_ENVS", "1"))
    
    @property
    def EPOCHS(self) -> int:
        """Number of training epochs."""
        return int(os.getenv("EPOCHS", "5"))
    
    @property
    def LOSS_WEIGHT(self) -> float:
        """Weight for losses in reward."""
        return float(os.getenv("LOSS_WEIGHT", "1.0"))
    
    @property
    def GAIN_WEIGHT(self) -> float:
        """Weight for gains in reward."""
        return float(os.getenv("GAIN_WEIGHT", "5.0"))
    
    @property
    def SHARPE_WEIGHT(self) -> float:
        """Weight for Sharpe ratio in reward."""
        return float(os.getenv("SHARPE_WEIGHT", "1.0"))
    
    @property
    def HOLD_REWARD(self) -> float:
        """Reward for holding position."""
        return float(os.getenv("HOLD_REWARD", "0.0"))
    
    @property
    def TRANSACTION_FEE(self) -> float:
        """
        Starting transaction fee percentage (0.0 = Zero).
        Curriculum learning starts with zero fees to let agent learn basic patterns.
        """
        return float(os.getenv("TRANSACTION_FEE", "0.0"))
    
    @property
    def TARGET_TRANSACTION_FEE(self) -> float:
        """
        Target transaction fee percentage (0.1% = 0.001).
        The final goal fee that will be reached by the end of training.
        """
        return float(os.getenv("TARGET_TRANSACTION_FEE", "0.001"))
    
    @property
    def INVALID_ACTION_PENALTY(self) -> float:
        """Penalty for trying to buy with no cash or sell with no stock."""
        # A 'buzzer' to tell the agent it pressed the wrong button.
        return float(os.getenv("INVALID_ACTION_PENALTY", "0.1"))
    
    @property
    def TRADE_PENALTY(self) -> float:
        """Penalty for trading."""
        # Restore a tiny friction cost to prevent cost-free churning.
        # Changed from 0.0 to 0.0005 (0.05% per trade).
        return float(os.getenv("TRADE_PENALTY", "0.0005"))
    
    @property
    def TRANSACTION_COST(self) -> float:
        """Transaction cost percentage (deprecated, use TRANSACTION_FEE)."""
        return float(os.getenv("TRANSACTION_COST", str(self.TRANSACTION_FEE)))
    
    @property
    def MIN_VOLATILITY(self) -> float:
        """Minimum volatility threshold for tradability. Baseline set to TRANSACTION_FEE * 1.5"""
        return float(os.getenv("MIN_VOLATILITY", "0.0015"))
    
    @property
    def ARTIFICIAL_DECAY(self) -> float:
        """Artificial decay to encourage action (negative value applies time decay urgency)."""
        return float(os.getenv("ARTIFICIAL_DECAY", "-0.0001"))
    
    @property
    def RISK_AVERSION(self) -> float:
        """Risk aversion factor."""
        return float(os.getenv("RISK_AVERSION", "0.0"))

    @property
    def INIT_LR(self) -> float:
        """Initial learning rate"""
        return float(os.getenv("INIT_LR", "0.0001"))

    @property
    def FINAL_LR(self) -> float:
        """Final learning rate"""
        return float(os.getenv("FINAL_LR", "0.00001"))

    @property
    def LR_DECAY(self) -> float:
        """Decay rate"""
        return float(os.getenv("LR_DECAY", "0.95"))
    
    @property
    def LEARNING_RATE_START(self) -> float:
        """Starting learning rate for linear decay schedule."""
        return float(os.getenv("LEARNING_RATE_START", "3e-4"))
    
    @property
    def LEARNING_RATE_END(self) -> float:
        """Ending learning rate for linear decay schedule."""
        return float(os.getenv("LEARNING_RATE_END", "1e-5"))
    
    @property
    def LR_WARMUP_START(self) -> float:
        """Starting learning rate for warmup (low initial value)."""
        return float(os.getenv("LR_WARMUP_START", "1e-5"))
    
    @property
    def LR_PEAK(self) -> float:
        """Peak learning rate after warmup (high value)."""
        return float(os.getenv("LR_PEAK", "3e-4"))
    
    @property
    def LR_WARMUP_FRACTION(self) -> float:
        """Fraction of training for warmup phase."""
        return float(os.getenv("LR_WARMUP_FRACTION", "0.1"))
    
    @property
    def LR_DECAY_FRACTION(self) -> float:
        """Fraction of training for decay phase (after warmup)."""
        return float(os.getenv("LR_DECAY_FRACTION", "0.8"))
    
    @property
    def ENTROPY_COEF(self) -> float:
        """
        Entropy coefficient.
        INCREASED from 0.05 to 0.2 to force the agent out of the "Wait" local optimum.
        """
        return float(os.getenv("ENTROPY_COEF", "0.05"))
    
    @property
    def CASH_DECAY(self) -> float:
        """
        Penalty for sitting in cash (Artificial Decay).
        Value: -0.001 (Acts like inflation/rent). 
        If the agent sits in cash, it bleeds slowly. It MUST trade to survive.
        """
        return float(os.getenv("CASH_DECAY", "-0.1"))

    @property
    def TRADING_THRESHOLD(self) -> float:
        """Threshold above or below which a buy or sell trade will be executed"""
        return float(os.getenv("TRADING_THRESHOLD", "0.10"))
    
    # =========================================================================
    # EXCHANGE CONFIGURATIONS
    # =========================================================================
    
    @property
    def STOCK_EXCHANGE_CONFIGS(self) -> Dict[str, Dict]:
        """Stock exchange configurations for symbol fetching."""
        configs = {}
        for name, cfg in self._config_data.get("exchanges", {}).items():
            configs[name] = {
                "exchange_code": cfg["exchange_code"],
                "output_exchange": cfg["output_exchange"],
                "description": cfg["description"],
                "allowed_exchanges": set(cfg["allowed_exchanges"]) if cfg["allowed_exchanges"] else None,
                "allowed_types": set(cfg["allowed_types"]),
            }
        return configs
    
    @property
    def MAJOR_CURRENCIES(self) -> List[Tuple[str, str]]:
        """Major currency pairs to include."""
        return [tuple(pair) for pair in self._config_data.get("currencies", [])]
    
    @property
    def MAJOR_CRYPTOS(self) -> List[Tuple[str, str]]:
        """Major cryptocurrencies to include."""
        return [tuple(pair) for pair in self._config_data.get("cryptos", [])]
    
    @property
    def SYMBOL_COUNTS(self) -> Dict[str, int]:
        """Estimated symbol counts for API usage calculations."""
        return self._config_data.get("api_usage_estimates", {}).get("symbol_counts", {})


# Global config instance for convenience
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

