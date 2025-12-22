"""
Centralized logging configuration for the intraday-price-fetch project.

Provides consistent logging setup across all modules with:
- Console and file handlers
- Colored output for console
- Rotating file logs
- Module-specific log levels
"""

import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ANSI color codes for console output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to log level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Add color to module name
        record.name = f"{Colors.BLUE}{record.name}{Colors.RESET}"
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_file_name: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. Defaults to ./logs
        log_to_file: Whether to log to file
        log_file_name: Custom log file name. Defaults to app_YYYYMMDD.log
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Root logger instance
    """
    # Get or create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Use an ASCII separator when stdout encoding is not UTF-capable (common on Windows cp1252)
    stdout_encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    sep = " │ " if ("utf" in stdout_encoding and os.name != "nt") else " | "

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        fmt=f"%(asctime)s{sep}%(levelname)-8s{sep}%(name)-25s{sep}%(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file_name is None:
            log_file_name = f"app_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = RotatingFileHandler(
            log_dir / log_file_name,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Configure third-party loggers to be less verbose
def configure_third_party_loggers(level: str = "WARNING") -> None:
    """Reduce verbosity of third-party loggers."""
    noisy_loggers = [
        "urllib3",
        "aiohttp",
        "asyncio",
        "h5py",
        "matplotlib",
        "PIL",
        "tensorboard",
    ]
    log_level = getattr(logging, level.upper(), logging.WARNING)
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(log_level)

