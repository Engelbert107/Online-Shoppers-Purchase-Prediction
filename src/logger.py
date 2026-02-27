# src/logger.py

import logging
import os
from datetime import datetime
from typing import Optional

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join( LOG_DIR, f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

_LOGGER_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = LOG_FILE):
    """Configure root logging once."""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers
    )

    _LOGGER_CONFIGURED = True


def get_logger(name: str):
    """Return a logger instance with global configuration."""
    setup_logging()
    return logging.getLogger(name)
