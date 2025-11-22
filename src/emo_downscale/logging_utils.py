# src/emo_downscale/logging_utils.py

import logging
import os
from datetime import datetime

LOGGER_NAME = "emo_downscale"  # <-- Unique logger for entire project

def setup_global_logger(run_name: str):
    """Configure a global logger shared across all modules."""
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{run_name}_{ts}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-runs
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # File handler
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False
    logger.info(f"Global logger initialized → {log_path}")

    return logger


def get_logger(module_name: str):
    """
    Child modules call this instead of logging.getLogger(__name__).
    Ensures they use the same parent logger.
    """
    return logging.getLogger(f"{LOGGER_NAME}.{module_name}")
