# src/emo_downscale/logging_utils.py

import logging
import os
from datetime import datetime

# Base logger name for the whole project
LOGGER_NAME = "emo_downscale"


def setup_global_logger(run_name: str):
    import warnings

    warnings.filterwarnings("default")

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{run_name}_{ts}.log")

    root_logger = logging.getLogger()
    # Only INFO and above
    root_logger.setLevel(logging.INFO)

    # Clear handlers from previous runs (critical!)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # FILE handler: INFO+
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    # CONSOLE handler: INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # Forward warnings as logging records
    logging.captureWarnings(True)

    # Make important external libs propagate to root
    for name in ["lightning", "pytorch_lightning"]:
        lg = logging.getLogger(name)
        lg.propagate = True
        # Ensure they don't spam DEBUG; respect INFO threshold
        if lg.level < logging.INFO:
            lg.setLevel(logging.INFO)

    root_logger.info(f"Global logger initialized (INFO) → {log_path}")
    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Child modules call this instead of logging.getLogger(__name__).

    Examples
    --------
    logger = get_logger("train")
    logger = get_logger("data.datamodule")
    logger = get_logger("models.unet")

    They end up as 'emo_downscale.<module_name>' and propagate to the
    root logger configured in setup_global_logger().
    """
    return logging.getLogger(f"{LOGGER_NAME}.{module_name}")
