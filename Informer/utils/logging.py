import logging
import os

def setup_logger(log_dir: str, log_name: str) -> logging.Logger:
    """
    Setup a logger that writes to a file and console.

    Args:
        log_dir (str): Directory to save the log file.
        log_name (str): Name of the log file (without extension).

    Returns:
        logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized: {log_path}")
    return logger
