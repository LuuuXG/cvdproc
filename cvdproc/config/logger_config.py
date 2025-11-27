import logging
import os
import sys

def setup_logger(name="cvdproc", log_file=None, level=logging.DEBUG):
    """
    Configure and return a consistent logger instance.

    :param name: str, logger name
    :param log_file: str, optional log file path (if provided, logs are also written to the file)
    :param level: logging.LEVEL, logging level (default DEBUG)
    :return: logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent messages from reaching the root logger (avoids duplicates)
    logger.propagate = False

    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console log handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File log handler (if a log file path is provided)
    if log_file:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global logger configuration (ready to use)
logger = setup_logger()
