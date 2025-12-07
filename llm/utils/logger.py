import logging
import os
from pathlib import Path

def setup_logging(output_dir, log_file='training_log.log', level=logging.INFO):
    """
    Set up logging to both console and a file.
    
    Args:
        output_dir (str): Directory to save the log file.
        log_file (str): Name of the log file.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplication (e.g., in Jupyter notebooks)
    logger.handlers.clear()

    # Create formatters
    # A more detailed formatter for the file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # A simpler formatter for the console
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # File handler (logs everything)
    file_handler = logging.FileHandler(log_path, mode='w') # 'w' to overwrite, 'a' to append
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (streams to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logging is set up. Log file: %s", log_path)
    return logger
