import logging
import os


def setup_logger(name: str) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    Args:
        name: Logger name 
    Returns:
        Configured logger instance
    """

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Get or create logger with given name
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Set minimum logging level
    logger.setLevel(logging.INFO)

    # Console handler - shows logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler - saves logs to file
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.INFO)

    # Log format: timestamp - module - level - message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger