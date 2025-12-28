"""
Logging configuration using loguru.
"""

import sys
from pathlib import Path
from loguru import logger

from config.settings import settings


def setup_logger():
    """Configure loguru logger with file and console output."""
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_path = Path(settings.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Console handler with colored output
    logger.add(
        sys.stdout,
        level=settings.logging.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler with rotation
    logger.add(
        settings.logging.log_file,
        level=settings.logging.level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="zip"
    )
    
    # Trade-specific log file (for audit trail)
    logger.add(
        "logs/trades.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        filter=lambda record: "TRADE" in record["message"],
        rotation="1 day",
        retention="30 days"
    )
    
    logger.info("Logger initialized")
    return logger


# Initialize logger on import
log = setup_logger()
