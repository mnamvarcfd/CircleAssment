"""
Centralized logging configuration.

This module provides a standardized logging setup that can be imported
and used across all modules in the project.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = 'main', log_level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure and return a logger that writes to a rotating file (no console output).
    
    Args:
        name (str): Name of the logger (typically the module name)
        log_level (int): Logging level (default: logging.DEBUG)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # If already configured, return existing
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    # Resolve logs directory relative to project root (one level up from src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # One log file per logger name
    log_file_path = os.path.join(logs_dir, f"{name}.log")

    # Rotating file handler: 5 MB per file, keep 5 backups
    # Set delay=False to ensure the file is created immediately
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=5 * 1024 * 1024, 
        backupCount=5, 
        encoding='utf-8',
        delay=False
    )
    file_handler.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    # Force immediate write by flushing (helps with debugging)
    file_handler.flush()

    return logger



