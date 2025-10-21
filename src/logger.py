"""
Logging Configuration Module

Provides centralized logging functionality for the entire project.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging format
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create logger instance
logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Parameters:
    -----------
    name : str
        Name of the logger (typically __name__)
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerConfig:
    """
    Logger configuration class for advanced logging setup.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Initialize logger configuration.
        
        Parameters:
        -----------
        log_dir : str
            Directory to store log files
        log_level : int
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.setup_logger()
    
    def setup_logger(self):
        """
        Set up logger with file and console handlers.
        """
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        log_path = self.log_dir / log_file
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(self.log_level)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(self.log_level)
        file_format = logging.Formatter(
            "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(levelname)s - %(name)s - %(message)s"
        )
        console_handler.setFormatter(console_format)
        
        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


# Initialize default logger
_default_logger_config = LoggerConfig()

