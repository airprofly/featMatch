from pathlib import Path
from typing import cast
from loguru import logger
from tqdm import tqdm

from configs.appConfig import  LoggingConfig


class LoggerConfig:
    def __init__(self, logging_config: LoggingConfig) -> None:
        """Initialize the logging system.
    
        Configures two handlers:
        1. tqdm handler: outputs logs to tqdm.write, avoiding conflicts with progress bars
        2. file handler: rotates log files by date, with auto-compression and cleanup.
    
        Configuration is read from APP_CONFIG.logging.
        """
        # Remove default stderr handler
        logger.remove()
    
        # Add tqdm-compatible handler (outputs logs above progress bar)
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            colorize=True
        )
    
        # Add file handler (rotate by date, auto-cleanup of expired logs)
        logger.add(
            cast(Path, logging_config.log_dir).joinpath(
                logging_config.file_pattern
            ),
            retention=logging_config.retention,
            level=logging_config.level,
            colorize=True
        )
    
        logger.info(f"\nLogging system initialized, output directory: {logging_config.log_dir}\n")
