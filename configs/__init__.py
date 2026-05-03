import os
from loguru import logger

from .app_config import AppConfig
from .logger_config import LoggerConfig
from .plt_config import PltConfig

# Global configuration instance (loaded once on module import, globally unique)
_temp_config_path = os.path.join(os.path.dirname(__file__), "app_config.yml")
APP_CONFIG = AppConfig.load_from_yaml(_temp_config_path)

# Initialize logging and plotting systems (executed automatically on module import)
logger_config = LoggerConfig(APP_CONFIG.logging)
plt_config = PltConfig()

# Log successful initialization of the configuration module
logger.success("\nConfiguration module initialized successfully. AppConfig loaded and logging/plotting systems set up.\n")

__all__ = ["APP_CONFIG", "LoggerConfig", "PltConfig"]
