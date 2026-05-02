from pathlib import Path
from typing import cast
from loguru import logger
from tqdm import tqdm

from configs.appConfig import APP_CONFIG

def init_logging() -> None:
    logger.remove()  # remove default logger to stderr
    logger.add(
        lambda msg: tqdm.write(msg, end=""), colorize=True
    )  # add logger that writes to tqdm.write for proper display with tqdm progress bars
    logger.add(
        cast(Path,APP_CONFIG.logging.log_dir).joinpath(APP_CONFIG.logging.file_pattern),
        retention=APP_CONFIG.logging.retention,
        compression=APP_CONFIG.logging.compression,
        level=APP_CONFIG.logging.level,
    )

# 初始化日志系统
init_logging()
