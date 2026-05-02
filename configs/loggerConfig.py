"""
日志配置模块

功能:
  - 初始化 loguru 日志系统
  - 添加 tqdm 兼容的 handler (避免进度条干扰)
  - 添加文件 handler (自动按日期分割、压缩、清理)

使用方式:
  from configs.loggerConfig import init_logging

  if __name__ == "__main__":
      init_logging()
      # ... 其他代码
"""

from pathlib import Path
from typing import cast

from loguru import logger
from tqdm import tqdm

from configs.appConfig import APP_CONFIG


def init_logging() -> None:
    """
    初始化日志系统.

    配置两个 handler:
    1. tqdm handler: 将日志输出到 tqdm.write, 避免与进度条显示冲突
    2. 文件 handler: 按日期分割日志文件, 自动压缩和清理

    Args:
        无参数, 使用 APP_CONFIG.logging 配置
    """
    # 移除默认的 stderr handler
    logger.remove()

    # 添加 tqdm 兼容 handler (日志输出到进度条上方)
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True
    )

    # 添加文件 handler (按日期分割, 自动清理过期日志)
    logger.add(
        cast(Path, APP_CONFIG.logging.log_dir).joinpath(
            APP_CONFIG.logging.file_pattern
        ),
        retention=APP_CONFIG.logging.retention,
        level=APP_CONFIG.logging.level,
        colorize=True
    )


# 初始化日志系统 (模块导入时自动执行)
init_logging()
logger.info(f"\n日志系统已初始化, 输出目录: {APP_CONFIG.logging.log_dir}\n")

if __name__ == "__main__":
    # 测试日志输出
    logger.debug("这是一个 DEBUG 级别的日志")
    logger.info("这是一个 INFO 级别的日志")
    logger.warning("这是一个 WARNING 级别的日志")
    logger.error("这是一个 ERROR 级别的日志")
    logger.critical("这是一个 CRITICAL 级别的日志")
