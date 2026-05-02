from loguru import logger
import matplotlib.pyplot as plt


class PltConfig:
    def __init__(self) -> None:
        """Initialize matplotlib interactive mode to display images dynamically during training."""
        plt.ion()
        logger.info("\nMatplotlib plt.ion() enabled, images will display dynamically during training without blocking execution\n")


