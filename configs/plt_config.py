from loguru import logger
import matplotlib.pyplot as plt


class PltConfig:
    def __init__(self) -> None:
        """Initialize matplotlib interactive mode to display images dynamically during training."""
        plt.ion()
        plt.rcParams["figure.autolayout"] = True # Automatically adjust subplot parameters to give specified padding
        plt.rcParams["axes.unicode_minus"] = False # Ensure that minus signs are displayed correctly in plots not using a grid
        logger.info("\nMatplotlib plt.ion() enabled, images will display dynamically during training without blocking execution\n")

