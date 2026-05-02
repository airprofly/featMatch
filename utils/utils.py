from email.mime import image
from pathlib import Path
import pickle
import sys

from matplotlib import pyplot as plt, scale
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.__config__ import show


def load_image(image_path: str | Path) -> NDArray[np.float32]:
    """
    Load an image from the specified path and convert it to a NumPy array. The image is normalized to the range [0, 1].
    Args:
        path (str): The file path to the image.
    Returns:
        np.ndarray: A NumPy array representing the image, with pixel values normalized to the range [0, 1].
    """
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)
    float_image = image.astype(np.float32) / 255.0
    return float_image


def save_image(image: NDArray[np.floating], path: str | Path) -> None:
    """
    Save a NumPy array as an image to the specified path. The pixel values are expected to be in the range [0, 1] and will be scaled to [0, 255] before saving.
    Args:
        image (np.ndarray): A NumPy array representing the image, with pixel values in the range [0, 1].
        path (str): The file path where the image will be saved.
    """
    scaled_image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(scaled_image)
    pil_image.save(path)


def show_image(axis: Axes, image: NDArray[np.floating], title: str) -> None:
    """
    Display an image on the specified matplotlib axis.

    Args:
        axis (matplotlib.axes.Axes): The matplotlib axis object to display the image on.
        image (NDArray[np.floating]): The image array to display.
        title (str): The title for the image.
    """
    axis.imshow(np.clip(image, 0, 1))
    axis.set_title(title)
    axis.axis("off")

def resize_image(image: NDArray[np.floating], new_size: tuple[int, int]) -> NDArray[np.floating]:
    """
    Resize an image to the specified new size.

    Args:
        image (NDArray[np.floating]): The input image array to resize.
        new_size (tuple[int, int]): The desired output size as a tuple (width, height).

    Returns:
        NDArray[np.floating]: The resized image array.
    """
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    resized_pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    resized_image = np.asarray(resized_pil_image).astype(np.float32) / 255.0
    return resized_image


if __name__ == "__main__":
    from configs.appConfig import APP_CONFIG
    scale_factor: float = 0.5
    image_a = load_image(APP_CONFIG.paths.image_pairs[0].src_path)
    image_b = load_image(APP_CONFIG.paths.image_pairs[0].dst_path)
    resized_image_a = resize_image(image_a, (int(image_a.shape[1] * scale_factor), int(image_a.shape[0] * scale_factor)))
    resized_image_b = resize_image(image_b, (int(image_b.shape[1] * scale_factor), int(image_b.shape[0] * scale_factor)))
    print(resized_image_a.shape, resized_image_b.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plt.ion()
    show_image(axes[0], resized_image_a, "Resized Image A")
    show_image(axes[1], resized_image_b, "Resized Image B")
    plt.show()