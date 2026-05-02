from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
from PIL import Image


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
    # 使用 'gray' 色图确保灰度图像正确显示，RGB图像不受影响
    axis.imshow(np.clip(image, 0, 1), cmap="gray")
    axis.set_title(title)
    axis.axis("off")


def resize_image(
    image: NDArray[np.floating], new_size: tuple[int, int]
) -> NDArray[np.floating]:
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


def rgb2gray(image: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Convert an RGB image to grayscale.

    Args:
        image (NDArray[np.floating]): The input RGB image array with shape (H, W, 3) and pixel values in the range [0, 1].

    Returns:
        NDArray[np.floating]: The output grayscale image array with shape (H, W) and pixel values in the range [0, 1].
    """
    if image.ndim == 2:  # Already grayscale
        return image
    if image.ndim == 3 and image.shape[2] == 3:  # RGB to Grayscale
        c = [0.2989, 0.5870, 0.1140]  # Standard weights for RGB to grayscale conversion
        return np.dot(image[..., :3], c)  # Apply the weights to convert to grayscale
    else:
        raise ValueError("Input image must be either a grayscale or RGB image.")


if __name__ == "__main__":
    from configs import APP_CONFIG
    from loguru import logger
    from torchvision.transforms import ToTensor, transforms

    scale_factor: float = 0.5
    image_a = load_image(APP_CONFIG.paths.image_pairs[0].src_path)
    image_b = load_image(APP_CONFIG.paths.image_pairs[0].dst_path)
    logger.info(
        f"Original Image A shape: {image_a.shape}, Image B shape: {image_b.shape}"
    )
    resized_image_a = resize_image(
        image_a,
        (int(image_a.shape[1] * scale_factor), int(image_a.shape[0] * scale_factor)),
    )
    resized_image_b = resize_image(
        image_b,
        (int(image_b.shape[1] * scale_factor), int(image_b.shape[0] * scale_factor)),
    )
    logger.info(
        f"Resized Image A shape: {resized_image_a.shape}, Image B shape: {resized_image_b.shape}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    show_image(axes[0], resized_image_a, "Resized Image A")
    show_image(axes[1], resized_image_b, "Resized Image B")
    image_a_gray = rgb2gray(resized_image_a)
    image_b_gray = rgb2gray(resized_image_b)
    logger.info(
        f"Grayscale Image A shape: {image_a_gray.shape}, Image B shape: {image_b_gray.shape}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    show_image(axes[0], image_a_gray, "Grayscale Image A")
    show_image(axes[1], image_b_gray, "Grayscale Image B")
    transform = transforms.Compose(
        [
            ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add batch dimension
        ]
    )
    image_a_tensor = transform(resized_image_a)
    image_b_tensor = transform(resized_image_b)
    logger.info(
        f"Tensor Image A shape: {image_a_tensor.shape}, Image B shape: {image_b_tensor.shape}"
    )
    plt.pause(5)