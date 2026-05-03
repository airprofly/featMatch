"""CS 4476 Project 2: Local Feature Matching.

Replicates the proj2.ipynb pipeline:
  1. Load & resize images
  2. Harris corner detection (interest points)
  3. SIFT feature description
  4. Feature matching (ratio test)
  5. Visualization & evaluation
"""

from pathlib import Path

import torch
from loguru import logger
from matplotlib import pyplot as plt

from configs import APP_CONFIG
from models.feature_match import match_features
from models.harrisNet import get_interest_points
from models.siftNet import get_siftnet_features
from utils.utils import (
    evaluate_correspondence,
    load_image,
    resize_image,
    rgb2gray,
    show_correspondence_circles,
    show_correspondence_lines,
    show_interest_points,
)


def main() -> None:
    pair = APP_CONFIG.paths.image_pairs[0]
    src_path = Path(pair.src_path)
    dst_path = Path(pair.dst_path)
    figures_dir: Path = Path(APP_CONFIG.paths.output.figure_dir)
    output_dir = Path(APP_CONFIG.paths.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing image pair: {src_path.name} <-> {dst_path.name}")

    # ------------------------------------------------------------------ #
    #  1. Load and preprocess images
    # ------------------------------------------------------------------ #
    image1 = load_image(src_path)
    image2 = load_image(dst_path)

    new_size = (
        int(image1.shape[1] * APP_CONFIG.experiment.scale_factor),
        int(image1.shape[0] * APP_CONFIG.experiment.scale_factor),
    )
    image1 = resize_image(image1, new_size)
    image2 = resize_image(image2, new_size)

    image1_bw = rgb2gray(image1)
    image2_bw = rgb2gray(image2)

    image_input1 = torch.tensor(image1_bw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image_input2 = torch.tensor(image2_bw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    logger.info(f"Image 1 size: {image1.shape}, Image 2 size: {image2.shape}")

    # ------------------------------------------------------------------ #
    #  2. Harris Corner Detector — interest points
    # ------------------------------------------------------------------ #
    logger.info("Detecting interest points with Harris corner detector ...")
    x1, y1, _ = get_interest_points(image_input1, num_points=APP_CONFIG.experiment.num_points)
    x2, y2, _ = get_interest_points(image_input2, num_points=APP_CONFIG.experiment.num_points)

    x1_np, x2_np = x1.detach().numpy(), x2.detach().numpy()
    y1_np, y2_np = y1.detach().numpy(), y2.detach().numpy()

    logger.info(f"{len(x1_np)} corners in image 1, {len(x2_np)} corners in image 2")

    # Visualize interest points
    c1 = show_interest_points(image1, x1_np, y1_np)
    c2 = show_interest_points(image2, x2_np, y2_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(c1)
    axes[0].set_title(f"{len(x1_np)} corners in image 1")
    axes[1].imshow(c2)
    axes[1].set_title(f"{len(x2_np)} corners in image 2")
    fig.tight_layout()
    fig.savefig(figures_dir / "interest_points.jpg", dpi=300)
    logger.info(f"Saved interest points figure to {figures_dir / 'interest_points.jpg'}")
    plt.show(block=False)
    plt.pause(1)

    # ------------------------------------------------------------------ #
    #  3. SIFT Feature Descriptor
    # ------------------------------------------------------------------ #
    logger.info("Extracting SIFT features ...")
    image1_features = get_siftnet_features(image_input1, x1_np, y1_np)
    image2_features = get_siftnet_features(image_input2, x2_np, y2_np)
    logger.info(f"Feature shapes: {image1_features.shape}, {image2_features.shape}")

    # ------------------------------------------------------------------ #
    #  4. Feature Matching (ratio test)
    # ------------------------------------------------------------------ #
    logger.info("Matching features with ratio test ...")
    matches, _ = match_features(
        image1_features, image2_features, x1_np, y1_np, x2_np, y2_np
    )
    logger.info(f"{len(matches)} matches from {len(x1_np)} corners")

    # ------------------------------------------------------------------ #
    #  5. Visualization
    # ------------------------------------------------------------------ #
    num_vis = min(APP_CONFIG.experiment.num_vis, len(matches))
    logger.info(f"Visualizing top {num_vis} matches ...")

    c_circles = show_correspondence_circles(
        image1,
        image2,
        x1_np[matches[:num_vis, 0]],
        y1_np[matches[:num_vis, 0]],
        x2_np[matches[:num_vis, 1]],
        y2_np[matches[:num_vis, 1]],
    )
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.imshow(c_circles)
    ax2.set_title(f"Correspondence circles (top {num_vis} matches)")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "vis_circles.jpg", dpi=1000)
    logger.info(f"Saved circles visualization to {figures_dir / 'vis_circles.jpg'}")

    c_lines = show_correspondence_lines(
        image1,
        image2,
        x1_np[matches[:num_vis, 0]],
        y1_np[matches[:num_vis, 0]],
        x2_np[matches[:num_vis, 1]],
        y2_np[matches[:num_vis, 1]],
    )
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.imshow(c_lines)
    ax3.set_title(f"Correspondence lines (top {num_vis} matches)")
    fig3.tight_layout()
    fig3.savefig(figures_dir / "vis_lines.jpg", dpi=1000)
    logger.info(f"Saved lines visualization to {figures_dir / 'vis_lines.jpg'}")

    # ------------------------------------------------------------------ #
    #  6. Evaluation (only for pairs with ground truth)
    # ------------------------------------------------------------------ #
    try:
        logger.info("Evaluating matches against ground truth ...")
        num_eval = min(num_vis, len(matches))
        accuracy, c_eval = evaluate_correspondence(
            image1,
            image2,
            pair.gt_path,
            APP_CONFIG.experiment.scale_factor,
            x1_np[matches[:num_eval, 0]],
            y1_np[matches[:num_eval, 0]],
            x2_np[matches[:num_eval, 1]],
            y2_np[matches[:num_eval, 1]],
        )
        logger.info(f"Evaluation accuracy = {accuracy:.4f}")

        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.imshow(c_eval)
        ax4.set_title(f"Evaluation (accuracy={accuracy:.3f})")
        fig4.tight_layout()
        fig4.savefig(figures_dir / "eval.jpg", dpi=1000)
        logger.info(f"Saved evaluation figure to {figures_dir / 'eval.jpg'}")

        plt.show(block=False)
        plt.pause(1)

    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Skipping evaluation — ground truth not available: {e}")

    logger.success("Pipeline complete.")
    input("Press Enter to close all figures ...")
    plt.close("all")


if __name__ == "__main__":
    main()
