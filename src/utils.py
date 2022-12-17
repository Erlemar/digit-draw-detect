from typing import List, Dict

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tomli as tomllib
import numpy.typing as npt


def plot_img_with_rects(
    img: npt.ArrayLike, boxes: List[List], threshold: float = 0.5, coef: int = 400
) -> matplotlib.figure.Figure:
    """
    Plot image with rectangles.

    Args:
        img: image as a numpy array
        boxes: the list of the bboxes
        threshold: threshold for bbox probability
        coef: coefficient to multiply images. Can be changed when the original image is a different size

    Returns:
        image with bboxes
    """
    fig, ax = plt.subplots(1, figsize=(4, 4))

    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch
    for _, rect in enumerate(b for b in boxes if b[1] > threshold):
        label, _, xc, yc, w, h = rect
        xc, yc, w, h = xc * coef, yc * coef, w * coef, h * coef
        # the coordinates from center-based to left top corner
        x = xc - w / 2
        y = yc - h / 2
        label = int(label)
        label = label if label != 10 else 'penis'
        label = label if label != 11 else 'junk'
        rect = [x, y, x + w, y + h]

        rect_ = patches.Rectangle(
            (rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], linewidth=2, edgecolor='blue', facecolor='none'
        )
        plt.text(rect[2], rect[1], f'{label}', color='blue')
        # Add the patch to the Axes
        ax.add_patch(rect_)
    return fig


def get_config() -> Dict:
    """
    Get dict from config.

    Returns:
        config
    """
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)

    return config
