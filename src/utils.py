import datetime
import json
import os
import uuid
from typing import List

import boto3
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy.typing as npt
import streamlit as st
import tomli

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
try:
    if st.secrets is not None:
        AWS_ACCESS_KEY_ID = st.secrets['AWS_ACCESS_KEY_ID']
        AWS_SECRET_ACCESS_KEY = st.secrets['AWS_SECRET_ACCESS_KEY']
except BaseException:
    pass

if os.path.exists('config.toml'):
    with open('config.toml', 'rb') as f:
        config = tomli.load(f)
        AWS_ACCESS_KEY_ID = config['AWS_ACCESS_KEY_ID']
        AWS_SECRET_ACCESS_KEY = config['AWS_SECRET_ACCESS_KEY']

client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


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


def save_object_to_s3(filename, s3_filename):
    client.upload_file(filename, 'digitdrawdetect', s3_filename)


@st.cache(show_spinner=False)
def save_image(image: npt.ArrayLike, pred: List[List]) -> str:
    """
    Save the image and upload the image with bboxes to s3.

    Args:
        image: np.array with image
        pred: bboxes

    Returns:
        image name

    """
    # create a figure and save it
    fig, ax = plt.subplots(1, figsize=(4, 4))
    ax.imshow(image)
    file_name = str(datetime.datetime.today().date()) + str(uuid.uuid1())
    fig.savefig(f'{file_name}.png')

    # dump bboxes in a local file
    with open(f'{file_name}.json', 'w') as f:
        json.dump({f'{file_name}.png': pred}, f)

    # upload the image and the bboxes to s3.
    save_object_to_s3(f'{file_name}.png', f'images/{file_name}.png')
    save_object_to_s3(f'{file_name}.json', f'labels/{file_name}.json')

    return file_name
