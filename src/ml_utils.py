import logging
from typing import List

import albumentations as A
import streamlit as st
import torch
from albumentations import pytorch

from src.model_architecture import Net

anchors = torch.tensor(
    [
        [[0.2800, 0.2200], [0.3800, 0.4800], [0.9000, 0.7800]],
        [[0.0700, 0.1500], [0.1500, 0.1100], [0.1400, 0.2900]],
        [[0.0200, 0.0300], [0.0400, 0.0700], [0.0800, 0.0600]],
    ]
)

transforms = A.Compose(
    [
        A.Resize(always_apply=False, p=1, height=192, width=192, interpolation=1),
        A.Normalize(),
        pytorch.transforms.ToTensorV2(),
    ]
)


def cells_to_bboxes(
    predictions: torch.Tensor, tensor_anchors: torch.Tensor, s: int, is_preds: bool = True
) -> List[List]:
    """
    Scale the predictions coming from the model_files to
    be relative to the entire image such that they for example later
    can be plotted or.
    Args:
        predictions: tensor of size (N, 3, S, S, num_classes+5)
        tensor_anchors: the anchors used for the predictions
        s: the number of cells the image is divided in on the width (and height)
        is_preds: whether the input is predictions or the true bounding boxes
    Returns:
        converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    batch_size = predictions.shape[0]
    num_anchors = len(tensor_anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        tensor_anchors = tensor_anchors.reshape(1, len(tensor_anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * tensor_anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = torch.arange(s).repeat(predictions.shape[0], 3, s, 1).unsqueeze(-1).to(predictions.device)
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / s * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * s * s, 6)
    return converted_bboxes.tolist()


def non_max_suppression(
    bboxes: List[List], iou_threshold: float, threshold: float, box_format: str = 'corners'
) -> List[List]:
    """
    Apply nms to the bboxes.

    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Args:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): 'midpoint' or 'corners' used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def intersection_over_union(
    boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format: str = 'midpoint'
) -> torch.Tensor:
    """
    Calculate iou.

    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Args:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def predict(
    model: torch.nn.Module, image: torch.Tensor, iou_threshold: float = 1.0, threshold: float = 0.05
) -> List[List]:
    """
    Apply the model_files to the predictions and to postprocessing
    Args:
        model: a trained pytorch model_files.
        image: image as a torch tensor
        iou_threshold: a threshold for intersection_over_union function
        threshold: a threshold for bbox probability

    Returns:
        predicted bboxes

    """
    # apply model_files. add a dimension to imitate a batch size of 1
    logits = model(image[None, :])
    logging.info('predicted')

    # postprocess. In fact, we could remove indexing with idx here, as there is a single image.
    # But I prefer to keep it so that this code could be easier changed for cases with batch size > 1
    bboxes: List[List] = [[] for _ in range(1)]
    idx = 0
    for i in range(3):
        S = logits[i].shape[2]
        # it could be better to initialize anchors inside the function, but I don't want to do it for every prediction.
        anchor = anchors[i] * S
        boxes_scale_i = cells_to_bboxes(logits[i], anchor, s=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
    logging.info('Starting nms')
    nms_boxes = non_max_suppression(
        bboxes[idx],
        iou_threshold=iou_threshold,
        threshold=threshold,
        box_format='midpoint',
    )

    return nms_boxes


@st.cache_data
def get_model():
    model_name = 'model_files/best_model.pth'

    model = Net()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    return model
