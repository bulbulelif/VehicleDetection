import cv2
import numpy as np
import os
import torch
from config import COUNTING_LINE_COORDS

def draw_counting_line(frame):
    """Draws the counting line on the frame."""
    p1 = (COUNTING_LINE_COORDS[0], COUNTING_LINE_COORDS[1])
    p2 = (COUNTING_LINE_COORDS[2], COUNTING_LINE_COORDS[3])
    cv2.line(frame, p1, p2, (0, 255, 255), 2) # Yellow line
    return frame

def intersects(bbox, line_coords):
    """
    Checks if a bounding box intersects with the counting line.
    bbox: (x1, y1, x2, y2)
    line_coords: (x1, y1, x2, y2)
    """
    box_x1, box_y1, box_x2, box_y2 = bbox
    line_x1, line_y1, line_x2, line_y2 = line_coords

    # Line equation: A*x + B*y + C = 0
    # A = y2 - y1
    # B = x1 - x2
    # C = -A*x1 - B*y1
    A = line_y2 - line_y1
    B = line_x1 - line_x2
    C = -A * line_x1 - B * line_y1

    # Check if any of the corners of the bounding box cross the line
    corners = [
        (box_x1, box_y1),
        (box_x2, box_y1),
        (box_x1, box_y2),
        (box_x2, box_y2)
    ]

    # Evaluate line equation for each corner
    # If signs are different, the line passes through the box
    signs = [np.sign(A * x + B * y + C) for x, y in corners]

    # If there are both positive and negative signs, the line crosses
    # Or if one corner is on the line (sign is 0) and another has a different sign
    if (1 in signs and -1 in signs) or \
       (0 in signs and (1 in signs or -1 in signs)):
        return True

    # More robust check: check if the line segment intersects with any of the box segments
    # This requires a line-segment intersection algorithm, which is more complex.
    # For simplicity, a basic check if the line's Y range overlaps with box Y range and X range overlaps
    # Or, if the line is horizontal/vertical, just check if the box crosses it.

    # For a horizontal line (common for counting in traffic)
    if line_y1 == line_y2:
        if (box_y1 < line_y1 < box_y2) or (box_y2 < line_y1 < box_y1): # Check if line is within box's y range
            if max(box_x1, line_x1) < min(box_x2, line_x2): # Check for x overlap
                return True
    # For a vertical line
    elif line_x1 == line_x2:
        if (box_x1 < line_x1 < box_x2) or (box_x2 < line_x1 < box_x1): # Check if line is within box's x range
            if max(box_y1, line_y1) < min(box_y2, line_y2): # Check for y overlap
                return True
    # General case (less common for counting lines)
    # This requires a full line-segment intersection check.
    # For project simplicity, assuming mostly horizontal/vertical lines for counting.
    return False

def xywhn2xyxy(x, w, h, img_w, img_h):
    """
    Converts normalized x, y, width, height to pixel x1, y1, x2, y2.
    x: normalized x_center
    y: normalized y_center
    w: normalized width
    h: normalized height
    img_w: image width
    img_h: image height
    """
    x_center, y_center, box_w, box_h = x, w, h # Renaming for clarity
    x1 = int((x_center - box_w / 2) * img_w)
    y1 = int((y_center - box_h / 2) * img_h)
    x2 = int((x_center + box_w / 2) * img_w)
    y2 = int((y_center + box_h / 2) * img_h)
    return x1, y1, x2, y2

def xyxy2xywh(bbox):
    """
    Converts (x1, y1, x2, y2) bounding box format to (x_center, y_center, width, height).
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    return x_center, y_center, width, height

def scale_coords(img1_shape, coords, img0_shape):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape.
    img1_shape: (height, width) of the image the coords are currently on.
    coords: numpy array of bounding boxes (x1, y1, x2, y2).
    img0_shape: (height, width) of the original image.
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clip(0, img0_shape[1])  # x1
    coords[:, 1].clip(0, img0_shape[0])  # y1
    coords[:, 2].clip(0, img0_shape[1])  # x2
    coords[:, 3].clip(0, img0_shape[0])  # y2
    return coords.astype(int)

def check_gpu():
    """Checks if CUDA is available and returns the device."""
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        return 'cuda'
    else:
        print("CUDA is not available. Using CPU.")
        return 'cpu'