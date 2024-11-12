import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import Gaze360, MPIIGaze

from models import (
    resnet18,
    resnet34,
    resnet50,
    mobilenet_v2,
    mobileone_s0,
    mobileone_s1,
    mobileone_s2,
    mobileone_s3,
    mobileone_s4
)


def get_model(arch, bins, pretrained=True, inference_mode=False):
    """Return the model based on the specified architecture."""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=bins)
    elif arch == "mobilenetv2":
        model = mobilenet_v2(pretrained=pretrained, num_classes=bins)
    elif arch == "mobileone_s0":
        model = mobileone_s0(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s1":
        model = mobileone_s1(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s2":
        model = mobileone_s2(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s3":
        model = mobileone_s3(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s4":
        model = mobileone_s4(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def angular_error(gaze_vector, label_vector):
    dot_product = np.dot(gaze_vector, label_vector)
    norm_product = np.linalg.norm(gaze_vector) * np.linalg.norm(label_vector)
    cosine_similarity = min(dot_product / norm_product, 0.9999999)

    return np.degrees(np.arccos(cosine_similarity))


def gaze_to_3d(gaze):
    yaw = gaze[0]   # Horizontal angle
    pitch = gaze[1]  # Vertical angle

    gaze_vector = np.zeros(3)
    gaze_vector[0] = -np.cos(pitch) * np.sin(yaw)
    gaze_vector[1] = -np.sin(pitch)
    gaze_vector[2] = -np.cos(pitch) * np.cos(yaw)

    return gaze_vector


def get_dataloader(params,  mode="train"):
    """Load dataset and return DataLoader."""

    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if params.dataset == "gaze360":
        dataset = Gaze360(params.data, transform, angle=params.angle, binwidth=params.binwidth, mode=mode)
    elif params.dataset == "mpiigaze":
        dataset = MPIIGaze(params.data, transform, angle=params.angle, binwidth=params.binwidth)
    else:
        raise ValueError("Supported dataset are `gaze360` and `mpiigaze`")

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=params.num_workers,
        pin_memory=True
    )
    return data_loader

def draw_gaze(frame, bbox, pitch, yaw, thickness=2, color=(0, 0, 255)):
    """Draws gaze direction on a frame given bounding box and gaze angles."""
    # Unpack bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    # Calculate center of the bounding box
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    # Handle grayscale frames by converting them to BGR
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Calculate the direction of the gaze
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))

    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)

    # Draw gaze direction
    cv2.circle(frame, (x_center, y_center), radius=4, color=color, thickness=-1)
    cv2.arrowedLine(
        frame,
        point1,
        point2,
        color=color,
        thickness=thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.25
    )



def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])

    width = x_max - x_min
    height = y_max - y_min

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    # Top-left corner
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)


def draw_bbox_gaze(frame: np.ndarray, bbox, pitch, yaw):
    draw_bbox(frame, bbox)
    draw_gaze(frame, bbox, pitch, yaw)
