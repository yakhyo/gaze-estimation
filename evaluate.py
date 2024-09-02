import os
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F

from config import data_config
from utils.helpers import angular_error, gaze_to_3d, get_dataloader, get_model

import warnings
warnings.filterwarnings("ignore")
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Gaze estimation evaluation")
    parser.add_argument("--data", type=str, default="data/Gaze360", help="Directory path for gaze images.")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name, available `gaze360`, `mpiigaze`")
    parser.add_argument("--weights", type=str, default="", help="Path to model weight for evaluation.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Network architecture, currently available: resnet18/34/50, mobilenetv2, mobileone_s0-s4."
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")

    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


@torch.no_grad()
def evaluate(params, model, data_loader, idx_tensor, device):
    """
    Evaluate the model on the test dataset.

    Args:
        params (argparse.Namespace): Parsed command-line arguments.
        model (nn.Module): The gaze estimation model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        idx_tensor (torch.Tensor): Tensor representing bin indices.
        device (torch.device): Device to perform evaluation on.
    """
    model.eval()
    average_error = 0
    total_samples = 0

    for images, labels_gaze, regression_labels_gaze, _ in tqdm(data_loader, total=len(data_loader)):
        total_samples += regression_labels_gaze.size(0)
        images = images.to(device)

        # Regression labels
        label_pitch = np.radians(regression_labels_gaze[:, 0], dtype=np.float32)
        label_yaw = np.radians(regression_labels_gaze[:, 1], dtype=np.float32)

        # Inference
        pitch, yaw = model(images)

        # Regression predictions
        pitch_predicted = F.softmax(pitch, dim=1)
        yaw_predicted = F.softmax(yaw, dim=1)

        # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * params.binwidth - params.angle
        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * params.binwidth - params.angle

        pitch_predicted = np.radians(pitch_predicted.cpu())
        yaw_predicted = np.radians(yaw_predicted.cpu())

        for p, y, pl, yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
            average_error += angular_error(gaze_to_3d([p, y]), gaze_to_3d([pl, yl]))

    logging.info(
        f"Dataset: {params.dataset} | "
        f"Total Number of Samples: {total_samples} | "
        f"Mean Angular Error: {average_error/total_samples}"
    )


def main():
    params = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = get_model(params.arch, params.bins, inference_mode=True)

    if os.path.exists(params.weights):
        model.load_state_dict(torch.load(params.weights, map_location=device, weights_only=True))
    else:
        raise ValueError(f"Model weight not found at {params.weights}")

    model.to(device)
    test_loader = get_dataloader(params, mode="test")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    logging.info("Start Evaluation")
    evaluate(params, model, test_loader, idx_tensor, device)


if __name__ == '__main__':
    main()
