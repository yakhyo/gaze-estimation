import os
import sys
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import data_config
from utils.helpers import angular_error, gaze_to_3d, get_model, get_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    # handlers=[
    #     logging.FileHandler("training.log"),
    #     logging.StreamHandler(sys.stdout)  # Display logs in terminal
    # ]
)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Gaze estimation training")
    parser.add_argument("--data", type=str, default="data", help="Directory path for gaze images.")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name, available `gaze360`, `mpiigaze`.")
    parser.add_argument("--output", type=str, default="output/", help="Path of output models.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for resuming training.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Network architecture, currently available: resnet18/34/50, mobilenetv2, mobileone_s0-s4."
    )
    parser.add_argument("--alpha", type=float, default=1, help="Regression loss coefficient.")
    parser.add_argument("--lr", type=float, default=0.00001, help="Base learning rate.")
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


def initialize_model(params, device):
    """
    Initialize the gaze estimation model, optimizer, and optionally load a checkpoint.

    Args:
        params (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to load the model and optimizer onto.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, int]: Initialized model, optimizer, and the starting epoch.
    """
    model = get_model(params.arch, params.bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    start_epoch = 0

    if params.checkpoint:
        checkpoint = torch.load(params.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint['epoch']
        logging.info(f'Resumed training from {params.checkpoint}, starting at epoch {start_epoch + 1}')

    return model.to(device), optimizer, start_epoch


def train_one_epoch(
    params,
    model,
    cls_criterion,
    reg_criterion,
    optimizer,
    data_loader,
    idx_tensor,
    device,
    epoch
):
    """
    Train the model for one epoch.

    Args:
        params (argparse.Namespace): Parsed command-line arguments.
        model (nn.Module): The gaze estimation model.
        cls_criterion (nn.Module): Loss function for classification.
        reg_criterion (nn.Module): Loss function for regression.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        data_loader (DataLoader): DataLoader for the training dataset.
        idx_tensor (torch.Tensor): Tensor representing bin indices.
        device (torch.device): Device to perform training on.
        epoch (int): The current epoch number.

    Returns:
        Tuple[float, float]: Average losses for pitch and yaw.
    """

    model.train()
    sum_loss_pitch, sum_loss_yaw = 0, 0

    for idx, (images, labels_gaze, regression_labels_gaze, _) in enumerate(data_loader):
        images = images.to(device)

        # Binned labels
        label_pitch = labels_gaze[:, 0].to(device)
        label_yaw = labels_gaze[:, 1].to(device)

        # Regression labels
        label_pitch_regression = regression_labels_gaze[:, 0].to(device)
        label_yaw_regression = regression_labels_gaze[:, 1].to(device)

        # Inference
        pitch, yaw = model(images)

        # Cross Entropy Loss
        loss_pitch = cls_criterion(pitch, label_pitch)
        loss_yaw = cls_criterion(yaw, label_yaw)

        # Mapping from binned (0 to 90) to angels (-180 to 180)
        pitch_predicted = torch.sum(F.softmax(pitch, dim=1) * idx_tensor, 1) * params.binwidth - params.angle
        yaw_predicted = torch.sum(F.softmax(yaw, dim=1) * idx_tensor, 1) * params.binwidth - params.angle

        # Mean Squared Error Loss
        loss_regression_pitch = reg_criterion(pitch_predicted, label_pitch_regression)
        loss_regression_yaw = reg_criterion(yaw_predicted, label_yaw_regression)

        # Calculate loss with regression alpha
        loss_pitch += params.alpha * loss_regression_pitch
        loss_yaw += params.alpha * loss_regression_yaw

        # Total loss for pitch and yaw
        loss = loss_pitch + loss_yaw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss_pitch += loss_pitch.item()
        sum_loss_yaw += loss_yaw.item()

        if (idx + 1) % 100 == 0:
            logging.info(
                f'Epoch [{epoch + 1}/{params.num_epochs}], Iter [{idx + 1}/{len(data_loader)}] '
                f'Losses: Gaze Yaw {sum_loss_yaw / (idx + 1):.4f}, Gaze Pitch {sum_loss_pitch / (idx + 1):.4f}'
            )
    avg_loss_pitch, avg_loss_yaw = sum_loss_pitch / len(data_loader), sum_loss_yaw / len(data_loader)

    return avg_loss_pitch, avg_loss_yaw


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
    return average_error/total_samples


def main():
    params = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_name = f'{params.dataset}_{params.arch}_{int(time.time())}'
    output = os.path.join(params.output, summary_name)
    if not os.path.exists(output):
        os.makedirs(output)
    torch.backends.cudnn.benchmark = True

    model, optimizer, start_epoch = initialize_model(params, device)
    data_loader = get_dataloader(params, mode="train")
    dataset = data_loader.dataset

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    best_avg_error = float('inf')
    k = 5  # number of folds
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_errors = []
    # K-Fold Cross Validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{k}")

        # Split data into training and validation sets for this fold
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create data loaders for the subsets
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=params.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=params.batch_size, shuffle=False)

        # Reset model and optimizer for each fold
        model, optimizer, start_epoch = initialize_model(params, device)

        for epoch in range(start_epoch, params.num_epochs):
            avg_loss_pitch, avg_loss_yaw = train_one_epoch(
                params,
                model,
                cls_criterion,
                reg_criterion,
                optimizer,
                train_loader,
                idx_tensor,
                device,
                epoch
            )

            logging.info(
                f'Epoch [{epoch + 1}/{params.num_epochs}] '
                f'Losses: Gaze Yaw {avg_loss_yaw:.4f}, Gaze Pitch {avg_loss_pitch:.4f}'
            )

            # checkpoint_path = os.path.join(output, f"checkpoint_fold_{fold+1}.ckpt")
            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': avg_loss_pitch + avg_loss_yaw,
            # }, checkpoint_path)
            # logging.info(f'Checkpoint saved at {checkpoint_path}')

        # Evaluate on validation set for the current fold
        avg_error = evaluate(params, model, val_loader, idx_tensor, device)  # Returns average error
        fold_errors.append(avg_error)

        logging.info(f'Fold {fold+1} average error: {avg_error:.4f}')

        # Save the best model for the fold
        if avg_error < best_avg_error:
            best_avg_error = avg_error
            best_model_path = os.path.join(output, f'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Best model saved for fold {fold+1} at {best_model_path}')

    # Calculate average error across all folds
    avg_error_overall = np.mean(fold_errors)
    logging.info(f'Average error across {k} folds: {avg_error_overall:.4f}')


if __name__ == '__main__':
    main()
