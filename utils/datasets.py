import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset


class Gaze360(Dataset):
    def __init__(self, root: str, transform=None, angle: int = 180, binwidth: int = 4, mode: str = 'train'):
        self.labels_dir = os.path.join(root, "Label")
        self.images_dir = os.path.join(root, "Image")

        if mode in ['train', 'test', 'val']:
            labels_file = os.path.join(self.labels_dir, f"{mode}.label")
        else:
            raise ValueError(f"{mode} must be in ['train','test', 'val']")

        self.transform = transform
        self.angle = angle if mode == "train" else 90
        self.binwidth = binwidth

        self.lines = []

        with open(labels_file) as f:
            lines = f.readlines()[1:]  # Skip the header line
            self.orig_list_len = len(lines)

            for line in tqdm(lines, desc="Loading Labels"):
                gaze2d = line.strip().split(" ")[5]
                label = np.array(gaze2d.split(",")).astype(float)
                pitch, yaw = label * 180 / np.pi

                if abs(pitch) <= self.angle and abs(yaw) <= self.angle:
                    self.lines.append(line)

        removed_items = self.orig_list_len - len(self.lines)
        print(f"{removed_items} items removed from dataset that have an angle > {self.angle}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(" ")

        image_path = line[0]
        filename = line[3]
        gaze2d = line[5]

        label = np.array(gaze2d.split(",")).astype(float)
        pitch, yaw = label * 180 / np.pi

        image = Image.open(os.path.join(self.images_dir, image_path))
        if self.transform is not None:
            image = self.transform(image)

        # bin values
        bins = np.arange(-self.angle, self.angle, self.binwidth)
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        # binned and regression labels
        binned_labels = torch.tensor(binned_pose, dtype=torch.long)
        regression_labels = torch.tensor([pitch, yaw], dtype=torch.float32)

        return image, binned_labels, regression_labels, filename


class MPIIGaze(Dataset):
    def __init__(self, root: str, transform=None, angle: int = 42, binwidth: int = 3):
        self.labels_dir = os.path.join(root, "Label")
        self.images_dir = os.path.join(root, "Image")

        label_files = [os.path.join(self.labels_dir, label) for label in os.listdir(self.labels_dir)]

        self.transform = transform
        self.orig_list_len = 0
        self.binwidth = binwidth
        self.angle = angle
        self.lines = []

        for label_file in label_files:
            with open(label_file) as f:
                lines = f.readlines()[1:]  # Skip the header line
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    pitch, yaw = label * 180 / np.pi

                    if abs(pitch) <= self.angle and abs(yaw) <= self.angle:
                        self.lines.append(line)

        removed_items = self.orig_list_len - len(self.lines)
        print(f"{removed_items} items removed from dataset that have an angle > {self.angle}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(" ")

        image_path = line[0]
        filename = line[3]
        gaze2d = line[7]

        label = np.array(gaze2d.split(",")).astype("float")
        pitch, yaw = label * 180 / np.pi

        image = Image.open(os.path.join(self.images_dir, image_path))
        if self.transform is not None:
            image = self.transform(image)

        # bin values
        bins = np.arange(-self.angle, self.angle, self.binwidth)
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        # binned and regression labels
        binned_labels = torch.tensor(binned_pose, dtype=torch.long)
        regression_labels = torch.tensor([pitch, yaw], dtype=torch.float32)

        return image, binned_labels, regression_labels, filename
