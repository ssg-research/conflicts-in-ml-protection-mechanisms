# Authors: Sebastian Szyller
# Copyright 2022 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import glob
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms


class GTSRB(Dataset):
    def __init__(self, root_dir: str, train: bool, transform: transforms.Compose, download: bool, use_probs=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_probs = use_probs

        self.data_frame = []
        self.label_frame = []
        idx = 0
        if train:
            for i in range(1, 44):
                dir_ = os.path.join(self.root_dir + "/GTSRB/train/" + str(i-1) + "/")
                for fimg in glob.glob(dir_ + '*.ppm'):
                    self.data_frame.append(fimg)
                    self.label_frame.append(i-1)
                    idx += 1
        else:
            for i in range(1, 44):
                dir_ = os.path.join(self.root_dir + "/GTSRB/test/" + str(i-1) + "/")
                for fimg in glob.glob(dir_ + '*.ppm'):
                    self.data_frame.append(fimg)
                    self.label_frame.append(i-1)
                    idx += 1

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.data_frame[idx]
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = self.transform(image)
        if self.use_probs:
            label = np.asarray(self.label_frame.iloc[idx,:], dtype=np.float32)
        else:
            label = self.label_frame[idx]
        sample = [image, label]
        return sample


def select_training_data(ds_name: str, data_path: Path, normalize_with_imagenet_vals: bool, log: logging.Logger) -> List[Dataset]:
    """Download a dataset to be used to training/testing.

    Args:
        ds_name (str): Name of the dataset to use for the primary training task.
        data_path (Path): Load data from/save data to this directory.
        log (logging.Logger): Logging facility.

    Raises:
        ValueError: Throw if a wrong dataset is provided.

    Returns:
        List[Dataset]: Train and test set.
    """

    supported_datasets: Dict[str, Callable[..., Dataset]] = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10,
        "STL10": datasets.STL10
        # "GTSRB": GTSRB
    }

    if ds_name not in supported_datasets:
        raise ValueError(f"Unsupported dataset specified {ds_name}. Supported datasets are: {list(supported_datasets.keys())}")

    dataset = supported_datasets[ds_name]

    train_transform, test_transform = select_transform(ds_name, normalize_with_imagenet_vals)
    if ds_name == "STL10":
        train_set = dataset(data_path, split="train", transform=train_transform, download=True)
        test_set = dataset(data_path, split="test", transform=test_transform, download=True)
    else:
        train_set = dataset(data_path, train=True, transform=train_transform, download=True)
        test_set = dataset(data_path, train=False, transform=test_transform, download=True)

    log.info(f"Selected training dataset: {ds_name}. Included training samples {len(train_set)}, and testing samples {len(test_set)}.") # type: ignore

    return [train_set, test_set]


def select_watermark_data(wm_name: str, watermark_size: int, data_path: Path, num_classes: int, normalize_with_imagenet_vals: bool, log: logging.Logger) -> Dataset:
    """Download dataset and select a random sample for the watermark/trigger.

    Args:
        wm_name (str): Name of the dataset to use as the watermark/trigger.
        watermark_size (int): Number of samples in the watermark/trigger.
        data_path (Path): Load data from/save data to this directory.
        log (logging.Logger): Logging facility.

    Raises:
        ValueError: Throw if a wrong dataset is provided.

    Returns:
        Dataset: Data to use as watermark/trigger.
    """
    supported_watermarks = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        # "CIFAR10": datasets.CIFAR10,
        "GTSRB": GTSRB
    }

    if wm_name not in supported_watermarks:
        raise ValueError(f"Unsupported dataset specified {wm_name}. Supported datasets are: {list(supported_watermarks.keys())}")

    dataset = supported_watermarks[wm_name]

    _, test_transform = select_transform(wm_name, normalize_with_imagenet_vals)
    test_set = dataset(str(data_path), train=False, transform=test_transform, download=True)
    samples_in_test = len(test_set)
    watermark_set, _ = random_split(test_set, [watermark_size, samples_in_test - watermark_size])

    # CIFAR10 has 10 and CIFAR100 has 100 classes while GTSRB has 45; need to give a new label of suitable
    if wm_name == "GTSRB":
        watermark_set = SimpleDataset([(img, another_label(label, num_classes)) for img, label in watermark_set])

    log.info(f"Selected watermark dataset: {wm_name}. Samples in trigger set: {len(watermark_set)}.")

    return watermark_set


def select_transform(ds_name: str, normalize_with_imagenet_vals: bool) -> List[transforms.Compose]:
    """Fetch train and test transforms for the given dataset.

    Args:
        ds_name (str): Corresponding dataset.
        normalize_with_imagenet_vals (bool): ImageNet-pretrained ResNets don't use [0.5, 0.5, 0.5] for normalization.

    Returns:
        List[transforms.Compose]: Train and test transforms for the given dataset.
    """

    supported_train = {
        "MNIST": transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]),

        "FashionMNIST": transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]),

        "CIFAR10": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5],
                std = [0.229, 0.224, 0.225] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5]
            )]),

        "STL10": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5],
                std = [0.229, 0.224, 0.225] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5]
            )]),

        "GTSRB": transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5],
                std = [0.229, 0.224, 0.225] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5]
            )])
    }

    supported_test = {
        "MNIST": transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]),

        "FashionMNIST": transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]),

        "CIFAR10": transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5],
                std = [0.229, 0.224, 0.225] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5]
            )]),

        "STL10": transforms.Compose([
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5],
                std = [0.229, 0.224, 0.225] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5]
            )]),

        "GTSRB": transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5],
                std = [0.229, 0.224, 0.225] if normalize_with_imagenet_vals else [0.5, 0.5, 0.5]
            )])
    }

    return [supported_train[ds_name], supported_test[ds_name]]


class SimpleDataset(Dataset):
    def __init__(self, dataset: List[Tuple[Any, int]]) -> None:
        self.data, self.labels = zip(*dataset)
        self.count = len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count


def another_label(real_label: int, number_of_classes: int) -> int:
    new_label = real_label
    while new_label == real_label:
        new_label = random.randint(0, number_of_classes - 1)
    return new_label
