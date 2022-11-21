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
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class MNIST_CNN(nn.Module):
    """
    A simple four-layer CNN with dropout.
    """
    def __init__(self) -> None:
        super().__init__()

        h, w = 28, 28
        # Reduction due to MaxPooling
        h, w = h // 4, w // 4

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Flatten(),

            nn.Linear(h * w * 64, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10))

    def forward(self, x):
        """Pass x forward through the model"""
        return self.net(x)

class FASHIONMNIST_CNN(nn.Module):
    """
    A simple 4-layer CNN with Tanh activations from papernot et al. paper
    """
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2), #Output: (13, 13, 16)
            nn.MaxPool2d(2, stride=1), #Output (12, 12, 16)
            nn.Tanh(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), #Output (5, 5, 32)
            nn.MaxPool2d(2, stride=1), #Output (4, 4, 32)
            nn.Tanh(),

            nn.Flatten(),

            nn.Linear(4 * 4 * 32, 32),
            nn.Tanh(),

            nn.Linear(32, 10))

    def forward(self, x):
        """Pass x forward through the model"""
        return self.net(x)

class CIFAR_CNN(nn.Module):
    """
    An 8-layer CNN with Tanh activations from papernot et al. paper
    """
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 32, 32)
            nn.Tanh(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 32, 32)
            nn.MaxPool2d(2, stride=2), # Output: (16, 16, 32)
            nn.Tanh(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (16, 16, 64)
            nn.Tanh(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Output: (16, 16, 64)
            nn.MaxPool2d(2, stride=2), # Output: (8, 8, 64)
            nn.Tanh(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: (8, 8, 128)
            nn.Tanh(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # Output: (8, 8, 128)
            nn.MaxPool2d(2, stride=2), # Output: (4, 4, 128)
            nn.Tanh(),

            nn.Flatten(),

            nn.Linear(4 * 4 * 128, 128),
            nn.Tanh(),

            nn.Linear(128, 10))

    def forward(self, x):
        """Pass x forward through the model"""
        return self.net(x)


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet CIFAR10/100 implementation from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

def resnet20(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def select_model(model_architecture: str, num_classes: int, log: logging.Logger) -> nn.Module:
    """Choose model architecture.

    Args:
        model_architecture (str): Model architecture to train with.
        log (logging.Logger): Logging facility.

    Raises:
        ValueError: Throws if wrong architecture specified.

    Returns:
        nn.Module: Instatiated model.
    """

    models = {
        "MNIST_BASE": MNIST_CNN,
        "FashionMNIST_BASE": MNIST_CNN,
        "RN20": resnet20,
        "RN32": resnet32
    }

    if model_architecture not in models:
        raise ValueError(f"Unsupported model specified {model_architecture}. Supported models are: {list(models.keys())}")

    if "RN" in model_architecture:
        model = models[model_architecture](num_classes)
    else:
        model = models[model_architecture]()

    log.info(f"Specified model: {model_architecture}.")

    return model


def save_model_if_better(net: nn.Module, curr_accuracy: float, best_accuracy: float, save_model_dir: Path, task: str, log: logging.Logger) -> float:
    """Saves model if accuracy is better than the best one so far.

    Args:
        model (torch.nn.Module): Current neural net
        curr_accuracy (float): The current accuracy (in this epoch)
        best_accuracy (float): Best accuracy so far
        save_model_dir: directory to save the models to
        label (str): either provide label for the model best_test/wm/adv_so_far or "final"
                    "best_test/wb/adv_so_far"signifies an intermediate model save
                    based on normal/wm/adv test accuracy.
                    "final" signifies saving the final trained model.

    Returns:
        [float]: The updated best accuracy so far.
    """
    if task == "final" or curr_accuracy > best_accuracy:
        save_model_path = (save_model_dir / f"{task}.pt").resolve()
        log.info(f"Saving {task} model to {save_model_path}")
        torch.save(net.state_dict(), save_model_path)
        return curr_accuracy

    return best_accuracy
