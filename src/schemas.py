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

from typing import Any, Optional

from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf


@dataclass
class LearnerSchema:
    """Schema that specifies training/watermark datasets and the model
    """
    training_data: str =  MISSING
    watermark_data: str = MISSING
    model_name: str = MISSING
    num_classes: int = MISSING
    normalize_with_imagenet_vals: bool = MISSING


@dataclass
class DPTrainingSchema:
    """Schema for training and watermarking with DP.
    """
    name: str = MISSING

    wm_train: bool = MISSING
    train_batch_size: int = MISSING
    test_batch_size: int = MISSING
    wm_batch_size: int = MISSING

    epochs: int = MISSING
    lr: float = MISSING

    log_interval: int = MISSING

    use_dp: bool = MISSING
    delta: float = MISSING
    eps: float = MISSING
    max_grad_norm: float = MISSING

    trigger_size: int = MISSING


@dataclass
class AdvTrainingSchema:
    """Schema for training and watermarking with adversarial training.
    """
    name: str = MISSING

    train_batch_size: int = MISSING
    test_batch_size: int = MISSING
    lr: float = MISSING
    epochs: int = MISSING

    log_interval: int = MISSING

    attack: str = MISSING

    eps: float = MISSING
    alpha: float = MISSING
    num_iter: int = MISSING
    randomize: bool = MISSING

    wm_train: bool = MISSING
    trigger_size: int = MISSING
    wm_batch_size: int = MISSING
    adv_wm_train: bool = MISSING


@dataclass
class Config:
    """Complete application config.
    """
    task: Any = MISSING
    data_path: str = MISSING
    learner: LearnerSchema = MISSING
    seed: Optional[int] = None
    gpu: int = MISSING



def yaml(cfg: Config) -> str:
    """Transform Config (a DictConfig) to yaml.
    Args:
        cfg (DictConfig): Omegaconf DictConfig (loaded from a file in $ROOT/conf).
    Returns:
        str: yaml representation of a loaded config.
    """

    return OmegaConf.to_yaml(cfg)


def to_dict(cfg: Config) -> str:
    """Transform Config (a DictConfig) to a Dict.
    Args:
        cfg (DictConfig): Omegaconf DictConfig (loaded from a file in $ROOT/configurations).
    Returns:
        str: Dict representation of a loaded config.
    """

    return OmegaConf.to_container(cfg)
