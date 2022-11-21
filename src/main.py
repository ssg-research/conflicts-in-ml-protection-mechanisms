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
import os
import random
import subprocess

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import rich.traceback
import torch
import wandb

from . import schemas
from . import os_layer
from .dp import dp_and_watermark
from .adv import adv_and_watermark

os.environ["HYDRA_FULL_ERROR"] = "1"

rich.traceback.install()

cs = ConfigStore.instance()
cs.store(name="config", node=schemas.Config)
cs.store(group="task", name="dp-wm", node=schemas.DPTrainingSchema)
cs.store(group="task", name="dp-only", node=schemas.DPTrainingSchema)
cs.store(group="task", name="wm-only", node=schemas.DPTrainingSchema)
cs.store(group="task", name="adv-wm", node=schemas.AdvTrainingSchema)
cs.store(group="task", name="adv-only", node=schemas.AdvTrainingSchema)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: schemas.Config) -> None:
    """Load the config file, parse args and launch specified training loop.
    Args:
        cfg (schemas.Config): Config for managing the experiment and corresponding variables.
    Raises:
        EnvironmentError: Mistake in the config file; it should not happen.
    """

    log = logging.getLogger(__name__)

    # Seed expriment consistently s.t. they can be repeated
    if cfg.seed is not None:
        log.info(f"Using provided seed: {cfg.seed}")
    else:
        seed = random.randrange(2**32 - 1)  # max for numpy.random.seed
        cfg.seed = seed
        log.info(f"Generating seed: {cfg.seed}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)  # seeding this is legacy and might not be needed on some systems

    # Train on GPU only
    if cfg.gpu == -1:
        gpu_idx = auto_alloc_gpu()
    else:
        gpu_idx = cfg.gpu

    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cpu'):
        raise EnvironmentError("GPU not available. Aborting.")

    log.info(f"Allocated job to GPU:{gpu_idx}")

    # Hydra overwrites cwd; get original and create directory for data, see conf/config.yaml for the hydra env var resolver
    data_path = os_layer.create_dir_if_doesnt_exist(cfg.data_path, log)
    if data_path is None:
        raise ValueError(f"Data directory specified by data_path does not exist {cfg.data_path}")

    log.info(f"Full data path: {data_path.resolve()}")

    wandb.init(
        project="ml-conf-interest",
        entity="YOUR ENTITY",
        config=schemas.to_dict(cfg)
    )

    if cfg.task.name in ["dp-wm", "dp-only", "wm-only"]:
        log.info("Maybe watermarking with or without DP.")
        dp_and_watermark.train(cfg.task, cfg.learner, data_path, device, log)

    elif cfg.task.name in ["adv-wm", "adv-only"]:
        log.info("Adversarial training with or without watermarking.")
        adv_and_watermark.train(cfg.task, cfg.learner, data_path, device, log)

    else:
        raise EnvironmentError(f"Unknown task name: {cfg.task.name}")

    wandb.finish()


def auto_alloc_gpu() -> int:
    cmd0 = "nvidia-smi -x -q | jc --xml -p | jq '.nvidia_smi_log.gpu[0].fb_memory_usage.used' | awk '{ print $1 }' | cut -c 2-"
    cmd1 = "nvidia-smi -x -q | jc --xml -p | jq '.nvidia_smi_log.gpu[1].fb_memory_usage.used' | awk '{ print $1 }' | cut -c 2-"

    ps0 = subprocess.Popen(cmd0, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ps1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output0 = int(ps0.communicate()[0])
    output1 = int(ps1.communicate()[0])

    if output0 > output1:
        return 1
    else:
        return 0


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
