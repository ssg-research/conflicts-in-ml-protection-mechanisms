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
from multiprocessing import cpu_count
import os
from pathlib import Path
from typing import Tuple

import torch

from torch import nn
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.utils.module_modification import convert_batchnorm_modules, replace_all_modules

import wandb

from ..schemas import DPTrainingSchema, LearnerSchema
from ..models import select_model, save_model_if_better
from ..data import select_training_data, select_watermark_data


def evaluate(model: nn.Module, num_classes: int, epoch: int, test_loader: Tuple[str, DataLoader],
             log: logging.Logger, device: torch.device) -> float:
    """
    Evaluate the performance of a model on test or watermark sets.
    """
    model.eval()

    with torch.no_grad():
        name, loader = test_loader
        correct, total = 0.0, 0.0
        confusion_matrix = torch.zeros(num_classes, num_classes) # per-class accuracy
        for X, y_true in loader:
            X, y_true = X.to(device), y_true.to(device)

            y_pred = model(X)
            predicted = torch.argmax(y_pred, dim=1)

            total += X.shape[0]
            correct += (predicted == y_true).sum().item()
            for t,p in zip(y_true.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        avg_accuracy = correct / total

        log.info(f"Epoch {epoch}: {name}/accuracy: {avg_accuracy}")
        wandb.log({ f"accuracy/{name}/avg": avg_accuracy, "epoch": epoch })

        # Log per-class accuracy only on test set
        if name == "test":
            log.info("Per class accuracy:")
            for i, acc in enumerate(confusion_matrix.diag() / confusion_matrix.sum(1)):
                log.info(f"Acc. of {i}: {acc:}.")
                wandb.log({ f"accuracy/{name}/class_{i}": acc, "epoch": epoch })


    return avg_accuracy


def train(cfg_dp: DPTrainingSchema, cfg_learner: LearnerSchema,
          data_path: Path, device: torch.device, log: logging.Logger) -> None:
    """Run experiments with watermarking and differential privacy.
    This script can run training without watermarking or DP, just watermarking, just DP or both.

    Args:
        cfg_dp (DPTrainingSchema):   Config for the DP training layer.
        cfg_learner (LearnerSchema): Config for the underlying model.
        data_path (Path):            Path to training data.
        device (torch.device):       Device to train on.
        log (logging.Logger):        Logging facility.
    """

    # Intialize neural network
    model = select_model(cfg_learner.model_name, cfg_learner.num_classes, log)
    if cfg_dp.use_dp:
        model = convert_batchnorm_modules(model)
        model = replace_all_modules(model, nn.Dropout, lambda _: nn.Identity())
    model = model.to(device)

    # Loading data
    cpu_cores = min(4, cpu_count())
    train_set, test_set = select_training_data(cfg_learner.training_data, data_path, cfg_learner.normalize_with_imagenet_vals, log)
    if cfg_dp.wm_train:
        wm_set = select_watermark_data(cfg_learner.watermark_data, cfg_dp.trigger_size, data_path, cfg_learner.num_classes, cfg_learner.normalize_with_imagenet_vals, log)

    if cfg_dp.use_dp:
        train_num_samples = len(train_set)
        train_sample_rate = cfg_dp.train_batch_size / train_num_samples
        train_loader = DataLoader(
            train_set,
            batch_sampler=UniformWithReplacementSampler(
                num_samples=train_num_samples, sample_rate=train_sample_rate,
            ),
            pin_memory=True, num_workers=cpu_cores)
    else:
        train_loader = DataLoader(train_set, cfg_dp.train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_cores)

    if cfg_dp.wm_train:
        if cfg_dp.use_dp:
            wm_num_samples = len(wm_set)
            wm_sample_rate = cfg_dp.wm_batch_size / wm_num_samples
            wm_train_loader = DataLoader(
                wm_set,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=wm_num_samples, sample_rate=wm_sample_rate,
                ),
                pin_memory=True, num_workers=cpu_cores)
        else:
            wm_train_loader = DataLoader(wm_set, cfg_dp.wm_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_cores)

    test_loader = DataLoader(test_set, cfg_dp.test_batch_size, shuffle=False, pin_memory=True, num_workers=cpu_cores)

    if cfg_dp.wm_train:
        wm_test_loader = DataLoader(wm_set, cfg_dp.wm_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_cores)

    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=cfg_dp.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg_dp.lr, div_factor=20, steps_per_epoch=len(train_loader), epochs=cfg_dp.epochs)

    if cfg_dp.use_dp:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=train_sample_rate,
            epochs=cfg_dp.epochs,
            target_epsilon=cfg_dp.eps,
            target_delta=cfg_dp.delta,
            max_grad_norm=cfg_dp.max_grad_norm,
        )
        privacy_engine.attach(opt)

    save_dir = Path(os.getcwd())

    best_test_accuracy_so_far, best_wm_accuracy_so_far = 0.0, 0.0

    batch = 1  # 1-based so that batch % n == 0 after n iterations
    for epoch in range(cfg_dp.epochs):
        # Train on training set

        model.train()

        epoch_loss, epoch_count = 0.0, 0.0
        interval_loss, interval_count = 0.0, 0.0

        for X, y_true in train_loader:
            X, y_true = X.to(device), y_true.to(device)

            opt.zero_grad()

            y_pred = model(X)
            loss = loss_func(y_pred, y_true)

            loss.backward()
            opt.step()
            sched.step() # we aren't stepping the schedule on watermark set, only on the train data; step here if using onecycle

            interval_loss += loss.item() * X.shape[0]
            interval_count += X.shape[0]

            # Full epoch takes too long; log more frequently
            if batch % cfg_dp.log_interval == 0:
                epoch_loss += interval_loss
                epoch_count += interval_count

                interval_loss /= interval_count

                wandb.log({ "running_loss": interval_loss, "batch_idx": batch })
                log.info(f"Running loss: {interval_loss}")

                interval_loss, interval_count = 0.0, 0.0

            batch += 1

        if cfg_dp.wm_train:
            for X, y_true in wm_train_loader:
                X, y_true = X.to(device), y_true.to(device)

                opt.zero_grad()

                y_pred = model(X)
                loss = loss_func(y_pred, y_true)

                loss.backward()
                opt.step()

                epoch_loss += loss.item() * X.shape[0]
                epoch_count += X.shape[0]


        epoch_loss /= epoch_count
        lr = opt.param_groups[0]["lr"]

        # sched.step() # step here if using schedulers other than the onecycle

        # Evaluate on test and watermark set
        # Saving model if accuracy has improved
        test_acc = evaluate(model, cfg_learner.num_classes, epoch, ["test", test_loader], log, device) # pylint: disable=unbalanced-tuple-unpacking
        best_test_accuracy_so_far = save_model_if_better(model, test_acc, best_test_accuracy_so_far, save_dir, "best_test_so_far", log)

        if cfg_dp.wm_train:
            wm_acc = evaluate(model, cfg_learner.num_classes, epoch, ["wm", wm_test_loader], log, device) # pylint: disable=unbalanced-tuple-unpacking
            best_wm_accuracy_so_far = save_model_if_better(model, wm_acc, best_wm_accuracy_so_far, save_dir, "best_wm_so_far", log)

        wandb.log({
            "epoch/loss": epoch_loss,
            "epoch/lr": lr,
            "epoch": epoch
        })

        log.info(f"epoch/loss: {epoch_loss}")
        log.info(f"epoch/lr: {lr}")

        if cfg_dp.use_dp:
            epsilon, delta = privacy_engine.get_privacy_spent(cfg_dp.delta)
            wandb.log({
                "epoch/epsilon": epsilon,
                "epoch/delta": delta,
                "epoch": epoch
            })
            log.info(f"epoch/epsilon: {epsilon}")
            log.info(f"epoch/delta: {delta}")

    # Saving final model
    save_model_if_better(model, test_acc, best_test_accuracy_so_far, save_dir, "final", log)
