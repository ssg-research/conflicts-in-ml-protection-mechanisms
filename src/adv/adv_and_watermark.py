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

from contextlib import nullcontext
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from ..schemas import AdvTrainingSchema, LearnerSchema
from ..models import select_model, save_model_if_better
from ..data import select_training_data, select_watermark_data


def pgd_linf(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, loss_function: Callable, cfg_adv: AdvTrainingSchema) -> torch.Tensor:
    """Construct PGD adversarial examples on the examples X.
       Referenced from: https://adversarial-ml-tutorial.org/adversarial_training/

    Args:
        model (torch.nn.Module): Neural model to train.
        X (torch.Tensor): Input image.
        y (torch.Tensor): Correct label.
        loss_function (Callable): Criterion to optimize.
        cfg_adv (AdvTrainingSchema): contains hyperparams for pgd

    Returns:
        [torch.Tensor]: Final perturbation.
    """

    if cfg_adv.randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * cfg_adv.eps - cfg_adv.eps
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for _ in range(cfg_adv.num_iter):
        loss = loss_function(model(X + delta), y)
        loss.backward()
        delta.data = (delta + cfg_adv.alpha*delta.grad.detach().sign()).clamp(-cfg_adv.eps,cfg_adv.eps)
        delta.grad.zero_()

    return delta.detach()

def evaluate(model: torch.nn.Module, num_classes: int, epoch: int, test_loader: Tuple[str, DataLoader],
             loss_function: Callable, cfg_adv: AdvTrainingSchema,
             log: logging.Logger, device: torch.device) -> float:
    """Helper function for testing the trained neural network

    Args:
        model (nn.Module): trained neural network (on training and adversarial datasets)
        num_classes (int): total number of classes
        epoch (int): the current epoch to evaluate on
        test_loader (Tuple[str, DataLoader]): dataloader for the dataset to evaluate model on
        loss_function (Callable): Loss function
        cfg_adv (AdvTrainingSchema, LearnerSchema): HydraConf/OmegaConf config.
        device (torch.device): Device to train on.
        log (logging.Logger): Logging facility.
    """
    model.eval()

    name, loader = test_loader
    grad_context = nullcontext() if name == "adv" else torch.no_grad()

    with grad_context:
        correct, total = 0.0, 0.0
        confusion_matrix = torch.zeros(num_classes, num_classes) # per-class accuracy
        for X, y_true in loader:
            X, y_true = X.to(device), y_true.to(device)

            if name == "adv":
                delta = torch.zeros_like(X, requires_grad=True)
                delta = pgd_linf(model, X, y_true, loss_function, cfg_adv)
                X += delta
                # clamp to [0,1] range for creating adversarial examples
                X = torch.clamp(X, 0, 1)

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


def train(cfg_adv: AdvTrainingSchema, cfg_learner: LearnerSchema, data_path: Path,
          device: torch.device, log: logging.Logger) -> None:
    """Run experiments with watermarking and differential privacy.

    Args:
        cfg_adv (AdvTrainingSchema): HydraConf/OmegaConf config for the task.
        cfg_learner (LearnerSchema): HydraConf/OmegaConf config for the datasets.
        data_path (Path): Path to training data.
        device (torch.device): Device to train on.
        log (logging.Logger): Logging facility.
    """

    # Intialize neural network
    model = select_model(cfg_learner.model_name, cfg_learner.num_classes, log).to(device)

    # Loading data
    cpu_cores = min(4, cpu_count())
    train_set, test_set = select_training_data(cfg_learner.training_data, data_path, cfg_learner.normalize_with_imagenet_vals, log)

    train_loader = DataLoader(train_set, batch_size=cfg_adv.train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_cores)
    test_loader = DataLoader(test_set, batch_size=cfg_adv.test_batch_size, shuffle=False, pin_memory=True, num_workers=cpu_cores)

    if cfg_adv.wm_train:
        wm_set = select_watermark_data(cfg_learner.watermark_data, cfg_adv.trigger_size, data_path, cfg_learner.num_classes, cfg_learner.normalize_with_imagenet_vals, log)
        wm_loader = DataLoader(wm_set, batch_size=cfg_adv.wm_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_cores)

    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=cfg_adv.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg_adv.lr, div_factor=20, steps_per_epoch=len(train_loader), epochs=cfg_adv.epochs)

    save_dir = Path(os.getcwd())

    best_test_accuracy_so_far, best_adv_accuracy_so_far, best_wm_accuracy_so_far = 0.0, 0.0, 0.0

    batch = 1  # 1-based so that batch % n == 0 after n iterations
    for epoch in range(cfg_adv.epochs):
        # Train on training set

        model.train()

        epoch_loss, epoch_count = 0.0, 0.0
        interval_loss, interval_count = 0.0, 0.0
        for X, y_true in train_loader:
            X, y_true = X.to(device), y_true.to(device)

            delta = torch.zeros_like(X, requires_grad=True)
            delta = pgd_linf(model, X, y_true, loss_func, cfg_adv)
            X += delta
            # clamp to [0,1] range for creating adversarial examples
            X = torch.clamp(X, 0, 1)

            opt.zero_grad()
        
            y_pred = model(X)
            loss = loss_func(y_pred, y_true)

            loss.backward()
            opt.step()
            sched.step() # we aren't stepping the schedule on watermark set, only on the train data; step here if using onecycle

            interval_loss += loss.item() * X.shape[0]
            interval_count += X.shape[0]

            # Full epoch takes too long; log more frequently
            if batch % cfg_adv.log_interval == 0:
                epoch_loss += interval_loss
                epoch_count += interval_count

                interval_loss /= interval_count

                wandb.log({ "running_loss": interval_loss, "batch_idx": batch })
                log.info(f"Running loss: {interval_loss}")

                interval_loss, interval_count = 0.0, 0.0

            batch += 1

        if cfg_adv.wm_train:
            for X, y_true in wm_loader:
                X, y_true = X.to(device), y_true.to(device)

                if cfg_adv.adv_wm_train:
                    delta = torch.zeros_like(X, requires_grad=True)
                    delta = pgd_linf(model, X, y_true, loss_func, cfg_adv)
                    X += delta
                    # clamp to [0,1] range for creating adversarial examples
                    X = torch.clamp(X, 0, 1)

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
        test_acc = evaluate(model, cfg_learner.num_classes, epoch, ["test", test_loader], loss_func, cfg_adv, log, device) # pylint: disable=unbalanced-tuple-unpacking
        best_test_accuracy_so_far = save_model_if_better(model, test_acc, best_test_accuracy_so_far, save_dir, "best_test_so_far", log)

        adv_acc = evaluate(model, cfg_learner.num_classes, epoch, ["adv", test_loader], loss_func, cfg_adv, log, device) # pylint: disable=unbalanced-tuple-unpacking
        best_adv_accuracy_so_far = save_model_if_better(model, adv_acc, best_adv_accuracy_so_far, save_dir, "best_adv_so_far", log)

        if cfg_adv.wm_train:
            wm_acc = evaluate(model, cfg_learner.num_classes, epoch, ["wm", wm_loader], loss_func, cfg_adv, log, device) # pylint: disable=unbalanced-tuple-unpacking
            best_wm_accuracy_so_far = save_model_if_better(model, wm_acc, best_wm_accuracy_so_far, save_dir, "best_wm_so_far", log)

        wandb.log({
            "epoch/loss": epoch_loss,
            "epoch/lr": lr,
            "epoch": epoch
        })

        log.info(f"epoch/lr: {lr}")

    # Saving final model
    save_model_if_better(model, test_acc, best_test_accuracy_so_far, save_dir, "final", log)
