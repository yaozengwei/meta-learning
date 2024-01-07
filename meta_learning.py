'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import argparse
import logging
import optim
import random
from pathlib import Path
from typing import List, Optional

from torch.utils.tensorboard import SummaryWriter
from models import MetaModel, Model, ResNet18
from optim import Eden, ScaledAdam
from utils import (
    AttributeDict,
    cal_meta_loss,
    duplicate_and_mask,
    fix_random_seed,
    get_parameter_groups_with_lrs,
    get_params_grads,
    get_params_lrs,
    load_checkpoint_if_available,
    save_checkpoint,
    setup_logger,
    str2bool,
)

LRSchedulerType = optim.LRScheduler


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--update-meta-every-k-epoch",
        type=int,
        default=1,
        help="Update the meta_model at every k epochs.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.02, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=2000,  # TODO: tune this
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="Number of epochs that affects how rapidly the learning rate decreases.",
    )

    parser.add_argument(
        "--main-batch-size",
        type=float,
        default=256,
        help="Batch size for training the main model",
    )

    parser.add_argument(
        "--meta-batch-size",
        type=float,
        default=512,
        help="Batch size for training the meta model",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default='data',
        help="Path to CIFAR10 dataset",
    )

    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.4,
        help="Mask ratio used to generate duplicated images",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.
    """
    params = AttributeDict(
        {
            "best_accuracy": 0,
            "best_epoch": -1,
            "batch_idx_main": 0,
            "batch_idx_meta": 0,
            "epoch_idx_meta": 0,
            "log_interval": 10,
            "dev_ratio": 0.1,
            "meta_in_channels": 6,
            "meta_hidden_channels": 32,
        }
    )

    return params


def get_data_loaders(params: AttributeDict):
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ori_train_set = torchvision.datasets.CIFAR10(
        root=params.data_root, train=True, download=True, transform=transform_train)

    tot_samples = len(ori_train_set)
    indexes = list(range(tot_samples))
    random.shuffle(indexes)
    train_sampels = int(tot_samples * (1 - params.dev_ratio))
    indexes_train = indexes[:train_sampels]
    indexes_dev = indexes[train_sampels:]

    train_set = torch.utils.data.Subset(ori_train_set, indexes_train)
    main_train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=params.main_batch_size, shuffle=True, num_workers=2)
    # meta_train_loader uses a larger batch_size
    meta_train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=params.meta_batch_size, shuffle=True, num_workers=2)

    # TODO: set shuffle to True or False
    dev_set = torch.utils.data.Subset(ori_train_set, indexes_dev)
    dev_loader = torch.utils.data.DataLoader(
        dev_set, batch_size=params.main_batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(
        root=params.data_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=params.main_batch_size, shuffle=False, num_workers=2)

    logging.info(
        f"Number of samples: train {len(train_set)}, dev {len(dev_set)}, test {len(test_set)}"
    )

    return main_train_loader, meta_train_loader, dev_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    main_optimizer: torch.optim.Optimizer,
    main_scheduler: LRSchedulerType,
    main_train_loader: torch.utils.data.DataLoader,
    params: AttributeDict,
    tb_writer: Optional[SummaryWriter] = None,
) -> None:
    model.train()
    device = next(model.parameters()).device

    tot_loss = 0
    tot_correct = 0
    tot_samples = 0
    for batch_idx, (inputs, targets) in enumerate(main_train_loader):
        params.batch_idx_main += 1

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = duplicate_and_mask(inputs, targets, params.mask_ratio)

        if params.batch_idx_meta > 0:
            main_out, aux_loss = model(inputs)
        else:
            main_out = model.main_model(inputs)
            aux_loss = torch.tensor(0.0)

        main_loss = F.cross_entropy(main_out, targets)
        loss = main_loss + aux_loss

        main_optimizer.zero_grad()
        loss.backward()

        main_scheduler.step_batch(params.batch_idx_main)
        # Only update the main_model's parameters
        main_optimizer.step()

        tot_loss += loss.item()
        _, predicted = main_out.max(1)
        tot_samples += targets.size(0)
        tot_correct += predicted.eq(targets).sum().item()

        if batch_idx % params.log_interval == 0:
            cur_lr = max(main_scheduler.get_last_lr())
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, loss {loss.item():.3}, "
                f"main_loss {main_loss.item():.3}, aux_loss {aux_loss.item():.3}, "
                f"tot_loss {(tot_loss/(batch_idx+1)):.3}, "
                f"tot_accuracy {(tot_correct/tot_samples):.3}, "
                f"lr: {cur_lr:.3}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train_main/loss", loss.item(), params.batch_idx_main
                )
                tb_writer.add_scalar(
                    "train_main/main_loss", main_loss.item(), params.batch_idx_main
                )
                tb_writer.add_scalar(
                    "train_main/aux_loss", aux_loss.item(), params.batch_idx_main
                )
                tb_writer.add_scalar(
                    "train_main/tot_loss", tot_loss / (batch_idx + 1), params.batch_idx_main
                )
                tb_writer.add_scalar(
                    "train_main/tot_accuracy", tot_correct / tot_samples, params.batch_idx_main
                )
                tb_writer.add_scalar("train_main/lr", cur_lr, params.batch_idx_main)


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    params: AttributeDict,
    tb_writer: Optional[SummaryWriter] = None,
) -> None:
    model.eval()
    device = next(model.parameters()).device

    tot_loss = 0
    tot_correct = 0
    tot_samples = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            main_out = model.main_model(inputs)
            main_loss = F.cross_entropy(main_out, targets)

            tot_loss += main_loss.item()
            _, predicted = main_out.max(1)
            tot_samples += targets.size(0)
            tot_correct += predicted.eq(targets).sum().item()

    accuracy = tot_correct / tot_samples
    if accuracy > params.best_accuracy:
        params.best_epoch = params.cur_epoch
        params.best_accuracy = accuracy

    logging.info(
        f"Epoch {params.cur_epoch}, main_loss {(tot_loss / (batch_idx + 1)):.3}, "
        f"accuracy {accuracy:.3}, "
    )
    if tb_writer is not None:
        tb_writer.add_scalar(
            "test/main_loss", tot_loss / (batch_idx + 1), params.cur_epoch
        )
        tb_writer.add_scalar("test/accuracy", accuracy, params.cur_epoch)


def accumulate_params_grads(
    model: nn.Module,
    main_optimizer: torch.optim.Optimizer,
    dev_loader: torch.utils.data.DataLoader,
    params: AttributeDict,
    main_params_names: List[str],
) -> List[torch.Tensor]:
    """Accumulate parameter gradients of the main_loss on dev set."""
    model.train()
    device = next(model.parameters()).device

    main_optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(dev_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = duplicate_and_mask(inputs, targets, params.mask_ratio)

        main_out = model.main_model(inputs)
        main_loss = F.cross_entropy(main_out, targets)
        main_loss.backward()

        if batch_idx % params.log_interval == 0:
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"main_loss {main_loss.item():.3}"
            )

    dev_params_grads = get_params_grads(main_optimizer, main_params_names)
    main_optimizer.zero_grad()
    return dev_params_grads


def train_meta_model(
    model: nn.Module,
    dev_params_grads: List[torch.Tensor],
    params_lrs: List[torch.Tensor],
    meta_optimizer: torch.optim.Optimizer,
    meta_scheduler: LRSchedulerType,
    meta_train_loader: torch.utils.data.DataLoader,
    main_params_names: List[str],
    params: AttributeDict,
    tb_writer: Optional[SummaryWriter] = None,
) -> List[torch.Tensor]:
    """Train the meta_model"""
    model.train()
    device = next(model.parameters()).device

    main_params = []
    for i, (name, param) in enumerate(model.main_model.named_parameters()):
        assert name == main_params_names[i]
        main_params.append(param)

    meta_params = list(model.meta_model.parameters())

    tot_loss = 0
    for batch_idx, (inputs, targets) in enumerate(meta_train_loader):
        params.batch_idx_meta += 1

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = duplicate_and_mask(inputs, targets, params.mask_ratio)

        main_out, aux_loss = model(inputs)
        main_loss = F.cross_entropy(main_out, targets)

        loss = main_loss + aux_loss

        train_params_grads = torch.autograd.grad(
            outputs=[loss],
            inputs=main_params,
            retain_graph=True,
            create_graph=True,
        )

        meta_loss = cal_meta_loss(train_params_grads, dev_params_grads, params_lrs)

        meta_optimizer.zero_grad()
        meta_loss.backward(inputs=meta_params)

        meta_scheduler.step_batch(params.batch_idx_meta)
        meta_optimizer.step()

        tot_loss += meta_loss.item()

        if batch_idx % params.log_interval == 0:
            cur_lr = max(meta_scheduler.get_last_lr())
            logging.info(
                f"Meta-epoch {params.epoch_idx_meta}, batch {batch_idx}, "
                f"meta_loss {meta_loss.item():.3}, tot_meta_loss {(tot_loss/(batch_idx+1)):.3}, "
                f"lr: {cur_lr:.3}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train_meta/meta_loss", meta_loss.item(), params.batch_idx_meta
                )
                tb_writer.add_scalar(
                    "train_meta/tot_meta_loss", tot_loss / (batch_idx + 1), params.batch_idx_meta
                )
                tb_writer.add_scalar("train_meta/lr", cur_lr, params.batch_idx_meta)


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    setup_logger(f"{params.exp_dir}/log/log-train")

    logging.info(params)
    logging.info("Training started")

    tb_writer = None
    if args.tensorboard:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    logging.info("Create models")

    main_model = ResNet18()
    num_param = sum([p.numel() for p in main_model.parameters()])
    logging.info(f"Number of model parameters in main_model: {num_param}")

    meta_model = MetaModel(params.meta_in_channels, params.meta_hidden_channels)
    num_param = sum([p.numel() for p in meta_model.parameters()])
    logging.info(f"Number of model parameters in meta_model: {num_param}")

    model = Model(main_model, meta_model, params.mask_ratio)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(params=params, model=model)
    model = model.to(device)

    main_optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(main_model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )
    main_scheduler = Eden(main_optimizer, params.lr_batches, params.lr_epochs)

    meta_optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(meta_model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )
    meta_scheduler = Eden(meta_optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "main_optimizer" in checkpoints:
        logging.info("Loading main_optimizer state dict")
        main_optimizer.load_state_dict(checkpoints["main_optimizer"])

    if (
        checkpoints
        and "main_scheduler" in checkpoints
        and checkpoints["main_scheduler"] is not None
    ):
        logging.info("Loading main_scheduler state dict")
        main_scheduler.load_state_dict(checkpoints["main_scheduler"])

    if checkpoints and "meta_optimizer" in checkpoints:
        logging.info("Loading meta_optimizer state dict")
        meta_optimizer.load_state_dict(checkpoints["meta_optimizer"])

    if (
        checkpoints
        and "meta_scheduler" in checkpoints
        and checkpoints["meta_scheduler"] is not None
    ):
        logging.info("Loading meta_scheduler state dict")
        meta_scheduler.load_state_dict(checkpoints["meta_scheduler"])

    logging.info("Preparing data")
    main_train_loader, meta_train_loader, dev_loader, test_loader = get_data_loaders(params)

    main_params_names = [name for name, param in model.main_model.named_parameters()]

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        fix_random_seed(epoch)
        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train_main/epoch", epoch, params.batch_idx_main)

        logging.info(f"Training epoch {epoch}")
        main_scheduler.step_epoch(epoch - 1)
        train_one_epoch(
            model=model,
            main_optimizer=main_optimizer,
            main_scheduler=main_scheduler,
            main_train_loader=main_train_loader,
            params=params,
            tb_writer=tb_writer,
        )

        logging.info(f"Testing epoch {epoch}")
        test(model=model, test_loader=test_loader, params=params, tb_writer=tb_writer)

        if epoch % params.update_meta_every_k_epoch == 0 and epoch >= 5:
            # start to optimize the meta_model
            params_lrs = get_params_lrs(main_optimizer, main_params_names)

            logging.info("Accumulating gradients on dev-set")
            dev_params_grads = accumulate_params_grads(
                model=model,
                main_optimizer=main_optimizer,
                dev_loader=dev_loader,
                params=params,
                main_params_names=main_params_names,
            )

            for _ in range(1):
                params.epoch_idx_meta += 1
                meta_scheduler.step_epoch(params.epoch_idx_meta - 1)
                train_meta_model(
                    model=model,
                    dev_params_grads=dev_params_grads,
                    params_lrs=params_lrs,
                    meta_optimizer=meta_optimizer,
                    meta_scheduler=meta_scheduler,
                    meta_train_loader=meta_train_loader,
                    main_params_names=main_params_names,
                    params=params,
                    tb_writer=tb_writer,
                )

        logging.info("Saving checkpoint")
        save_checkpoint(
            params=params,
            model=model,
            main_optimizer=main_optimizer,
            main_scheduler=main_scheduler,
            meta_optimizer=meta_optimizer,
            meta_scheduler=meta_scheduler,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
