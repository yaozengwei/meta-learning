'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import argparse
import logging
import optim
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
from models import ResNet18
from optim import Eden, ScaledAdam
from utils import (
    AttributeDict,
    fix_random_seed,
    get_parameter_groups_with_lrs,
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
        "--num-epochs",
        type=int,
        default=100,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.04, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,  # TODO: tune this
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=50,
        help="Number of epochs that affects how rapidly the learning rate decreases.",
    )

    parser.add_argument(
        "--main-batch-size",
        type=int,
        default=256,
        help="Batch size for training the main model",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default='data',
        help="Path to CIFAR10 dataset",
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
            "log_interval": 10,
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

    train_set = torchvision.datasets.CIFAR10(
        root=params.data_root, train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=params.main_batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(
        root=params.data_root, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=params.main_batch_size, shuffle=False, num_workers=2)

    logging.info(
        f"Number of samples: train {len(train_set)}, test {len(test_set)}"
    )

    return train_loader, test_loader


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

        main_out = model(inputs)
        loss = F.cross_entropy(main_out, targets)

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
                f"tot_loss {(tot_loss/(batch_idx+1)):.3}, "
                f"tot_accuracy {(tot_correct/tot_samples):.3}, "
                f"lr: {cur_lr:.3}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train_main/loss", loss.item(), params.batch_idx_main
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
            main_out = model(inputs)
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
        f"best accuracy {params.best_accuracy:.3}, best_epoch {params.best_epoch}"
    )
    if tb_writer is not None:
        tb_writer.add_scalar(
            "test/main_loss", tot_loss / (batch_idx + 1), params.cur_epoch
        )
        tb_writer.add_scalar("test/accuracy", accuracy, params.cur_epoch)


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

    model = ResNet18()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters in main_model: {num_param}")

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(params=params, model=model)
    model = model.to(device)

    main_optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )
    main_scheduler = Eden(main_optimizer, params.lr_batches, params.lr_epochs)

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

    logging.info("Preparing data")
    train_loader, test_loader = get_data_loaders(params)

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
            main_train_loader=train_loader,
            params=params,
            tb_writer=tb_writer,
        )

        logging.info(f"Testing epoch {epoch}")
        test(model=model, test_loader=test_loader, params=params, tb_writer=tb_writer)

        logging.info("Saving checkpoint")
        save_checkpoint(
            params=params,
            model=model,
            main_optimizer=main_optimizer,
            main_scheduler=main_scheduler,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
