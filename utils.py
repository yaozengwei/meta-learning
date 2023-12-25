# from https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import argparse
import os
import sys
import logging
import time
import numpy as np
import random
import uuid

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

# use duck typing for LRScheduler since we have different possibilities, see
# our class LRScheduler.
LRSchedulerType = object

Pathlike = Union[str, Path]


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# This is a utility that generates uuid4's and is set when the user calls
# the ``fix_random_seed`` function.
# Python's uuid module is not affected by the ``random.seed(value)`` call,
# so we work around it to provide deterministic ID generation when requested.
_lhotse_uuid: Optional[Callable] = None


def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules that Lhotse interacts with.
    Includes the ``random`` module, numpy, torch, and ``uuid4()`` function defined in this file.
    """
    global _lhotse_uuid
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    # Ensure deterministic ID creation
    rd = random.Random()
    rd.seed(random_seed)
    _lhotse_uuid = lambda: uuid.UUID(int=rd.getrandbits(128))


def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_checkpoint(
    filename: Path,
    model: Union[nn.Module, DDP],
    main_optimizer: Optional[Optimizer] = None,
    meta_optimizer: Optional[Optimizer] = None,
    main_scheduler: Optional[LRSchedulerType] = None,
    meta_scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    TODO: document it
    """
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    if next(iter(checkpoint["model"])).startswith("module."):
        logging.info("Loading checkpoint saved by DDP")

        dst_state_dict = model.state_dict()
        src_state_dict = checkpoint["model"]
        for key in dst_state_dict.keys():
            src_key = "{}.{}".format("module", key)
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        model.load_state_dict(dst_state_dict, strict=strict)
    else:
        model.load_state_dict(checkpoint["model"], strict=strict)

    checkpoint.pop("model")

    def load(name, obj):
        s = checkpoint.get(name, None)
        if obj and s:
            obj.load_state_dict(s)
            checkpoint.pop(name)

    load("main_optimizer", main_optimizer)
    load("main_scheduler", main_scheduler)
    load("meta_optimizer", meta_optimizer)
    load("meta_scheduler", meta_scheduler)
    load("grad_scaler", scaler)

    return checkpoint


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    main_optimizer: Optional[torch.optim.Optimizer] = None,
    meta_optimizer: Optional[torch.optim.Optimizer] = None,
    main_scheduler: Optional[LRSchedulerType] = None,
    meta_scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        main_optimizer=main_optimizer,
        main_scheduler=main_scheduler,
        meta_optimizer=meta_optimizer,
        meta_scheduler=meta_scheduler,
    )

    keys = [
        "best_epoch",
        "best_accuracy",
        "batch_idx_main",
        "batch_idx_meta",
        "epoch_idx_meta",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def save_checkpoint_impl(
    filename: Path,
    model: Union[nn.Module, DDP],
    params: Optional[Dict[str, Any]] = None,
    main_optimizer: Optional[torch.optim.Optimizer] = None,
    meta_optimizer: Optional[torch.optim.Optimizer] = None,
    main_scheduler: Optional[LRSchedulerType] = None,
    meta_scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    """Save training information to a file.

    Args:
      filename:
        The checkpoint filename.
      model:
        The model to be saved. We only save its `state_dict()`.
      model_avg:
        The stored model averaged from the start of training.
      params:
        User defined parameters, e.g., epoch, loss.
      optimizer:
        The optimizer to be saved. We only save its `state_dict()`.
      scheduler:
        The scheduler to be saved. We only save its `state_dict()`.
      scalar:
        The GradScaler to be saved. We only save its `state_dict()`.
      rank:
        Used in DDP. We save checkpoint only for the node whose rank is 0.
    Returns:
      Return None.
    """
    logging.info(f"Saving checkpoint to {filename}")

    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "main_optimizer": main_optimizer.state_dict() if main_optimizer is not None else None,
        "main_scheduler": main_scheduler.state_dict() if main_scheduler is not None else None,
        "meta_optimizer": meta_optimizer.state_dict() if meta_optimizer is not None else None,
        "meta_scheduler": meta_scheduler.state_dict() if meta_scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
    }

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    torch.save(checkpoint, filename)


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    main_optimizer: Optional[torch.optim.Optimizer] = None,
    meta_optimizer: Optional[torch.optim.Optimizer] = None,
    main_scheduler: Optional[LRSchedulerType] = None,
    meta_scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        params=params,
        main_optimizer=main_optimizer,
        main_scheduler=main_scheduler,
        meta_optimizer=meta_optimizer,
        meta_scheduler=meta_scheduler,
        scaler=scaler,
    )

    if params.best_epoch == params.cur_epoch:
        best_filename = params.exp_dir / "best-accuracy.pt"
        copyfile(src=filename, dst=best_filename)


def get_parameter_groups_with_lrs(
    model: nn.Module,
    lr: float,
    include_names: bool = False,
    freeze_modules: List[str] = [],
) -> List[dict]:
    """
    This is for use with the ScaledAdam optimizers (more recent versions that accept lists of
    named-parameters; we can, if needed, create a version without the names).

    It provides a way to specify learning-rate scales inside the module, so that if
    any nn.Module in the hierarchy has a floating-point parameter 'lr_scale', it will
    scale the LR of any parameters inside that module or its submodules.  Note: you
    can set module parameters outside the __init__ function, e.g.:
      >>> a = nn.Linear(10, 10)
      >>> a.lr_scale = 0.5

    Returns: a list of dicts, of the following form:
      if include_names == False:
        [  { 'params': [ tensor1, tensor2, ... ], 'lr': 0.01 },
           { 'params': [ tensor3, tensor4, ... ], 'lr': 0.005 },
         ...   ]
      if include_names == true:
        [  { 'named_params': [ (name1, tensor1, (name2, tensor2), ... ], 'lr': 0.01 },
           { 'named_params': [ (name3, tensor3), (name4, tensor4), ... ], 'lr': 0.005 },
         ...   ]

    """
    named_modules = list(model.named_modules())

    # flat_lr_scale just contains the lr_scale explicitly specified
    # for each prefix of the name, e.g. 'encoder.layers.3', these need
    # to be multiplied for all prefix of the name of any given parameter.
    flat_lr_scale = defaultdict(lambda: 1.0)
    names = []
    for name, m in model.named_modules():
        names.append(name)
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale

    # lr_to_parames is a dict from learning rate (floating point) to: if
    # include_names == true, a list of (name, parameter) for that learning rate;
    # otherwise a list of parameters for that learning rate.
    lr_to_params = defaultdict(list)

    for name, parameter in model.named_parameters():
        split_name = name.split(".")
        # caution: as a special case, if the name is '', split_name will be [ '' ].
        prefix = split_name[0]
        if prefix == "module":  # DDP
            module_name = split_name[1]
            if module_name in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        else:
            if prefix in freeze_modules:
                logging.info(f"Remove {name} from parameters")
                continue
        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

    if include_names:
        return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
    else:
        return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]


def get_params_lrs(
    optim: torch.optim.Optimizer, params_names: List[str]
) -> List[torch.Tensor]:
    named_params_lrs = {}
    with torch.no_grad():
        for group, group_params_names in zip(optim.param_groups, optim.parameters_names):
            with optim.batched_params(group["params"], group_params_names) as batches:
                for p, state, p_names in batches:
                    lrs = state["param_rms"] / (state["exp_avg_sq"].sqrt() + group["eps"])
                    lrs = lrs.unbind(0)
                    named_params_lrs.update({n: lr for n, lr in zip(p_names, lrs)})

    params_lrs = [named_params_lrs[n] for n in params_names]
    return params_lrs


def get_params_grads(
    optim: torch.optim.Optimizer, params_names: List[str]
) -> List[torch.Tensor]:
    named_params_grads = {}
    with torch.no_grad():
        for group, group_params_names in zip(optim.param_groups, optim.parameters_names):
            with optim.batched_params(group["params"], group_params_names) as batches:
                for p, state, p_names in batches:
                    grads = p.grad.unbind(0)
                    named_params_grads.update({n: g for n, g in zip(p_names, grads)})

    params_grads = [named_params_grads[n] for n in params_names]
    return params_grads


def cos_similarity(
    grad_a: torch.Tensor, grad_b: torch.Tensor, lr: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    num = (grad_a * lr * grad_b).sum()
    denom = ((grad_a * lr * grad_a).sum() * (grad_b * lr * grad_b).sum()).sqrt() + eps
    return num / denom


def cal_meta_loss(
    train_param_grads: List[torch.Tensor],
    dev_param_grads: List[torch.Tensor],
    param_lrs: List[torch.Tensor],
):
    assert len(train_param_grads) == len(dev_param_grads) == len(param_lrs)
    total = 0
    for grad_a, grad_b, lr in zip(train_param_grads, dev_param_grads, param_lrs):
        total = total + cos_similarity(grad_a, grad_b, lr)

    return -total / len(train_param_grads)


def duplicate_and_mask(
    x: torch.Tensor, targets: torch.Tensor, mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = x.shape
    mask = torch.rand(B, 2, 1, H, W) > mask_ratio
    masked_x = (x.unsqueeze(1) * mask.to(x.device)).view(B * 2, C, H, W)

    targets = targets.unsqueeze(1).expand(B, 2).contiguous().view(B * 2)

    return masked_x, targets
