import os
import random

from loguru import logger
import numpy as np
import torch
import torch.nn as nn


def get_dtype(use_gpu: bool = True, use_bf16: bool = True) -> tuple[torch.dtype, bool]:
    """
    Get the dtype to use for training.

    Args:
        use_gpu (bool): Whether to use GPU if possible.
        use_bf16 (bool): Whether to use bfloat16 if possible.

    Returns:
        tuple[torch.dtype, bool]: The dtype to use and whether to use GradScaler.
    """
    if use_gpu and torch.cuda.is_available():
        support_native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        use_bf16 = use_bf16 and support_native_bf16

        if use_bf16:
            return torch.bfloat16, False

        if not support_native_bf16:
            logger.warning("bfloat16 is not natively supported. Falling back to float16.")

        return torch.float16, True

    return torch.float32, False


def split_parameters_for_weight_decay(
    model: nn.Module, weight_decay: float, no_decay_layer_types: tuple = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
) -> list[dict]:
    """
    Splits model parameters into two groups:
    1. Parameters to apply weight decay to (typically weights of Linear/Conv layers).
    2. Parameters to exclude from weight decay (typically biases and normalization weights).

    Strategy:
    - Decay: Weights with ndim >= 2 (Linear, Conv, Embedding).
    - No Decay: Biases (names ending in .bias) and 1D parameters (Norms).

    Args:
        model: The Pytorch model.
        weight_decay: The target weight decay value.
        no_decay_layer_types: Explicit types of layers to exclude from decay (optional safeguard).

    Returns:
        List of dictionaries suitable for torch.optim.Optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias"):
            no_decay_params.append(param)
            continue

        parent_name = name.rsplit(".", 1)[0]
        parent_module = model.get_submodule(parent_name)
        if isinstance(parent_module, no_decay_layer_types):
            no_decay_params.append(param)
            continue

        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    assert len(set(decay_params)) + len(set(no_decay_params)) == len([p for p in model.parameters() if p.requires_grad]), (
        "Some parameters were missed in the split logic!"
    )

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def log_model_size(model: nn.Module) -> None:
    param_count, size_all_mb = get_model_size(model)
    logger.info(f"Model {param_count} parameters and size of {size_all_mb:.2f} MB")


def get_model_size(model: nn.Module) -> tuple[int, float]:
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return param_count, size_all_mb


def set_seed(seed: int = 2048, deterministic: bool = False) -> None:
    """
    Sets seeds for all random number generators.

    Args:
        seed (int): The seed value.
        deterministic (bool): If True, sets flags that ensure reproducibility
                              at the cost of performance (slower training).
                              If False, prioritizes speed (enables cudnn.benchmark).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
