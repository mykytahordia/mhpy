import os
import random

from loguru import logger
import numpy as np
import torch
import torch.nn as nn

from mhpy.utils.format import fcount
from mhpy.utils.format import fsize


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


def log_model_size(model: nn.Module) -> None:
    param_count, size_bytes = get_model_size(model)
    logger.info(f"Params: {fcount(param_count)} | Memory: {fsize(size_bytes)}")


def get_model_size(model: nn.Module) -> tuple[int, int]:
    param_size, param_count = 0, 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    return param_count, param_size + buffer_size


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
