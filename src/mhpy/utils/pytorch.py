from contextlib import AbstractContextManager
from contextlib import nullcontext
from dataclasses import dataclass
import os
import random

from loguru import logger
import numpy as np
import torch
import torch.nn as nn

from mhpy.utils.format import fcount
from mhpy.utils.format import fsize


@dataclass
class HardwarePermissions:
    # --- Device ---
    cuda: bool = True
    mps: bool = True

    # --- Dtype ---
    # BFloat16
    bf16: bool = True
    bf16_emulation: bool = False  # For CUDA devices
    bf16_cpu: bool = False  # Allows BF16 on CPU (requires AVX512_BF16/AMX)
    bf16_mps: bool = False  # Allows BF16 on MPS (M2+ chips)

    # Float16
    fp16_mps: bool = True  # Standard FP16 on MPS

    # --- Automatic Mixed Precision ---
    amp: bool = True

    # --- TF32 Specifics ---
    tf32: bool = False
    tf32_matmul: bool = True
    tf32_cudnn: bool = True


def get_device(allow_cuda: bool = True, allow_mps: bool = True) -> torch.device:
    """
    Get the optimal device to use for training with given constraints.

    Args:
        allow_cuda (bool): Whether to use CUDA if available.
        allow_mps (bool): Whether to use MPS if available.

    Returns:
        torch.device: The device to use.
    """
    if allow_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logger.warning("CUDA not available. Falling back to CPU.")
    if allow_mps:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            if not torch.backends.mps.is_built():
                logger.warning("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                logger.warning(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
    return torch.device("cpu")


def get_amp_dtype(
    device_type: str,
    allow_bf16: bool = True,
    allow_bf16_emulation: bool = False,
    allow_bf16_on_cpu: bool = False,
    allow_fp16_on_mps: bool = True,
    allow_bf16_on_mps: bool = False,
) -> torch.dtype:
    """
    Get the optimal dtype to use for AMP training with given constraints.

    Args:
        device_type (str): The device to use.
        allow_bf16 (bool): Whether to use bfloat16 if possible.
        allow_bf16_emulation (bool): Whether to allow bfloat16 emulation.
        allow_bf16_on_cpu (bool): Whether to allow bfloat16 on CPU. (if AVX-512/AMX supported)
        allow_fp16_on_mps (bool): Whether to allow float16 on MPS.
        allow_bf16_on_mps (bool): Whether to allow bfloat16 on MPS.

    Returns:
        torch.dtype: The dtype to use.
    """
    if device_type == "cuda":
        if allow_bf16:
            is_native_supported = torch.cuda.is_bf16_supported(including_emulation=False)
            if is_native_supported:
                return torch.bfloat16

            if allow_bf16_emulation:
                is_emulation_supported = torch.cuda.is_bf16_supported(including_emulation=True)
                if is_emulation_supported:
                    return torch.bfloat16
                else:
                    logger.warning("bfloat16 is not supported even with emulation. Falling back to float16.")
                    return torch.float16
            else:
                logger.warning("bfloat16 is not enabled. Falling back to float16.")
                return torch.float16
        else:
            return torch.float16

    if device_type == "mps":
        if allow_bf16_on_mps:
            return torch.bfloat16
        elif allow_fp16_on_mps:
            return torch.float16
        else:
            return torch.float32

    if device_type == "cpu":
        if allow_bf16_on_cpu:
            return torch.bfloat16
        else:
            return torch.float32

    return torch.float32


def use_amp(dtype: torch.dtype) -> bool:
    if dtype in [torch.float16, torch.bfloat16]:
        return True
    return False


def use_grad_scaler(dtype: torch.dtype) -> bool:
    if dtype == torch.float16:
        return True
    return False


def enable_tf32(device: torch.device, matmul: bool = True, cudnn: bool = True) -> None:
    if device.type != "cuda":
        logger.warning(f"TF32 is not supported on {device.type}.")
        return

    major, minor = torch.cuda.get_device_capability(device.index)
    if major < 8:
        logger.warning(f"TF32 is not supported on devices with CUDA capability < 8.0. Found {major}.{minor}. on {device}")
        return

    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def auto_environment(
    seed: int,
    perms: HardwarePermissions,
    deterministic: bool = False,
) -> tuple[torch.device, AbstractContextManager, bool]:
    """
    Setup optimal training environment to use with given hardware permissions (contraints).

    Args:
        seed (int): The seed value.
        perms (HardwarePermissions): The hardware permissions.
        deterministic (bool): If True, sets flags that ensure reproducibility
                              at the cost of performance (slower training).

    Returns:
        tuple: (device, context, enable_grad_scaler) tuple of device, context and whether to use GradScaler.
    """

    set_seed(seed, deterministic)
    device = get_device(perms.cuda, perms.mps)

    context = nullcontext()
    enable_grad_scaler = False

    if perms.amp:
        amp_dtype = get_amp_dtype(device.type, perms.bf16, perms.bf16_emulation, perms.bf16_cpu, perms.fp16_mps, perms.bf16_mps)
        if use_amp(amp_dtype):
            context = torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
            enable_grad_scaler = use_grad_scaler(amp_dtype)
        else:
            logger.warning(f"AMP can not be enabled for {amp_dtype} on {device.type}.")

    if perms.tf32:
        enable_tf32(device, perms.tf32_matmul, perms.tf32_cudnn)

    logger.info(f"Training on {device.type} with {amp_dtype if perms.amp else 'FP32'} (enable_grad_scaler: {enable_grad_scaler})")
    return device, context, enable_grad_scaler


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
