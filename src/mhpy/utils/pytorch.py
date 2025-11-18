from loguru import logger
import torch.nn as nn


def get_model_size(model: nn.Module) -> int:
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    logger.info(f"Total Parameters: {param_count:,}")
    logger.info(f"Model Size (RAM): {size_all_mb:.2f} MB")
    return param_count
