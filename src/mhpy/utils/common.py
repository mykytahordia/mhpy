import os
from pathlib import Path
import random
import sys

import hydra
from loguru import logger
import numpy as np
import torch


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


def configure_logger(save_logs: bool = True) -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        enqueue=True,
        diagnose=False,
    )

    if save_logs:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        log_file_path = Path(hydra_cfg.runtime.output_dir) / f"{hydra_cfg.job.name}.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
            enqueue=True,
            diagnose=True,
        )

        logger.info(f"Logging to {log_file_path}")
