from pathlib import Path
import sys

import hydra
from loguru import logger


def configure_logger(debug=False, save_logs: bool = True) -> None:
    level = "DEBUG" if debug else "INFO"
    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
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
            level=level,
            rotation="100 MB",
            enqueue=True,
            diagnose=True,
        )

        logger.info(f"Logging to {log_file_path}")


def launch_debugger(ui: bool = True) -> None:
    try:
        if ui:
            import pudb as debugger

            logger.info("Using PuDB (Visual Debugger)")
        else:
            raise ImportError
    except ImportError:
        import pdb as debugger

        logger.info("Using PDB (Standard Debugger) - PuDB not installed")
    debugger.post_mortem()
