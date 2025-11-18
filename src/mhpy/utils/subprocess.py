import subprocess
import sys

from loguru import logger


def run_cmd(command: str, error_msg: str) -> None:
    logger.info(f"Running: {command}")
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error - {error_msg}: {e.stderr}")
        sys.exit(1)
