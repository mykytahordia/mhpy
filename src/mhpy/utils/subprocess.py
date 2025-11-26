import os
import subprocess

from loguru import logger


def run_cmd(command: list[str], error_msg: str, env: dict = {}) -> None:
    logger.info(f"Running: {' '.join(command)}" + (f" with env: {env}" if env else ""))

    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        subprocess.run(
            command,
            shell=False,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=run_env,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error - {error_msg}, {e.stderr}")
        raise e
