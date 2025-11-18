from pathlib import Path
import sys
import time

from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.multiprocessing import Queue
import wandb


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def get_wandb_run(run_path: str) -> wandb.Run:
    return wandb.Api().run(run_path)


def get_forked_run(run: wandb.Run, id: int) -> wandb.Run:
    path = Path(run.path).parent / str(id)
    return get_wandb_run(str(path))


def track_permormance(run: wandb.Run, it: int, total_runtime: float, step_timer: Timer, cpu_timer: Timer, queue: Queue):
    run.log(
        {
            "performance/step_runtime": step_timer.interval,
            "performance/step_runtime_cum": total_runtime,
            "performance/data_latency": cpu_timer.interval,
            "performance/data_queue_size": queue.qsize(),
        },
        step=it,
    )


def capture_args(cfg: dict) -> dict:
    cfg["command"] = " ".join(sys.argv)
    return cfg


def config_from_run(run: wandb.Run) -> DictConfig:
    return OmegaConf.create(dict(run.config))
