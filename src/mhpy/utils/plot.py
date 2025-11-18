from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def plot_lr_finder(lrs: list[float], losses: list[float], artifacts_dir: str, warmup: int) -> Path:
    fig_path = Path(artifacts_dir) / "lr_finder_plot.png"

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Smoothed Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True, which="both", ls="--")

    if len(losses) > 1:
        min_grad_idx = np.gradient(np.array(losses[warmup:])).argmin() + warmup
        suggested_lr = lrs[min_grad_idx]
        logger.info(f"Steepest gradient at LR = {suggested_lr:.2e}. A good LR is often 10x smaller.")
        plt.axvline(x=suggested_lr, color="r", linestyle="--", label=f"Steepest point: {suggested_lr:.2e}")
        plt.legend()

    plt.savefig(fig_path)
    return fig_path
