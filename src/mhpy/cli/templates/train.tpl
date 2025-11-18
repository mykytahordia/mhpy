import hydra
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

# Import and configure logger
# Note: Relative import works because this script is run as a module
# from ..logger import configure_logger
# configure_logger()


@hydra.main(version_base=None, config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training script.
    """
    logger.info("🚀 Starting training...")
    logger.debug(f"Full config: \n{OmegaConf.to_yaml(cfg)}")

    # Your training logic starts here
    # Example:
    # model_cfg = cfg.model
    # data_cfg = cfg.data
    # logger.info(f"Using model: {model_cfg.name}")

    logger.success("✅ Training complete.")


if __name__ == "__main__":
    train()
