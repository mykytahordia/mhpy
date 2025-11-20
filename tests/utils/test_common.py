import os
import random
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import torch

from mhpy.utils.common import configure_logger
from mhpy.utils.common import set_seed


class TestSetSeed:
    def test_set_seed_default(self):
        set_seed()

        py_random = random.random()
        np_random = np.random.rand()
        torch_random = torch.rand(1).item()

        set_seed()

        assert random.random() == py_random
        assert np.random.rand() == np_random
        assert torch.rand(1).item() == torch_random

    def test_set_seed_custom(self):
        custom_seed = 42
        set_seed(custom_seed)

        py_random = random.random()
        np_random = np.random.rand()
        torch_random = torch.rand(1).item()

        set_seed(custom_seed)

        assert random.random() == py_random
        assert np.random.rand() == np_random
        assert torch.rand(1).item() == torch_random

    def test_set_seed_different_seeds(self):
        set_seed(42)
        value1 = random.random()

        set_seed(123)
        value2 = random.random()

        assert value1 != value2

    def test_set_seed_environment_variable(self):
        seed = 999
        set_seed(seed)
        assert os.environ["PYTHONHASHSEED"] == str(seed)


class TestConfigureLogger:
    @patch("mhpy.utils.common.logger")
    def test_configure_logger_no_save(self, mock_logger):
        configure_logger(save_logs=False)

        mock_logger.remove.assert_called_once()

        assert mock_logger.add.call_count == 1

    @patch("mhpy.utils.common.logger")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_configure_logger_with_save(self, mock_hydra_get, mock_logger):
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = "/tmp/test_output"
        mock_hydra_cfg.job.name = "test_job"
        mock_hydra_get.return_value = mock_hydra_cfg

        configure_logger(save_logs=True)

        mock_logger.remove.assert_called_once()

        assert mock_logger.add.call_count == 2

        mock_logger.info.assert_called_once()

    @patch("mhpy.utils.common.logger")
    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_configure_logger_creates_directory(self, mock_hydra_get, mock_logger):
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.runtime.output_dir = "/tmp/test_nonexistent_dir"
        mock_hydra_cfg.job.name = "test_job"
        mock_hydra_get.return_value = mock_hydra_cfg

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            configure_logger(save_logs=True)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
