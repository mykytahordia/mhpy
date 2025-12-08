from unittest.mock import MagicMock
from unittest.mock import patch

from mhpy.utils.common import configure_logger


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
