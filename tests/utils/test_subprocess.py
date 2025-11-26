import subprocess
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from mhpy.utils.subprocess import run_cmd


class TestRunCmd:
    @patch("mhpy.utils.subprocess.logger")
    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_failure(self, mock_run, mock_logger):
        command = ["false"]
        error_msg = "Command failed"

        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=command, stderr="error output")

        with pytest.raises(subprocess.CalledProcessError):
            run_cmd(command, error_msg)

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_with_complex_command(self, mock_run):
        command = ["sh", "-c", "ls -la | grep test | wc -l"]
        error_msg = "Pipeline failed"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == command

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_shell_false(self, mock_run):
        command = ["echo", "test"]
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["shell"] is False

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_check_true(self, mock_run):
        command = ["echo", "test"]
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["check"] is True

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_captures_output(self, mock_run):
        command = ["echo", "test"]
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["stdout"] == subprocess.PIPE
        assert mock_run.call_args[1]["stderr"] == subprocess.PIPE

    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_text_mode(self, mock_run):
        command = ["echo", "test"]
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)

        assert mock_run.call_args[1]["text"] is True

    @patch("mhpy.utils.subprocess.logger")
    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_empty_stderr(self, mock_run, mock_logger):
        command = ["false"]
        error_msg = "Command failed"

        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=command, stderr="")

        with pytest.raises(subprocess.CalledProcessError):
            run_cmd(command, error_msg)

    @patch("mhpy.utils.subprocess.logger")
    @patch("mhpy.utils.subprocess.subprocess.run")
    def test_run_cmd_logs_command(self, mock_run, mock_logger):
        command = ["custom_command", "--flag", "value"]
        error_msg = "Error"

        mock_run.return_value = MagicMock(returncode=0)

        run_cmd(command, error_msg)
