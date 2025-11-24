import time
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest

from mhpy.utils.tracking import GitStatusError
from mhpy.utils.tracking import Timer
from mhpy.utils.tracking import assert_clean_git
from mhpy.utils.tracking import capture_args
from mhpy.utils.tracking import config_from_run
from mhpy.utils.tracking import get_forked_run
from mhpy.utils.tracking import get_wandb_run
from mhpy.utils.tracking import track_performance


class TestTimer:
    def test_timer_basic(self):
        with Timer() as timer:
            time.sleep(0.01)

        assert hasattr(timer, "interval")
        assert timer.interval >= 0.01
        assert timer.interval < 0.1

    def test_timer_start_time(self):
        with Timer() as timer:
            assert hasattr(timer, "start")
            assert timer.start > 0

    def test_timer_end_time(self):
        with Timer() as timer:
            pass

        assert hasattr(timer, "end")
        assert timer.end > timer.start

    def test_timer_interval_calculation(self):
        with Timer() as timer:
            time.sleep(0.02)

        assert timer.interval == pytest.approx(timer.end - timer.start, rel=1e-6)

    def test_timer_multiple_uses(self):
        with Timer() as timer1:
            time.sleep(0.01)

        with Timer() as timer2:
            time.sleep(0.02)

        # Second timer should have longer interval
        assert timer2.interval > timer1.interval


class TestGetWandbRun:
    @patch("mhpy.utils.tracking.wandb.Api")
    def test_get_wandb_run(self, mock_api):
        run_path = "entity/project/run_id"
        mock_run = MagicMock()
        mock_api.return_value.run.return_value = mock_run

        result = get_wandb_run(run_path)

        assert result == mock_run
        mock_api.return_value.run.assert_called_once_with(run_path)

    @patch("mhpy.utils.tracking.wandb.Api")
    def test_get_wandb_run_different_paths(self, mock_api):
        run_path1 = "entity1/project1/run1"
        run_path2 = "entity2/project2/run2"

        get_wandb_run(run_path1)
        get_wandb_run(run_path2)

        assert mock_api.return_value.run.call_count == 2


class TestGetForkedRun:
    @patch("mhpy.utils.tracking.get_wandb_run")
    def test_get_forked_run(self, mock_get_run):
        mock_run = MagicMock()
        type(mock_run).path = PropertyMock(return_value="entity/project/original_run")

        mock_forked_run = MagicMock()
        mock_get_run.return_value = mock_forked_run

        result = get_forked_run(mock_run, 42)

        expected_path = "entity/project/42"
        mock_get_run.assert_called_once_with(expected_path)
        assert result == mock_forked_run

    @patch("mhpy.utils.tracking.get_wandb_run")
    def test_get_forked_run_different_ids(self, mock_get_run):
        mock_run = MagicMock()
        type(mock_run).path = PropertyMock(return_value="entity/project/run")

        get_forked_run(mock_run, 1)
        get_forked_run(mock_run, 2)

        assert mock_get_run.call_count == 2
        mock_get_run.assert_any_call("entity/project/1")
        mock_get_run.assert_any_call("entity/project/2")


class TestTrackPerformance:
    def test_track_performance_basic(self):
        mock_run = MagicMock()
        mock_queue = MagicMock()
        mock_queue.qsize.return_value = 5

        step_timer = MagicMock()
        step_timer.interval = 0.1

        cpu_timer = MagicMock()
        cpu_timer.interval = 0.05

        track_performance(mock_run, 100, 10.5, step_timer, cpu_timer, mock_queue)

        mock_run.log.assert_called_once()

        log_call = mock_run.log.call_args
        logged_data = log_call[0][0]

        assert logged_data["performance/step_runtime"] == 0.1
        assert logged_data["performance/step_runtime_cum"] == 10.5
        assert logged_data["performance/data_latency"] == 0.05
        assert logged_data["performance/data_queue_size"] == 5

        assert log_call[1]["step"] == 100


class TestCaptureArgs:
    @patch("mhpy.utils.tracking.sys.argv", ["script.py", "--arg1", "value1", "--arg2"])
    def test_capture_args_basic(self):
        cfg = {"param1": "value1", "param2": 42}

        result = capture_args(cfg)

        assert "command" in result
        assert result["command"] == "script.py --arg1 value1 --arg2"
        assert result["param1"] == "value1"
        assert result["param2"] == 42

    @patch("mhpy.utils.tracking.sys.argv", ["main.py"])
    def test_capture_args_no_arguments(self):
        cfg = {}

        result = capture_args(cfg)

        assert result["command"] == "main.py"

    @patch("mhpy.utils.tracking.sys.argv", ["script.py", "arg1", "arg2", "arg3"])
    def test_capture_args_multiple_arguments(self):
        cfg = {"existing": "value"}

        result = capture_args(cfg)

        assert result["command"] == "script.py arg1 arg2 arg3"
        assert result["existing"] == "value"

    @patch("mhpy.utils.tracking.sys.argv", ["test.py", "--config", "path/to/config.yaml"])
    def test_capture_args_preserves_existing_config(self):
        cfg = {"model": "resnet", "lr": 0.001}

        result = capture_args(cfg)

        assert result["model"] == "resnet"
        assert result["lr"] == 0.001
        assert "command" in result


class TestConfigFromRun:
    @patch("mhpy.utils.tracking.OmegaConf.create")
    def test_config_from_run(self, mock_omega_create):
        mock_run = MagicMock()
        mock_run.config = {"param1": "value1", "param2": 42}

        mock_dict_config = MagicMock()
        mock_omega_create.return_value = mock_dict_config

        result = config_from_run(mock_run)

        mock_omega_create.assert_called_once_with(dict(mock_run.config))
        assert result == mock_dict_config

    @patch("mhpy.utils.tracking.OmegaConf.create")
    def test_config_from_run_empty_config(self, mock_omega_create):
        mock_run = MagicMock()
        mock_run.config = {}

        config_from_run(mock_run)

        mock_omega_create.assert_called_once_with({})


class TestAssertCleanGit:
    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_clean_repo(self, mock_repo_class):
        mock_repo = MagicMock()
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        assert_clean_git("test_project", repo_path=".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_dirty_code(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "src/test_project/main.py"
        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_config_changes_allowed(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "src/test_project/config/experiment.yaml"
        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        assert_clean_git("test_project", repo_path=".", ignore_submodules=["config"])

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_untracked_files(self, mock_repo_class):
        mock_repo = MagicMock()
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = ["src/test_project/new_file.py"]
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_untracked_config_allowed(self, mock_repo_class):
        mock_repo = MagicMock()
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = ["src/test_project/config/new_config.yaml"]
        mock_repo_class.return_value = mock_repo

        assert_clean_git("test_project", repo_path=".", ignore_submodules=["config"])

    @patch("mhpy.utils.tracking.logger")
    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_invalid_repo(self, mock_repo_class, mock_logger):
        import git

        mock_repo_class.side_effect = git.InvalidGitRepositoryError("This directory is not a Git repository.")

        assert_clean_git("test_project", repo_path=".")

        mock_logger.error.assert_called_once_with("This directory is not a Git repository. Skipping check.")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_custom_project_name(self, mock_repo_class):
        mock_repo = MagicMock()
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        assert_clean_git("custom_project")

        mock_repo_class.assert_called_once_with(".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_multiple_dirty_files(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item1 = MagicMock()
        mock_diff_item1.a_path = "src/test_project/main.py"
        mock_diff_item2 = MagicMock()
        mock_diff_item2.a_path = "src/test_project/utils.py"
        mock_repo.index.diff.return_value = [mock_diff_item1, mock_diff_item2]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_mixed_dirty_and_config(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item1 = MagicMock()
        mock_diff_item1.a_path = "src/test_project/conf/exp.yaml"
        mock_diff_item2 = MagicMock()
        mock_diff_item2.a_path = "src/test_project/main.py"
        mock_repo.index.diff.return_value = [mock_diff_item1, mock_diff_item2]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_untracked_outside_config(self, mock_repo_class):
        mock_repo = MagicMock()
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = ["tests/test_new.py", "src/test_project/conf/new.yaml"]
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".")

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_custom_ignore_submodules(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "src/test_project/data/dataset.py"
        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        assert_clean_git("test_project", repo_path=".", ignore_submodules=["config", "data"])

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_custom_ignore_submodules_violation(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "src/test_project/models/model.py"
        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".", ignore_submodules=["config", "data"])

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_empty_ignore_submodules(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item = MagicMock()
        mock_diff_item.a_path = "src/test_project/config/exp.yaml"
        mock_repo.index.diff.return_value = [mock_diff_item]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        with pytest.raises(GitStatusError):
            assert_clean_git("test_project", repo_path=".", ignore_submodules=[])

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_multiple_ignore_submodules(self, mock_repo_class):
        mock_repo = MagicMock()

        mock_diff_item1 = MagicMock()
        mock_diff_item1.a_path = "src/test_project/config/exp.yaml"
        mock_diff_item2 = MagicMock()
        mock_diff_item2.a_path = "src/test_project/data/loader.py"
        mock_diff_item3 = MagicMock()
        mock_diff_item3.a_path = "src/test_project/utils/helper.py"
        mock_repo.index.diff.return_value = [mock_diff_item1, mock_diff_item2, mock_diff_item3]
        mock_repo.untracked_files = []
        mock_repo_class.return_value = mock_repo

        assert_clean_git("test_project", repo_path=".", ignore_submodules=["config", "data", "utils"])

    @patch("mhpy.utils.tracking.git.Repo")
    def test_assert_clean_git_untracked_with_custom_ignore(self, mock_repo_class):
        mock_repo = MagicMock()
        mock_repo.index.diff.return_value = []
        mock_repo.untracked_files = ["src/test_project/scripts/new_script.py"]
        mock_repo_class.return_value = mock_repo

        assert_clean_git("test_project", repo_path=".", ignore_submodules=["config", "scripts"])
