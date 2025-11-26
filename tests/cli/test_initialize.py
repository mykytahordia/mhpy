from pathlib import Path
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

from mhpy.cli.commands.initialize import _cleanup
from mhpy.cli.commands.initialize import _dvc
from mhpy.cli.commands.initialize import _git
from mhpy.cli.commands.initialize import _hydra_configs
from mhpy.cli.commands.initialize import _makefile
from mhpy.cli.commands.initialize import _other_dirs
from mhpy.cli.commands.initialize import _pre_commit
from mhpy.cli.commands.initialize import _prompting
from mhpy.cli.commands.initialize import _py_templates
from mhpy.cli.commands.initialize import _tests
from mhpy.cli.commands.initialize import _uv
from mhpy.cli.commands.initialize import _wandb
from mhpy.cli.commands.initialize import append_template_to_file
from mhpy.cli.commands.initialize import create_file_from_template
from mhpy.cli.commands.initialize import init


@pytest.fixture
def temp_project_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_cfg():
    """Create a mock configuration for testing."""
    cfg_dict = {
        "package_name": "test_project",
        "python_version": "3.12",
        "run": {"code_privacy_confirmation": False, "remote_url_prompt": False},
        "python_submodules": ["data", "models", "utils"],
        "uv": {"timeout": 100000, "packages": ["pytest", "numpy"]},
        "mhpy_url": "mhpy[ml] @ git+https://github.com/mykytahordia/mhpy.git",
        "data_states": ["raw", "interim", "processed"],
        "hydra": {"submodule": "config", "configs": ["model", "train", "data"]},
        "other_dirs": ["notebooks", "scripts"],
        "debug": False,
    }
    return OmegaConf.create(cfg_dict)


class TestCreateFileFromTemplate:
    @patch("mhpy.cli.commands.initialize.jinja_env")
    @patch("mhpy.cli.commands.initialize.logger")
    def test_create_file_from_template_basic(self, mock_logger, mock_jinja_env, temp_project_dir):
        mock_template = MagicMock()
        mock_template.render.return_value = "test content"
        mock_jinja_env.get_template.return_value = mock_template

        filepath = temp_project_dir / "test.txt"
        create_file_from_template(filepath, "test_template.jinja", {"key": "value"})

        assert filepath.exists()
        assert filepath.read_text() == "test content"
        mock_jinja_env.get_template.assert_called_once_with("test_template.jinja")
        mock_template.render.assert_called_once_with(key="value")

    @patch("mhpy.cli.commands.initialize.jinja_env")
    def test_create_file_from_template_creates_parent_dirs(self, mock_jinja_env, temp_project_dir):
        mock_template = MagicMock()
        mock_template.render.return_value = "nested content"
        mock_jinja_env.get_template.return_value = mock_template

        filepath = temp_project_dir / "nested" / "dir" / "file.txt"
        create_file_from_template(filepath, "template.jinja")

        assert filepath.exists()
        assert filepath.parent.exists()


class TestAppendTemplateToFile:
    @patch("mhpy.cli.commands.initialize.jinja_env")
    def test_append_template_to_file(self, mock_jinja_env, temp_project_dir):
        mock_template = MagicMock()
        mock_template.render.return_value = "appended content"
        mock_jinja_env.get_template.return_value = mock_template

        filepath = temp_project_dir / "test.txt"
        filepath.write_text("initial content\n")

        append_template_to_file(filepath, "template.jinja", {"key": "value"})

        content = filepath.read_text()
        assert "initial content" in content
        assert "appended content" in content


class TestPrompting:
    @patch("mhpy.cli.commands.initialize.input")
    def test_prompting_no_confirmation_no_remote(self, mock_input):
        cfg = OmegaConf.create({"run": {"code_privacy_confirmation": False, "remote_url_prompt": False}})

        result = _prompting("test_pkg", cfg)

        assert result == {"remote_url": None}
        mock_input.assert_not_called()

    @patch("mhpy.cli.commands.initialize.input")
    def test_prompting_with_remote_url(self, mock_input):
        mock_input.return_value = "https://github.com/user/repo.git"
        cfg = OmegaConf.create({"run": {"code_privacy_confirmation": False, "remote_url_prompt": True}})

        result = _prompting("test_pkg", cfg)

        assert result["remote_url"] == "https://github.com/user/repo.git"
        mock_input.assert_called_once()

    @patch("mhpy.cli.commands.initialize.input")
    def test_prompting_skip_remote_url(self, mock_input):
        mock_input.return_value = ""
        cfg = OmegaConf.create({"run": {"code_privacy_confirmation": False, "remote_url_prompt": True}})

        result = _prompting("test_pkg", cfg)

        assert result["remote_url"] == ""

    @patch("mhpy.cli.commands.initialize._assert_no_code_leakage")
    @patch("mhpy.cli.commands.initialize.input")
    def test_prompting_with_code_privacy_confirmation(self, mock_input, mock_assert_no_code_leakage):
        mock_input.return_value = ""
        cfg = OmegaConf.create({"run": {"code_privacy_confirmation": True, "remote_url_prompt": False}})

        result = _prompting("test_pkg", cfg)

        mock_assert_no_code_leakage.assert_called_once_with("test_pkg")
        assert result["remote_url"] is None


class TestGit:
    @patch("mhpy.cli.commands.initialize.run_cmd")
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_git_init_new_repo(self, mock_create_file, mock_run_cmd, temp_project_dir):
        _git(temp_project_dir, None)

        mock_run_cmd.assert_any_call(["git", "init"], "Failed to initialize git")
        mock_run_cmd.assert_any_call(["git", "branch", "-M", "main"], "Failed to create main branch")
        mock_create_file.assert_called_once()

    @patch("mhpy.cli.commands.initialize.run_cmd")
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    @patch("mhpy.cli.commands.initialize.logger")
    def test_git_existing_repo(self, mock_logger, mock_create_file, mock_run_cmd, temp_project_dir):
        (temp_project_dir / ".git").mkdir()

        _git(temp_project_dir, None)

        mock_logger.info.assert_any_call("Git repository already exists.")

    @patch("mhpy.cli.commands.initialize.run_cmd")
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_git_with_remote_url(self, mock_create_file, mock_run_cmd, temp_project_dir):
        remote_url = "https://github.com/user/repo.git"

        _git(temp_project_dir, remote_url)

        mock_run_cmd.assert_any_call(["git", "remote", "add", "origin", remote_url], "Failed to add remote origin")
        mock_run_cmd.assert_any_call(["git", "push", "-u", "origin", "main"], f"Failed to push to origin/main {remote_url}")


class TestUv:
    @patch("mhpy.cli.commands.initialize.run_cmd")
    @patch("mhpy.cli.commands.initialize.append_template_to_file")
    @patch("mhpy.cli.commands.initialize.logger")
    def test_uv_setup(self, mock_logger, mock_append_template, mock_run_cmd, temp_project_dir, mock_cfg):
        package_root = temp_project_dir / "src" / "test_project"

        _uv(temp_project_dir, package_root, "test_project", mock_cfg)

        assert (package_root / "__init__.py").exists()
        assert (package_root / "data" / "__init__.py").exists()
        assert (package_root / "models" / "__init__.py").exists()
        assert (package_root / "utils" / "__init__.py").exists()

        mock_run_cmd.assert_any_call(["uv", "init", "--bare", f"--python={mock_cfg.python_version}"], "Failed to initialize uv")
        mock_append_template.assert_called_once()

    @patch("mhpy.cli.commands.initialize.run_cmd")
    @patch("mhpy.cli.commands.initialize.append_template_to_file")
    def test_uv_installs_packages(self, mock_append_template, mock_run_cmd, temp_project_dir, mock_cfg):
        package_root = temp_project_dir / "src" / "test_project"

        _uv(temp_project_dir, package_root, "test_project", mock_cfg)

        mock_run_cmd.assert_any_call(
            ["uv", "add", "pytest", "numpy"],
            "Failed to install Python packages",
            env={"UV_HTTP_TIMEOUT": str(mock_cfg.uv.timeout)},
        )


class TestDvc:
    @patch("mhpy.cli.commands.initialize.run_cmd")
    def test_dvc_initialization(self, mock_run_cmd, temp_project_dir, mock_cfg):
        _dvc(temp_project_dir, mock_cfg)

        assert (temp_project_dir / "data" / "raw").exists()
        assert (temp_project_dir / "data" / "interim").exists()
        assert (temp_project_dir / "data" / "processed").exists()
        assert (temp_project_dir / ".local_dvc_storage").exists()

        mock_run_cmd.assert_any_call(["uv", "run", "dvc", "init"], "Failed to initialize DVC")
        mock_run_cmd.assert_any_call(["uv", "run", "dvc", "config", "core.autostage", "true"], "Failed to set DVC autostage")


class TestWandb:
    @patch("mhpy.cli.commands.initialize.logger")
    def test_wandb_setup(self, mock_logger, temp_project_dir):
        _wandb(temp_project_dir)

        assert (temp_project_dir / "models").exists()
        assert (temp_project_dir / "reports").exists()


class TestPreCommit:
    @patch("mhpy.cli.commands.initialize.run_cmd")
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_pre_commit_setup(self, mock_create_file, mock_run_cmd, temp_project_dir):
        _pre_commit(temp_project_dir)

        mock_create_file.assert_called_once()
        mock_run_cmd.assert_called_once_with(["uv", "run", "pre-commit", "install"], "Failed to install pre-commit hooks")


class TestMakefile:
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_makefile_creation(self, mock_create_file, temp_project_dir):
        _makefile(temp_project_dir, "test_project", "config")

        mock_create_file.assert_called_once_with(
            temp_project_dir / "Makefile",
            "Makefile.jinja",
            {"PACKAGE_NAME": "test_project", "HYDRA_SUBMODULE": "config"},
        )


class TestHydraConfigs:
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_hydra_configs_creation(self, mock_create_file, temp_project_dir, mock_cfg):
        package_root = temp_project_dir / "src" / "test_project"

        _hydra_configs(package_root, mock_cfg)

        hydra_dir = package_root / "config"
        assert hydra_dir.exists()
        assert (hydra_dir / "__init__.py").exists()
        assert (hydra_dir / "model").exists()
        assert (hydra_dir / "train").exists()
        assert (hydra_dir / "data").exists()

        mock_create_file.assert_called_once()


class TestTests:
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_tests_directory_creation(self, mock_create_file, temp_project_dir):
        _tests(temp_project_dir)

        assert (temp_project_dir / "tests").exists()
        assert (temp_project_dir / "tests" / "__init__.py").exists()
        mock_create_file.assert_called_once()


class TestOtherDirs:
    def test_other_dirs_creation(self, temp_project_dir, mock_cfg):
        _other_dirs(temp_project_dir, mock_cfg)

        assert (temp_project_dir / "notebooks").exists()
        assert (temp_project_dir / "scripts").exists()


class TestPyTemplates:
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_py_templates_creation(self, mock_create_file, temp_project_dir):
        package_root = temp_project_dir / "src" / "test_project"

        _py_templates(package_root, "test_project")

        mock_create_file.assert_called_once_with(package_root / "train.py", "train.py.jinja", {"PACKAGE_NAME": "test_project"})


class TestCleanup:
    def test_cleanup_removes_all_files(self, temp_project_dir):
        (temp_project_dir / "file1.txt").write_text("content")
        (temp_project_dir / "file2.txt").write_text("content")
        (temp_project_dir / "dir1").mkdir()
        (temp_project_dir / "dir1" / "nested.txt").write_text("content")

        _cleanup(temp_project_dir)

        assert len(list(temp_project_dir.iterdir())) == 0


class TestInit:
    @patch("mhpy.cli.commands.initialize.logger")
    def test_init_non_empty_directory(self, mock_logger, temp_project_dir, mock_cfg):
        (temp_project_dir / "existing_file.txt").write_text("content")

        with patch("mhpy.cli.commands.initialize.Path.cwd", return_value=temp_project_dir):
            init(mock_cfg)

        mock_logger.error.assert_called_once()
        assert "not empty" in mock_logger.error.call_args[0][0]

    @patch("mhpy.cli.commands.initialize._print_summary")
    @patch("mhpy.cli.commands.initialize._final_commit")
    @patch("mhpy.cli.commands.initialize._ipynb_templates")
    @patch("mhpy.cli.commands.initialize._py_templates")
    @patch("mhpy.cli.commands.initialize._other_dirs")
    @patch("mhpy.cli.commands.initialize._tests")
    @patch("mhpy.cli.commands.initialize._hydra_configs")
    @patch("mhpy.cli.commands.initialize._makefile")
    @patch("mhpy.cli.commands.initialize._pre_commit")
    @patch("mhpy.cli.commands.initialize._wandb")
    @patch("mhpy.cli.commands.initialize._dvc")
    @patch("mhpy.cli.commands.initialize._uv")
    @patch("mhpy.cli.commands.initialize._git")
    @patch("mhpy.cli.commands.initialize._prompting")
    def test_init_success(
        self,
        mock_prompting,
        mock_git,
        mock_uv,
        mock_dvc,
        mock_wandb,
        mock_pre_commit,
        mock_makefile,
        mock_hydra_configs,
        mock_tests,
        mock_other_dirs,
        mock_py_templates,
        mock_ipynb_templates,
        mock_final_commit,
        mock_print_summary,
        temp_project_dir,
        mock_cfg,
    ):
        mock_prompting.return_value = {"remote_url": None}

        with patch("mhpy.cli.commands.initialize.Path.cwd", return_value=temp_project_dir):
            init(mock_cfg)

        mock_prompting.assert_called_once()
        mock_git.assert_called_once()
        mock_uv.assert_called_once()
        mock_dvc.assert_called_once()
        mock_wandb.assert_called_once()
        mock_pre_commit.assert_called_once()
        mock_makefile.assert_called_once()
        mock_hydra_configs.assert_called_once()
        mock_tests.assert_called_once()
        mock_other_dirs.assert_called_once()
        mock_py_templates.assert_called_once()
        mock_ipynb_templates.assert_called_once()
        mock_final_commit.assert_called_once()
        mock_print_summary.assert_called_once()

    @patch("mhpy.cli.commands.initialize._cleanup")
    @patch("mhpy.cli.commands.initialize._prompting")
    @patch("mhpy.cli.commands.initialize._git")
    @patch("mhpy.cli.commands.initialize.logger")
    def test_init_error_with_cleanup(self, mock_logger, mock_git, mock_prompting, mock_cleanup, temp_project_dir, mock_cfg):
        mock_prompting.return_value = {"remote_url": None}
        mock_git.side_effect = Exception("Git initialization failed")

        with patch("mhpy.cli.commands.initialize.Path.cwd", return_value=temp_project_dir):
            init(mock_cfg)

        mock_logger.error.assert_called()
        mock_cleanup.assert_called_once_with(temp_project_dir)

    @patch("mhpy.cli.commands.initialize._cleanup")
    @patch("mhpy.cli.commands.initialize._prompting")
    @patch("mhpy.cli.commands.initialize._git")
    @patch("mhpy.cli.commands.initialize.logger")
    def test_init_error_debug_mode_no_cleanup(self, mock_logger, mock_git, mock_prompting, mock_cleanup, temp_project_dir):
        cfg_dict = {
            "package_name": "test_project",
            "python_version": "3.12",
            "run": {"code_privacy_confirmation": False, "remote_url_prompt": False},
            "python_submodules": ["data"],
            "uv": {"timeout": 100000, "packages": []},
            "mhpy_url": "mhpy",
            "data_states": ["raw"],
            "hydra": {"submodule": "config", "configs": []},
            "other_dirs": [],
            "debug": True,
        }
        debug_cfg = OmegaConf.create(cfg_dict)

        mock_prompting.return_value = {"remote_url": None}
        mock_git.side_effect = Exception("Git initialization failed")

        with patch("mhpy.cli.commands.initialize.Path.cwd", return_value=temp_project_dir):
            init(debug_cfg)

        mock_logger.error.assert_called()
        mock_cleanup.assert_not_called()


class TestAssertNoCodeLeakage:
    @patch("mhpy.cli.commands.initialize.uuid")
    @patch("mhpy.cli.commands.initialize.input")
    def test_assert_no_code_leakage_correct_code(self, mock_input, mock_uuid):
        from mhpy.cli.commands.initialize import _assert_no_code_leakage

        mock_uuid_obj = MagicMock()
        mock_uuid_obj.__str__ = MagicMock(return_value="abcd-1234-5678-90ab")
        mock_uuid.uuid4.return_value = mock_uuid_obj
        mock_input.return_value = "test_pkg_abcd"

        _assert_no_code_leakage("test_pkg")

        mock_input.assert_called_once()

    @patch("mhpy.cli.commands.initialize.uuid")
    @patch("mhpy.cli.commands.initialize.logger")
    @patch("mhpy.cli.commands.initialize.input")
    def test_assert_no_code_leakage_retry(self, mock_input, mock_logger, mock_uuid):
        from mhpy.cli.commands.initialize import _assert_no_code_leakage

        mock_uuid_obj = MagicMock()
        mock_uuid_obj.__str__ = MagicMock(return_value="abcd-1234-5678-90ab")
        mock_uuid.uuid4.return_value = mock_uuid_obj
        mock_input.side_effect = ["wrong_code", "test_pkg_abcd"]

        _assert_no_code_leakage("test_pkg")

        assert mock_input.call_count == 2
        mock_logger.info.assert_called_with("Try again...")


class TestIpynbTemplates:
    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_ipynb_templates_with_notebooks_dir(self, mock_create_file, temp_project_dir):
        from mhpy.cli.commands.initialize import _ipynb_templates

        cfg = OmegaConf.create({"other_dirs": ["notebooks", "scripts"]})

        _ipynb_templates(temp_project_dir, cfg)

        mock_create_file.assert_called_once_with(temp_project_dir / "notebooks" / "EDA.ipynb", "EDA.ipynb.jinja")

    @patch("mhpy.cli.commands.initialize.create_file_from_template")
    def test_ipynb_templates_without_notebooks_dir(self, mock_create_file, temp_project_dir):
        from mhpy.cli.commands.initialize import _ipynb_templates

        cfg = OmegaConf.create({"other_dirs": ["scripts"]})

        _ipynb_templates(temp_project_dir, cfg)

        mock_create_file.assert_not_called()


class TestFinalCommit:
    @patch("mhpy.cli.commands.initialize.run_cmd")
    def test_final_commit(self, mock_run_cmd):
        from mhpy.cli.commands.initialize import _final_commit

        _final_commit()

        mock_run_cmd.assert_any_call(["git", "add", "."], "Failed to add all new files to git")
        mock_run_cmd.assert_any_call(
            ["git", "commit", "-m", "feat: Initial project setup from mhpy template"],
            "Failed to create final commit",
        )
