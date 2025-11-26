import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytest

from mhpy.cli.commands.initialize import init


@pytest.fixture
def integration_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["git", "config", "--global", "user.email", "test@example.com"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "--global", "user.name", "Test User"],
            check=False,
            capture_output=True,
        )
        yield Path(tmpdir)


@pytest.fixture
def minimal_config():
    cfg_dict = {
        "package_name": "test_ml_project",
        "python_version": "3.12",
        "run": {
            "code_privacy_confirmation": False,
            "remote_url_prompt": False,
        },
        "python_submodules": ["data", "models"],
        "uv": {
            "timeout": 100000,
            "packages": ["pytest"],
        },
        "mhpy_url": "",
        "data_states": ["raw", "processed"],
        "hydra": {
            "submodule": "config",
            "configs": ["model", "train"],
        },
        "other_dirs": ["notebooks"],
        "debug": False,
    }
    return OmegaConf.create(cfg_dict)


def run_init_in_dir(temp_dir: Path, config: DictConfig, user_input: str = ""):
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)

        original_run = subprocess.run

        def mock_subprocess_run(cmd, **kwargs):
            if isinstance(cmd, list) and len(cmd) >= 2:
                if cmd[0] == "uv" and cmd[1] in ["add", "pip"]:
                    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return original_run(cmd, **kwargs)

        with patch("mhpy.cli.commands.initialize.input", return_value=user_input):
            with patch("subprocess.run", side_effect=mock_subprocess_run):
                init(config)
    finally:
        os.chdir(original_cwd)


@pytest.mark.integration
class TestInitIntegration:
    def test_init_creates_complete_project_structure(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        package_dir = integration_temp_dir / "src" / minimal_config.package_name
        assert package_dir.exists()
        assert package_dir.is_dir()

        assert (package_dir / "data").exists()
        assert (package_dir / "data" / "__init__.py").exists()
        assert (package_dir / "models").exists()
        assert (package_dir / "models" / "__init__.py").exists()

        data_dir = integration_temp_dir / "data"
        assert data_dir.exists()
        assert (data_dir / "raw").exists()
        assert (data_dir / "processed").exists()

        assert (integration_temp_dir / "notebooks").exists()

        assert (integration_temp_dir / ".gitignore").exists()
        assert (integration_temp_dir / "Makefile").exists()
        assert (integration_temp_dir / ".pre-commit-config.yaml").exists()

        config_dir = package_dir / "config"
        assert config_dir.exists()
        assert (config_dir / "config.yaml").exists()
        assert (config_dir / "model" / "default.yaml").exists()
        assert (config_dir / "train" / "default.yaml").exists()

        tests_dir = integration_temp_dir / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "__init__.py").exists()

    def test_init_creates_valid_git_repository(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        assert (integration_temp_dir / ".git").exists()

        result = subprocess.run(
            ["git", "status"],
            cwd=integration_temp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=integration_temp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Initial commit" in result.stdout or "Initial project setup" in result.stdout

    def test_init_creates_valid_python_files(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        package_dir = integration_temp_dir / "src" / minimal_config.package_name
        init_file = package_dir / "__init__.py"
        assert init_file.exists()

        with open(init_file) as f:
            code = f.read()
            compile(code, str(init_file), "exec")

        train_file = package_dir / "train.py"
        assert train_file.exists()
        with open(train_file) as f:
            code = f.read()
            compile(code, str(train_file), "exec")

    def test_init_creates_valid_makefile(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        makefile = integration_temp_dir / "Makefile"
        assert makefile.exists()

        content = makefile.read_text()
        assert "install:" in content or ".PHONY" in content
        assert len(content) > 0

    def test_init_creates_valid_gitignore(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        gitignore = integration_temp_dir / ".gitignore"
        assert gitignore.exists()

        content = gitignore.read_text()
        assert "__pycache__" in content or "*.pyc" in content
        assert len(content) > 0

    def test_init_creates_dvc_structure(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        dvc_dir = integration_temp_dir / ".dvc"
        assert dvc_dir.exists()
        assert (dvc_dir / "config").exists()

    def test_init_with_remote_url(self, integration_temp_dir, minimal_config):
        remote_url = "https://github.com/test/repo.git"

        minimal_config.run.remote_url_prompt = True

        original_cwd = os.getcwd()
        try:
            os.chdir(integration_temp_dir)

            original_run = subprocess.run

            def mock_subprocess_run(cmd, **kwargs):
                if isinstance(cmd, list) and len(cmd) >= 2:
                    if cmd[0] == "uv" and cmd[1] in ["add", "pip"]:
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                    if cmd[0] == "git" and cmd[1] == "push":
                        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
                return original_run(cmd, **kwargs)

            with patch("mhpy.cli.commands.initialize.input", return_value=remote_url):
                with patch("subprocess.run", side_effect=mock_subprocess_run):
                    init(minimal_config)
        finally:
            os.chdir(original_cwd)

        result = subprocess.run(
            ["git", "remote", "-v"],
            cwd=integration_temp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert remote_url in result.stdout

    def test_init_rejects_non_empty_directory(self, integration_temp_dir, minimal_config):
        (integration_temp_dir / "existing_file.txt").write_text("content")

        run_init_in_dir(integration_temp_dir, minimal_config)

        package_dir = integration_temp_dir / "src" / minimal_config.package_name
        assert not package_dir.exists()

    def test_init_creates_wandb_config(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        models_dir = integration_temp_dir / "models"
        reports_dir = integration_temp_dir / "reports"
        assert models_dir.exists()
        assert reports_dir.exists()

    def test_init_creates_hydra_configs_with_correct_structure(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        config_dir = integration_temp_dir / "src" / minimal_config.package_name / "config"

        main_config = config_dir / "config.yaml"
        assert main_config.exists()
        content = main_config.read_text()
        assert len(content) > 0
        OmegaConf.create(content)

        for config_name in ["model", "train"]:
            config_subdir = config_dir / config_name
            assert config_subdir.exists()
            assert config_subdir.is_dir()
            default_file = config_subdir / "default.yaml"
            assert default_file.exists()

    def test_init_creates_notebook_template(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        notebooks_dir = integration_temp_dir / "notebooks"
        assert notebooks_dir.exists()

        eda_notebook = notebooks_dir / "EDA.ipynb"
        assert eda_notebook.exists()

        import json

        with open(eda_notebook) as f:
            notebook_data = json.load(f)
            assert "cells" in notebook_data
            assert "metadata" in notebook_data

    def test_init_project_passes_ruff_checks(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        ruff_result = subprocess.run(
            ["uv", "run", "ruff", "format", "--check", "src/"],
            cwd=integration_temp_dir,
            capture_output=True,
            text=True,
        )
        assert ruff_result.returncode == 0, f"Ruff format check failed:\n{ruff_result.stdout}\n{ruff_result.stderr}"

        ruff_lint_result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/"],
            cwd=integration_temp_dir,
            capture_output=True,
            text=True,
        )
        assert ruff_lint_result.returncode == 0, f"Ruff lint check failed:\n{ruff_lint_result.stdout}\n{ruff_lint_result.stderr}"

    def test_init_project_passes_precommit_checks(self, integration_temp_dir, minimal_config):
        run_init_in_dir(integration_temp_dir, minimal_config)

        result = subprocess.run(
            ["uv", "run", "pre-commit", "run", "--all-files"],
            cwd=integration_temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Pre-commit checks failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
