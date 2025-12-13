import os
from pathlib import Path
import shutil
import uuid

from jinja2 import Environment
from jinja2 import PackageLoader
from jinja2 import StrictUndefined
from loguru import logger
from omegaconf import DictConfig

from mhpy.utils.common import launch_debugger
from mhpy.utils.subprocess import run_cmd

jinja_env = Environment(loader=PackageLoader("mhpy.cli", "templates"), undefined=StrictUndefined, keep_trailing_newline=True)


def create_file_from_template(filepath: Path, template_name: str, replacements: dict = {}) -> None:
    content = jinja_env.get_template(template_name).render(**replacements)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    logger.info(f"Created: {filepath}")


def append_template_to_file(filepath: Path, template_name: str, replacements: dict = {}) -> None:
    content = jinja_env.get_template(template_name).render(**replacements)
    with filepath.open("a") as f:
        f.write(content)


def prepend_template_to_file(filename: Path, template_name: str, replacements: dict = {}):
    content = jinja_env.get_template(template_name).render(**replacements)
    file_path = Path(filename)
    original_content = file_path.read_text()

    new_content = content + original_content
    file_path.write_text(new_content)


def _assert_no_code_leakage(package_name: str) -> None:
    while True:
        confirmation_code = f"{package_name}_{str(uuid.uuid4())[:4]}"
        response_code = input(
            f"‼️ Make sure your AI tool settings are configured so that no code is used for training by third parties. Type '{confirmation_code}' to confim: "
        )
        if response_code == confirmation_code:
            return
        else:
            logger.info("Try again...")


def _prompting(package_game: str, cfg: DictConfig) -> dict:
    if cfg.run.code_privacy_confirmation:
        _assert_no_code_leakage(package_game)

    remote_url = input("🔗 Enter remote repository URL (Enter to skip): ").strip() if cfg.run.remote_url_prompt else None

    return {"remote_url": remote_url}


def _git(project_root: Path, remote_url: str) -> None:
    logger.info("Initializing Git...")
    if not (project_root / ".git").exists():
        run_cmd(["git", "init"], "Failed to initialize git")
        run_cmd(["git", "branch", "-M", "main"], "Failed to create main branch")
    else:
        logger.info("Git repository already exists.")

    create_file_from_template(project_root / ".gitignore", "gitignore.jinja")
    run_cmd(["git", "add", ".gitignore"], "Failed to add .gitignore")
    run_cmd(["git", "commit", "-m", "Initial commit: Add .gitignore"], "Failed to commit .gitignore")

    if remote_url:
        run_cmd(["git", "remote", "add", "origin", remote_url], "Failed to add remote origin")
        logger.info(f"✅ Remote 'origin' set to {remote_url}")
        run_cmd(["git", "push", "-u", "origin", "main"], f"Failed to push to origin/main {remote_url}")
    else:
        logger.info("Skipping remote configuration.")


def _uv(project_root: Path, package_root: Path, package_name: str, cfg: DictConfig) -> None:
    logger.info("Setting up Python environment with uv...")
    run_cmd(["uv", "init", "--bare", f"--python={cfg.python_version}"], "Failed to initialize uv")
    os.environ["VIRTUAL_ENV"] = str(project_root / ".venv")
    run_cmd(["uv", "python", "pin", cfg.python_version], f"Failed to pin python version: {cfg.python_version}")

    for dir in cfg.python_submodules:
        submodule = package_root / dir
        submodule.mkdir(parents=True, exist_ok=True)
        (submodule / "__init__.py").touch()

    (package_root / "__init__.py").touch()
    logger.info(f"Created src structure at: {project_root / 'src'}")

    append_template_to_file(project_root / "pyproject.toml", "pyproject_suffix.toml.jinja", {"PACKAGE_NAME": package_name})
    prepend_template_to_file(project_root / "pyproject.toml", "pyproject_prefix.toml.jinja")
    logger.info("Updated: pyproject.toml")

    run_cmd(
        ["uv", "add"] + cfg.uv.packages,
        "Failed to install Python packages",
        env={"UV_HTTP_TIMEOUT": str(cfg.uv.timeout)},
    )
    run_cmd(["uv", "add", cfg.mhpy_url], "Failed at adding mhpy library as python package")

    logger.info("✅ Virtual environment created and project installed.")
    logger.info("Run 'source .venv/bin/activate' to activate it.")


def _dvc(project_root: Path, cfg: DictConfig) -> None:
    logger.info("Initializing DVC...")
    run_cmd(["uv", "run", "dvc", "init"], "Failed to initialize DVC")
    run_cmd(["uv", "run", "dvc", "config", "core.autostage", "true"], "Failed to set DVC autostage")

    for state in cfg.data_states:
        (project_root / "data" / state).mkdir(parents=True, exist_ok=True)
    (project_root / ".local_dvc_storage").mkdir(exist_ok=True)

    run_cmd(["uv", "run", "dvc", "remote", "add", "local", "./.local_dvc_storage"], "Failed to add DVC local remote")
    run_cmd(["uv", "run", "dvc", "remote", "default", "local"], "Failed to set DVC default remote")
    logger.info("DVC initialized with a local remote.")


def _wandb(project_root: Path) -> None:
    logger.info("Setting up W&B folders...")
    (project_root / "models").mkdir(exist_ok=True)
    (project_root / "reports").mkdir(exist_ok=True)
    logger.info("Created 'models' and 'reports' directories.")
    logger.info("Run 'wandb login' to log in to Weights & Biases.")


def _pre_commit(project_root: Path) -> None:
    logger.info("Setting up pre-commit hooks...")
    create_file_from_template(project_root / ".pre-commit-config.yaml", "pre-commit.yaml.jinja")
    run_cmd(["uv", "run", "pre-commit", "install"], "Failed to install pre-commit hooks")


def _makefile(project_root: Path, package_name: str, hydra_subdir: str) -> None:
    logger.info("Creating Makefile...")
    create_file_from_template(project_root / "Makefile", "Makefile.jinja", {"HYDRA_PATH": hydra_subdir})


def _hydra_configs(project_root: Path, cfg: DictConfig) -> None:
    logger.info("Creating default hydra configs...")
    hydra_dir = project_root / cfg.hydra.subdir

    hydra_dir.mkdir(parents=True, exist_ok=True)

    create_file_from_template(hydra_dir / "config.yaml", "hydra_config.yaml.jinja")

    for dir in cfg.hydra.configs:
        (hydra_dir / dir).mkdir(exist_ok=True)
        (hydra_dir / dir / "default.yaml").touch()


def _tests(project_root: Path) -> None:
    logger.info("Creating tests...")
    create_file_from_template(project_root / "pytest.ini", "pytest.ini.jinja")
    tests_dir = project_root / "tests"
    tests_dir.mkdir(exist_ok=True, parents=True)
    (tests_dir / "__init__.py").touch()


def _other_dirs(project_root: Path, cfg: DictConfig) -> None:
    logger.info("Creating remaining directories...")
    for dir in cfg.other_dirs:
        (project_root / dir).mkdir(exist_ok=True)


def _py_templates(project_root: Path, cfg: DictConfig) -> None:
    if "scripts" in cfg.other_dirs:
        create_file_from_template(project_root / "scripts" / "train.py", "train.py.jinja")


def _ipynb_templates(project_root: Path, cfg: DictConfig) -> None:
    if "notebooks" in cfg.other_dirs:
        create_file_from_template(project_root / "notebooks" / "EDA.ipynb", "EDA.ipynb.jinja")


def _final_commit() -> None:
    logger.info("Finalizing project setup...")
    run_cmd(["git", "add", "."], "Failed to add all new files to git")
    run_cmd(
        ["git", "commit", "-m", "Initial project setup from mhpy template"],
        "Failed to create final commit",
    )


def _cleanup(project_root: Path) -> None:
    for item in project_root.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _print_summary() -> None:
    logger.info("\n")
    logger.info("🎉 Project setup complete! 🎉")
    logger.info("Next steps:")
    logger.info("1. Activate the environment: source .venv/bin/activate")
    logger.info("2. Log in to W&B: wandb login")


def init(cfg: DictConfig) -> None:
    package_name = cfg.package_name
    project_root = Path.cwd()
    package_root = project_root / "src" / package_name

    if any(project_root.iterdir()):
        logger.error(f"Directory {project_root} is not empty. Aborting init to prevent data loss.")
        return

    info = _prompting(package_name, cfg)

    logger.info(f"🚀 Starting new ML project '{package_name}'...")

    try:
        _git(project_root, info["remote_url"])
        _uv(project_root, package_root, package_name, cfg)
        _dvc(project_root, cfg)
        _wandb(project_root)
        _pre_commit(project_root)
        _makefile(project_root, package_name, cfg.hydra.subdir)
        _hydra_configs(project_root, cfg)
        _tests(project_root)
        _other_dirs(project_root, cfg)
        _py_templates(project_root, cfg)
        _ipynb_templates(project_root, cfg)
        _final_commit()
        _print_summary()
    except Exception as e:
        logger.error(f"Error: {e}")
        if not cfg.debug:
            _cleanup(project_root)
            logger.info(f"All changes in {project_root} have been removed.")
        else:
            launch_debugger()
