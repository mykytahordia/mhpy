from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
import uuid

from loguru import logger

from mhpy.utils.subprocess import run_cmd

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
PYTHON_SUBMODULES = ["data", "train", "models", "utils"]
MHPY_URL = "https://github.com/NikitaGordia/mhpy.git"
UV_TIMEOUT = 100000
PYTHON_VERSION = "3.12"
UV_PACKAGES = [
    "dvc",
    "ruff",
    "pre-commit",
    "wandb",
    "torch",
    "numpy",
    "scikit-learn",
    "ipython",
    "jupyter",
    "tqdm",
    "matplotlib",
    "seaborn",
    "hydra-core",
    "omegaconf",
    "loguru",
    "pandas",
    "torcheval",
    "pytest",
    "pytest-cov",
]
DATA_STATES = ["raw", "interim", "processed"]
CONFIG_DIRS = ["model", "train", "data", "env", "exp"]
OTHER_DIRS = ["notebooks", "scripts", "tests"]


def create_file_from_template(filepath: Path, template_name: str, replacements: dict | None = None) -> None:
    content = (TEMPLATE_DIR / template_name).read_text()
    if replacements:
        for key, value in replacements.items():
            content = content.replace(f"{{{{{key}}}}}", value)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content)
    logger.info(f"Created: {filepath}")


def register_init_args(subparsers: ArgumentParser) -> None:
    init_parser = subparsers.add_parser("init", help="Initializes a new ML project in the current directory.")

    init_parser.add_argument(
        "package_name",
        type=str,
        help="The name of the main Python package under 'src/'",
    )

    init_parser.set_defaults(func=init)


def _assert_no_code_leakage(package_name: str) -> None:
    while True:
        confirmation_code = f"{package_name}_{str(uuid.uuid4())[:4]}"
        response_code = input(
            f"‼️ Make sure your AI tool settings are configured so that no code is used for training by third parties. Type '{confirmation_code}' to confim: "
        )
        if response_code == confirmation_code:
            return True
        else:
            logger.info("Try again...")


def _prompting(package_game: str) -> dict:
    _assert_no_code_leakage(package_game)

    remote_url = input("🔗 Enter remote repository URL (Enter to skip): ").strip()

    return {"remote_url": remote_url}


def _git(project_root: Path, remote_url: str) -> None:
    logger.info("Initializing Git...")
    if not (project_root / ".git").exists():
        run_cmd("git init", "Failed to initialize git")
        run_cmd("git branch -M main", "Failed to create main branch")
    else:
        logger.info("Git repository already exists.")

    create_file_from_template(project_root / ".gitignore", "gitignore.tpl")
    run_cmd("git add .gitignore", "Failed to add .gitignore")
    run_cmd('git commit -m "Initial commit: Add .gitignore"', "Failed to commit .gitignore")

    if remote_url:
        run_cmd(f"git remote add origin {remote_url}", "Failed to add remote origin")
        logger.info(f"✅ Remote 'origin' set to {remote_url}")
        run_cmd("git push -u origin main", f"Failed to push to origin/main {remote_url}")
    else:
        logger.info("Skipping remote configuration.")


def _uv(project_root: Path, package_root: Path, package_name: str) -> None:
    logger.info("Setting up Python environment with uv...")
    run_cmd(f"uv init --bare --python={PYTHON_VERSION}", "Failed to initialize uv")
    run_cmd(f"uv python pin {PYTHON_VERSION}", f"Failed to pin python version: {PYTHON_VERSION}")

    for dir in PYTHON_SUBMODULES:
        submodule = package_root / dir
        submodule.mkdir(parents=True, exist_ok=True)
        (submodule / "__init__.py").touch()

    (package_root / "config").mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").touch()
    logger.info(f"Created src structure at: {project_root / 'src'}")

    pyproject_append_content = (TEMPLATE_DIR / "pyproject_append.tpl").read_text()
    pyproject_append_content = pyproject_append_content.replace("{{PACKAGE_NAME}}", package_name)
    with (project_root / "pyproject.toml").open("a") as f:
        f.write("")
        f.write(pyproject_append_content)
    logger.info("Updated: pyproject.toml")

    packages = f"dvc pre-commit {' '.join(UV_PACKAGES)}"
    run_cmd(
        f"export UV_HTTP_TIMEOUT={UV_TIMEOUT} && uv add {packages}",
        "Failed to install Python packages",
    )
    run_cmd(f"uv add {MHPY_URL}", "Failed at adding mhpy library as python package")

    run_cmd("uv pip install -e .", "Failed to install project in editable mode")
    logger.info("✅ Virtual environment created and project installed.")
    logger.info("Run 'source .venv/bin/activate' to activate it.")


def _dvc(project_root: Path) -> None:
    logger.info("Initializing DVC...")
    run_cmd("dvc init", "Failed to initialize DVC")
    run_cmd("dvc config core.autostage true", "Failed to set DVC autostage")

    for state in DATA_STATES:
        (project_root / "data" / state).mkdir(parents=True, exist_ok=True)
    (project_root / ".local_dvc_storage").mkdir(exist_ok=True)

    run_cmd("dvc remote add local ./.local_dvc_storage", "Failed to add DVC local remote")
    run_cmd("dvc remote default local", "Failed to set DVC default remote")
    logger.info("DVC initialized with a local remote.")


def _wandb(project_root: Path) -> None:
    logger.info("Setting up W&B...")
    (project_root / "models").mkdir(exist_ok=True)
    (project_root / "reports").mkdir(exist_ok=True)
    logger.info("Created 'models' and 'reports' directories.")
    logger.info("Run 'wandb login' to log in to Weights & Biases.")


def _pre_commit(project_root: Path) -> None:
    logger.info("Setting up pre-commit hooks...")
    create_file_from_template(project_root / ".pre-commit-config.yaml", "pre-commit.tpl")
    run_cmd("pre-commit install", "Failed to install pre-commit hooks")


def _makefile(project_root: Path, package_name: str) -> None:
    logger.info("Creating Makefile...")
    create_file_from_template(project_root / "Makefile", "Makefile.tpl", {"PACKAGE_NAME": package_name})


def _hydra_configs(package_root: Path, package_name: str) -> None:
    logger.info("Creating default hydra configs...")
    create_file_from_template(package_root / "config" / "config.yaml", "hydra_config.tpl")
    for dir in CONFIG_DIRS:
        (package_root / "config" / dir).mkdir(exist_ok=True)
        (package_root / "config" / dir / "default.yaml").touch()

    create_file_from_template(
        package_root / "train" / "train.py",
        "train.tpl",
        {"PACKAGE_NAME": package_name},
    )


def _tests(project_root: Path) -> None:
    logger.info("Creating tests...")
    create_file_from_template(project_root / "pytest.ini", "pytest.tpl")


def _other_dirs(project_root: Path) -> None:
    logger.info("Creating remaining directories...")
    for dir in OTHER_DIRS:
        (project_root / dir).mkdir(exist_ok=True)


def _final_commit() -> None:
    logger.info("Finalizing project setup...")
    run_cmd("git add .", "Failed to add all new files to git")
    run_cmd(
        'git commit -m "feat: Initial project setup from mhpy template"',
        "Failed to create final commit",
    )


def _print_summary() -> None:
    logger.info("\n")
    logger.info("🎉 Project setup complete! 🎉")
    logger.info("Next steps:")
    logger.info("1. Activate the environment: source .venv/bin/activate")
    logger.info("2. Log in to W&B: wandb login")


def init(args: Namespace) -> None:
    package_name = args.package_name
    project_root = Path.cwd()
    package_root = project_root / "src" / package_name

    info = _prompting(package_name)

    logger.info(f"🚀 Starting new ML project '{package_name}'...")
    _git(project_root, info["remote_url"])
    _uv(project_root, package_root, package_name)
    _dvc(project_root)
    _wandb(project_root)
    _pre_commit(project_root)
    _makefile(project_root, package_name)
    _hydra_configs(package_root, package_name)
    _tests(project_root)
    _other_dirs(project_root)
    _final_commit()

    _print_summary()
