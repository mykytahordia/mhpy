# mhpy

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Hydra-1.3.2+-89b8cd.svg)](https://hydra.cc/)
[![Weights & Biases](https://img.shields.io/badge/W&B-0.23.0+-FFBE00.svg)](https://wandb.ai/)
[![Loguru](https://img.shields.io/badge/Loguru-0.7.3+-00ADD8.svg)](https://github.com/Delgan/loguru)
[![Jinja2](https://img.shields.io/badge/Jinja2-3.1.6+-B41717.svg)](https://jinja.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24.4+-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.5+-11557c.svg)](https://matplotlib.org/)
[![GitPython](https://img.shields.io/badge/GitPython-3.1.45+-F05032.svg)](https://gitpython.readthedocs.io/)
[![pytest](https://img.shields.io/badge/pytest-8.3.5+-0A9EDC.svg)](https://pytest.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-4.4.0+-FAB040.svg)](https://pre-commit.com/)

A scaffolding library and CLI for quickly spinning up ML projects with all the good stuff already configured.

## What is this?

`mhpy` (MykytaHordiaPY) is a project scaffolder that sets up a complete ML project structure with:
- **Hydra** for config management
- **W&B** for experiment tracking
- **DVC** for data versioning
- **pre-commit** hooks for code quality
- **pytest** for testing
- Common ML utilities (PyTorch helpers, metrics, plotting, etc.)

Basically, it saves you from copy-pasting the same boilerplate across projects.

## Installation

### Basic install (just the CLI)
pip:
```bash
pip install git+https://github.com/NikitaGordia/mhpy.git
```
uv:
```bash
uv tool install git+https://github.com/NikitaGordia/mhpy.git
```

### With ML dependencies and utils (PyTorch, W&B, etc.)

```bash
pip install "mhpy[ml] @ git+https://github.com/NikitaGordia/mhpy.git"
```

### With dev dependencies (pytest, pre-commit, etc.)

```bash
pip install "mhpy[dev] @ git+https://github.com/NikitaGordia/mhpy.git"
```

### Everything

```bash
pip install "mhpy[all] @ git+https://github.com/NikitaGordia/mhpy.git"
```

### For development

If you want to try it youself:

```bash
git clone https://github.com/NikitaGordia/mhpy.git
cd mhpy
pip install uv  # if you don't have it
uv sync --all-extras
```

## Usage

### Initialize a new ML project

```bash
mhpy package_name=my_awesome_project
```

This will scaffold a complete project with:
- Git repo initialized
- Virtual environment with `uv`
- Project structure (`src/`, `tests/`, `notebooks/`, `data/`, etc.)
- Hydra configs
- DVC setup
- pre-commit hooks
- Makefile with common commands
- All dependencies installed

After running, just:
```bash
source .venv/bin/activate
wandb login  # if you use W&B
```

## Requirements

- Python ≥3.12
- `uv` package manager

## License

Check the LICENSE file in the repo.