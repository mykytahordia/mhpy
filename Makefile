VIRTUAL_ENV ?= .venv
CONFIG_DIR ?= src/mhpy/config
HYDRA ?=

.PHONY: reinstall

reinstall: 
	VIRTUAL_ENV=$(VIRTUAL_ENV) uv pip install -e ".[all]"

clean:
	rm -rf __pycache__ .pytest_cache dist build *.egg-info

format:
	ruff check --fix src