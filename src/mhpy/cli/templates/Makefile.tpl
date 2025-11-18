VENV_PYTHON = .venv/bin/python
CONFIG_DIR ?= src/{{PACKAGE_NAME}}/config
HYDRA ?=

.PHONY: train

train: 
	$(VENV_PYTHON) -m {{PACKAGE_NAME}} --config-dir $(CONFIG_DIR) $(HYDRA)

clean:
	rm -rf __pycache__ .pytest_cache dist build *.egg-info

format:
	ruff check --fix src