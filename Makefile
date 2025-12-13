CONFIG_DIR ?= src/mhpy/config
HYDRA ?=

.PHONY: reinstall clean format commit-check

commit-check:
	uv run pre-commit run --all-files

reinstall: 
	uv pip install -e ".[all]"

clean:
	rm -rf __pycache__ .pytest_cache dist build *.egg-info

format:
	uv run ruff check --fix src