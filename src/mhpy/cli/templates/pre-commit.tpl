repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.14.5
  hooks:
    # Run the linter.
  - id: ruff
    types_or: [python, pyi]
    args: [--fix]
    # Run the formatter.
  - id: ruff-format
    types_or: [python, pyi]
- repo: local
  hooks:
  - id: dvc-strict-check
    name: DVC strict check
    entry: dvc status --quiet
    language: system
    stages: [pre-commit]
    pass_filenames: false
- repo: https://github.com/gitleaks/gitleaks
  rev: v8.29.0
  hooks:
    - id: gitleaks
