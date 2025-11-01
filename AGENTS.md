# Repository Guidelines

## Project Structure & Module Organization
- `docs/` stores research narratives and outlines; keep section numbering consistent with existing files.
- Place implementation under `src/plora_cl/` with subpackages for `data`, `models`, `training`, and `evaluation`.
- Version experiment configs inside `experiments/<task>-<date>/` accompanied by a short README noting datasets and metrics.
- Exclude large checkpoints via `.gitignore`; point to external storage in `artifacts/README.md`.

## Build, Test, and Development Commands
- Pin the toolchain with `uv python install 3.11` to align everyone on the same interpreter.
- Create or refresh the environment via `uv venv` and sync dependencies with `uv pip install -r requirements.txt`.
- Run training locally using `uv run python -m plora_cl.cli.train --config experiments/<task>/config.yaml` once the CLI module lands.
- Execute fast checks with `uv run pytest tests -q`; before merging run `uv run pytest --maxfail=1 --disable-warnings --cov=plora_cl`.

## Coding Style & Naming Conventions
- Follow PEP 8, four-space indentation, and descriptive snake_case; reserve PascalCase for classes and configs.
- Type-hint public functions and favor dataclasses for configuration objects.
- Format code with `ruff format` and lint using `ruff check src tests`; auto-apply fixes when available.
- Include brief module docstrings explaining continual-learning assumptions or adapter wiring decisions.

## Testing Guidelines
- Mirror the `src/` layout under `tests/` using filenames like `test_lora_scheduler.py`.
- Write deterministic fixtures and synthetic samples so suites finish in under two minutes.
- Cover adapters, replay utilities, and evaluation metrics with both unit and integration tests.
- Track catastrophic forgetting metrics by snapshotting baseline scores and asserting acceptable deltas.

## Commit & Pull Request Guidelines
- Use imperative commit subjects under 50 characters, e.g., `Add orthogonal adapter scheduler`.
- Reference issues or discussion threads in the body and note datasets touched or artifacts regenerated.
- PR descriptions should summarize scope, include experiment IDs, and attach key metrics or plots.
- Confirm `pytest` results or CI links before requesting review; flag follow-up work in a closing checklist.

## Research Assets & Data Handling
- Record dataset provenance and preprocessing steps in `docs/` or experiment READMEs.
- Store credentials and API keys in environment variables, documenting required names in `.env.example`.
