## Project Overview

**Sunstone** is a Python‑based AI-driven desktop journaling toolkit that:

* **hear/**: Captures system audio and transcribes it via an external AI API.
* **see/**: Takes screenshots and analyzes them with AI vision models.
* **think/**: Post‑processes and summarizes captured data (clustering, topic extraction).
* **dream/**: A web interface for navigating and interacting with journaled content.

Entry points for each package are defined in `pyproject.toml` under `[tool.poetry.scripts]`. Each package has its own `README.md` with deeper usage examples.

---

## Project Structure

```
sunstone/
├── hear/           # Audio capture & transcription
├── see/            # Screenshot capture & image analysis
├── think/          # Data post‑processing & summarization
├── dream/          # Web app frontend & backend
├── tests/          # pytest test suites
├── JOURNAL.md      # Domain model for journal directories
├── README.md       # Helpful project overview
├── CRUMBS.md       # Definition of the .crumb file format
└── AGENTS.md       # AI & contributor guidance (this file)
```

* **Packages**: Each top‑level folder is a Python package with an `__init__.py`. Use absolute imports (e.g. `from sunstone.hear import recorder`).
* **Journal**: Data is organized under a root `journal/` with the location always specified in a .env as `JOURNAL_PATH`, with subfolders per date (“day”) as `YYYYMMDD`. See `JOURNAL.md` for details.

---

## Coding Standards & Style

* **Language**: Python 3.9+
* **Formatter**: Black (`black .`) and isort (`isort .`) with settings in `pyproject.toml`.
* **Linting**: flake8 (`flake8 .`), MyPy (`mypy .`). All new code must pass these checks.
* **Naming Conventions**:

  * **Modules & packages**: snake\_case
  * **Classes**: PascalCase
  * **Functions & variables**: snake\_case
  * **Constants**: UPPER\_SNAKE\_CASE
* **Docstrings**: Google style or NumPy style. Include parameter and return descriptions.
* **Imports**: Absolute imports only, grouped in the order: standard library, third‑party, local.

---

## Testing & CI

* **Test Framework**: pytest. Place tests under `tests/`, matching module structure.
* **Coverage**: We include `pytest-cov` in dev dependencies. Aim to maintain or improve existing coverage.
* **Commands**:

  ```bash
  pip install -e .[dev]
  pytest -q --cov=.
  flake8 .
  mypy .
  ```
* **Continuous Integration**: GitHub Actions runs Black, isort, flake8, mypy, and pytest on every pull request. Ensure local checks pass before pushing.

---

## Security & Secrets

* **No hard‑coded secrets**: All credentials must come from environment variables or arguments.
* **Input Validation**: Sanitize and validate all external inputs (file paths, user data).
* **Error Handling**: Raise exceptions for unexpected states; avoid silent failures.

---

## Dependencies Management

* **Standard Library Preferred**: Avoid adding heavy dependencies for simple tasks.
* **Adding New Dependencies**: Must update `pyproject.toml` under `[tool.poetry.dependencies]` and `[tool.poetry.dev-dependencies]` if for tests or tooling.
* **Optional Heavy Packages**: Use extras (`.[full]`) to install GPU/torch if needed. Document any optional features.

---

## Shared Utilities

* Check for shared function or common utilities:
  * `think/utils.py` available for any script
  * `dream/utils.py` for the dream app
  * whenever you create a new shared utility anywhere else, add a note here to make it more visible
  * `think/extract.py` hosts the `dream-extract` CLI for chunking media into the journal
