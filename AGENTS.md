# Repository Guide for Agents

Welcome to **sunstone**, a Python 3 project providing utilities for audio capture,
screenshot processing and AI‑driven analysis. The code base is organised into
four main packages:

- **`hear/`** – audio recording and transcription using the Gemini API.
- **`see/`** – screenshot capture, comparison and image analysis via Gemini.
- **`think/`** – post‑processing utilities for clustering and summarising data.
- **`dream/`** - a web app for interacting with the journal data.

Each of these packages contains a `README.md` that explains its usage and
installation. Entry points are defined in `pyproject.toml` with interdependencies across packages.
The following command installs all tools in editable mode:

```bash
pip install -e .
```

Refer to the top level `README.md` for a feature overview or individual package `README.md` for more specifics. Update these when helpful.

## Terminology

The directory containing all dated folders is called the **journal**.  A single
`YYYYMMDD` folder inside the journal is referred to as a **day**. More details
on these are available in the `JOURNAL.md` file.

## Development guidelines

- Use **Python 3.9+** and keep code formatted with `black` (line length 100) and
  `isort`. The configuration resides in `pyproject.toml`.
- Linting with `flake8` and type checking with `mypy` are recommended.
- If tests are added under `tests/`, run `pytest` before committing.
- The dependencies declared in `pyproject.toml` include heavy optional
  packages (for example `torch` and `PyGObject`).  If `pip install -e .[dev]`
  fails you can install only the minimal set needed for the tests:

```bash
pip install -e . --no-deps
pip install numpy Pillow Flask Markdown pytest pytest-cov
pytest
```
- Prompt .txt files in `hear/`, `see/` and `think/` provide system instructions
  for Gemini. Modify them only when a task explicitly requires it.
- Use absolute imports from the package root (for example,
  `from hear.audio_utils import SAMPLE_RATE`) so that scripts can be run directly
  as a cli or as a module, don't wrap imports in `try`/`except` blocks.

## Environment notes

The screenshot utilities depend on GNOME DBus and GObject; these only run on a Linux desktop environment.

## Contributing

Keep commits focused and descriptive, minimal comments. Ensure code is formatted and any available
checks pass before submitting a pull request, fix failed checks if relevant to your changes.
