# Repository Guide for Agents

Welcome to **sunstone**, a Python 3 project providing utilities for audio capture,
screenshot processing and AI‑driven analysis. The code base is organised into
three main packages:

- **`hear/`** – audio recording and transcription using the Gemini API.
- **`see/`** – screenshot capture, comparison and image analysis via Gemini.
- **`think/`** – post‑processing utilities for clustering and summarising data.

Each of these packages contains a `README.md` that explains its usage and
installation. Entry points are defined in `pyproject.toml` so the following
command installs all tools in editable mode:

```bash
pip install -e .
```

Refer to the project README for a feature overview.

## Terminology

The directory containing all dated folders is called the **journal**.  A single
`YYYYMMDD` folder inside the journal is referred to as a **day**.

## Development guidelines

- Use **Python 3.9+** and keep code formatted with `black` (line length 100) and
  `isort`. The configuration resides in `pyproject.toml`.
- Linting with `flake8` and type checking with `mypy` are recommended.
- If tests are added under `tests/`, run `pytest` before committing.
- Avoid committing large binary artifacts or API keys.
- Prompt text files in `hear/`, `see/` and `think/` provide system instructions
  for Gemini. Modify them only when a task explicitly requires it.
- Use absolute imports from the package root (for example,
  `from hear.audio_utils import SAMPLE_RATE`) rather than wrapping imports in
  `try`/`except` blocks.

## Environment notes

Several tools rely on the Gemini API. Set the `GOOGLE_API_KEY` environment
variable before running commands such as `gemini-mic` or `screen-watch` as
described in the package READMEs.
The screenshot utilities depend on GNOME DBus and GObject; these run on Linux.

## Useful commands

- `gemini-mic <dir>` – record audio and save transcriptions.
- `screen-watch <dir>` – capture screenshots and detect changes.
- `ponder-day <folder>` – summarise a day's recordings using Gemini.
- `cluster-day <folder>` – build a markdown report from audio and screen files.
- `reduce-screen <folder>` – summarise screen diff JSON files.

Commands `gemini-hear` and `gemini-see` run the audio and visual capture loops
continuously by wrapping the lower-level tools.

## Contributing

Keep commits focused and descriptive. Ensure code is formatted and any available
checks pass before submitting a pull request.
