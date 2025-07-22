# sunstone-see

Utilities for capturing and analysing screenshots on Linux desktops. The package provides helpers for taking screenshots via DBus, comparing sequential images, and describing regions of interest using the Gemini API. Screenshots are organised under a **journal** directory containing daily folders.

## Installation

```bash
pip install -e .
```

Dependencies are defined in `pyproject.toml`. You will also need system packages for GObject introspection and DBus when running on Linux.

## Usage

The package exposes two commands driven by the `scan.py` and `describe.py` modules:

- `see-scan` captures the current screen state and compares it with the last run.
  When differences are found the full screenshot and a `_box.json` file are saved
  for later processing. Previous screenshots and the last run timestamp are stored
  under `~/.cache/sunstone/see`.

- `see-describe` watches the journal's current day folder for these `_box.json`
  files and uses Gemini to produce a JSON description for each diff image. The
  region is highlighted with a red box before calling Gemini. Optional flags
  enable verbose output or allow repair/scan modes for a given day.

```bash
see-scan [--verbose] [--min PIXELS]
see-describe [-v] [--scan DAY] [--repair DAY]
```

The `see-runner` command runs `see-scan` and `see-describe` together. Screenshots
are captured at a configurable interval (default 5 seconds):

```bash
see-runner [interval_seconds] [scan/describe args]
```

Use `see-reduce` to summarise diff descriptions into five minute Markdown blocks. Example:

```bash
see-reduce YYYYMMDD [--force] [--start HH:MM] [--end HH:MM] [-j THREADS]
```

Set the `GOOGLE_API_KEY` environment variable (or place it in a `.env` file)
before running `see-describe`. These utilities only run on Linux desktops
with a GNOME environment as they rely on DBus and GObject.
