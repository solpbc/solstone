# sunstone-see

Utilities for capturing and analysing screenshots on Linux desktops. The package provides helpers for taking screenshots via DBus, comparing sequential images, and describing regions of interest using the Gemini API. Screenshots are organised under a **journal** directory containing daily folders.

## Installation

```bash
pip install -e .
```

Dependencies are defined in `pyproject.toml`. You will also need system packages for GObject introspection and DBus when running on Linux.

## Usage

The package exposes two commands driven by the `scan.py` and `describe.py` modules:

- `screen-watch` captures the current screen state and compares it with the last
  run. When differences are found an annotated image and a `_box.json` file are
  saved for later processing. Previous screenshots and the last run timestamp
  are stored under `~/.cache/sunstone/see`.

- `screen-describe` watches the journal's current day folder for these `_box.json`
  files and uses Gemini to produce a JSON description for each diff image.
  Optional flags allow specifying an entities file, the polling interval and
  verbose logging.

```bash
screen-watch <journal> [--verbose] [--min PIXELS]
screen-describe <journal> [-e ENTITIES] [-i SECONDS] [-v]
```

Set the `GOOGLE_API_KEY` environment variable before running `screen-describe`.
