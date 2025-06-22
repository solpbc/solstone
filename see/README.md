# sunstone-see

Utilities for capturing and analyzing screenshots on Linux desktops. The module provides helpers for taking screenshots via DBus, comparing sequential images, and describing regions of interest using the Gemini API.

## Installation

```bash
pip install -e .
```

Dependencies are defined in `pyproject.toml`. You will also need system packages for GObject introspection and DBus when running on Linux.

## Usage

The package exposes two commands:

- `screen-watch` captures the current screen state and compares it with the last
  run. When differences are found an annotated image and a `_box.json` file are
  saved for later processing. Previous screenshots and the last run timestamp
  are stored under `~/.cache/sunstone/see`.

- `screen-describe` monitors a directory for these `_box.json` files and uses
  Gemini to produce a JSON description for each diff image.

```bash
screen-watch <output-directory> [--verbose] [--min PIXELS]
screen-describe <day-directory>
```

Set the `GOOGLE_API_KEY` environment variable before running `screen-describe`.
