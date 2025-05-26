# sunstone-see

Utilities for capturing and analyzing screenshots on Linux desktops. The module provides helpers for taking screenshots via DBus, comparing sequential images, and describing regions of interest using the Gemini API.

## Installation

```bash
pip install -e .
```

Dependencies are defined in `pyproject.toml`. You will also need system packages for GObject introspection and DBus when running on Linux.

## Usage

The package exposes a `screen-watch` command that periodically captures the screen, detects changes and optionally queries Gemini for a description:

```bash
screen-watch <interval> <output-directory> [--verbose] [--min PIXELS] [-g]
```

Set the `GOOGLE_API_KEY` environment variable before invoking Gemini functionality.
