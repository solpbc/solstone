# sunstone-think

Post-processing utilities for clustering, summarising and repairing captured data. The tools leverage the Gemini API to analyse transcriptions and screenshots. All commands work with a **journal** directory that holds daily folders in `YYYYMMDD` format.

## Installation

```bash
pip install -e .
```

All dependencies are listed in `pyproject.toml`.

## Usage

The package exposes several commands:

- `ponder-day` builds a Markdown summary of a day's recordings using a Gemini prompt.
- `cluster-day` groups audio and screen JSON files into report sections.
- `reduce-screen` condenses screenshot diff descriptions into shorter text.
- `see-repair` and `hear-repair` fix partial outputs from the visual and audio tools.
- `entity-roll` collects entities across days and writes a rollup file.
- `cluster-glob` summarises multiple folders matching a pattern.

```bash
ponder-day <day-folder> [-f PROMPT] [-p MODEL]
cluster-day <day-folder>
reduce-screen <day-folder>
see-repair <day-folder>
hear-repair <day-folder>
entity-roll <journal>
cluster-glob <pattern>
```

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.
