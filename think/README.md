# sunstone-think

Post-processing utilities for clustering, summarising and repairing captured data. The tools leverage the Gemini API to analyse transcriptions and screenshots. All commands work with a **journal** directory that holds daily folders in `YYYYMMDD` format.

## Installation

```bash
pip install -e .
```

All dependencies are listed in `pyproject.toml`.

## Usage

The package exposes several commands:

- `ponder` builds a Markdown summary of a day's recordings using a Gemini prompt.
- `cluster` groups audio and screen JSON files into report sections.
- `see-repair` and `hear-repair` fix partial outputs from the visual and audio tools.
- `entity-roll` collects entities across days and writes a rollup file.
- `process-day` runs the above tools for a single day.
 - `ponder-mcp` starts an OAuth-enabled server exposing search capabilities over MCP for both ponder text and structured occurrences.

```bash
ponder YYYYMMDD [-f PROMPT] [-p MODEL]
cluster YYYYMMDD
see-repair YYYYMMDD
hear-repair YYYYMMDD
entity-roll
process-day [--day YYYYMMDD] [--force] [--repair] [--rebuild]
 ponder-mcp [--port PORT]
```

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.

## Automating daily processing

The `process-day` command can be triggered by a systemd timer. Below is a
minimal service and timer that process yesterday's folder every morning at
06:00:

```ini
[Unit]
Description=Process sunstone journal

[Service]
Type=oneshot
ExecStart=/usr/local/bin/process-day --repair

[Install]
WantedBy=multi-user.target
```

```ini
[Unit]
Description=Run process-day daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true
Unit=process-day.service

[Install]
WantedBy=timers.target
```
