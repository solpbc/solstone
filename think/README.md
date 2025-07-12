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
- `screen-describe` and `gemini-transcribe` include a `--repair` option to process
  any missing screenshot or audio descriptions for a day.
- `entity-roll` collects entities across days and writes a rollup file.
- `process-day` runs the above tools for a single day.
 - `ponder-mcp` starts an OAuth-enabled server exposing search capabilities over MCP for both ponder text and structured occurrences.

```bash
ponder YYYYMMDD [-f PROMPT] [-p] [-c] [--force] [-v]
cluster YYYYMMDD
entity-roll
process-day [--day YYYYMMDD] [--force] [--repair] [--rebuild]
 ponder-mcp [--port PORT]
```

`-p` is a switch enabling the Gemini Pro model. Use `-c` to count tokens only,
`--force` to overwrite existing files and `-v` for verbose logs.

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.
`JOURNAL_PATH` and `GOOGLE_API_KEY` can also be provided in a `.env` file which
is loaded automatically by most commands.

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

## CLI Agent

`think.agent` provides a small command line interface around an OpenAI agent.
It can search across ponder summaries, search the occurrences index and read
full Markdown files from the journal using three custom tools.

```bash
python -m think.agent path/to/task.txt [--model MODEL] [--max-tokens N]
```

Set `OPENAI_API_KEY` and `JOURNAL_PATH` in your environment so the agent can
query your journal index. The tools available to the agent are:

- **search_ponder** – full text search across `ponder_*.md` sentences.
- **search_occurrences** – keyword search over `occurrences.json` entries.
- **read_markdown** – return the contents of any `journal/YYYYMMDD/*.md` file.

The agent will loop automatically and print its final answer to `stdout`.
