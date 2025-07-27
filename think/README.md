# sunstone-think

Post-processing utilities for clustering, summarising and repairing captured data. The tools leverage the Gemini API to analyse transcriptions and screenshots. All commands work with a **journal** directory that holds daily folders in `YYYYMMDD` format.

## Installation

```bash
pip install -e .
```

All dependencies are listed in `pyproject.toml`.

## Usage

The package exposes several commands:

- `think-ponder` builds a Markdown summary of a day's recordings using a Gemini prompt.
- `think-cluster` groups audio and screen JSON files into report sections. Use `--start` and
  `--length` to limit the report to a specific time range.
- `see-describe` and `hear-transcribe` include a `--repair` option to process
  any missing screenshot or audio descriptions for a day.
- `think-entity-roll` collects entities across days and writes a rollup file.
- `think-process-day` runs the above tools for a single day.
- `think-supervisor` monitors hear and see heartbeats. Use `--no-runners` to skip starting them automatically.
- `think-mcp-server` starts an OAuth-enabled server exposing search capabilities over MCP for both ponder text and raw transcripts.

```bash
think-ponder YYYYMMDD [-f PROMPT] [-p] [-c] [--force] [-v]
think-cluster YYYYMMDD [--start HHMMSS --length MINUTES]
think-entity-roll
think-process-day [--day YYYYMMDD] [--force] [--repair] [--rebuild]
think-supervisor [--no-runners]
 think-mcp-server [--port PORT]
```

`-p` is a switch enabling the Gemini Pro model. Use `-c` to count tokens only,
`--force` to overwrite existing files and `-v` for verbose logs.

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.
`JOURNAL_PATH` and `GOOGLE_API_KEY` can also be provided in a `.env` file which
is loaded automatically by most commands.

## Automating daily processing

The `think-process-day` command can be triggered by a systemd timer. Below is a
minimal service and timer that process yesterday's folder every morning at
06:00:

```ini
[Unit]
Description=Process sunstone journal

[Service]
Type=oneshot
ExecStart=/usr/local/bin/think-process-day --repair

[Install]
WantedBy=multi-user.target
```

```ini
[Unit]
Description=Run think-process-day daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true
Unit=think-process-day.service

[Install]
WantedBy=timers.target
```

## CLI Agent

Two command line tools offer the same interface using different LLM backends:

```bash
think-agent  [TASK_FILE] [--model MODEL] [--max-tokens N] [-o OUT_FILE]
think-google [TASK_FILE] [--model MODEL] [--max-tokens N] [-o OUT_FILE]
```

`think-agent` relies on the OpenAI Agents API while `think-google` talks to
Gemini via the `google-genai` library. Both start a local MCP server so tools
like topic search are available during a run. If `TASK_FILE` is omitted an
interactive prompt is started.

Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` (depending on the command) along with
`JOURNAL_PATH` so the agent can query your journal index. The agent prints its
final answer to `stdout`; `-o` or `--out` writes all JSON events to a file.

### Common interface

The `AgentSession` context manager powers both CLIs. Use
`think.openai.AgentSession` or `think.google.AgentSession` depending on the
backend. The shared `BaseAgentSession` interface lives in `think.agents`:

```python
async with AgentSession() as agent:
    agent.add_history("user", "previous message")
    result = await agent.run("new request")
    print(agent.history)
```

`run()` returns the final text result. `add_history()` queues prior messages to
provide context and `history` exposes all messages seen during the session. The
same code works with either implementation, allowing you to choose between
OpenAI or Gemini at runtime.

## Topic map keys

`think.utils.get_topics()` reads the prompt files under `think/topics` and
returns a dictionary keyed by topic name. Each entry contains:

- `path` – the prompt text file path
- `color` – UI color hex string
- `mtime` – modification time of the `.txt` file
- Any additional keys from the matching `<topic>.json` metadata file such as
  `title`, `description` or `occurrences`
