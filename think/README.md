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
- `think-mcp-tools` starts an MCP server exposing search capabilities for both ponder text and raw transcripts.
- `think-cortex` starts a WebSocket API server for managing AI agent instances.

```bash
think-ponder YYYYMMDD [-f PROMPT] [-p] [-c] [--force] [-v]
think-cluster YYYYMMDD [--start HHMMSS --length MINUTES]
think-entity-roll
think-process-day [--day YYYYMMDD] [--force] [--repair] [--rebuild]
think-supervisor [--no-runners]
think-mcp-tools [--transport http] [--port PORT] [--path PATH]
think-cortex [--host HOST] [--port PORT] [--path PATH]
```

`-p` is a switch enabling the Gemini Pro model. Use `-c` to count tokens only,
`--force` to overwrite existing files and `-v` for verbose logs.

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.
`JOURNAL_PATH` and `GOOGLE_API_KEY` can also be provided in a `.env` file which
is loaded automatically by most commands.

## Service Discovery

When HTTP services start up, they write their active URIs to files in the journal's `agents/` directory for automated discovery:

- `think-mcp-tools --transport http` writes to `<journal>/agents/mcp.uri` (default: `http://127.0.0.1:6270/mcp/`)
- `think-cortex` writes to `<journal>/agents/cortex.uri` (default: `ws://127.0.0.1:2468/ws/cortex`)

These URI files allow other components to automatically discover running services without hardcoded addresses.

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

## Agent System

### Cortex: Central Agent Manager

The Cortex service (`think-cortex`) is the central system for managing AI agent instances. It monitors the journal's `agents/` directory for new requests and manages agent execution. All agent spawning should go through Cortex for proper event tracking and management.

To spawn agents programmatically, use the `CortexClient`:

```python
from think.cortex_client import CortexClient

async with CortexClient() as client:
    agent_id = await client.spawn(
        prompt="Your task here",
        persona="default",
        backend="openai"  # or "google", "anthropic", "claude"
    )
    result = await client.wait_for_completion(agent_id)
```

### Direct CLI Usage (Testing Only)

The `think-agents` command is primarily used internally by Cortex. For testing purposes, it can be invoked directly:

```bash
think-agents [TASK_FILE] [--backend PROVIDER] [--model MODEL] [--max-tokens N] [-o OUT_FILE]
```

The provider can be ``openai`` (default), ``google`` or ``anthropic``. Set the corresponding API key environment variable (`OPENAI_API_KEY`,
`GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`) along with `JOURNAL_PATH`.

### Common interface

The `AgentSession` context manager powers all the CLIs. Use
`think.openai.AgentSession`, `think.google.AgentSession` or
`think.anthropic.AgentSession` depending on the backend. The shared
`BaseAgentSession` interface lives in `think.agents`:

```python
async with AgentSession() as agent:
    agent.add_history("user", "previous message")
    result = await agent.run("new request")
    print(agent.history)
```

`run()` returns the final text result. `add_history()` queues prior messages to
provide context and `history` exposes all messages seen during the session. The
same code works with any implementation, allowing you to choose between OpenAI,
Gemini or Claude at runtime.

## Topic map keys

`think.utils.get_topics()` reads the prompt files under `think/topics` and
returns a dictionary keyed by topic name. Each entry contains:

- `path` – the prompt text file path
- `color` – UI color hex string
- `mtime` – modification time of the `.txt` file
- Any additional keys from the matching `<topic>.json` metadata file such as
  `title`, `description` or `occurrences`

## Cortex API

Cortex is the central agent management system that all agent spawning should go through. See [CORTEX.md](/CORTEX.md) for complete documentation of the Cortex WebSocket API and agent event structures.

### Using CortexClient

The `think.cortex_client` module provides a Python client for interacting with Cortex:

```python
from think.cortex_client import run_agent

# Simple helper for one-shot agent runs
result = await run_agent("Your prompt", persona="default", backend="openai")
```
