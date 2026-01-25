# solstone-think

Post-processing utilities for clustering and summarising captured data. The tools leverage the Gemini API to analyse transcriptions and screenshots. All commands work with a **journal** directory that holds daily folders in `YYYYMMDD` format.

## Installation

```bash
pip install -e .
```

All dependencies are listed in `pyproject.toml`.

## Usage

The package exposes several commands:

- `sol insight` builds a Markdown summary of a day's recordings using a Gemini prompt.
- `sol cluster` groups audio and screen JSON files into report sections. Use `--start` and
  `--length` to limit the report to a specific time range.
- `sol dream` runs the above tools for a single day.
- `sol supervisor` monitors observation heartbeats. Use `--no-observers` to disable local capture (sense still runs for remote uploads and imports).
- `sol mcp` starts an MCP server exposing search capabilities for both summary text and raw transcripts.
- `sol cortex` starts a WebSocket API server for managing AI agent instances.

```bash
sol insight YYYYMMDD -f PROMPT [--segment HHMMSS_LEN] [--segments SEG1,SEG2 -o OUT] [--force] [-v]
sol cluster YYYYMMDD [--start HHMMSS --length MINUTES]
sol dream [--day YYYYMMDD] [--segment HHMMSS_LEN] [--force] [--skip-insights] [--skip-agents]
sol supervisor [--no-observers]
sol mcp [--transport http] [--port PORT] [--path PATH]
sol cortex [--host HOST] [--port PORT] [--path PATH]
```

Use `--segment` to process a single segment, or `--segments` with `-o` to process
multiple specific segments (comma-separated). Use `-o` to override the output path
for any mode.

Use `-c` to count tokens only, `--force` to overwrite existing files, and `-v` for
verbose logs.

Set `GOOGLE_API_KEY` before running any command that contacts Gemini.
`JOURNAL_PATH` and `GOOGLE_API_KEY` can also be provided in a `.env` file which
is loaded automatically by most commands.

## Service Discovery

The MCP HTTP server now runs inside Cortex itself. When Cortex starts it passes
the URL directly to each agent request (`mcp_server_url`). Utilities that need
tool metadata, such as `sol planner`, query the registered tools directly and
no discovery files or environment variables are required.

## Automating daily processing

The `sol dream` command can be triggered by a systemd timer. Below is a
minimal service and timer that process yesterday's folder every morning at
06:00:

```ini
[Unit]
Description=Process solstone journal

[Service]
Type=oneshot
ExecStart=/usr/local/bin/sol dream

[Install]
WantedBy=multi-user.target
```

```ini
[Unit]
Description=Run sol dream daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true
Unit=sol-dream.service

[Install]
WantedBy=timers.target
```

## Agent System

### Cortex: Central Agent Manager

The Cortex service (`sol cortex`) is the central system for managing AI agent instances. It monitors the journal's `agents/` directory for new requests and manages agent execution. All agent spawning should go through Cortex for proper event tracking and management.

To spawn agents programmatically, use the cortex_client functions:

```python
from muse.cortex_client import cortex_request
from think.callosum import CallosumConnection

# Create a request
agent_id = cortex_request(
    prompt="Your task here",
    persona="default",
    provider="openai"  # or "google", "anthropic", "claude"
)

# Watch for agent events via Callosum
def on_event(message):
    # Filter for cortex tract events
    if message.get('tract') != 'cortex':
        return

    print(f"Event: {message['event']}")
    if message.get('event') == 'finish':
        print(f"Result: {message.get('result')}")

watcher = CallosumConnection(callback=on_event)
watcher.connect()
# ... later, when done:
watcher.close()
```

### Direct CLI Usage (Testing Only)

The `sol agents` command is primarily used internally by Cortex. For testing purposes, it can be invoked directly:

```bash
sol agents [TASK_FILE] [--provider PROVIDER] [--model MODEL] [--max-tokens N] [-o OUT_FILE]
```

The provider can be ``openai`` (default), ``google`` or ``anthropic``. Set the corresponding API key environment variable (`OPENAI_API_KEY`,
`GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`) along with `JOURNAL_PATH`.

### Provider modules

Each provider lives in `muse/providers/` and exposes a common interface:

- `generate()` - Sync text generation
- `agenerate()` - Async text generation
- `run_agent()` - Agent execution with MCP tools and event streaming

For direct LLM calls, use `muse.models.generate()` or `muse.models.agenerate()`
which automatically routes to the configured provider based on context.

## Insight map keys

`think.utils.get_insights()` reads the `.md` prompt files under `think/insights` and
returns a dictionary keyed by insight name. Each entry contains:

- `path` – the prompt file path
- `color` – UI color hex string
- `mtime` – modification time of the `.md` file
- Additional keys from JSON frontmatter such as `title`, `description`, `hook`, or `instructions`

The `hook` field enables event extraction by invoking named hooks like `"occurrence"` or `"anticipation"`.
The `instructions` key allows customizing system prompts and source filtering.
See [APPS.md](APPS.md#instructions-configuration) for the full schema.

## Cortex API

Cortex is the central agent management system that all agent spawning should go through. See [CORTEX.md](CORTEX.md) for complete documentation of the Cortex WebSocket API and agent event structures.

### Using cortex_client

The `muse.cortex_client` module provides functions for interacting with Cortex:

```python
from muse.cortex_client import cortex_request, cortex_agents

# Create an agent request
request_file = cortex_request(
    prompt="Your prompt",
    persona="default",
    provider="openai"
)

# List running and completed agents
agents_info = cortex_agents(limit=10, agent_type="live")
print(f"Found {agents_info['live_count']} running agents")
```
