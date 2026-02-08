# solstone-think

Post-processing utilities for clustering and summarising captured data. The tools leverage the Gemini API to analyse transcriptions and screenshots. All commands work with a **journal** directory that holds daily folders in `YYYYMMDD` format.

## Installation

```bash
make install
```

All dependencies are listed in `pyproject.toml`.

## Usage

The package exposes several commands:

- `sol call transcripts read` groups audio and screen transcripts into report sections. Use `--start` and
  `--length` to limit the report to a specific time range. See `sol call transcripts --help` for additional commands.
- `sol dream` runs generators and agents for a single day via Cortex.
- `sol agents` is the unified CLI for tool agents and generators (spawned by Cortex, NDJSON protocol).
- `sol supervisor` monitors observation heartbeats. Use `--no-observers` to disable local capture (sense still runs for remote uploads and imports).
- `sol mcp` starts an MCP server exposing search capabilities for both summary text and raw transcripts.
- `sol cortex` starts a Callosum-based service for managing AI agent instances and generators.
- `sol muse` lists available agents and generators with their configuration. Use `sol muse <name>` to see details, and `sol muse <name> --prompt` to see the fully composed prompt that would be sent to the LLM.

```bash
sol call transcripts read YYYYMMDD [--start HHMMSS --length MINUTES]
sol dream [--day YYYYMMDD] [--segment HHMMSS_LEN] [--force] [--run NAME]
sol supervisor [--no-observers]
sol mcp [--transport http] [--port PORT] [--path PATH]
sol cortex [--host HOST] [--port PORT] [--path PATH]
sol muse [--schedule daily|segment] [--json]
sol muse <name> [--prompt] [--day YYYYMMDD] [--segment HHMMSS_LEN] [--full]
```

Use `--force` to overwrite existing files, and `-v` for verbose logs.

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

### Unified Priority Execution

All scheduled prompts (both generators and tool-using agents) share a unified priority system. The `sol dream` command executes prompts ordered by priority, from lowest (runs first) to highest (runs last).

**Priority is required for all scheduled prompts.** Prompts without a `priority` field will fail validation. Suggested priority bands:

| Band | Range | Use Case |
|------|-------|----------|
| Generators | 10-30 | Content-producing prompts that create `.md` files |
| Analysis Agents | 40-60 | Agents that analyze generated content |
| Late-stage | 90+ | Agents that run after most others complete |
| Fun/Optional | 99 | Low-priority or experimental prompts |

After each generator completes and creates output, the indexer runs `--rescan-file` for incremental indexing. A full `--rescan` runs in the post phase.

### Cortex: Central Agent Manager

The Cortex service (`sol cortex`) is the central system for managing AI agent instances and generators. It monitors the journal's `agents/` directory for new requests and manages execution. All agent spawning should go through Cortex for proper event tracking and management.

Cortex routes requests based on configuration:
- Requests with `tools` field → tool-using agents (`sol agents`)
- Requests with `output` field (no `tools`) → generators (`sol agents`)

Both types are handled by the unified `sol agents` CLI which routes internally.

To spawn agents programmatically, use the cortex_client functions:

```python
from think.cortex_client import cortex_request
from think.callosum import CallosumConnection

# Create a request
agent_id = cortex_request(
    prompt="Your task here",
    name="default",
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

watcher = CallosumConnection()
watcher.start(callback=on_event)
# ... later, when done:
watcher.stop()
```

### Spawning Generators via Cortex

Generators can also be spawned via `cortex_request` by including an `output` field:

```python
from think.cortex_client import cortex_request, wait_for_agents

# Spawn a generator
agent_id = cortex_request(
    prompt="",  # Generators don't use prompts
    name="activity",
    config={
        "day": "20250109",
        "output": "md",
        "force": True,  # Regenerate even if output exists
    }
)

# Wait for completion
completed, timed_out = wait_for_agents([agent_id], timeout=300)
```

### Direct CLI Usage (Testing Only)

The `sol agents` command is primarily used internally by Cortex. For testing purposes, it can be invoked directly:

```bash
sol agents [TASK_FILE] [--provider PROVIDER] [--model MODEL] [--max-tokens N] [-o OUT_FILE]
```

The provider can be ``openai`` (default), ``google`` or ``anthropic``. Set the corresponding API key environment variable (`OPENAI_API_KEY`,
`GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`) along with `JOURNAL_PATH`.

### Provider modules

Each provider lives in `think/providers/` and exposes a common interface:

- `run_generate()` - Sync text generation, returns `GenerateResult`
- `run_agenerate()` - Async text generation, returns `GenerateResult`
- `run_tools()` - Tool-calling execution with MCP integration and event streaming

For direct LLM calls, use `think.models.generate()` or `think.models.agenerate()`
which automatically routes to the configured provider based on context.

## Generator map keys

`think.muse.get_muse_configs(has_tools=False)` reads the `.md` prompt files under `muse/` and
returns a dictionary keyed by generator name. Each entry contains:

- `path` – the prompt file path
- `color` – UI color hex string
- `mtime` – modification time of the `.md` file
- Additional keys from JSON frontmatter such as `title`, `description`, `hook`, or `instructions`

The `hook` field enables event extraction by invoking named hooks like `"occurrence"` or `"anticipation"`.
The `instructions` key allows customizing system prompts and source filtering.
See [APPS.md](APPS.md#instructions-configuration) for the full schema.

## Cortex API

Cortex is the central agent management system that all agent spawning should go through. See [CORTEX.md](CORTEX.md) for complete documentation of the Cortex API and agent event structures.

### Using cortex_client

The `think.cortex_client` module provides functions for interacting with Cortex:

```python
from think.cortex_client import cortex_request, cortex_agents

# Create an agent request
request_file = cortex_request(
    prompt="Your prompt",
    name="default",
    provider="openai"
)

# List running and completed agents
agents_info = cortex_agents(limit=10, agent_type="live")
print(f"Found {agents_info['live_count']} running agents")
```
# Muse Module

AI agent system and MCP tooling for solstone.

## Commands

| Command | Purpose |
|---------|---------|
| `sol cortex` | Agent orchestration service |
| `sol mcp` | MCP tool server (runs inside Cortex) |
| `sol agents` | Direct agent invocation (testing only) |

## Architecture

```
Cortex (orchestrator)
   ├── Callosum connection (events)
   ├── MCP HTTP server (tools)
   └── Agent subprocess management
          ↓
   Providers (openai, google, anthropic)
```

## Providers

| Provider | Module | Features |
|----------|--------|----------|
| OpenAI | `think/providers/openai.py` | GPT models via Agents SDK |
| Google | `think/providers/google.py` | Gemini models |
| Anthropic | `think/providers/anthropic.py` | Claude via Anthropic SDK |

Providers implement `run_generate()`, `run_agenerate()`, and `run_tools()` functions. See [PROVIDERS.md](PROVIDERS.md) for implementation details.

## Key Components

- **cortex.py** - Central agent manager, file watcher, event distribution, spawns agents.py
- **cortex_client.py** - Client functions: `cortex_request()`, `cortex_agents()`, `wait_for_agents()`
- **mcp.py** - FastMCP server with journal search tools
- **agents.py** - Unified CLI entry point for both tool-using agents and generators (NDJSON protocol)
- **models.py** - Unified `generate()`/`agenerate()` API, provider routing, token logging
- **batch.py** - `Batch` class for concurrent LLM requests with dynamic queuing

## Agent Personas

System prompts in `muse/*.md` (markdown with JSON frontmatter). Apps can add custom agents in `apps/{app}/muse/`.

JSON metadata supports `title`, `provider`, `model`, `tools`, `schedule`, `priority`, `multi_facet`, and `instructions` keys.

**Important:** The `priority` field is **required** for all prompts with a `schedule`. Prompts without explicit priority will fail validation. See the [Unified Priority Execution](#unified-priority-execution) section for priority bands.

See [APPS.md](APPS.md#instructions-configuration) for the `instructions` schema that controls system prompts, facet context, and source filtering.

## Documentation

- [PROVIDERS.md](PROVIDERS.md) - Provider implementation guide
- [CORTEX.md](CORTEX.md) - Full API, event schemas, request format
- [CALLOSUM.md](CALLOSUM.md) - Message bus protocol
- [THINK.md](THINK.md) - Cortex usage examples
