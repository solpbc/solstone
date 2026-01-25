# Cortex API and Eventing

The Cortex system manages AI agent execution through the Callosum message bus with file-based persistence. It acts as a process manager for agent instances, receiving requests via Callosum and writing execution events to both JSONL files (for persistence) and the message bus (for real-time distribution).

For details on the Callosum protocol and message format, see [CALLOSUM.md](CALLOSUM.md).

## Architecture

### Event Flow
1. **Request Creation**: Client calls `cortex_request()` which broadcasts to Callosum (`tract="cortex"`, `event="request"`)
2. **Request Reception**: Cortex receives message via Callosum callback and creates `<timestamp>_active.jsonl`
3. **Agent Spawning**: Cortex spawns agent process via `sol agents` with merged configuration
4. **Event Emission**: Agents write JSON events to stdout (captured by Cortex)
5. **Event Distribution**: Cortex appends events to JSONL file AND broadcasts to Callosum
6. **Agent Completion**: Cortex renames file to `<timestamp>.jsonl` when agent finishes

### Key Components
- **Message Bus Integration**: Cortex connects to Callosum to receive requests and broadcast events
- **Configuration Loading**: Cortex loads and merges persona configuration with request parameters
- **Process Management**: Spawns agent subprocesses via the `sol agents` command with merged configuration
- **Event Capture**: Monitors agent stdout/stderr and appends to JSONL files
- **Dual Event Distribution**: Events go to both persistent files and real-time message bus
- **NDJSON Input Mode**: Agent processes accept newline-delimited JSON via stdin containing the full merged configuration

### File States
- `<timestamp>_active.jsonl`: Agent currently executing (Cortex is appending events)
- `<timestamp>.jsonl`: Agent completed (contains full event history)

**Note**: Files provide persistence and historical record, while Callosum provides real-time event distribution to all interested services.

## Request Format

Requests are created via `cortex_request()` from `think.cortex_client`, which broadcasts to Callosum. The request message follows this format:

```json
{
  "event": "request",
  "ts": 1234567890123,              // Required: millisecond timestamp (must match filename)
  "prompt": "Analyze this code for security issues",  // Required: the task or question
  "persona": "default",              // Optional: agent persona from muse/*.md
  "provider": "openai",              // Optional: override provider (openai, google, anthropic)
  "max_tokens": 8192,               // Optional: token limit (if supported)
  "disable_mcp": false,             // Optional: disable MCP tools for this request
  "continue_from": "1234567890122",  // Optional: continue from previous agent
  "facet": "my-project",          // Optional: project context
  "output": "md",                     // Optional: output format ("md" or "json"), writes to insights/
  "day": "20250109",                  // Optional: YYYYMMDD format, defaults to current day
  "env": {                           // Optional: environment variables for subprocess
    "API_KEY": "secret",
    "DEBUG": "true"
  },
  "handoff": {                       // Optional: chain to another agent on completion
    "persona": "reviewer",
    "prompt": "Review the analysis",
    "provider": "openai"
  },
  "handoff_from": "1234567890122"   // Optional: present when spawned via handoff
}
```

The model is automatically resolved based on the agent context (`agent.{app}.{persona}`)
and the configured tier in `journal.json`. Provider can optionally be overridden at
request time, which will resolve the appropriate model for that provider at the same tier.

### Conversation Continuations

All providers (Anthropic, OpenAI, Google) support continuing conversations from previous
agent runs. Include a `continue_from` field in your request with the `<timestamp>`
identifier of any completed agent run. The provider will load the conversation history
from the agent's event log and continue from where it left off. This works seamlessly
across all providers - you can even switch providers mid-conversation (e.g., start with
OpenAI, continue with Anthropic).

## Agent Event Format

All subsequent lines are JSON objects with `event` and millisecond `ts` fields. The `ts` field is automatically added by Cortex if not provided by the provider. Additionally, Cortex automatically adds an `agent_id` field (matching the timestamp from the filename) to all events for tracking purposes.

### request
The initial spawn request (first line of file, written by client).
```json
{
  "event": "request",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "prompt": "User's task or question",
  "provider": "openai",
  "persona": "default",
  "output": "md",
  "day": "20250109",
  "handoff": {},
  "handoff_from": "1234567890122"
}
```

### start
Emitted when an agent run begins.
```json
{
  "event": "start",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "persona": "default",
  "model": "gpt-4o"
}
```

### tool_start
Emitted when a tool execution begins.
```json
{
  "event": "tool_start",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "tool": "search_journal",
  "args": {"query": "search terms", "limit": 10},
  "call_id": "search_journal-1"
}
```

### tool_end
Emitted when a tool execution completes.
```json
{
  "event": "tool_end",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "tool": "search_journal",
  "args": {"query": "search terms"},
  "result": ["result", "array", "or", "object"],
  "call_id": "search_journal-1"
}
```

### thinking
Emitted when the model produces reasoning/thinking content (model-dependent, primarily o1 models).
```json
{
  "event": "thinking",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "summary": "Model's internal reasoning about the task...",
  "model": "o1-mini"
}
```

### agent_updated
Emitted when control is handed off to a different agent (multi-agent scenarios).
```json
{
  "event": "agent_updated",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "agent": "SpecializedAgent"
}
```

### finish
Emitted when the agent run completes successfully.
```json
{
  "event": "finish",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "result": "Final response text to the user",
  "handoff": {                     // Optional: triggers next agent
    "prompt": "Continue with next task",
    "persona": "specialist",
    "provider": "openai"
  }
}
```

### error
Emitted when an error occurs during execution.
```json
{
  "event": "error",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "error": "Error message",
  "trace": "Full stack trace..."
}
```

### info
Emitted when non-JSON output is captured from agent stdout.
```json
{
  "event": "info",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "message": "Non-JSON output line from agent"
}
```

## Tool Call Tracking

Tool events use `call_id` to pair `tool_start` and `tool_end` events. This allows tracking:
- Which tools are currently running
- Tool execution duration
- Tool inputs and outputs
- Concurrent tool executions

The frontend uses this to show real-time status updates as tools execute, changing from "running..." to "✓" when complete.

## Agent Output

When an agent completes successfully, its result can be automatically written to a file. This uses the same output path logic as insights.

- Include an `output` field in the agent's frontmatter with the format ("md" or "json")
- Output path is derived from persona name + format + schedule:
  - Daily agents: `YYYYMMDD/insights/{persona}.{ext}`
  - Segment agents: `YYYYMMDD/{segment}/{persona}.{ext}`
- Writing occurs before any handoff processing
- Write failures are logged but don't interrupt the agent flow
- Commonly used for scheduled agents that generate daily reports

## Agent Handoff

Agents can transfer control to other agents for specialized tasks. When an agent completes with a handoff configuration, Cortex automatically spawns the next agent in the chain.

- The `finish` event may include a `handoff` field specifying the next agent
- The subsequent request includes `handoff_from` with the originating agent ID
- Handoff agents automatically inherit the parent agent's configuration (provider, model, etc.) unless explicitly overridden
- This enables multi-step workflows and agent specialization with consistent configuration

## Agent Personas

Agents use persona configurations stored in the `muse/` directory. Each persona is a `.md` file containing:
- JSON frontmatter with metadata and configuration
- The agent-specific prompt and instructions in the content

When spawning an agent:
1. Cortex loads the persona configuration using `get_agent()` from `think/utils.py`
2. The configuration is built with three instruction components:
   - `system_instruction`: `journal.md` (shared base prompt, cacheable)
   - `extra_context`: Runtime context (facets, insights list, datetime)
   - `user_instruction`: The agent's `.md` file content
3. Request parameters override persona defaults in the merged configuration
4. The full configuration is passed to the agent process

Personas define specialized behaviors, tool usage patterns, and facet expertise. Available personas can be discovered using the `get_agents()` function or by listing files in the `muse/` directory (agents are `.md` files with a `tools` field).

### Persona Configuration Options

The JSON frontmatter for a persona can include:
- `max_tokens`: Maximum response token limit
- `tools`: MCP tools configuration (string or array)
  - String: Comma-separated pack names (e.g., `"journal"`, `"journal, todo"`) - expanded via `get_tools()`
  - Available packs: `journal`, `todo`, `facets`, `entities`, `apps`
  - Array: Explicit list of tool names (e.g., `["search_insights", "get_facet"]`)
  - If omitted, defaults to "default" pack (alias for "journal")
- `schedule`: Scheduling configuration for automated execution
  - `"daily"`: Run automatically at midnight each day
- `priority`: Execution order for scheduled agents (integer, default: 50)
  - Lower numbers run first (e.g., priority 10 runs before priority 50)
  - Used to control the order when multiple agents have the same schedule
- `multi_facet`: Boolean flag for facet-aware agents (default: false)
  - When true, the agent is spawned once for each **active** facet (see Multi-Facet Agents section)
  - Each instance receives a facet-specific prompt with the facet name
  - Useful for creating per-facet reports, newsletters, or analyses
- `always`: Override active facet detection for multi-facet agents (default: false)
  - When true, agent runs for all non-muted facets regardless of activity
- `env`: Environment variables to set for the agent subprocess (object)
  - Keys are variable names, values are coerced to strings
  - Request-level `env` overrides persona defaults
  - Inherited by handoff agents unless explicitly overridden
  - Note: `JOURNAL_PATH` cannot be overridden (always set by Cortex)

### Model Resolution

Models are resolved automatically based on context and tier:
1. Each agent has a context pattern: `agent.{app}.{persona}` (e.g., `agent.system.default`)
2. The context determines the tier (pro/flash/lite) from `journal.json` or system defaults
3. The tier + provider determines the actual model to use

This allows controlling model selection via tier configuration rather than hardcoding models:
```json
{
  "providers": {
    "contexts": {
      "agent.system.default": {"tier": 1},
      "agent.*": {"tier": 2}
    }
  }
}
```

## MCP Tools Integration

The Model Context Protocol (MCP) provides tools for agent-journal interaction:

### Backend Support
- **OpenAI, Anthropic, Google**: Full MCP tool support via HTTP transport

### Tool Discovery
MCP tools are provided by the `think.mcp_tools` FastMCP server, which:
- Runs inside Cortex as a background HTTP service
- Shares its URL directly with agent runs (`mcp_server_url`) so no discovery file is needed
- Exposes journal search and retrieval capabilities
- Available tools can be discovered via the MCP service endpoint

## Agent Providers

The system supports multiple AI providers, each implementing the same event interface:

- **OpenAI** (`think/providers/openai.py`): GPT models with OpenAI Agents SDK
- **Google** (`think/providers/google.py`): Gemini models with Google AI SDK
- **Anthropic** (`think/providers/anthropic.py`): Claude models with Anthropic SDK

All providers:
- Emit JSON events to stdout (one per line)
- Are spawned as subprocesses by Cortex
- Use consistent event structures across providers
- Process events are written to stdout for Cortex to capture

## Scheduled Agents

Agents with `"schedule": "daily"` run automatically via `sol dream` at midnight each day:

### Execution Order
Scheduled agents run in priority order (lower numbers first):
1. Agents are sorted by their `priority` field (default: 50)
2. Agents with the same priority run in alphabetical order by filename
3. Each agent completes before the next begins

### Multi-Facet Agents
When an agent has `"multi_facet": true`:
1. The agent is spawned once for each **active** facet
2. Each instance receives a prompt including the facet name
3. The agent should call `get_facet(facet_name)` to load facet context
4. This enables per-facet reports, newsletters, and analyses

**Active Facet Detection**: By default, multi-facet agents only run for facets that had activity the previous day. Activity is determined by the presence of occurrence events (not anticipations) in `facets/{facet}/events/{day}.jsonl`. This prevents unnecessary agent runs for inactive facets.

To force an agent to run for all facets regardless of activity, set `"always": true`:

```json
{
  "title": "Facet Newsletter Generator",
  "schedule": "daily",
  "priority": 10,
  "multi_facet": true,
  "tools": "journal,facets"
}
```

```json
{
  "title": "Facet Auditor",
  "schedule": "daily",
  "multi_facet": true,
  "always": true,
  "tools": "journal,facets"
}
```

## Process Management

The `sol supervisor` command provides process management for the Cortex ecosystem:
- Starts and monitors the Cortex file watcher service
- Starts and monitors the MCP tools HTTP server
- Handles process restarts on failure
- Monitors system health indicators
- Triggers `sol dream` at midnight for daily processing (insights + agents)

This is distinct from agent lifecycle management, which Cortex handles internally through file state transitions.
