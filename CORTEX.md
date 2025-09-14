# Cortex API and Eventing

The Cortex system manages AI agent execution through a file-based architecture. It acts as a process manager for agent instances, monitoring the journal's `agents/` directory for new requests and appending execution events to JSONL files.

## Architecture

### Event Flow
1. **Request Creation**: Client writes spawn request to `<timestamp>_pending.jsonl`
2. **Agent Activation**: Client renames file to `<timestamp>_active.jsonl` (atomic handoff)
3. **Agent Spawning**: Cortex detects new active file via watchdog and spawns agent process
4. **Event Emission**: Agents write JSON events to stdout (captured by Cortex)
5. **Event Storage**: Cortex appends events to the active JSONL file with timestamps
6. **Agent Completion**: Cortex renames file to `<timestamp>.jsonl` when agent finishes

### Key Components
- **File Watching**: Cortex uses watchdog to monitor for new `*_active.jsonl` files
- **Configuration Loading**: Cortex loads and merges persona configuration with request parameters
- **Process Management**: Spawns agent subprocesses via the `think-agents` command with merged configuration
- **Event Capture**: Monitors agent stdout/stderr and appends to JSONL files
- **Atomic Operations**: File renames provide race-free state transitions
- **NDJSON Input Mode**: Agent processes accept newline-delimited JSON via stdin containing the full merged configuration

### File States
- `<timestamp>_pending.jsonl`: Request written by client, awaiting processing
- `<timestamp>_active.jsonl`: Agent currently executing (Cortex is appending events)
- `<timestamp>.jsonl`: Agent completed (contains full history)

## Request Format

The first line of a request file must be a JSON object with `event: "request"`:

```json
{
  "event": "request",
  "ts": 1234567890123,              // Required: millisecond timestamp (must match filename)
  "prompt": "Analyze this code for security issues",  // Required: the task or question
  "backend": "openai",              // Required: openai, google, anthropic, or claude
  "persona": "default",              // Optional: agent persona from think/agents/*.txt
  "config": {                        // Optional: backend-specific configuration
    "model": "gpt-4o",              // Optional: model override
    "max_tokens": 8192,             // Optional: token limit
    "domain": "my-project"          // Required for Claude backend only
  },
  "save": "analysis.md",             // Optional: save result to file in day directory
  "day": "20250109",                  // Optional: YYYYMMDD format, defaults to current day
  "handoff": {                       // Optional: chain to another agent on completion
    "persona": "reviewer",
    "prompt": "Review the analysis",
    "backend": "openai"
  },
  "handoff_from": "1234567890122"   // Optional: present when spawned via handoff
}
```

## Agent Event Format

All subsequent lines are JSON objects with `event` and millisecond `ts` fields. The `ts` field is automatically added by Cortex if not provided by the backend. Additionally, Cortex automatically adds an `agent_id` field (matching the timestamp from the filename) to all events for tracking purposes.

### request
The initial spawn request (first line of file, written by client).
```json
{
  "event": "request",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "prompt": "User's task or question",
  "backend": "openai",
  "persona": "default",
  "config": {},
  "save": "output.md",
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
  "tool": "search_summaries",
  "args": {"query": "search terms", "limit": 10},
  "call_id": "search_summaries-1"
}
```

### tool_end
Emitted when a tool execution completes.
```json
{
  "event": "tool_end",
  "ts": 1234567890123,
  "agent_id": "1234567890123",
  "tool": "search_summaries",
  "args": {"query": "search terms"},
  "result": ["result", "array", "or", "object"],
  "call_id": "search_summaries-1"
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
    "backend": "openai"
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

## Agent Result Saving

When an agent completes successfully, its result can be automatically saved to a file in the journal's day directory.

- Include a `save` field in the request with the desired filename
- Optional `day` field specifies the target day in YYYYMMDD format (defaults to current day)
- The result from the `finish` event is written to `<journal>/<day>/<filename>`
- Saving occurs before any handoff processing
- Save failures are logged but don't interrupt the agent flow
- Commonly used for scheduled agents that generate daily reports (e.g., TODO.md)

## Agent Handoff

Agents can transfer control to other agents for specialized tasks. When an agent completes with a handoff configuration, Cortex automatically spawns the next agent in the chain.

- The `finish` event may include a `handoff` field specifying the next agent
- The subsequent request includes `handoff_from` with the originating agent ID
- Handoff agents automatically inherit the parent agent's configuration (backend, model, etc.) unless explicitly overridden
- This enables multi-step workflows and agent specialization with consistent configuration

## Agent Personas

Agents use persona configurations stored in the `think/agents/` directory. Each persona consists of:
- A `.txt` file containing system instructions and prompts
- An optional `.json` file with metadata and configuration

When spawning an agent:
1. Cortex loads the persona configuration using `get_agent()` from `think/utils.py`
2. The persona's instruction text and JSON metadata are merged into a complete configuration
3. Request parameters override persona defaults in the merged configuration
4. The full configuration (including instruction text) is passed to the agent process

Personas define specialized behaviors, tool usage patterns, and domain expertise. Available personas can be discovered using the `get_agents()` function or by listing files in the `think/agents/` directory.

### Persona Configuration Options

The `.json` file for a persona can include:
- `backend`: Default backend (openai, google, anthropic, claude)
- `model`: Default model name for the backend
- `max_tokens`: Maximum response token limit
- `tools`: MCP tools configuration (string or array)
  - String: Tool pack name (e.g., "default", "journal") - expanded via `get_tools()`
  - Array: Explicit list of tool names (e.g., `["search_summaries", "get_domain"]`)
  - If omitted, defaults to "default" pack

## MCP Tools Integration

The Model Context Protocol (MCP) provides tools for agent-journal interaction:

### Backend Support
- **OpenAI, Anthropic, Google**: Full MCP tool support via HTTP transport
- **Claude**: Uses filesystem tools instead; requires `domain` configuration in spawn request

### Tool Discovery
MCP tools are provided by the `think-mcp-tools --transport http` service, which:
- Writes its URI to `<journal>/agents/mcp.uri` for automatic discovery
- Exposes journal search and retrieval capabilities
- Available tools can be discovered via the MCP service endpoint

## Agent Backends

The system supports multiple AI backends, each implementing the same event interface:

- **OpenAI** (`think/openai.py`): GPT models with OpenAI Agents SDK
- **Google** (`think/google.py`): Gemini models with Google AI SDK
- **Anthropic** (`think/anthropic.py`): Claude models with Anthropic SDK
- **Claude** (`think/claude.py`): Claude models via Claude Code SDK
  - Uses filesystem tools (Read, Write, Edit, etc.) instead of MCP
  - Requires `domain` configuration specifying journal domain directory
  - Operates within domain-scoped file permissions

All backends:
- Emit JSON events to stdout (one per line)
- Are spawned as subprocesses by Cortex
- Use consistent event structures across providers
- Process events are written to stdout for Cortex to capture

## Process Management

The `think-supervisor` command provides process management for the Cortex ecosystem:
- Starts and monitors the Cortex file watcher service
- Starts and monitors the MCP tools HTTP server
- Handles process restarts on failure
- Monitors system health indicators

This is distinct from agent lifecycle management, which Cortex handles internally through file state transitions.

