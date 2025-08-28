# Cortex API and Eventing

The Cortex system provides a WebSocket API for spawning and managing actively running AI agent instances. It acts as a process manager and event relay for live agent executions only. Historical agent data is stored in `<journal>/agents/<timestamp>.jsonl` files and should be accessed directly from those files.

## Service Discovery

When services start up, they write their active URIs to files in the journal's `agents/` directory for automated discovery:

- `think-cortex` writes its WebSocket URI to `<journal>/agents/cortex.uri` (default: `ws://127.0.0.1:2468/ws/cortex`)
- `think-mcp-tools --transport http` writes to `<journal>/agents/mcp.uri` (default: `http://127.0.0.1:6270/mcp/`)

Use the `JOURNAL_PATH` environment variable to locate the journal and these URI files.

## Architecture

### Event Flow
1. **Agent Spawning**: Cortex spawns agent processes via `python -m think.agents`
2. **Event Emission**: Agents write JSON events to stdout (captured by Cortex)
3. **Event Storage**: Cortex writes events to `<journal>/agents/<timestamp>.jsonl`
4. **Event Relay**: Cortex broadcasts events to attached WebSocket clients
5. **Agent Completion**: When agents finish, they are removed from Cortex memory

### Key Components
- **RunningAgent**: In-memory representation of active agent processes with event buffer
- **Event Monitoring**: Cortex monitors agent stdout/stderr and manages the lifecycle
- **No Historical Support**: Cortex only tracks running agents; finished agents exist only in `.jsonl` files
- **NDJSON Input Mode**: Agent processes accept newline-delimited JSON via stdin for batch processing

## WebSocket API

Connect to the URI and send JSON messages with an `action` field:

### List Agents
Request:
```json
{"action": "list", "limit": 10, "offset": 0}
```
Response (only running agents):
```json
{
  "type": "agent_list",
  "agents": [
    {
      "id": "1234567890123",
      "status": "running",
      "started_at": 1234567890123,
      "pid": 12345,
      "metadata": {}
    }
  ],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 2,
    "has_more": false
  }
}
```

### Attach to Agent
Request:
```json
{"action": "attach", "agent_id": "1234567890123"}
```
Response (only for running agents):
```json
{"type": "attached", "agent_id": "1234567890123"}
```
Streams in-memory buffered events and live updates. Returns error if agent is not running.

### Detach from Agent
Request:
```json
{"action": "detach"}
```
Response:
```json
{"type": "detached"}
```

### Spawn New Agent
Request:
```json
{
  "action": "spawn",
  "prompt": "Analyze this code for security issues",
  "backend": "openai",
  "persona": "default",
  "config": {
    "model": "gpt-4o",
    "max_tokens": 8192,
    "domain": "my-project"  // Required for Claude backend
  },
  "handoff": {
    "persona": "reviewer",
    "prompt": "Review the analysis"
  }
}
```
Response:
```json
{"type": "agent_spawned", "agent_id": "1234567890123"}
```

Cortex immediately:
1. Creates the `.jsonl` file with a start event
2. Spawns the agent subprocess
3. Monitors stdout/stderr for events
4. Writes events to both the `.jsonl` file and broadcasts to watchers

During an active attachment, the server streams:

- `{"type": "agent_event", "agent_id": "123", "event": {...}}` for each agent JSON event
- `{"type": "agent_finished", "agent_id": "123"}` when an agent exits
- `{"type": "error", "message": "..."}` for protocol errors

## History Management

Agent history is stored exclusively in `<journal>/agents/<timestamp>.jsonl` files:

- **Running Agents**: Cortex captures stdout from agent processes and writes events to `.jsonl` files in real-time
- **In-Memory Buffer**: Running agents maintain an in-memory event buffer for fast replay to new attachments
- **Agent Completion**: Once finished, agents are removed from Cortex memory entirely
- **Historical Access**: Finished agent data must be accessed directly from `.jsonl` files, not through Cortex
- **Event Sources**: 
  - Agent stdout → parsed as JSON events
  - Agent stderr → converted to error events
  - Non-JSON output → wrapped as info events
  - Process exit → generates finish/error event

## Agent Event Format

All agent events are JSON objects with `event` and millisecond `ts` fields. The `ts` field is automatically added if not provided by the backend. Backends (OpenAI, Gemini, Claude) emit the following event types:

### start
Emitted when an agent run begins.
```json
{
  "event": "start",
  "ts": 1234567890123,
  "prompt": "User's request text",
  "persona": "default",
  "model": "gpt-4o",
  "handoff_from": "1234567890122"  // Present when spawned via handoff
}
```

### tool_start
Emitted when a tool execution begins.
```json
{
  "event": "tool_start",
  "ts": 1234567890123,
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
  "agent": "SpecializedAgent"
}
```

### finish
Emitted when the agent run completes successfully.
```json
{
  "event": "finish",
  "ts": 1234567890123,
  "result": "Final response text to the user",
  "handoff": {  // Optional: triggers next agent
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

## Agent Handoff

Agents can transfer control to other agents for specialized tasks. When an agent completes with a handoff configuration, Cortex automatically spawns the next agent in the chain.

- The `finish` event may include a `handoff` field specifying the next agent
- The subsequent `start` event includes `handoff_from` with the originating agent ID
- This enables multi-step workflows and agent specialization

## Agent Personas

Agents use persona configurations stored in the `think/agents/` directory. Each persona consists of:
- A `.txt` file containing system instructions and prompts
- An optional `.json` file with metadata and configuration

Personas define specialized behaviors, tool usage patterns, and domain expertise. Available personas can be discovered by listing files in the `think/agents/` directory.

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

## Frontend Integration

The Dream web app (`dream/views/chat.py`) connects to Cortex via WebSocket to:
1. Spawn new agents in response to user queries
2. Receive real-time events during agent execution
3. Display tool usage, thinking summaries, and results in the chat interface

Note: Historical agent runs must be accessed directly from `.jsonl` files, not through Cortex.

Events are forwarded through the WebSocket with type `agent_event`:
```json
{
  "type": "agent_event",
  "event": {
    "event": "tool_start",
    "tool": "search_summaries",
    "args": {"query": "example"},
    "call_id": "search_summaries-1",
    "ts": 1234567890123
  }
}
```

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
- Starts and monitors the Cortex WebSocket server
- Starts and monitors the MCP tools HTTP server
- Handles process restarts on failure
- Monitors system health indicators

This is distinct from agent lifecycle management, which Cortex handles internally

