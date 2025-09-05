# Cortex API and Eventing

The Cortex system manages AI agent execution through a file-based architecture. It acts as a process manager for agent instances, monitoring the journal's `agents/` directory for new requests and appending execution events to JSONL files.

## Architecture

### Event Flow
1. **Request Creation**: Client writes spawn request to `<timestamp>_pending.jsonl`
2. **Agent Activation**: Client renames file to `<timestamp>_active.jsonl` (atomic handoff)
3. **Agent Spawning**: Cortex detects new active file via inotify and spawns agent process
4. **Event Emission**: Agents write JSON events to stdout (captured by Cortex)
5. **Event Storage**: Cortex appends events to the active JSONL file with timestamps
6. **Agent Completion**: Cortex renames file to `<timestamp>.jsonl` when agent finishes

### Key Components
- **File Watching**: Cortex uses inotify to monitor for new `*_active.jsonl` files
- **Process Management**: Spawns agent subprocesses via `python -m think.agents`
- **Event Capture**: Monitors agent stdout/stderr and appends to JSONL files
- **Atomic Operations**: File renames provide race-free state transitions
- **NDJSON Input Mode**: Agent processes accept newline-delimited JSON via stdin for batch processing

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
  "handoff": {                       // Optional: chain to another agent on completion
    "persona": "reviewer",
    "prompt": "Review the analysis",
    "backend": "openai"
  },
  "handoff_from": "1234567890122"   // Optional: present when spawned via handoff
}
```

## Agent Event Format

All subsequent lines are JSON objects with `event` and millisecond `ts` fields. The `ts` field is automatically added by Cortex if not provided by the backend.

### request
The initial spawn request (first line of file, written by client).
```json
{
  "event": "request",
  "ts": 1234567890123,
  "prompt": "User's task or question",
  "backend": "openai",
  "persona": "default",
  "config": {},
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
- The subsequent request includes `handoff_from` with the originating agent ID
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

