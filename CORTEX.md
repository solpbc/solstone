# Cortex API and Eventing

The Cortex system provides a WebSocket API for managing and monitoring AI agent instances. Agents emit structured JSON events during execution that are stored in `<journal>/agents/<timestamp>.jsonl` files and can be consumed via the WebSocket API or directly from the files.

## Service Discovery

When services start up, they write their active URIs to files in the journal's `agents/` directory for automated discovery:

- `think-cortex` writes its WebSocket URI to `<journal>/agents/cortex.uri` (default: `ws://127.0.0.1:2468/ws/cortex`)
- `think-mcp-tools --transport http` writes to `<journal>/agents/mcp.uri` (default: `http://127.0.0.1:6270/mcp/`)

Use the `JOURNAL_PATH` environment variable to locate the journal and these URI files.

## WebSocket API

Connect to the URI and send JSON messages with an `action` field:

### List Agents
Request:
```json
{"action": "list", "limit": 10, "offset": 0}
```
Response:
```json
{
  "type": "agent_list",
  "agents": [
    {
      "id": "1234567890123",
      "status": "finished",
      "started_at": 1234567890123,
      "metadata": {
        "prompt": "User's request",
        "persona": "default",
        "model": "gpt-4o"
      }
    }
  ],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 42,
    "has_more": true
  }
}
```

### Attach to Agent
Request:
```json
{"action": "attach", "agent_id": "1234567890123"}
```
Response:
```json
{"type": "attached", "agent_id": "1234567890123"}
```
Followed by historical `agent_event` messages replaying the agent's execution.

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
  "model": "gpt-4o",
  "persona": "default",
  "max_tokens": 8192
}
```
Response:
```json
{"type": "agent_spawned", "agent_id": "1234567890123"}
```

During an active attachment, the server streams:

- `{"type": "agent_event", "agent_id": "123", "event": {...}}` for each agent JSON event
- `{"type": "agent_finished", "agent_id": "123"}` when an agent exits
- `{"type": "error", "message": "..."}` for protocol errors

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
  "result": "Final response text to the user"
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

## Tool Call Tracking

Tool events use `call_id` to pair `tool_start` and `tool_end` events. This allows tracking:
- Which tools are currently running
- Tool execution duration
- Tool inputs and outputs
- Concurrent tool executions

The frontend uses this to show real-time status updates as tools execute, changing from "running..." to "✓" when complete.

## Frontend Integration

The Dream web app (`dream/views/chat.py`) connects to Cortex via WebSocket to:
1. Spawn new agents in response to user queries
2. Receive real-time events during agent execution
3. Display tool usage, thinking summaries, and results in the chat interface
4. Allow viewing historical agent runs by attaching to completed agents

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

All backends emit consistent event structures, ensuring uniform behavior across different AI providers.

