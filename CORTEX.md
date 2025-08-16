# Cortex API and Eventing

## Service Discovery

`think-cortex` writes its WebSocket URI to `<journal>/agents/cortex.uri`. The
file contains a single URI such as `ws://127.0.0.1:2468/ws/cortex`. Use the
`JOURNAL_PATH` environment variable to locate the journal.

## WebSocket API

Connect to the URI and send JSON messages with an `action` field:

- `{"action": "list", "limit": 10, "offset": 0}` →
  `{"type": "agent_list", "agents": [...], "pagination": {...}}`
- `{"action": "attach", "agent_id": "123"}` →
  `{"type": "attached", "agent_id": "123"}` followed by historical
  `agent_event` messages
- `{"action": "detach"}` → `{"type": "detached"}`
- `{"action": "spawn", "prompt": "hi", "backend": "openai",
  "model": "gpt", "persona": "default", "max_tokens": 0}` →
  `{"type": "agent_spawned", "agent_id": "123"}`

During an active attachment, the server streams:

- `{"type": "agent_event", "agent_id": "123", "event": ...}` for each
  agent JSON event
- `{"type": "agent_finished", "agent_id": "123"}` when an agent exits
- `{"type": "error", "message": "..."}` for protocol errors

## Agent Event Format

All agent events are JSON objects with `event` and millisecond `ts` fields.
Backends (OpenAI, Gemini, Claude) emit the following event types:

- **start** – `prompt`, `persona`, `model`
- **agent_updated** – `agent`
- **tool_start** – `tool`, optional `args`, optional `call_id`
- **tool_end** – `tool`, optional `args`, `result`, optional `call_id`
- **thinking** – `summary`, optional `model`
- **finish** – `result`
- **error** – `error`, optional `trace`

`call_id` links a tool's start and end events. All events include `ts` even
when not supplied by the backend.

