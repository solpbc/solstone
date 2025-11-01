# Callosum Message Specification

Callosum is a Unix domain socket broadcast bus for real-time event distribution across Sunstone services.

**Socket:** `$JOURNAL_PATH/health/callosum.sock`

**Protocol:** JSON-per-line broadcast. No routing, no filtering.

## Message Format

All messages follow this structure:

```json
{
  "tract": "string",
  "event": "string",
  // ... tract-specific fields
}
```

**Required fields:**
- `tract` - Source service/subsystem identifier
- `event` - Event type within the tract

**Auto-added fields:**
- `ts` - Timestamp in milliseconds (added by server if not present)

All other fields are tract-specific and documented below.

## Client Usage

**Unified bidirectional connections:**

All connections can both emit and receive messages. A background receive loop always runs to drain the socket buffer.

**Emit-only usage (no callback):**
```python
from think.callosum import CallosumConnection

client = CallosumConnection()
client.emit("cortex", "agent_start", agent_id="123", persona="analyst")
client.close()
```

**Listen-only usage (with callback):**
```python
from think.callosum import CallosumConnection

def handle_message(msg):
    print(f"Received: {msg}")

listener = CallosumConnection(callback=handle_message)
listener.connect()  # Starts background receive loop
# Messages are now processed in background
# ...
listener.close()  # Stops receive loop
```

**Both emit and receive:**
```python
from think.callosum import CallosumConnection

messages_received = []
conn = CallosumConnection(callback=lambda msg: messages_received.append(msg))
conn.emit("cortex", "status", message="active")  # Auto-connects
# Connection drains broadcasts in background
conn.close()
```

## Notes

- Messages are JSON objects, one per line
- All connections are bidirectional (emit + receive)
- Background receive loop prevents TCP backpressure
- Clients auto-reconnect on failure
- Missing `tract` or `event` fields are rejected

---

# Tract Specifications

## `"tract": "cortex"`

Agent execution events from the Muse cortex service.

For detailed cortex tract implementation, agent lifecycle management, personas, MCP tools, backends, scheduling, and usage patterns, see [CORTEX.md](CORTEX.md).

**Event types:** `request`, `start`, `thinking`, `tool_start`, `tool_end`, `finish`, `error`, `agent_updated`

**Common fields (all events):**
- `agent_id` - Unique agent identifier (timestamp-based)
- `ts` - Timestamp in milliseconds

**Event-specific fields:**

### `request`
Agent request sent by client via `cortex_request()`.

Required:
- `prompt` - Task prompt for the agent
- `persona` - Agent persona name (e.g., "default", "analyst")
- `backend` - AI backend (openai, google, anthropic)

Optional:
- `handoff_from` - Previous agent ID if this is a handoff
- `save` - Filename to save result to in day directory
- `model` - Model name override
- `max_tokens` - Token limit override
- `timeout_seconds` - Timeout in seconds (default 600)
- Additional backend-specific configuration

### `start`
Agent process has started executing.

- `prompt` - The task prompt
- `persona` - Agent persona name
- `model` - Model being used
- `backend` - AI backend

### `thinking`
Agent reasoning/thinking summary (for models that support extended thinking).

- `summary` - Thinking summary text
- `model` - Model name (optional)

### `tool_start`
Agent is starting a tool call.

- `tool` - Name of the tool being called
- `call_id` - Unique identifier for this tool call
- `args` - Tool arguments (optional)

### `tool_end`
Agent tool call has completed.

- `tool` - Name of the tool
- `call_id` - Matches the call_id from tool_start
- `args` - Tool arguments (optional)
- `result` - Tool result

### `finish`
Agent has completed successfully.

- `result` - Agent output/result text
- `usage` - Token usage statistics (optional)
  - `input_tokens` - Tokens in input
  - `output_tokens` - Tokens in output
  - `total_tokens` - Total tokens used
- `conversation_id` - OpenAI conversation ID (optional, for continuations)

### `error`
Agent encountered an error.

- `error` - Error message
- `trace` - Stack trace (optional)
- `exit_code` - Process exit code (optional)

### `agent_updated`
Agent context or state has changed.

- `agent` - Agent description or identifier

### `info`
Non-JSON output from agent process (captured as info event).

- `message` - The info message text

## `"tract": "indexer"` (future)

Database indexing events from the think.indexer service.

**Event types:** `scan_start`, `scan_progress`, `scan_complete`, `scan_error`

**Example fields:**
- `index_type` - Type of index (transcripts, events, summaries, entities, news)
- `changes` - Number of changes detected
- `day` - Day being indexed (YYYYMMDD format)

## `"tract": "supervisor"` (future)

Process supervision events from think.supervisor.

**Event types:** `process_start`, `process_exit`, `heartbeat_stale`, `heartbeat_ok`

**Example fields:**
- `process_name` - Name of the managed process
- `exit_code` - Exit code for process_exit events
- `pid` - Process ID

## `"tract": "observe"` (future)

Observation events from the observe subsystem.

**Event types:** `capture_start`, `capture_complete`, `vad_activity`, `vad_silence`

**Example fields:**
- `source` - Capture source (screen, mic, etc.)
- `file_path` - Path to captured file
