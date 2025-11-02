# Callosum Protocol

Callosum is a JSON-per-line message bus for real-time event distribution across Sunstone services.

## Protocol

**Transport:** Unix domain socket at `$JOURNAL_PATH/health/callosum.sock`

**Format:** Newline-delimited JSON. Broadcast to all connected clients.

**Message Structure:**
```json
{
  "tract": "source_subsystem",
  "event": "event_type",
  "ts": 1234567890123,
  // ... tract-specific fields
}
```

**Required Fields:**
- `tract` - Source subsystem identifier (string)
- `event` - Event type within tract (string)
- `ts` - Timestamp in milliseconds (auto-added if missing)

**Behavior:**
- All connections are bidirectional (can emit and receive)
- No routing, no filtering - all messages broadcast to all clients
- Clients should drain socket continuously to prevent backpressure

---

## Tract Registry

> **Note:** This registry is kept intentionally high-level. For detailed event schemas, field specifications, and usage patterns, always refer to the source files listed for each tract.

### `cortex` - Agent execution events
**Source:** `muse/cortex.py`
**Events:** `request`, `start`, `thinking`, `tool_start`, `tool_end`, `finish`, `error`, `agent_updated`, `info`
**Details:** See [CORTEX.md](CORTEX.md) for agent lifecycle, personas, and event schemas

### `supervisor` - Process lifecycle management
**Source:** `think/supervisor.py`
**Events:** `request`, `started`, `stopped`, `restarting`, `status`
**Fields:** `ref`, `service`, `cmd`, `pid`, `exit_code`
**Purpose:** Unified lifecycle events for all supervised processes (services and tasks)

### `logs` - Process output streaming
**Source:** `think/runner.py`
**Events:** `exec`, `line`, `exit`
**Fields:** `ref`, `name`, `pid`, `cmd`, `stream`, `line`, `exit_code`, `duration_ms`, `log_path`
**Purpose:** Real-time stdout/stderr streaming and process exit events

### `observe` - Multimodal capture processing events
**Source:** `observe/sense.py`, `observe/describe.py`, `observe/transcribe.py`, `observe/reduce.py`
**Events:** `detected`, `described`, `transcribed`, `reduced`
**Fields:**
- `detected`: `file`, `handler`, `ref` - File detected and handler spawned
- `described`/`transcribed`/`reduced`: `input`, `output`, `duration_ms` - Processing complete
**Purpose:** Track observation pipeline from file detection through processing completion
**Path Format:** Relative to `JOURNAL_PATH` (e.g., `20251102/163045_screen.webm`, `20251102/seen/163045_screen.webm`)
**Correlation:** `detected.ref` matches `logs.exec.ref` for the same handler process

---

## Key Concepts

**Correlation ID (`ref`):** Universal identifier for process instances, used across all tracts to correlate events. Auto-generated as epoch milliseconds if not provided by client.

**Field Semantics:**
- `service` - Human-readable name (e.g., "cortex", "think-importer")
- `ref` - Unique instance ID (changes on each restart)
- `pid` - Operating system process ID

---

## Implementation

**Client Library:** `think/callosum.py` `CallosumConnection` class
**Server:** `think/callosum.py` `CallosumServer` class

See code documentation for usage patterns and examples.
