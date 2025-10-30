# Callosum Message Specification

Callosum is a WebSocket broadcast bus for real-time event distribution across Sunstone services.

**Endpoint:** `ws://localhost:PORT/callosum`

**Protocol:** All messages are broadcast to all connected clients. No routing, no filtering.

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

All other fields are tract-specific and documented below.

# `"tract": "cortex"`

Agent execution events from the Muse cortex service.

**Event types:** `request`, `start`, `thinking`, `tool_start`, `tool_end`, `finish`, `error`, `agent_updated`

**Notes:**
- `usage` field on `finish` contains token statistics when available
- `error` events include optional `trace` (string) and `exit_code` (integer)
