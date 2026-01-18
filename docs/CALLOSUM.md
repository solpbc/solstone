# Callosum Protocol

Callosum is a JSON-per-line message bus for real-time event distribution across solstone services.

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
**Events:** `request`, `start`, `thinking`, `tool_start`, `tool_end`, `finish`, `error`, `agent_updated`, `info`, `status`
**Details:** See [CORTEX.md](CORTEX.md) for agent lifecycle, personas, and event schemas

### `supervisor` - Process lifecycle management
**Source:** `think/supervisor.py`
**Events:** `started`, `stopped`, `restarting`, `status`
**Listens for:** `request` (task spawn), `restart` (service restart)
**Fields:** `ref`, `service`, `cmd`, `pid`, `exit_code`
**Purpose:** Unified lifecycle events for all supervised processes (services and tasks)

### `logs` - Process output streaming
**Source:** `think/runner.py`
**Events:** `exec`, `line`, `exit`
**Fields:** `ref`, `name`, `pid`, `cmd`, `stream`, `line`, `exit_code`, `duration_ms`, `log_path`
**Purpose:** Real-time stdout/stderr streaming and process exit events

### `observe` - Multimodal capture processing events
**Source:** `observe/observer.py` (delegates to `observe/linux/observer.py` or `observe/macos/observer.py`), `observe/sense.py`, `observe/describe.py`, `observe/transcribe.py`
**Events:** `status`, `observing`, `detected`, `described`, `transcribed`, `observed`
**Fields:**
- `status`: Periodic state (every 5s while running)
  - From `observer.py`: `screencast`, `audio`, `activity` - Live capture state (for UI/debugging)
  - From `sense.py`: `describe`, `transcribe` - Processing pipeline state (with `running`/`queued` sub-fields)
- `observing`: `day`, `segment`, `files` - Recording window boundary crossed with saved files
  - Remote events include `remote` (remote name) from `apps/remote/routes.py`
- `detected`: `day`, `segment`, `file`, `handler`, `ref`, `remote` - File detected and handler spawned
- `described`/`transcribed`: `day`, `segment`, `input`, `output`, `duration_ms`, `remote` - Processing complete
- `observed`: `day`, `segment`, `duration` - All files for segment fully processed
  - Batch mode (--day) events include `batch=true` to indicate non-live origin
  - Remote events include `remote` (remote name)
- Observer events (`status`, `observing`) include `host` (hostname) and `platform` ("linux"/"darwin") for multi-host support
**Purpose:** Track observation pipeline from live capture state through processing completion
**Health Model:** Fail-fast - observers exit if capture process dies. Supervisor checks event freshness only.
**Path Format:** Relative to `JOURNAL_PATH` (e.g., `20251102/163045_300_center_DP-3_screen.webm` for multi-monitor recordings)
**Correlation:** `detected.ref` matches `logs.exec.ref` for the same handler process; `observed.segment` groups all files from same capture window
**Event Log:** Any observe event with `day` + `segment` fields is logged to `<day>/<segment>/events.jsonl` by supervisor (if directory exists)

### `importer` - Media import and transcription processing
**Source:** `think/importer.py`
**Events:** `started`, `status`, `completed`, `error`
**Fields:**
- `started`: `import_id`, `input_file`, `file_type`, `day`, `facet`, `setting`, `options`, `stage`
- `status`: `import_id`, `stage`, `elapsed_ms`, `stage_elapsed_ms` - Periodic progress (every 5s)
- `completed`: `import_id`, `stage`, `duration_ms`, `total_files_created`, `output_files`, `metadata_file`, `stages_run`
- `error`: `import_id`, `stage`, `error`, `duration_ms`, `partial_outputs`
**Stages:** `initialization`, `transcribing`, `segmenting`, `summarizing`
**Purpose:** Track media file import and transcription progress from start to completion
**Correlation:** `import_id` correlates all events for a single import operation

### `dream` - Dream processing lifecycle
**Source:** `think/dream.py`
**Events:** `started`, `command`, `insights_completed`, `agents_started`, `group_started`, `group_completed`, `agents_completed`, `completed`
**Fields:**
- `mode` - Processing mode: "daily" or "segment"
- `day` - Day being processed (YYYYMMDD)
- `segment` - Segment key (only present when mode="segment")
- `command`, `index`, `total` - Command execution details
- `priority`, `count`, `completed`, `timed_out` - Agent group details
- `success`, `failed`, `duration_ms` - Phase completion metrics
**Purpose:** Track dream processing from insights through scheduled agents
**Correlation:** `day` + `segment` (when present) correlates all events for a single processing run

---

## Key Concepts

**Correlation ID (`ref`):** Universal identifier for process instances, used across all tracts to correlate events. Auto-generated as epoch milliseconds if not provided by client.

**Field Semantics:**
- `service` - Human-readable name (e.g., "cortex", "sol import")
- `ref` - Unique instance ID (changes on each restart)
- `pid` - Operating system process ID

---

## Implementation

**Client Library:** `think/callosum.py` `CallosumConnection` class
**Server:** `think/callosum.py` `CallosumServer` class

**Convey Integration:**
- `convey.emit()` - Non-blocking event emission from route handlers (uses shared bridge connection)
- `apps.events` - Server-side event handlers via `@on_event` decorator (dispatched in thread pool)

See [APPS.md](APPS.md) for app event handler documentation and code for usage patterns.
