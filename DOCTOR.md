# Sunstone Diagnostic Guide

Quick reference for debugging and diagnosing issues. For detailed specifications, see linked documentation.

## Prerequisites

Always get the journal path first:

```bash
export JOURNAL_PATH=$(grep JOURNAL_PATH .env | cut -d= -f2)
```

---

## Quick Health Check

```bash
# Check if supervisor services are running
pgrep -af "observe-gnome|observe-sense|think-supervisor"

# Check heartbeat freshness (should be recent)
ls -la $JOURNAL_PATH/health/*.up

# Check Callosum socket exists
ls -la $JOURNAL_PATH/health/callosum.sock

# Check for stuck agents (should be empty or short-lived)
ls $JOURNAL_PATH/$(date +%Y%m%d)/agents/*_active.jsonl 2>/dev/null
```

**Healthy state:**
- All three processes running
- `.up` files modified within last 60 seconds
- `callosum.sock` exists
- No `_active.jsonl` files older than a few minutes

---

## Service Architecture

The supervisor (`think-supervisor`) manages these services:

| Service | Command | Purpose | Auto-restart |
|---------|---------|---------|--------------|
| Callosum | (in-process) | Message bus for inter-service events | No |
| Observer | `observe-gnome` | Screen/audio capture | Yes |
| Sense | `observe-sense` | File detection, processing dispatch | Yes |

Cortex (agent execution) connects to Callosum but runs independently via `muse-cortex`.

See [CALLOSUM.md](CALLOSUM.md) for message protocol and [CORTEX.md](CORTEX.md) for agent system.

---

## Log Locations

| What | Where |
|------|-------|
| Current service logs | `$JOURNAL_PATH/health/{service}.log` (symlinks) |
| Day's process logs | `$JOURNAL_PATH/{YYYYMMDD}/health/{ref}_{name}.log` |
| Agent execution | `$JOURNAL_PATH/{YYYYMMDD}/agents/*.jsonl` |
| Journal task log | `$JOURNAL_PATH/task_log.txt` |

**Symlink structure:** Journal-level symlinks point to current day's logs. Day-level symlinks point to current process instance (by ref).

```bash
# Tail current observer log
tail -f $JOURNAL_PATH/health/observe-gnome.log

# Find today's logs
ls -la $JOURNAL_PATH/$(date +%Y%m%d)/health/
```

---

## Health Signals

### Heartbeat Files

| File | Updated by | Meaning |
|------|------------|---------|
| `health/see.up` | Observer | Screen capture active |
| `health/hear.up` | Observer | Audio capture active |

Staleness threshold: 60 seconds (configurable). Supervisor checks these and alerts if stale.

### Callosum Status Events

Services emit periodic status to Callosum (every 5 seconds when active):

- `observe.status` - Capture state (screencast, audio, activity)
- `cortex.status` - Running agents list
- `supervisor.status` - Service health, stale heartbeats

See [CALLOSUM.md](CALLOSUM.md) Tract Registry for event schemas.

---

## Reading Agent Files

**Location:** `$JOURNAL_PATH/{YYYYMMDD}/agents/`

**File states:**
- `{timestamp}_active.jsonl` - Agent currently running
- `{timestamp}.jsonl` - Agent completed

**Event sequence** (JSONL, one event per line):

1. `request` - Initial spawn request (prompt, backend, persona)
2. `start` - Agent began execution (model info)
3. `tool_start`/`tool_end` - Tool calls (paired by `call_id`)
4. `thinking` - Model reasoning (if supported)
5. `finish` or `error` - Final result or failure

```bash
# View an agent's final result
jq -r 'select(.event=="finish") | .result' $JOURNAL_PATH/$(date +%Y%m%d)/agents/1234567890123.jsonl

# List today's agents with their prompts
for f in $JOURNAL_PATH/$(date +%Y%m%d)/agents/*.jsonl; do
  echo "=== $(basename $f) ==="
  head -1 "$f" | jq -r '.prompt[:80]'
done
```

See [CORTEX.md](CORTEX.md) for complete event schemas and agent configuration.

---

## Common Issues

### Observer not capturing

```bash
# Check heartbeats
ls -la $JOURNAL_PATH/health/*.up

# Check observer log for errors
tail -50 $JOURNAL_PATH/health/observe-gnome.log | grep -i error
```

Causes: DBus issues, screencast permissions, audio device unavailable.

### Agent appears stuck

```bash
# Find active agents
ls -la $JOURNAL_PATH/$(date +%Y%m%d)/agents/*_active.jsonl

# Check last event in active agent
tail -1 $JOURNAL_PATH/$(date +%Y%m%d)/agents/*_active.jsonl | jq .
```

Causes: Backend timeout, tool hanging, network issues.

### No Callosum events

```bash
# Verify socket exists
ls -la $JOURNAL_PATH/health/callosum.sock

# Check supervisor is running
pgrep -af think-supervisor
```

Causes: Supervisor not started, socket path permissions.

### Processing backlog

```bash
# Check sense log for queue status
grep -i "queue" $JOURNAL_PATH/health/observe-sense.log | tail -10
```

Causes: Slow transcription, describe API rate limits.

---

## Useful Commands

```bash
# Watch all service logs
tail -f $JOURNAL_PATH/health/*.log

# Count today's agents by status
echo "Completed: $(ls $JOURNAL_PATH/$(date +%Y%m%d)/agents/*.jsonl 2>/dev/null | grep -v _active | wc -l)"
echo "Running: $(ls $JOURNAL_PATH/$(date +%Y%m%d)/agents/*_active.jsonl 2>/dev/null | wc -l)"

# Find agents that errored today
grep -l '"event":"error"' $JOURNAL_PATH/$(date +%Y%m%d)/agents/*.jsonl

# Check token usage for today
wc -l $JOURNAL_PATH/tokens/$(date +%Y%m%d).jsonl

# Find errors in today's logs
grep -i error $JOURNAL_PATH/$(date +%Y%m%d)/health/*.log

# Watch Callosum events in real-time
socat - UNIX-CONNECT:$JOURNAL_PATH/health/callosum.sock
```

---

## See Also

- [JOURNAL.md](JOURNAL.md) - Directory structure and file formats
- [CORTEX.md](CORTEX.md) - Agent system, events, personas
- [CALLOSUM.md](CALLOSUM.md) - Message bus protocol
- [CRUMBS.md](CRUMBS.md) - Transcript format
