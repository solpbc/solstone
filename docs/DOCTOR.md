# solstone Diagnostic Guide

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
pgrep -af "observer|observe-sense|think-supervisor"

# Check Callosum socket exists
ls -la $JOURNAL_PATH/health/callosum.sock

# Check for stuck agents (should be empty or short-lived)
ls $JOURNAL_PATH/$(date +%Y%m%d)/agents/*_active.jsonl 2>/dev/null
```

**Healthy state:**
- All three processes running
- `callosum.sock` exists
- `supervisor.status` events show no stale heartbeats
- No `_active.jsonl` files older than a few minutes

---

## Service Architecture

The supervisor (`think-supervisor`) manages these services:

| Service | Command | Purpose | Auto-restart |
|---------|---------|---------|--------------|
| Callosum | (in-process) | Message bus for inter-service events | No |
| Observer | `observer` | Screen/audio capture (platform-detected) | Yes |
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
tail -f $JOURNAL_PATH/health/observer.log

# Find today's logs
ls -la $JOURNAL_PATH/$(date +%Y%m%d)/health/
```

---

## Health Signals

Health uses a **fail-fast model**: observers exit if they detect problems, and supervisor restarts them. Health is simply whether the observer is running and sending status events.

| Signal | Healthy when | Stale when |
|--------|--------------|------------|
| `hear` | Status received within threshold | No status for 60+ seconds |
| `see` | Status received within threshold | No status for 60+ seconds |

Both signals track the same thing: is the observer alive and communicating? If the observer has capture problems (e.g., screencast files not growing), it exits gracefully and supervisor restarts it.

Staleness threshold: 60 seconds (configurable via `--threshold`).

### Callosum Status Events

Services emit periodic status to Callosum (every 5 seconds when active):

- `observe.status` - Capture state (screencast, audio, activity, files_growing)
- `cortex.status` - Running agents list
- `supervisor.status` - Service health, stale heartbeats

The supervisor checks for `observe.status` event freshness and includes `stale_heartbeats` in its own status.

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
# Check observer log for errors
tail -50 $JOURNAL_PATH/health/observer.log | grep -i error

# Check if observer is emitting status (supervisor.status will show stale_heartbeats)
# Health is derived from observe.status Callosum events
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
