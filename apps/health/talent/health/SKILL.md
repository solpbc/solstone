---
name: health
description: >
  Diagnose solstone service health, inspect agent run logs, and check system
  status. View service uptimes, crashes, queue depths, recent errors, and
  agent run costs. Includes a journal layout reference for navigating data
  files. Use when the owner reports issues, asks about service health, agent
  costs, pipeline status, or when troubleshooting capture gaps and processing
  failures.
  TRIGGER: health, status, is it running, something broke, service down,
  errors, agent runs, costs, logs, pipeline, diagnostics, system check.
---

# Health CLI Skill

Use these commands to check service health, view logs, and inspect agent runs.

**Typical workflow**: `sol health` to check service status → `sol health logs` to inspect recent log output → `sol talent logs` to review agent runs → `sol talent log <ID>` for run details.

## status

```bash
sol health
```

Show current supervisor status: running services (names, PIDs, uptimes), crashed services, active tasks, queue depths, heartbeat health, and callosum client count.

Connects to `journal/health/callosum.sock` with a 10-second timeout.

Example:

```bash
sol health
```

## logs

```bash
sol health logs [-c N] [-f] [--since TIME] [--service NAME] [--grep PATTERN]
```

View service health logs from today's log files.

- `-c N`: lines per service (default `5`).
- `-f`: follow mode — tail all logs continuously.
- `--since TIME`: filter by time. Accepts relative (`30m`, `2h`, `1d`) or absolute (`4pm`, `16:00`).
- `--service NAME`: filter to one service.
- `--grep PATTERN`: filter lines matching a Python regex.

Behavior notes:

- Reads symlinked logs from `journal/YYYYMMDD/health/*.log`.
- Includes `journal/health/supervisor.log` when no filters are active.
- Log line format: `ISO8601 [service:stream] LEVEL:logger:message`.
- `-f` mode handles symlink target rotation at midnight.

Examples:

```bash
sol health logs
sol health logs -c 20 --service cortex
sol health logs --since 30m --grep "ERROR"
sol health logs -f
```

## agent runs

```bash
sol talent logs [AGENT] [-c COUNT] [--day YYYYMMDD] [--daily] [--errors] [--summary]
```

List recent agent runs.

- `AGENT`: optional agent name filter.
- `-c, --count`: max runs shown (default `20`; `50` when `--daily`).
- `--day YYYYMMDD`: show only runs from a specific day.
- `--daily`: show only daily-scheduled runs.
- `--errors`: show only error runs.
- `--summary`: show grouped aggregation instead of individual lines.

Flags compose with AND logic. For example, `--daily --errors` shows only daily runs that errored.

Output columns: use_id, time, name, status, runtime, cost, events, tools, output_size, model, facet.

Examples:

```bash
sol talent logs
sol talent logs activity -c 10
sol talent logs --daily
sol talent logs --daily --summary
sol talent logs --day 20260228
sol talent logs --daily --errors
```

## agent run detail

```bash
sol talent log <ID> [--json] [--full]
```

Show events for a single agent run.

- `ID`: agent run ID (from `sol talent logs` output).
- `--json`: raw JSONL events.
- `--full`: expanded event detail (no truncation).

Without flags, shows a one-line-per-event timeline: timestamp, event type, detail.

Examples:

```bash
sol talent log 1700000000001
sol talent log 1700000000001 --json
sol talent log 1700000000001 --full
```

## journal layout

Reference map of key paths. `journal/` is the journal root.

### journal level

| Path | Purpose |
|------|---------|
| `health/` | Service logs: `<service>.log` symlinks, `callosum.sock`, `supervisor.log` |
| `talents/` | Agent run logs: `<name>/<id>.jsonl`, `<name>/<id>_active.jsonl`, `<name>.log` symlink, `<day>.jsonl` day index |
| `config/` | `journal.json`, `convey.json`, `schedules.json`, `actions/YYYYMMDD.jsonl` |
| `facets/<facet>/` | Per-facet data: `facet.json`, `entities/`, `todos/`, `events/`, `news/`, `logs/` |
| `entities/<id>/` | Canonical entity records: `entity.json` |
| `tokens/` | Token usage: `YYYYMMDD.jsonl` per day |
| `indexer/` | Search index: `journal.sqlite` (FTS5) |
| `streams/` | Stream state: `<name>.json` |
| `imports/` | Imported audio and processing artifacts |

### day level (`YYYYMMDD/`)

| Path | Purpose |
|------|---------|
| `<stream>/HHMMSS_LEN/` | Segment folders (captures, extracts, agent outputs) |
| `talents/` | Daily agent outputs: `<name>.md`, `<name>.json` |
| `health/` | Service logs for that day: `<ref>_<service>.log` (symlinked from journal-level `health/`) |
| `stats.json` | Day statistics |

### segment level (`YYYYMMDD/<stream>/HHMMSS_LEN/`)

| Path | Purpose |
|------|---------|
| `audio.*` | Audio captures (`.flac`, `.m4a`, `.ogg`, `.opus`) |
| `<pos>_<connector>_screen.*` | Screen captures (`.webm`, `.mov`, `.mp4`) |
| `audio.jsonl` | Audio transcript extract |
| `<pos>_<connector>_screen.jsonl` | Screen analysis extract |
| `stream.json` | Segment metadata and stream linkage |
| `*.md` | Segment-level agent outputs |

## services

Which services write where:

| Service | Writes to |
|---------|-----------|
| Observer | Audio/video captures in segment folders |
| Sense | Transcripts + screen analysis (JSONL) in segment folders |
| Cortex | Agent JSONL in `talents/<name>/`, outputs in segment/day dirs |
| Indexer | `indexer/journal.sqlite` |
| Supervisor | `health/supervisor.log`, service logs in `YYYYMMDD/health/` |

## Troubleshooting

### `sol health` returns "Connection refused" or times out
The supervisor is not running. Check if `sol supervisor` is active. The owner may need to start solstone with `sol start` or `make dev`.

### Agent run shows "error" status in `sol talent logs`
Run `sol talent log <ID> --full` to see the complete event timeline including the error. Common causes:
- API key issues (rate limits, expired keys)
- Prompt too large (context overflow)
- Network connectivity

### Missing segments or capture gaps
1. Run `sol health` to check observer service status
2. Run `sol health logs --service sense --since 2h` to check for transcription errors
3. Check if the stream is active: `sol streams`

### High agent costs
Run `sol talent logs --summary` for aggregated cost view. Filter by agent: `sol talent logs <agent-name> --summary`.
