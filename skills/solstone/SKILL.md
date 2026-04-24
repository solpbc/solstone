---
name: solstone
version: 1.0.0
description: >
  Read-only query of the solstone journal from any project. Look up people
  and relationships, today's events, todos; read transcripts; intelligence
  briefings on entities. TRIGGER: solstone, my journal, search my memory,
  what happened, who is, meeting with, co-brain, recall, sol call
  journal/entities/transcripts/todos.
---

# solstone — journal query interface

Read-only query interface to your solstone journal. Invoke via Bash: `sol call <subcommand> [flags]`.

Use this skill to search memories, look up people, check today's events, read transcripts, and get relationship briefings — all from any project context.

## Prerequisites

The `sol` CLI must be on PATH. Quick check:

```bash
sol help
```

If this fails, solstone is not installed. Install it from the solstone project: `make install-service`.

## Capabilities

### recall — search your memory

Search the journal index for anything matching a query.

```bash
sol call journal search "<query>"
sol call journal search "<query>" --day 20260327
sol call journal search "<query>" --facet work
sol call journal search "<query>" --day-from 20260320 --day-to 20260327
```

Dates use `YYYYMMDD` format. Omit `--day` to search all days. Omit `--facet` to search all facets.

### who — entity intelligence

Look up what the journal knows about a person, company, or project.

```bash
# Full intelligence briefing for a named entity
sol call entities intelligence "<name>"

# Search for entities by text, type, or activity
sol call entities search --query "<query>"

# See observations recorded for an entity (requires facet)
sol call entities observations "<entity_name>" --facet "<facet>"
```

`intelligence` gives the richest answer — use it first, then `search` if the entity isn't attached.

### today — what's happening now

Combine these commands to get a full picture of the current day:

```bash
# Upcoming todos
sol call todos upcoming

# Calendar events for today
sol call activities list --source anticipated

# Latest facet news
sol call journal news "<facet>"
```

### transcript — meeting transcripts

Read what was said during meetings or any recorded time.

```bash
# List transcript coverage ranges for a day
sol call transcripts scan
sol call transcripts scan 20260327

# Read transcript content for a specific time range
sol call transcripts read
sol call transcripts read 20260327
sol call transcripts read 20260327 --start 14 --length 2

# Transcript stats for a month
sol call transcripts stats
```

`scan` first to see what's available, then `read` with `--start` (hour, 24h format) and `--length` (hours) to narrow down.

### people — relationship strength

See who the strongest relationships are, or who's been active recently.

```bash
# Overall relationship strength ranking
sol call entities strength

# Strength since a specific date
sol call entities strength --since 20260320

# Strength within a facet
sol call entities strength --facet work
```

### status — system health

Check if solstone is running and how much data exists.

```bash
# Journal storage summary (days, facets, size)
sol call journal storage-summary

# Local diagnostics (no network)
sol call support diagnose
```

## Paths

`sol root` prints the solstone repo root — useful for scripting: `cd $(sol root)`, `SOL=$(sol root)`.

## Environment

The `sol` CLI uses three environment variables that default sensibly:

- `SOL_DAY` — defaults to today (`YYYYMMDD`)
- `SOL_FACET` — defaults to all facets
- `SOL_SEGMENT` — defaults to no segment

External callers should never need to set these. The commands above use explicit flags (`--day`, `--facet`) when narrowing scope is needed.

## Composing queries

For richer answers, combine multiple commands:

**"Brief me on today"** — events + todos + calendar + recent entity strength:
```bash
sol call todos upcoming
sol call activities list --source anticipated
sol call entities strength --since $(date +%Y%m%d)
```

**"Prep me for a meeting with X"** — entity intelligence + recent transcript mentions + relationship strength:
```bash
sol call entities intelligence "<name>"
sol call journal search "<name>" --day-from 20260320
sol call entities strength --since 20260301
```

**"What did I miss yesterday?"** — yesterday's events + transcripts + news:
```bash
sol call journal search "" -d 20260326 -a meetings
sol call transcripts scan 20260326
sol call journal news "<facet>" --day 20260326
```

## Output format

Most commands output plain text by default. Many support `--json` for structured output. Prefer plain text for human-readable answers; use `--json` when you need to process the data further.

## What you cannot do

This is a **read-only** interface. The journal is the person's private space. You cannot:

- Create, delete, or modify facets
- Add or complete todos
- Attach or modify entities
- Write news or observations
- Run pipeline operations (think, indexer, transcribe)
- Access internal agent state or orchestration

If a task requires writing to the journal, it must be done from within the solstone project context using sol's internal skills.

## Error handling

If `sol` is not found on PATH or returns an error:

- **"command not found: sol"** — solstone is not installed. The user needs to run `make install-service` in their solstone project.
- **"journal not found"** or empty output — the journal directory doesn't exist or has no data yet. solstone may be installed but not yet initialized.
- **Connection errors from `sol call support`** — `diagnose` is local-only and should always work. Other support commands (`search`, `article`) contact the support portal and may fail if offline.

Do not retry failed commands. Report the error clearly so the user can investigate.
