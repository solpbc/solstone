---
name: journal
description: Search and browse journal content using sol call journal commands. Find, query, and look up information across transcripts, insights, events, entities, and todos. Get facet overviews and news feeds.
---

# Journal CLI Skill

Use these commands to explore journal content from the terminal.

**Environment defaults**: When `SOL_DAY` is set, commands that take a DAY argument will use it automatically. Same for `SOL_SEGMENT` and `SOL_FACET`.

Common pattern:

```bash
sol call journal <command> [args...]
```

**Typical workflow**: `search` to find content across all types → `events` or `facet` for structured detail on a specific day or project.

## search

```bash
sol call journal search [QUERY] [-n LIMIT] [--offset N] [-d DAY] [--day-from DAY] [--day-to DAY] [-f FACET] [-a AGENT]
```

Search the journal index across insights, transcripts, events, entities, and todos.

- `QUERY`: optional text query. Defaults to empty string (`""`), which works as browse mode when filters are provided.
- `-n, --limit`: max results (default `10`).
- `--offset`: skip N results (default `0`).
- `-d, --day`: exact day filter (`YYYYMMDD`).
- `--day-from`, `--day-to`: inclusive date-range filters (`YYYYMMDD`).
- `-f, --facet`: facet filter (for example `work`, `personal`).
- `-a, --agent`: agent/content filter (for example `flow`, `event`, `news`, `entity:detected`).

Behavior notes:

- FTS5 query syntax:
- Terms are `AND`'d by default.
- Use `OR` for alternatives: `apple OR orange`.
- Use quotes for exact phrases: `"weekly sync"`.
- Use `*` for prefix matching: `migrat*`.
- Use either `--day` or date range flags; do not combine exact day with range filters.

Examples:

```bash
sol call journal search "incident review" -n 20 -f work
sol call journal search "standup OR sync" --day-from 20260101 --day-to 20260107
sol call journal search "" -d 20260115 -a audio
```

## events

```bash
sol call journal events [DAY] [-f FACET]
```

List structured events for a day.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: optional facet filter.

Use this when you need full event records (titles, summaries, times, participants), not just search snippets.

Examples:

```bash
sol call journal events 20260115
sol call journal events 20260115 -f work
```

## facet

```bash
sol call journal facet [NAME]
```

Show a comprehensive facet summary.

- `NAME`: facet name (default: `SOL_FACET` env).

Use this for a quick overview of facet metadata, entities, and current state.

Example:

```bash
sol call journal facet work
sol call journal facet          # uses SOL_FACET
```

## agents

```bash
sol call journal agents [DAY] [-s SEGMENT]
```

List available agent outputs for a day.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-s, --segment`: optional segment key (default: `SOL_SEGMENT` env).

Without `--segment`, lists daily agent outputs and per-segment outputs. With `--segment`, lists only that segment's outputs.

Example:

```bash
sol call journal agents 20260115
sol call journal agents -s 091500_300
```

## read

```bash
sol call journal read AGENT [-d DAY] [-s SEGMENT] [--max BYTES]
```

Read full content of an agent output.

- `AGENT`: agent name, e.g. `flow`, `meetings`, `activity` (positional argument).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-s, --segment`: optional segment key (default: `SOL_SEGMENT` env).
- `--max`: max output bytes (default `16384`, `0` for unlimited).

Without `--segment`, reads from the daily agents directory. With `--segment`, reads from that segment's agents directory.

Examples:

```bash
sol call journal read flow -d 20260115
sol call journal read meetings
sol call journal read activity -s 091500_300
```

## news

```bash
sol call journal news [NAME] [-d DAY] [-n LIMIT] [--cursor CURSOR] [-w]
```

Read or write facet news entries.

- `NAME`: facet name (default: `SOL_FACET` env).
- `-d, --day`: optional specific day (`YYYYMMDD`, default: `SOL_DAY` env).
- `-n, --limit`: max days to return (default `5`).
- `--cursor`: optional pagination cursor (typically a `YYYYMMDD` cutoff for older entries).
- `-w, --write`: write mode — reads markdown from stdin and saves as news for the given day.

Behavior notes:

- Without `--write`: reads and displays existing news entries. Uses `SOL_DAY` to filter to a specific day when set.
- With `--write`: requires `--day` (or `SOL_DAY` env), reads markdown content from stdin, saves to facet news directory.

Examples:

```bash
sol call journal news work -n 3
sol call journal news -d 20260115          # uses SOL_FACET
sol call journal news work --cursor 20260110 -n 5
```
