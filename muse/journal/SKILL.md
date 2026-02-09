---
name: journal
description: Search and browse journal content using sol call journal commands. Use this when you need to find information across transcripts, insights, events, entities, and todos, or get facet overviews and news.
---

# Journal CLI Skill

Use these commands to explore journal content from the terminal.
Common pattern:

```bash
sol call journal <command> [args...]
```

## search

```bash
sol call journal search [QUERY] [-n LIMIT] [--offset N] [-d DAY] [--day-from DAY] [--day-to DAY] [-f FACET] [-t TOPIC]
```

Search the journal index across insights, transcripts, events, entities, and todos.

- `QUERY`: optional text query. Defaults to empty string (`""`), which works as browse mode when filters are provided.
- `-n, --limit`: max results (default `10`).
- `--offset`: skip N results (default `0`).
- `-d, --day`: exact day filter (`YYYYMMDD`).
- `--day-from`, `--day-to`: inclusive date-range filters (`YYYYMMDD`).
- `-f, --facet`: facet filter (for example `work`, `personal`).
- `-t, --topic`: topic/content filter (for example `flow`, `event`, `news`, `entity:detected`).

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
sol call journal search "" -d 20260115 -t audio
```

## events

```bash
sol call journal events DAY [-f FACET]
```

List structured events for a day.

- `DAY`: required day in `YYYYMMDD`.
- `-f, --facet`: optional facet filter.

Use this when you need full event records (titles, summaries, times, participants), not just search snippets.

Examples:

```bash
sol call journal events 20260115
sol call journal events 20260115 -f work
```

## facet

```bash
sol call journal facet NAME
```

Show a comprehensive facet summary.

- `NAME`: facet name.

Use this for a quick overview of facet metadata, entities, and current state.

Example:

```bash
sol call journal facet work
```

## news

```bash
sol call journal news NAME [-d DAY] [-n LIMIT] [--cursor CURSOR]
```

Read facet news entries (read-only feed in CLI).

- `NAME`: facet name.
- `-d, --day`: optional specific day (`YYYYMMDD`).
- `-n, --limit`: max days to return (default `5`).
- `--cursor`: optional pagination cursor (typically a `YYYYMMDD` cutoff for older entries).

Behavior note:

- CLI `news` reads content only. It does not provide write behavior.

Examples:

```bash
sol call journal news work -n 3
sol call journal news work -d 20260115
sol call journal news work --cursor 20260110 -n 5
```
