---
name: journal
description: >
  Search the journal, list facets, and explain how the journal is laid out
  on disk — captures, extracts, talent outputs, apps, facets, and the
  search index. Also covers the `sol call journal` CLI.
  TRIGGER: journal, journal layout, search journal, find meeting, list
  facets, show agent output, captures, extracts, talents, apps, facet,
  indexer, activity records, sol call journal, sol call journal search,
  sol call journal facet.
---

# Journal Skill

Explore journal layout and run `sol call journal` CLI work. Invoke via Bash: `sol call journal <command> [args...]`.

## Overview

A journal is the on-disk record of captures, extracts, facet data, app storage, and talent outputs.

```
┌──────────────────────┐
│ LAYER 3: OUTPUTS     │ talents/*.md, segment *.md
├──────────────────────┤
│ LAYER 2: EXTRACTS    │ *.jsonl transcripts, frames, events
├──────────────────────┤
│ LAYER 1: CAPTURES    │ audio/video files
└──────────────────────┘
```

For the full pipeline, see [captures](references/captures.md).

## Vocabulary

| Term | Definition | Examples |
|------|------------|----------|
| **Day** | 24-hour activity directory | `20250119/` |
| **Segment** | Timestamped capture window | `143022_300/` |
| **Facet** | Project/context scope | `#work`, `#personal` |
| **Entity** | Tracked person/project/tool | People, companies, tools |
| **Activity** | Completed span of one activity type | Meeting, coding session, review |

## Top-Level Layout

| Path | Purpose |
|------|---------|
| `chronicle/` | Daily capture folders |
| `entities/` | Journal-level entity records |
| `facets/` | Facet data: entities, todos, events, news, logs |
| `talents/` | Talent run logs and outputs |
| `solstone/apps/` | App-specific journal storage |
| `imports/` | Imported audio and artifacts |
| `indexer/` | Search index |
| `config/` | Journal configuration and action logs |

For the full table, see [storage](references/storage.md).

## References

- [CLI reference](references/cli.md) — `sol call journal` commands
- [Configuration](references/config.md) — `journal.json`, providers, retention
- [Facets](references/facets.md) — facet folders, entities, news, todos
- [Captures and Extracts](references/captures.md) — layers, imports, segment layout
- [Logs](references/logs.md) — action logs, token usage, talent logs, health
- [Storage](references/storage.md) — top-level layout, app storage, search index
