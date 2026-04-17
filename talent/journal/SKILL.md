---
name: journal
description: >
  Search, find, list, and show journal data, and explain journal layout,
  structure, architecture, storage, facets, and `sol call journal` CLI
  commands. Use when the owner asks what a journal is, where content is
  stored, to search for meetings, list facets, or show agent output.
---

# Journal Skill

> **First rule for AI agents in a journal**: before doing anything else, run `sol call identity` to hydrate Sol's self, partner, agency, and awareness. The output of that command tells you who you are, who you're working with, and what's currently on your plate. Everything below describes the journal's *layout* — the dynamic identity context comes from the CLI.

Use this skill for both journal layout questions and `sol call journal` CLI work.

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
| **Occurrence** | Time-based event | Meetings, messages, files |

## Top-Level Layout

| Path | Purpose |
|------|---------|
| `chronicle/` | Daily capture folders |
| `entities/` | Journal-level entity records |
| `facets/` | Facet data: entities, todos, events, news, logs |
| `talents/` | Talent run logs and outputs |
| `apps/` | App-specific journal storage |
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
