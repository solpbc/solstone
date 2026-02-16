---
name: todos
description: Manage todo checklists using sol call todos commands. List, add, complete, and cancel tasks and action items organized by facet and day. Review upcoming scheduled items.
---

# Todos CLI Skill

Use these commands to manage checklist entries from the terminal.

**Environment defaults**: When `SOL_DAY` is set, commands that take a DAY argument will use it automatically. Same for `SOL_FACET` where FACET is required.

Common pattern:

```bash
sol call todos <command> [args...]
```

## list

```bash
sol call todos list [DAY] [-f FACET] [--to DAY]
```

Show checklist entries for one day or an inclusive day range.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: optional facet filter. Omit to show all facets.
- `--to`: optional inclusive range end day in `YYYYMMDD`.

Behavior notes:

- When `--to` is set and differs from `DAY`, output is grouped by day headers.
- Cancelled items are shown with strikethrough so numbering remains stable.

Examples:

```bash
sol call todos list 20260115 -f work
sol call todos list 20260101 --to 20260107 -f work
sol call todos list 20260115
```

## add

```bash
sol call todos add TEXT [-d DAY] [-f FACET]
```

Add a new todo item.

- `TEXT`: todo text (positional argument).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Behavior notes:

- Line number is auto-calculated by the CLI; do not provide one.
- You can include time in the text as `(HH:MM)` suffix.
- Before adding a future todo, check `upcoming` first to avoid duplicates already scheduled on other days.

Examples:

```bash
sol call todos add "Draft Q1 plan" -d 20260115 -f work
sol call todos add "Team sync prep (14:30)" -d 20260115 -f work
```

## done

```bash
sol call todos done LINE_NUMBER [-d DAY] [-f FACET]
```

Mark a todo as complete.

- `LINE_NUMBER`: 1-based line number from `list` output (positional argument).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Example:

```bash
sol call todos done 2 -d 20260115 -f work
```

## cancel

```bash
sol call todos cancel LINE_NUMBER [-d DAY] [-f FACET]
```

Cancel (soft-delete) a todo.

- `LINE_NUMBER`: 1-based line number from `list` output (positional argument).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Behavior notes:

- Cancellation keeps line numbering continuity; entries stay in storage but are treated as inactive.

Example:

```bash
sol call todos cancel 4 -d 20260115 -f work
```

## upcoming

```bash
sol call todos upcoming [-l LIMIT] [-f FACET]
```

Show future todos grouped by facet and day.

- `-l, --limit`: max items (default `20`).
- `-f, --facet`: optional facet filter. Omit to include all facets.

Behavior notes:

- Run this before adding future-day todos to avoid cross-day duplicates.

Examples:

```bash
sol call todos upcoming
sol call todos upcoming -l 50 -f work
```
