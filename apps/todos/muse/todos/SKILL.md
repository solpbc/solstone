---
name: todos
description: Manage todo checklists using sol call todos commands. Use this when you need to list, add, complete, cancel, or review upcoming todos organized by facet and day.
---

# Todos CLI Skill

Use these commands to manage checklist entries from the terminal.
Common pattern:

```bash
sol call todos <command> [args...]
```

## list

```bash
sol call todos list DAY [-f FACET] [--to DAY]
```

Show checklist entries for one day or an inclusive day range.

- `DAY`: required day in `YYYYMMDD`.
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
sol call todos add DAY TEXT -f FACET
```

Add a new todo item.

- `DAY`: required day in `YYYYMMDD`; must be today or in the future.
- `TEXT`: todo text.
- `-f, --facet`: required facet name.

Behavior notes:

- Line number is auto-calculated by the CLI; do not provide one.
- You can include time in the text as `(HH:MM)` suffix.
- Before adding a future todo, check `upcoming` first to avoid duplicates already scheduled on other days.

Examples:

```bash
sol call todos add 20260115 "Draft Q1 plan" -f work
sol call todos add 20260115 "Team sync prep (14:30)" -f work
```

## done

```bash
sol call todos done DAY LINE_NUMBER -f FACET
```

Mark a todo as complete.

- `DAY`: required day in `YYYYMMDD`.
- `LINE_NUMBER`: 1-based line number from `list` output.
- `-f, --facet`: required facet name.

Example:

```bash
sol call todos done 20260115 2 -f work
```

## cancel

```bash
sol call todos cancel DAY LINE_NUMBER -f FACET
```

Cancel (soft-delete) a todo.

- `DAY`: required day in `YYYYMMDD`.
- `LINE_NUMBER`: 1-based line number from `list` output.
- `-f, --facet`: required facet name.

Behavior note:

- Cancellation keeps line numbering continuity; entries stay in storage but are treated as inactive.

Example:

```bash
sol call todos cancel 20260115 4 -f work
```

## upcoming

```bash
sol call todos upcoming [-l LIMIT] [-f FACET]
```

Show future todos grouped by facet and day.

- `-l, --limit`: max items (default `20`).
- `-f, --facet`: optional facet filter. Omit to include all facets.

Workflow note:

- Run this before adding future-day todos to avoid cross-day duplicates.

Examples:

```bash
sol call todos upcoming
sol call todos upcoming -l 50 -f work
```
