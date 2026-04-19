---
name: todos
description: >
  Manage todo checklists organized by facet and day. List, add, complete,
  cancel, move tasks, schedule nudges (reminders), and review upcoming items.
  Use when the owner mentions tasks, to-do items, action items, checklists,
  reminders, or nudges, or asks to add, complete, cancel, or review todos.
  TRIGGER: todo, task, action item, checklist, reminder, nudge, scheduled
  reminder, upcoming items, remind me, sol call todos list,
  sol call todos add, sol call todos done, sol call todos cancel,
  sol call todos upcoming, sol call todos list-nudges-due,
  sol call todos dispatch-nudges.
---

# Todos CLI Skill

Manage checklist entries and reminders. Invoke via Bash: `sol call todos <command> [args...]`.

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
sol call todos add TEXT [-d DAY] [-f FACET] [-n NUDGE] [--force]
```

Add a new todo item, optionally with a nudge (reminder).

- `TEXT`: todo text (positional argument).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-n, --nudge`: optional reminder time. Accepted formats: `HH:MM` (today), `tomorrow HH:MM`, `YYYYMMDDTHH:MM` (absolute), or `now`.
- `--force`: skip the cross-facet duplicate check.

Behavior notes:

- Line number is auto-calculated by the CLI; do not provide one.
- You can include time in the text as `(HH:MM)` suffix.
- Before adding a future todo, check `upcoming` first to avoid duplicates already scheduled on other days.
- The command checks for fuzzy duplicates (≥70% similarity) across other facets within ±1 day. If a match is found, the add is rejected with a match report. Use `--force` to override.

Examples:

```bash
sol call todos add "Draft Q1 plan" -d 20260115 -f work
sol call todos add "Team sync prep (14:30)" -d 20260115 -f work
sol call todos add "Call Alicia back" -f work -n "15:30"
sol call todos add "Submit expense report" -f work -n "tomorrow 09:00"
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

## move

```bash
sol call todos move LINE_NUMBER --day DAY --from SOURCE --to DEST [--consent]
```

Move an open todo from one facet to another.

- `LINE_NUMBER`: 1-based line number from `list` output (positional argument).
- `--day`: day in `YYYYMMDD` (required).
- `--from`: source facet name.
- `--to`: destination facet name.
- `--consent`: required when called by a proactive agent.

Example:

```bash
sol call todos move 3 --day 20260115 --from personal --to work --consent
```

## list-nudges-due

```bash
sol call todos list-nudges-due [-f FACET] [--json]
```

List todo items whose nudge time is due and have not yet been notified.

- `-f, --facet`: optional facet filter. Omit to check all facets.
- `--json`: emit JSON for programmatic use.

Behavior notes:

- Reads state only; does not mark items as notified or send notifications.
- Groups by facet when multiple facets have due items.

Example:

```bash
sol call todos list-nudges-due
sol call todos list-nudges-due -f work --json
```

## dispatch-nudges

```bash
sol call todos dispatch-nudges [-f FACET]
```

Send desktop notifications for all due, unnotified nudges and mark them notified so they won't fire again.

- `-f, --facet`: optional facet filter. Omit to dispatch for all facets.

Behavior notes:

- Uses `sol notify` to deliver. Each successful dispatch updates the item's notified state.
- Intended for a periodic caller (routine or launchd); manual invocation is fine for testing.

Example:

```bash
sol call todos dispatch-nudges
```

## Gotchas

- **Duplicate-check is fuzzy and silent-looking.** `add` rejects items with ≥70% similarity to an existing item across other facets within ±1 day. The rejection prints matches to stderr; if you miss the message, the add looks like it silently failed. Pass `--force` when you know the duplication is intentional.
- **`list-nudges-due` is non-mutating; `dispatch-nudges` marks notified.** Don't expect `list-nudges-due` to clear the queue — use it for inspection.
