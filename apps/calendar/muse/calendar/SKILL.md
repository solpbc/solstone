---
name: calendar
description: >
  Manage calendar events organized by facet and day. List, create, update,
  cancel, and move events including scheduling with participants and times.
  Use when the owner mentions calendar events, scheduling, appointments,
  meetings, or wants to create, reschedule, or cancel events.
  TRIGGER: calendar, schedule, appointment, meeting, event, reschedule.
---

# Calendar CLI Skill

Use these commands to manage calendar events from the terminal.

**Environment defaults**: When `SOL_DAY` is set, commands that take a DAY argument will use it automatically. Same for `SOL_FACET` where FACET is required.

Common pattern:

```bash
sol call calendar <command> [args...]
```

## list

```bash
sol call calendar list [DAY] --facet FACET
```

List events for a day.

- `DAY`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--facet`: facet name (default: `SOL_FACET` env).

Example:

```bash
sol call calendar list 20260115 --facet work
```

## create

```bash
sol call calendar create TITLE --start HH:MM --day DAY --facet FACET [--end HH:MM] [--summary TEXT] [--participants NAMES]
```

Create a calendar event.

- `TITLE`: event title (positional argument).
- `--start`: start time in `HH:MM` (required).
- `--day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--facet`: facet name (default: `SOL_FACET` env).
- `--end`: optional end time in `HH:MM`.
- `--summary`: optional event description.
- `--participants`: optional comma-separated participant names.

Example:

```bash
sol call calendar create "Team standup" --start 09:00 --end 09:30 --day 20260115 --facet work --participants "Alicia, Ben"
```

## update

```bash
sol call calendar update LINE --day DAY --facet FACET [--title TEXT] [--start HH:MM] [--end HH:MM] [--summary TEXT] [--participants NAMES]
```

Update an existing calendar event.

- `LINE`: 1-based line number from `list` output (positional argument).
- `--day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--facet`: facet name (default: `SOL_FACET` env).
- `--title`: new event title.
- `--start`: new start time in `HH:MM`.
- `--end`: new end time in `HH:MM`.
- `--summary`: new event description.
- `--participants`: new comma-separated participant names.

Example:

```bash
sol call calendar update 2 --day 20260115 --facet work --start 10:00 --end 10:30
```

## cancel

```bash
sol call calendar cancel LINE --day DAY --facet FACET
```

Cancel a calendar event.

- `LINE`: 1-based line number from `list` output (positional argument).
- `--day`: day in `YYYYMMDD` (default: `SOL_DAY` env).
- `--facet`: facet name (default: `SOL_FACET` env).

Behavior notes:

- Cancelled events remain visible in listings to preserve line numbering.

Example:

```bash
sol call calendar cancel 3 --day 20260115 --facet work
```

## move

```bash
sol call calendar move LINE --day YYYYMMDD --from SOURCE --to DEST [--consent]
```

Move a non-cancelled calendar event to another facet.

- `LINE`: 1-based line number from `list` output (positional argument).
- `--day`: day in `YYYYMMDD` (required).
- `--from`: source facet name.
- `--to`: destination facet name.
- `--consent`: required when called by a proactive agent. Must have explicit owner approval before calling with this flag.

Behavior notes:

- Only non-cancelled events can be moved.
- The `--consent` flag signals that the owner has explicitly approved this action. Proactive agents must obtain owner approval before including it.

Example:

```bash
sol call calendar move 1 --day 20260115 --from personal --to work --consent
```
