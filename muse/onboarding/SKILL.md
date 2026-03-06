---
name: onboarding
description: Set up a new journal — welcome choice, facet creation, and entity seeding.
---

# Onboarding CLI Skill

Use these commands to guide first-time setup.

## awareness onboarding

```bash
sol call awareness onboarding [--path a|b] [--skip] [--complete]
```

- `--path a`: Start Path A (passive observation).
- `--path b`: Start Path B (conversational interview).
- `--skip`: Skip onboarding entirely.
- `--complete`: Mark onboarding as complete.
- No flags: Read current onboarding state.

## awareness status

```bash
sol call awareness status [SECTION]
```

- `SECTION`: Optional section name (e.g., `onboarding`). Omit for full state.

## awareness log-read

```bash
sol call awareness log-read [DAY] [--kind KIND] [--limit N]
```

- `DAY`: Day in YYYYMMDD format (defaults to today).
- `--kind`: Filter by entry kind (e.g., `observation`, `nudge`, `state`).
- `--limit`: Max entries to return (0 = all).

## facet create

```bash
sol call journal facet create <title> [--emoji EMOJI] [--color COLOR] [--description DESC]
```

- `title`: Display title for the new facet.
- `--emoji`: Optional facet icon (default: box emoji).
- `--color`: Optional hex color (default: #667eea).
- `--description`: Optional description.

Example:

```bash
sol call journal facet create "Work" --emoji "briefcase emoji" --color "#667eea" --description "Client deliverables and meetings"
```

## facets

```bash
sol call journal facets [--all]
```

- `--all`: Include muted facets.

## attach

```bash
sol call entities attach TYPE ENTITY DESCRIPTION --facet FACET
```

- `TYPE`: One of `Person`, `Company`, `Project`, `Tool`.
- `ENTITY`: Entity identifier/name.
- `DESCRIPTION`: Persistent description to store for the entity.
- `--facet`: Facet to attach the entity to.

Example:

```bash
sol call entities attach "Person" "Alex Chen" "Product manager for onboarding" --facet work
```
