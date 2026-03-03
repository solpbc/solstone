---
name: onboarding
description: Set up a new journal by creating facets and seeding entities through guided conversation.
---

# Onboarding CLI Skill

Use these commands to guide first-time setup.

## facet create

```bash
sol call journal facet create <title> [--emoji EMOJI] [--color COLOR] [--description DESC]
```

- `title`: Display title for the new facet.
- `--emoji`: Optional facet icon (default: 📦).
- `--color`: Optional hex color (default: #667eea).
- `--description`: Optional description.

Example:

```bash
sol call journal facet create "Work" --emoji "💼" --color "#667eea" --description "Client deliverables and meetings"
```

## facets

```bash
sol call journal facets [--all]
```

- `--all`: Include muted facets.

Example:

```bash
sol call journal facets
```

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
