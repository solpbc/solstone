---
name: entities
description: >
  Manage tracked entities for people, companies, projects, and tools within
  facets. Detect, attach, update, alias, search, and record observations.
  Query relationship strength and get full intelligence briefings.
  Use when the owner asks about people, contacts, companies, or projects
  tracked in the journal, or wants to add, update, or search entities.
  TRIGGER: entity, person, company, project, relationship, observation,
  who is, contact, knowledge graph, intelligence briefing.
---

# Entities CLI Skill

Use these commands to maintain facet-scoped entity memory from the terminal.

**Environment defaults**: When `SOL_FACET` is set, all commands use it automatically. Same for `SOL_DAY` where DAY is accepted.

Common pattern:

```bash
sol call entities <command> [args...]
```

## Entity Lifecycle

- **Detected**: day-scoped, ephemeral observations captured for a specific day.
- **Attached**: persistent entities tracked long-term in a facet.
- **Blocked**: entities the owner has blocked; do not detect, attach, or reuse.
- **Detached**: entities the owner removed; do not re-attach automatically.

Use `detect` for day-specific sightings and `attach` for long-term tracking.

## list

```bash
sol call entities list [FACET] [-d DAY]
```

List entities for a facet.

- `FACET`: facet name (default: `SOL_FACET` env).
- `-d, --day`: optional day (`YYYYMMDD`).

Behavior notes:

- Without `--day`: lists attached (permanent) entities.
- With `--day`: lists detected entities for that day.

Examples:

```bash
sol call entities list work
sol call entities list work -d 20260115
```

## detect

```bash
sol call entities detect TYPE ENTITY DESCRIPTION [-f FACET] [-d DAY]
```

Record a detected entity for a day.

- `TYPE`: entity type (alphanumeric + spaces, minimum 3 chars).
- `ENTITY`: entity id, full name, or alias.
- `DESCRIPTION`: day-scoped description.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).

Behavior notes:

- If `ENTITY` matches an attached entity, detection uses its canonical name.
- Blocked entities are rejected.
- Duplicate detections for the same day are rejected.

Example:

```bash
sol call entities detect "Person" "Alicia Chen" "Led architecture review" -f work -d 20260115
```

## attach

```bash
sol call entities attach TYPE ENTITY DESCRIPTION [-f FACET]
```

Attach an entity permanently to a facet.

- `TYPE`: required type.
- `ENTITY`: id, name, or alias reference.
- `DESCRIPTION`: persistent description.
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Behavior notes:

- If already attached, command reports existing entity.
- Blocked entities are rejected.
- Previously detached entities are rejected.

Example:

```bash
sol call entities attach "Company" "Acme Corp" "Primary platform vendor" -f work
```

## update

```bash
sol call entities update ENTITY DESCRIPTION [-f FACET] [-d DAY]
```

Update entity description.

- `ENTITY`: entity id, name, or alias for attached entities; exact name for day-scoped detected updates.
- `DESCRIPTION`: new description.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `-d, --day`: optional day (`YYYYMMDD`) to update a detected entity.

Behavior notes:

- Without `--day`: updates an attached entity.
- With `--day`: updates a detected entity for that day.

Examples:

```bash
sol call entities update "acme_corp" "Primary vendor for identity services" -f work
sol call entities update "Alicia Chen" "Discussed migration plan" -f work -d 20260115
```

## aka

```bash
sol call entities aka ENTITY AKA [-f FACET]
```

Add an alias to an attached entity.

- `ENTITY`: entity id, name, or alias reference.
- `AKA`: alias to add.
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Behavior notes:

- Automatically skips aliases that equal the first word of the entity name.
- Automatically deduplicates existing aliases.
- Validates alias uniqueness across entities in the facet.

Example:

```bash
sol call entities aka "Federal Aviation Administration" "FAA" -f work
```

## observations

```bash
sol call entities observations ENTITY [-f FACET]
```

List durable observations for an attached entity.

- `ENTITY`: entity id, name, or alias.
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Output is numbered for quick review.

Example:

```bash
sol call entities observations "Alicia Chen" -f work
```

## observe

```bash
sol call entities observe ENTITY CONTENT [-f FACET] [--source-day DAY]
```

Add a durable observation to an attached entity.

- `ENTITY`: entity id, name, or alias.
- `CONTENT`: observation text.
- `-f, --facet`: facet name (default: `SOL_FACET` env).
- `--source-day`: optional day (`YYYYMMDD`) when this was observed.

Behavior notes:

- Observation number is auto-calculated by the CLI.

### Observation Quality Guidance

Good observations (durable factoids):

- "Prefers async communication over meetings"
- "Works PST timezone, typically available after 10am"
- "Has deep expertise in distributed systems and Rust"
- "Reports to Sarah Chen on the platform team"

Bad observations (day-specific activity; use `detect` instead):

- "Discussed API migration today"
- "Sent contract for review"

Example:

```bash
sol call entities observe "Alicia Chen" "Prefers design docs before implementation" -f work --source-day 20260115
```

## strength

```bash
sol call entities strength [FACET] [-n LIMIT]
```

Rank entities by relationship strength score within a facet.

- `FACET`: facet name (default: `SOL_FACET` env).
- `-n, --limit`: max results (default `10`).

Use this to find the most significant relationships in a facet.

Example:

```bash
sol call entities strength work -n 20
```

## search

```bash
sol call entities search [QUERY] [--type TYPE] [--facet FACET] [--active-days N]
```

Search entities by text, type, facet, or recent activity.

- `QUERY`: optional text query.
- `--type`: filter by entity type (e.g., `Person`, `Company`).
- `--facet`: filter by facet.
- `--active-days`: filter to entities active within N days.

Examples:

```bash
sol call entities search "Chen"
sol call entities search --type Person --facet work
sol call entities search --active-days 7
```

## intelligence

```bash
sol call entities intelligence ENTITY [-f FACET]
```

Get a full intelligence briefing for an entity — relationship history, observations, activity timeline, and cross-facet presence.

- `ENTITY`: entity id, name, or alias.
- `-f, --facet`: facet name (default: `SOL_FACET` env).

Example:

```bash
sol call entities intelligence "Alicia Chen" -f work
```
