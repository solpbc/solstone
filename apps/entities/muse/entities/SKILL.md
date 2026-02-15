---
name: entities
description: Manage tracked entities with sol call entities commands. List, detect, attach, update, alias, and record observations for people, companies, projects, and tools. Track relationships and knowledge within a facet.
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
- **Blocked**: entities the user has blocked; do not detect, attach, or reuse.
- **Detached**: entities the user removed; do not re-attach automatically.

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
