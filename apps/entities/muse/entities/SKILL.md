---
name: entities
description: Manage tracked entities with sol call entities commands. List, detect, attach, update, alias, and record observations for people, companies, projects, and tools. Track relationships and knowledge within a facet.
---

# Entities CLI Skill

Use these commands to maintain facet-scoped entity memory from the terminal.

**Environment defaults**: When `SOL_FACET` is set, commands that take a FACET argument will use it automatically. Same for `SOL_DAY` where DAY is accepted.

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
sol call entities detect FACET TYPE ENTITY DESCRIPTION [-d DAY]
```

Record a detected entity for a day.

- `FACET`: required facet name (positional argument).
- `TYPE`: entity type (alphanumeric + spaces, minimum 3 chars).
- `ENTITY`: entity id, full name, or alias.
- `DESCRIPTION`: day-scoped description.
- `-d, --day`: day in `YYYYMMDD` (default: `SOL_DAY` env).

Behavior notes:

- If `ENTITY` matches an attached entity, detection uses its canonical name.
- Blocked entities are rejected.
- Duplicate detections for the same day are rejected.

Example:

```bash
sol call entities detect work "Person" "Alicia Chen" "Led architecture review" -d 20260115
```

## attach

```bash
sol call entities attach FACET TYPE ENTITY DESCRIPTION
```

Attach an entity permanently to a facet.

- `FACET`: required facet name.
- `TYPE`: required type.
- `ENTITY`: id, name, or alias reference.
- `DESCRIPTION`: persistent description.

Behavior notes:

- If already attached, command reports existing entity.
- Blocked entities are rejected.
- Previously detached entities are rejected.

Example:

```bash
sol call entities attach work "Company" "Acme Corp" "Primary platform vendor"
```

## update

```bash
sol call entities update FACET ENTITY DESCRIPTION [-d DAY]
```

Update entity description.

- `FACET`: required facet name.
- `ENTITY`: entity id, name, or alias for attached entities; exact name for day-scoped detected updates.
- `DESCRIPTION`: new description.
- `-d, --day`: optional day (`YYYYMMDD`) to update a detected entity.

Behavior notes:

- Without `--day`: updates an attached entity.
- With `--day`: updates a detected entity for that day.

Examples:

```bash
sol call entities update work "acme_corp" "Primary vendor for identity services"
sol call entities update work "Alicia Chen" "Discussed migration plan" -d 20260115
```

## aka

```bash
sol call entities aka FACET ENTITY AKA
```

Add an alias to an attached entity.

- `FACET`: required facet name.
- `ENTITY`: entity id, name, or alias reference.
- `AKA`: alias to add.

Behavior notes:

- Automatically skips aliases that equal the first word of the entity name.
- Automatically deduplicates existing aliases.
- Validates alias uniqueness across entities in the facet.

Example:

```bash
sol call entities aka work "Federal Aviation Administration" "FAA"
```

## observations

```bash
sol call entities observations FACET ENTITY
```

List durable observations for an attached entity.

- `FACET`: required facet name.
- `ENTITY`: entity id, name, or alias.

Output is numbered for quick review.

Example:

```bash
sol call entities observations work "Alicia Chen"
```

## observe

```bash
sol call entities observe FACET ENTITY CONTENT [--source-day DAY]
```

Add a durable observation to an attached entity.

- `FACET`: required facet name.
- `ENTITY`: entity id, name, or alias.
- `CONTENT`: observation text.
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
sol call entities observe work "Alicia Chen" "Prefers design docs before implementation" --source-day 20260115
```
