# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity saving functions.

This module handles saving entities to storage:
- save_entities: Save attached or detected entities for a facet
- save_detected_entity: Concurrency-safe single entity detection with file locking
"""

import fcntl
import json
import random
import time

from think.entities.core import EntityDict, atomic_write, entity_slug
from think.entities.journal import get_or_create_journal_entity, save_journal_entity
from think.entities.loading import detected_entities_path, load_entities
from think.entities.relationships import save_facet_relationship


def _save_entities_detected(facet: str, entities: list[EntityDict], day: str) -> None:
    """Save detected entities to day-specific JSONL file."""
    path = detected_entities_path(facet, day)

    # Ensure id field is present
    for entity in entities:
        name = entity.get("name", "")
        expected_id = entity_slug(name)
        if entity.get("id") != expected_id:
            entity["id"] = expected_id

    # Sort by type, then name for consistency
    sorted_entities = sorted(
        entities, key=lambda e: (e.get("type", ""), e.get("name", ""))
    )

    # Format as JSONL and write atomically
    content = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in sorted_entities)
    atomic_write(path, content, prefix="entities_")


def _save_entities_attached(facet: str, entities: list[EntityDict]) -> None:
    """Save attached entities to new structure (journal entities + facet relationships)."""
    # Validate uniqueness
    seen_names: set[str] = set()
    seen_ids: set[str] = set()

    for entity in entities:
        name = entity.get("name", "")
        expected_id = entity_slug(name)

        # Set or update id
        if entity.get("id") != expected_id:
            entity["id"] = expected_id

        name_lower = name.lower()
        if name_lower in seen_names:
            raise ValueError(f"Duplicate entity name '{name}' in facet '{facet}'")
        seen_names.add(name_lower)

        if expected_id in seen_ids:
            raise ValueError(
                f"Duplicate entity id '{expected_id}' in facet '{facet}' "
                f"(names may slugify to same value)"
            )
        seen_ids.add(expected_id)

    # Fields that belong to journal entity (identity)
    journal_fields = {"id", "name", "type", "aka", "is_principal", "created_at"}

    # Process each entity
    for entity in entities:
        entity_id = entity["id"]
        name = entity.get("name", "")
        entity_type = entity.get("type", "")
        aka = entity.get("aka")
        is_detached = entity.get("detached", False)

        # Ensure journal entity exists (creates if needed, preserves if exists)
        # Skip principal flagging for detached entities
        journal_entity = get_or_create_journal_entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            aka=aka if isinstance(aka, list) else None,
            skip_principal=is_detached,
        )

        # Update journal entity if name/type/aka changed
        journal_updated = False
        if journal_entity.get("name") != name:
            journal_entity["name"] = name
            journal_updated = True
        if journal_entity.get("type") != entity_type:
            journal_entity["type"] = entity_type
            journal_updated = True
        if aka and isinstance(aka, list):
            # Merge aka lists (union)
            existing_aka = set(journal_entity.get("aka", []))
            new_aka = existing_aka | set(aka)
            if new_aka != existing_aka:
                journal_entity["aka"] = sorted(new_aka)
                journal_updated = True
        # Only propagate is_principal if explicitly set and entity not detached
        if (
            entity.get("is_principal")
            and not is_detached
            and not journal_entity.get("is_principal")
        ):
            journal_entity["is_principal"] = True
            journal_updated = True

        if journal_updated:
            save_journal_entity(journal_entity)

        # Build relationship record (all non-identity fields)
        relationship: EntityDict = {
            "entity_id": entity_id,
        }
        for key, value in entity.items():
            if key not in journal_fields:
                relationship[key] = value

        # Save facet relationship
        save_facet_relationship(facet, entity_id, relationship)


def save_entities(
    facet: str, entities: list[EntityDict], day: str | None = None
) -> None:
    """Save entities to storage.

    For detected entities (day provided), writes to day-specific JSONL files.
    For attached entities (day=None), writes to:
    - Journal-level entity files: entities/<id>/entity.json (identity)
    - Facet relationship files: facets/<facet>/entities/<id>/entity.json

    Ensures all entities have an `id` field (generates from name if missing).
    For attached entities, validates name uniqueness within the facet.

    Args:
        facet: Facet name
        entities: List of entity dictionaries (must have type, name, description keys;
                  attached entities may also have id, attached_at, updated_at timestamps)
        day: Optional day in YYYYMMDD format for detected entities

    Raises:
        ValueError: If duplicate names found in attached entities (day=None)
    """
    if day is not None:
        _save_entities_detected(facet, entities, day)
    else:
        _save_entities_attached(facet, entities)


def _locked_modify_detected(
    facet: str,
    day: str,
    modify_fn: callable,
    max_retries: int = 3,
) -> list[EntityDict]:
    """Perform a locked read-modify-write on detected entities.

    Acquires an exclusive file lock, loads current state, applies the
    mutation function, and writes back atomically. Retries with randomized
    backoff on transient OS errors.

    Args:
        facet: Facet name
        day: Day in YYYYMMDD format
        modify_fn: Called with current entity list, must return the new list.
                   May raise ValueError for logical errors (not retried).
        max_retries: Maximum attempts (default 3)

    Returns:
        The entity list as written

    Raises:
        ValueError: From modify_fn (logical errors, not retried)
        OSError: If all retries exhausted on transient errors
    """
    path = detected_entities_path(facet, day)
    lock_path = path.parent / f"{path.name}.lock"

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Fresh load inside lock — sees all prior writers' changes
                    entities = load_entities(facet, day)
                    entities = modify_fn(entities)
                    _save_entities_detected(facet, entities, day)
                    return entities
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except ValueError:
            raise  # Logical errors (duplicate, not found) — don't retry
        except OSError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))

    raise last_error  # type: ignore[misc]


def save_detected_entity(
    facet: str,
    day: str,
    entity_type: str,
    name: str,
    description: str,
) -> EntityDict:
    """Add a single detected entity with concurrency-safe file locking.

    Uses exclusive file locking to serialize concurrent writers to the same
    facet+day file, preventing lost updates. Retries with randomized backoff
    on transient OS errors.

    Args:
        facet: Facet name
        day: Day in YYYYMMDD format
        entity_type: Entity type (e.g. "Person", "Company")
        name: Entity name
        description: Entity description

    Returns:
        The saved entity dict (with generated id)

    Raises:
        ValueError: If entity with same name already detected for this day
        OSError: If all retries exhausted
    """
    new_entity: EntityDict = {
        "type": entity_type,
        "name": name,
        "description": description,
    }
    name_lower = name.lower()

    def _add_entity(entities: list[EntityDict]) -> list[EntityDict]:
        for e in entities:
            if e.get("name", "").lower() == name_lower:
                raise ValueError(f"Entity '{name}' already detected for {day}")
        entities.append(new_entity)
        return entities

    _locked_modify_detected(facet, day, _add_entity)

    # Return with id filled in (set by _save_entities_detected)
    return new_entity


def update_detected_entity(
    facet: str,
    day: str,
    name: str,
    description: str,
) -> EntityDict:
    """Update a detected entity's description with concurrency-safe locking.

    Args:
        facet: Facet name
        day: Day in YYYYMMDD format
        name: Entity name to find
        description: New description

    Returns:
        The updated entity dict

    Raises:
        ValueError: If entity not found
        OSError: If all retries exhausted
    """

    def _update_entity(entities: list[EntityDict]) -> list[EntityDict]:
        for e in entities:
            if e.get("name") == name:
                e["description"] = description
                return entities
        raise ValueError(f"Entity '{name}' not found for {day}")

    result = _locked_modify_detected(facet, day, _update_entity)
    return next(e for e in result if e.get("name") == name)
