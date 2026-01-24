# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Journal-level entity management.

Journal entities are the canonical identity records stored at:
    entities/<id>/entity.json

They contain identity fields: id, name, type, aka, is_principal, created_at, blocked.
Facet-specific data (description, timestamps) is stored in facet relationships.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any

from think.entities.core import EntityDict, atomic_write, get_identity_names
from think.utils import get_journal


def journal_entity_path(entity_id: str) -> Path:
    """Return path to journal-level entity file.

    Args:
        entity_id: Entity ID (slug)

    Returns:
        Path to entities/<id>/entity.json
    """
    return Path(get_journal()) / "entities" / entity_id / "entity.json"


def load_journal_entity(entity_id: str) -> EntityDict | None:
    """Load a journal-level entity by ID.

    Args:
        entity_id: Entity ID (slug)

    Returns:
        Entity dict with id, name, type, aka, is_principal, created_at fields,
        or None if not found.
    """
    path = journal_entity_path(entity_id)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure id is present
        data["id"] = entity_id
        return data
    except (json.JSONDecodeError, OSError):
        return None


def save_journal_entity(entity: EntityDict) -> None:
    """Save a journal-level entity using atomic write.

    The entity must have an 'id' field. Creates the directory if needed.

    Args:
        entity: Entity dict with id, name, type, aka (optional), is_principal (optional),
                created_at fields.

    Raises:
        ValueError: If entity has no id field
    """
    entity_id = entity.get("id")
    if not entity_id:
        raise ValueError("Entity must have an 'id' field")

    path = journal_entity_path(entity_id)
    content = json.dumps(entity, ensure_ascii=False, indent=2) + "\n"
    atomic_write(path, content, prefix=".entity_")


def scan_journal_entities() -> list[str]:
    """List all entity IDs from journal-level entities.

    Scans entities/ directory for subdirectories containing entity.json.

    Returns:
        List of entity IDs (directory names)
    """
    entities_dir = Path(get_journal()) / "entities"
    if not entities_dir.exists():
        return []

    entity_ids = []
    for entry in entities_dir.iterdir():
        if entry.is_dir() and (entry / "entity.json").exists():
            entity_ids.append(entry.name)

    return sorted(entity_ids)


def load_all_journal_entities() -> dict[str, EntityDict]:
    """Load all journal-level entities.

    Returns:
        Dict mapping entity_id to entity dict
    """
    entity_ids = scan_journal_entities()
    entities = {}
    for entity_id in entity_ids:
        entity = load_journal_entity(entity_id)
        if entity:
            entities[entity_id] = entity
    return entities


def has_journal_principal() -> bool:
    """Check if any journal entity is already flagged as principal.

    Returns:
        True if a principal entity exists, False otherwise
    """
    for entity_id in scan_journal_entities():
        entity = load_journal_entity(entity_id)
        if entity and entity.get("is_principal"):
            return True
    return False


def _should_be_principal(name: str, aka: list[str] | None) -> bool:
    """Check if an entity should be flagged as principal based on identity config.

    Args:
        name: Entity name
        aka: Optional list of aliases

    Returns:
        True if the entity matches identity config, False otherwise
    """
    identity_names = get_identity_names()
    if not identity_names:
        return False

    # Check if name or any aka matches identity
    names_to_check = [name.lower()]
    if aka:
        names_to_check.extend(a.lower() for a in aka)

    for identity_name in identity_names:
        if identity_name.lower() in names_to_check:
            return True

    return False


def get_or_create_journal_entity(
    entity_id: str,
    name: str,
    entity_type: str,
    aka: list[str] | None = None,
    *,
    skip_principal: bool = False,
) -> EntityDict:
    """Get existing journal entity or create new one.

    If entity exists, returns it unchanged (does not update fields).
    If entity doesn't exist, creates it with provided values.

    Args:
        entity_id: Entity ID (slug)
        name: Entity name
        entity_type: Entity type (e.g., "Person", "Company")
        aka: Optional list of aliases
        skip_principal: If True, don't flag as principal even if matches identity

    Returns:
        The existing or newly created entity dict
    """
    existing = load_journal_entity(entity_id)
    if existing:
        return existing

    # Create new entity
    entity: EntityDict = {
        "id": entity_id,
        "name": name,
        "type": entity_type,
        "created_at": int(time.time() * 1000),
    }
    if aka:
        entity["aka"] = aka

    # Check if this should be the principal
    # Only flag if: matches identity, no existing principal, and not skipped
    if (
        not skip_principal
        and _should_be_principal(name, aka)
        and not has_journal_principal()
    ):
        entity["is_principal"] = True

    save_journal_entity(entity)
    return entity


def block_journal_entity(entity_id: str) -> dict[str, Any]:
    """Block a journal entity and detach all facet relationships.

    Sets `blocked: true` on the journal entity and `detached: true` on all
    facet relationships. This is a soft disable that hides the entity from
    active use while preserving all data.

    Args:
        entity_id: Entity ID (slug)

    Returns:
        Dict with:
            - success: True if blocked
            - facets_detached: List of facet names where relationships were detached

    Raises:
        ValueError: If entity not found or is the principal entity
    """
    # Import here to avoid circular dependency
    from think.entities.relationships import load_facet_relationship, save_facet_relationship

    journal_entity = load_journal_entity(entity_id)
    if not journal_entity:
        raise ValueError(f"Entity '{entity_id}' not found")

    if journal_entity.get("is_principal"):
        raise ValueError("Cannot block the principal (self) entity")

    # Set blocked flag on journal entity
    journal_entity["blocked"] = True
    journal_entity["updated_at"] = int(time.time() * 1000)
    save_journal_entity(journal_entity)

    # Detach all facet relationships
    facets_detached = []
    facets_dir = Path(get_journal()) / "facets"
    if facets_dir.exists():
        for facet_path in facets_dir.iterdir():
            if not facet_path.is_dir():
                continue
            facet_name = facet_path.name

            relationship = load_facet_relationship(facet_name, entity_id)
            if relationship and not relationship.get("detached"):
                relationship["detached"] = True
                relationship["updated_at"] = int(time.time() * 1000)
                save_facet_relationship(facet_name, entity_id, relationship)
                facets_detached.append(facet_name)

    return {"success": True, "facets_detached": facets_detached}


def unblock_journal_entity(entity_id: str) -> dict[str, Any]:
    """Unblock a journal entity.

    Clears the `blocked` flag on the journal entity. Does NOT automatically
    reattach facet relationships - the user must do that manually per-facet.

    Args:
        entity_id: Entity ID (slug)

    Returns:
        Dict with:
            - success: True if unblocked

    Raises:
        ValueError: If entity not found or not blocked
    """
    journal_entity = load_journal_entity(entity_id)
    if not journal_entity:
        raise ValueError(f"Entity '{entity_id}' not found")

    if not journal_entity.get("blocked"):
        raise ValueError(f"Entity '{entity_id}' is not blocked")

    # Clear blocked flag
    journal_entity.pop("blocked", None)
    journal_entity["updated_at"] = int(time.time() * 1000)
    save_journal_entity(journal_entity)

    return {"success": True}


def delete_journal_entity(entity_id: str) -> dict[str, Any]:
    """Permanently delete a journal entity and all facet relationships.

    This is a destructive operation that removes:
    - The journal entity directory (entities/<id>/)
    - All facet relationship directories (facets/*/entities/<id>/)
    - All entity memory (voiceprints, observations) in those directories

    Args:
        entity_id: Entity ID (slug)

    Returns:
        Dict with:
            - success: True if deleted
            - facets_deleted: List of facet names where relationships were deleted

    Raises:
        ValueError: If entity not found or is the principal entity
    """
    journal_entity = load_journal_entity(entity_id)
    if not journal_entity:
        raise ValueError(f"Entity '{entity_id}' not found")

    if journal_entity.get("is_principal"):
        raise ValueError("Cannot delete the principal (self) entity")

    facets_deleted = []

    # Delete all facet relationship directories
    facets_dir = Path(get_journal()) / "facets"
    if facets_dir.exists():
        for facet_path in facets_dir.iterdir():
            if not facet_path.is_dir():
                continue
            facet_name = facet_path.name

            # Check for relationship directory (contains entity.json and memory)
            rel_dir = facet_path / "entities" / entity_id
            if rel_dir.exists() and rel_dir.is_dir():
                shutil.rmtree(rel_dir)
                facets_deleted.append(facet_name)

    # Delete journal entity directory
    journal_dir = Path(get_journal()) / "entities" / entity_id
    if journal_dir.exists() and journal_dir.is_dir():
        shutil.rmtree(journal_dir)

    return {"success": True, "facets_deleted": facets_deleted}
