# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Facet relationship management and entity memory.

Facet relationships link journal entities to specific facets with context:
    facets/<facet>/entities/<id>/entity.json

Entity memory (observations, voiceprints) is stored alongside relationships:
    facets/<facet>/entities/<id>/observations.jsonl
    facets/<facet>/entities/<id>/voiceprints.npz
"""

import json
import shutil
from pathlib import Path

from think.entities.core import EntityDict, atomic_write, entity_slug
from think.utils import get_journal


def facet_relationship_path(facet: str, entity_id: str) -> Path:
    """Return path to facet relationship file.

    Args:
        facet: Facet name
        entity_id: Entity ID (slug)

    Returns:
        Path to facets/<facet>/entities/<id>/entity.json
    """
    return (
        Path(get_journal()) / "facets" / facet / "entities" / entity_id / "entity.json"
    )


def load_facet_relationship(facet: str, entity_id: str) -> EntityDict | None:
    """Load a facet relationship for an entity.

    Args:
        facet: Facet name
        entity_id: Entity ID (slug)

    Returns:
        Relationship dict with entity_id, description, timestamps, etc.,
        or None if not found.
    """
    path = facet_relationship_path(facet, entity_id)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure entity_id is present
        data["entity_id"] = entity_id
        return data
    except (json.JSONDecodeError, OSError):
        return None


def save_facet_relationship(
    facet: str, entity_id: str, relationship: EntityDict
) -> None:
    """Save a facet relationship using atomic write.

    Creates the directory if needed.

    Args:
        facet: Facet name
        entity_id: Entity ID (slug)
        relationship: Relationship dict with description, timestamps, etc.
    """
    path = facet_relationship_path(facet, entity_id)

    # Ensure entity_id is in the relationship
    relationship["entity_id"] = entity_id

    content = json.dumps(relationship, ensure_ascii=False, indent=2) + "\n"
    atomic_write(path, content, prefix=".relationship_")


def scan_facet_relationships(facet: str) -> list[str]:
    """List all entity IDs with relationships in a facet.

    Scans facets/<facet>/entities/ for subdirectories containing entity.json.

    Args:
        facet: Facet name

    Returns:
        List of entity IDs (directory names)
    """
    entities_dir = Path(get_journal()) / "facets" / facet / "entities"
    if not entities_dir.exists():
        return []

    entity_ids = []
    for entry in entities_dir.iterdir():
        if entry.is_dir() and (entry / "entity.json").exists():
            entity_ids.append(entry.name)

    return sorted(entity_ids)


def enrich_relationship_with_journal(
    relationship: EntityDict,
    journal_entity: EntityDict | None,
) -> EntityDict:
    """Merge journal entity fields into relationship for unified view.

    Creates a combined entity dict that has identity fields (name, type, aka,
    is_principal) from journal and relationship fields (description, timestamps,
    etc.) from facet.

    Args:
        relationship: Facet relationship dict
        journal_entity: Journal-level entity dict (or None)

    Returns:
        Merged entity dict with all fields
    """
    # Start with relationship data
    result = dict(relationship)

    # Add identity fields from journal entity
    if journal_entity:
        result["id"] = journal_entity.get("id", relationship.get("entity_id", ""))
        result["name"] = journal_entity.get("name", "")
        result["type"] = journal_entity.get("type", "")
        if journal_entity.get("aka"):
            result["aka"] = journal_entity["aka"]
        if journal_entity.get("is_principal"):
            result["is_principal"] = True
    else:
        # No journal entity - use entity_id as id
        result["id"] = relationship.get("entity_id", "")

    # Remove entity_id from result (use id instead)
    result.pop("entity_id", None)

    return result


def entity_memory_path(facet: str, name: str) -> Path:
    """Return path to entity's memory folder.

    Entity memory folders store persistent data about attached entities:
    observations (durable facts), voiceprints (voice recognition), etc.

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name (will be slugified)

    Returns:
        Path to facets/{facet}/entities/{entity_slug}/

    Raises:
        ValueError: If name slugifies to empty string
    """
    slug = entity_slug(name)
    if not slug:
        raise ValueError(f"Entity name '{name}' slugifies to empty string")

    return Path(get_journal()) / "facets" / facet / "entities" / slug


def ensure_entity_memory(facet: str, name: str) -> Path:
    """Create entity memory folder if needed, return path.

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name (will be slugified)

    Returns:
        Path to the created/existing folder

    Raises:
        ValueError: If name slugifies to empty string
    """
    folder = entity_memory_path(facet, name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def rename_entity_memory(facet: str, old_name: str, new_name: str) -> bool:
    """Rename entity memory folder if it exists.

    Called when an entity is renamed to keep folder in sync.

    Args:
        facet: Facet name
        old_name: Previous entity name
        new_name: New entity name

    Returns:
        True if folder was renamed, False if old folder didn't exist
        or names slugify to the same value

    Raises:
        ValueError: If either name slugifies to empty string
        OSError: If rename fails (e.g., target exists)
    """
    old_folder = entity_memory_path(facet, old_name)
    new_folder = entity_memory_path(facet, new_name)

    # No rename needed if slugified names are the same
    if old_folder == new_folder:
        return False

    if not old_folder.exists():
        return False

    if new_folder.exists():
        raise OSError(f"Target folder already exists: {new_folder}")

    shutil.move(str(old_folder), str(new_folder))
    return True
