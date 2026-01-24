# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity management with journal-wide identity and facet-scoped relationships.

Entity System Architecture:
- Journal-level entities: entities/<id>/entity.json - canonical identity (name, type, aka)
- Facet relationships: facets/<facet>/entities/<id>/entity.json - per-facet data
- Detected entities: facets/<facet>/entities/<day>.jsonl - ephemeral daily discoveries
- Entity memory: facets/<facet>/entities/<id>/ - voiceprints, observations (per-facet)

The system supports both the new structure and legacy entities.jsonl files for
backwards compatibility during migration.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from slugify import slugify

from think.utils import get_config, get_journal

# Default timestamp for entities without activity data (Jan 1 2026 00:00:00 UTC)
# Used as fallback in entity_last_active_ts() to ensure all entities have a sortable value
DEFAULT_ACTIVITY_TS = 1767225600000

# Standard entity types - used for UI suggestions and documentation.
# Custom types are still allowed (validated by is_valid_entity_type regex).
ENTITY_TYPES = [
    {"name": "Person"},
    {"name": "Company"},
    {"name": "Project"},
    {"name": "Tool"},
]


def get_identity_names() -> list[str]:
    """Get all names/aliases for the journal principal from identity config.

    Returns a list of names to match against entities, in display priority order:
    1. identity.preferred (nickname/preferred name) - best for display
    2. identity.name (full name)
    3. identity.aliases (list of alternative names)

    The first element (if any) is the best name for display purposes.
    Returns empty list if identity is not configured.
    """
    config = get_config()
    identity = config.get("identity", {})

    names: list[str] = []

    # Preferred name first (best for display)
    preferred = identity.get("preferred", "").strip()
    if preferred:
        names.append(preferred)

    # Full name
    name = identity.get("name", "").strip()
    if name and name not in names:
        names.append(name)

    # Aliases
    aliases = identity.get("aliases", [])
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str):
                alias = alias.strip()
                if alias and alias not in names:
                    names.append(alias)

    return names


def entity_last_active_ts(entity: dict[str, Any]) -> int:
    """Get entity's last activity timestamp with fallback chain.

    Returns a Unix timestamp (milliseconds) representing when the entity was
    last active, using the following priority:
    1. last_seen (YYYYMMDD string, converted to local midnight)
    2. updated_at (Unix ms)
    3. attached_at (Unix ms)
    4. DEFAULT_ACTIVITY_TS (Jan 1 2026)

    This ensures all entities have a sortable timestamp value.

    Args:
        entity: Entity dictionary with optional last_seen, updated_at, attached_at fields

    Returns:
        Unix timestamp in milliseconds

    Examples:
        >>> entity_last_active_ts({"last_seen": "20260115"})  # Jan 15 2026 local midnight
        >>> entity_last_active_ts({"updated_at": 1700000000000})
        1700000000000
        >>> entity_last_active_ts({})
        1767225600000  # DEFAULT_ACTIVITY_TS (Jan 1 2026 UTC)
    """
    from datetime import datetime

    # Priority 1: last_seen (YYYYMMDD string)
    last_seen = entity.get("last_seen")
    if last_seen and isinstance(last_seen, str) and len(last_seen) == 8:
        try:
            dt = datetime.strptime(last_seen, "%Y%m%d")
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass  # Malformed, fall through

    # Priority 2: updated_at
    updated_at = entity.get("updated_at")
    if updated_at and isinstance(updated_at, int) and updated_at > 0:
        return updated_at

    # Priority 3: attached_at
    attached_at = entity.get("attached_at")
    if attached_at and isinstance(attached_at, int) and attached_at > 0:
        return attached_at

    # Priority 4: Default
    return DEFAULT_ACTIVITY_TS


def is_valid_entity_type(etype: str) -> bool:
    """Validate entity type: alphanumeric and spaces only, at least 3 characters."""
    if not etype or len(etype.strip()) < 3:
        return False
    # Must contain only alphanumeric and spaces, and at least one alphanumeric character
    return bool(
        re.match(r"^[A-Za-z0-9 ]+$", etype) and re.search(r"[A-Za-z0-9]", etype)
    )


# Maximum length for entity slug before truncation
MAX_ENTITY_SLUG_LENGTH = 200


def entity_slug(name: str) -> str:
    """Generate a stable slug identifier for an entity name.

    The slug is used as:
    - The `id` field stored in entity records
    - Folder names for entity memory storage
    - URL-safe programmatic references

    Uses python-slugify to convert names to lowercase with underscores.
    Long names are truncated with a hash suffix to ensure uniqueness.

    Args:
        name: Entity name (e.g., "Alice Johnson", "Acme Corp")

    Returns:
        Slug identifier (e.g., "alice_johnson", "acme_corp")

    Examples:
        >>> entity_slug("Alice Johnson")
        'alice_johnson'
        >>> entity_slug("O'Brien")
        'o_brien'
        >>> entity_slug("AT&T")
        'at_t'
        >>> entity_slug("José García")
        'jose_garcia'
    """
    if not name or not name.strip():
        return ""

    # Use slugify with underscore separator
    slug = slugify(name, separator="_")

    # Handle very long names - truncate and add hash suffix
    if len(slug) > MAX_ENTITY_SLUG_LENGTH:
        # Create hash of full name for uniqueness
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        # Truncate and append hash
        slug = slug[: MAX_ENTITY_SLUG_LENGTH - 9] + "_" + name_hash

    return slug


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


# -----------------------------------------------------------------------------
# Journal-Level Entity Functions
# -----------------------------------------------------------------------------


def journal_entity_path(entity_id: str) -> Path:
    """Return path to journal-level entity file.

    Args:
        entity_id: Entity ID (slug)

    Returns:
        Path to entities/<id>/entity.json
    """
    return Path(get_journal()) / "entities" / entity_id / "entity.json"


def load_journal_entity(entity_id: str) -> dict[str, Any] | None:
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


def save_journal_entity(entity: dict[str, Any]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".entity_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entity, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


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


def load_all_journal_entities() -> dict[str, dict[str, Any]]:
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
) -> dict[str, Any]:
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
    entity = {
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


# -----------------------------------------------------------------------------
# Facet Relationship Functions
# -----------------------------------------------------------------------------


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


def load_facet_relationship(facet: str, entity_id: str) -> dict[str, Any] | None:
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
    facet: str, entity_id: str, relationship: dict[str, Any]
) -> None:
    """Save a facet relationship using atomic write.

    Creates the directory if needed.

    Args:
        facet: Facet name
        entity_id: Entity ID (slug)
        relationship: Relationship dict with description, timestamps, etc.
    """
    path = facet_relationship_path(facet, entity_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure entity_id is in the relationship
    relationship["entity_id"] = entity_id

    # Atomic write
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".relationship_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(relationship, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


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


def _enrich_relationship_with_journal(
    relationship: dict[str, Any],
    journal_entity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge journal entity fields into relationship for unified view.

    Creates a combined entity dict that looks like the legacy format,
    with identity fields (name, type, aka, is_principal) from journal
    and relationship fields (description, timestamps, etc.) from facet.

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


def parse_entity_file(
    file_path: str, *, validate_types: bool = True
) -> list[dict[str, Any]]:
    """Parse entities from a JSONL file.

    This is the low-level file parsing function used by all entity loading code.
    Each line in the file should be a JSON object with type, name, and description fields.

    Generates `id` field (slug) for entities that don't have one, enabling
    lazy migration of existing entity files.

    Args:
        file_path: Absolute path to entities.jsonl file
        validate_types: If True, filters out invalid entity types (default: True)

    Returns:
        List of entity dictionaries with id, type, name, and description keys

    Example:
        >>> parse_entity_file("/path/to/entities.jsonl")
        [{"id": "john_smith", "type": "Person", "name": "John Smith", "description": "Friend from college"}]
    """
    if not os.path.isfile(file_path):
        return []

    entities = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                etype = data.get("type", "")
                name = data.get("name", "")
                desc = data.get("description", "")

                # Validate if requested
                if validate_types and not is_valid_entity_type(etype):
                    continue

                # Generate id from name if not present (lazy migration)
                entity_id = data.get("id") or entity_slug(name)

                # Preserve all fields from JSON, ensuring core fields exist
                # Put id first for readability in JSONL output
                entity = {
                    "id": entity_id,
                    "type": etype,
                    "name": name,
                    "description": desc,
                }
                # Add any additional fields from the JSON
                for key, value in data.items():
                    if key not in entity:
                        entity[key] = value

                entities.append(entity)
            except (json.JSONDecodeError, AttributeError):
                continue  # Skip malformed lines

    return entities


def entity_file_path(facet: str, day: Optional[str] = None) -> Path:
    """Return path to entity file for a facet.

    Args:
        facet: Facet name (e.g., "personal", "work")
        day: Optional day in YYYYMMDD format for detected entities

    Returns:
        Path to entities.jsonl (attached) or entities/YYYYMMDD.jsonl (detected)
    """
    facet_path = Path(get_journal()) / "facets" / facet

    if day is None:
        # Attached entities
        return facet_path / "entities.jsonl"
    else:
        # Detected entities for specific day
        return facet_path / "entities" / f"{day}.jsonl"


def _load_entities_new_structure(
    facet: str, *, include_detached: bool = False
) -> list[dict[str, Any]] | None:
    """Load attached entities from new structure (facet relationships + journal entities).

    Returns None if no new-structure entities exist (fall back to legacy).
    Returns list of enriched entities if any new-structure relationships found.
    """
    entity_ids = scan_facet_relationships(facet)
    if not entity_ids:
        return None  # No new structure, fall back to legacy

    # Load all journal entities for enrichment
    journal_entities = load_all_journal_entities()

    entities = []
    for entity_id in entity_ids:
        relationship = load_facet_relationship(facet, entity_id)
        if relationship is None:
            continue

        # Skip detached if not requested
        if not include_detached and relationship.get("detached"):
            continue

        # Enrich with journal entity data
        journal_entity = journal_entities.get(entity_id)
        enriched = _enrich_relationship_with_journal(relationship, journal_entity)
        entities.append(enriched)

    return entities


def _load_entities_legacy(
    facet: str, *, include_detached: bool = False
) -> list[dict[str, Any]]:
    """Load attached entities from legacy entities.jsonl file.

    Includes deduplication logic for self-healing corrupted files.
    """
    import logging

    path = entity_file_path(facet, day=None)
    entities = parse_entity_file(str(path))

    # Deduplicate by name (case-insensitive) to self-heal
    seen: dict[str, dict[str, Any]] = {}
    duplicates_found = []

    for entity in entities:
        name = entity.get("name", "")
        name_lower = name.lower()

        if name_lower in seen:
            # Duplicate found - keep the one with most recent activity
            existing = seen[name_lower]
            existing_time = entity_last_active_ts(existing)
            current_time = entity_last_active_ts(entity)

            if current_time > existing_time:
                duplicates_found.append(existing.get("name", ""))
                seen[name_lower] = entity
            else:
                duplicates_found.append(name)
        else:
            seen[name_lower] = entity

    if duplicates_found:
        logging.info(
            f"Healed {len(duplicates_found)} duplicate entities in facet "
            f"'{facet}': {duplicates_found}"
        )

    entities = list(seen.values())

    # Filter out detached if not requested
    if not include_detached:
        entities = [e for e in entities if not e.get("detached")]

    return entities


def load_entities(
    facet: str, day: Optional[str] = None, *, include_detached: bool = False
) -> list[dict[str, Any]]:
    """Load entities from facet.

    For attached entities (day=None), tries new structure first (facet relationships
    enriched with journal entities), then falls back to legacy entities.jsonl.

    For detected entities (day provided), loads from day-specific JSONL files.

    Args:
        facet: Facet name
        day: Optional day in YYYYMMDD format for detected entities
        include_detached: If True, includes entities with detached=True.
                         Default False excludes detached entities.
                         Only applies to attached entities (day=None).

    Returns:
        List of entity dictionaries with id, type, name, description, and other fields.

    Example:
        >>> load_entities("personal")
        [{"id": "john_smith", "type": "Person", "name": "John Smith", "description": "Friend"}]
    """
    # For detected entities, use day-specific files (unchanged)
    if day is not None:
        path = entity_file_path(facet, day)
        return parse_entity_file(str(path))

    # For attached entities, try new structure first
    entities = _load_entities_new_structure(facet, include_detached=include_detached)
    if entities is not None:
        return entities

    # Fall back to legacy structure
    return _load_entities_legacy(facet, include_detached=include_detached)


def _ensure_principal_flag(entities: list[dict[str, Any]]) -> None:
    """Ensure exactly one entity is flagged as principal if one matches identity.

    Checks if any entity already has is_principal=True. If not, attempts to
    find an entity matching the journal identity config (name, preferred, aliases)
    and flags it as principal.

    This is called during save_entities() for attached entities only.
    Modifies entities in place.

    Args:
        entities: List of attached entity dicts (modified in place)
    """
    # Check if any entity already has is_principal flag
    for entity in entities:
        if entity.get("is_principal"):
            return  # Already have a principal, nothing to do

    # No principal flagged - try to find one matching identity
    identity_names = get_identity_names()
    if not identity_names:
        return  # No identity configured

    # Build lookup for case-insensitive matching
    # Maps lowercase name/aka -> entity
    name_map: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if entity.get("detached"):
            continue  # Skip detached entities

        name = entity.get("name", "")
        if name:
            name_map[name.lower()] = entity

        # Also check akas
        aka_list = entity.get("aka", [])
        if isinstance(aka_list, list):
            for aka in aka_list:
                if aka:
                    name_map[aka.lower()] = entity

    # Try to match identity names against entities
    for identity_name in identity_names:
        identity_lower = identity_name.lower()
        if identity_lower in name_map:
            # Found a match - flag as principal
            name_map[identity_lower]["is_principal"] = True
            return


def _save_entities_detected(
    facet: str, entities: list[dict[str, Any]], day: str
) -> None:
    """Save detected entities to day-specific JSONL file."""
    path = entity_file_path(facet, day)
    path.parent.mkdir(parents=True, exist_ok=True)

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
    lines = [json.dumps(e, ensure_ascii=False) + "\n" for e in sorted_entities]

    fd, temp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".entities_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def _save_entities_attached(facet: str, entities: list[dict[str, Any]]) -> None:
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
        relationship = {
            "entity_id": entity_id,
        }
        for key, value in entity.items():
            if key not in journal_fields:
                relationship[key] = value

        # Save facet relationship
        save_facet_relationship(facet, entity_id, relationship)


def save_entities(
    facet: str, entities: list[dict[str, Any]], day: Optional[str] = None
) -> None:
    """Save entities to new structure.

    For detected entities (day provided), writes to day-specific JSONL files.
    For attached entities (day=None), writes to:
    - Journal-level entity files: entities/<id>/entity.json (identity)
    - Facet relationship files: facets/<facet>/entities/<id>/entity.json

    Ensures all entities have an `id` field (generates from name if missing).
    For attached entities, validates name uniqueness within the facet and
    ensures the principal entity is flagged at the journal level.

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


def update_entity_description(
    facet: str,
    name: str,
    old_description: str,
    new_description: str,
    day: Optional[str] = None,
) -> dict[str, Any]:
    """Update an entity's description after validating current state.

    Sets updated_at timestamp to current time on successful update.

    Args:
        facet: Facet name
        name: Entity name to match (unique within facet)
        old_description: Current description (guard - must match)
        new_description: New description to set
        day: Optional day for detected entities

    Returns:
        The updated entity dict

    Raises:
        ValueError: If entity not found or guard mismatch
    """
    # Load ALL entities including detached to avoid data loss on save
    # For attached entities (day=None), we need include_detached=True
    entities = (
        load_entities(facet, day, include_detached=True)
        if day is None
        else load_entities(facet, day)
    )

    for entity in entities:
        # Skip detached entities when searching
        if entity.get("detached"):
            continue
        if entity.get("name") == name:
            current_desc = entity.get("description", "")
            if current_desc != old_description:
                raise ValueError(
                    f"Description mismatch for '{name}': expected '{old_description}', "
                    f"found '{current_desc}'"
                )
            entity["description"] = new_description
            entity["updated_at"] = int(time.time() * 1000)
            save_entities(facet, entities, day)
            return entity

    raise ValueError(f"Entity '{name}' not found in facet '{facet}'")


def load_all_attached_entities(
    *,
    sort_by: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load all attached entities from all facets with deduplication.

    Iterates facets in sorted (alphabetical) order. When the same entity
    ID appears in multiple facets, keeps the first occurrence.

    Uses load_entities() for each facet, which handles both new structure
    (journal entities + facet relationships) and legacy entities.jsonl.

    Args:
        sort_by: Optional field to sort by. Currently supports "last_seen"
                 which sorts by recency (entities without the field go to end).
        limit: Optional maximum number of entities to return (applied after
               deduplication and sorting).

    Returns:
        List of entity dictionaries, deduplicated by id

    Example:
        >>> load_all_attached_entities()
        [{"id": "john_smith", "type": "Person", "name": "John Smith", ...}, ...]

        >>> load_all_attached_entities(sort_by="last_seen", limit=20)
        # Returns 20 most recently seen entities

    Note:
        Used for agent context loading. Provides deterministic behavior
        despite allowing independent entity descriptions across facets.
    """
    facets_dir = Path(get_journal()) / "facets"
    if not facets_dir.exists():
        return []

    # Track seen IDs for deduplication (use ID instead of name for uniqueness)
    seen_ids: set[str] = set()
    all_entities: list[dict[str, Any]] = []

    # Process facets in sorted order for deterministic results
    for facet_path in sorted(facets_dir.iterdir()):
        if not facet_path.is_dir():
            continue

        facet_name = facet_path.name

        # Use load_entities which handles both new and legacy structures
        for entity in load_entities(facet_name, include_detached=False):
            entity_id = entity.get("id", "")
            # Keep first occurrence only (deduplicate by ID)
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                all_entities.append(entity)

    # Sort if requested
    if sort_by == "last_seen":
        # Sort by activity timestamp descending (uses full fallback chain)
        all_entities.sort(
            key=entity_last_active_ts,
            reverse=True,
        )

    # Apply limit if requested
    if limit is not None and limit > 0:
        all_entities = all_entities[:limit]

    return all_entities


def _extract_spoken_names(entities: list[dict[str, Any]]) -> list[str]:
    """Extract spoken-form names from entity list.

    Extracts shortened forms optimized for audio transcription:
    - First word from base name (without parentheses)
    - All items from within parentheses (comma-separated)

    Examples:
        - "Ryan Reed (R2)" → ["Ryan", "R2"]
        - "Federal Aviation Administration (FAA)" → ["Federal", "FAA"]
        - "Acme Corp" → ["Acme"]

    Args:
        entities: List of entity dictionaries with "name" and optional "aka" fields

    Returns:
        List of unique spoken names, preserving insertion order
    """
    spoken_names: list[str] = []

    def add_name_variants(name: str) -> None:
        """Extract and add first word + parenthetical items from a name."""
        if not name:
            return

        # Get base name (without parens) and extract first word
        base_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
        first_word = base_name.split()[0] if base_name else None

        # Add first word
        if first_word and first_word not in spoken_names:
            spoken_names.append(first_word)

        # Extract and add all items from parens (comma-separated)
        paren_match = re.search(r"\(([^)]+)\)", name)
        if paren_match:
            paren_items = [item.strip() for item in paren_match.group(1).split(",")]
            for item in paren_items:
                if item and item not in spoken_names:
                    spoken_names.append(item)

    for entity in entities:
        name = entity.get("name", "")
        if name:
            add_name_variants(name)

        # Process aka list with same logic
        aka_list = entity.get("aka", [])
        if isinstance(aka_list, list):
            for aka_name in aka_list:
                add_name_variants(aka_name)

    return spoken_names


def load_entity_names(
    *,
    facet: str | None = None,
    spoken: bool = False,
) -> str | list[str] | None:
    """Load entity names from entities.jsonl for AI transcription context.

    This function extracts just the entity names (no types or descriptions) from
    entity files. When spoken=False (default), returns them as a
    semicolon-delimited string. When spoken=True, returns a list of shortened forms
    optimized for audio transcription.

    When facet is None, loads and merges entities from ALL facets with
    deduplication (first occurrence wins when same name appears in multiple facets).
    Falls back to top-level entities.jsonl if no facets exist.

    When spoken=True, uses uniform processing for all entity types:
    - Extracts first word from base name (without parentheses)
    - Extracts all items from within parentheses (comma-separated)
    - Examples:
      - "Ryan Reed (R2)" → ["Ryan", "R2"]
      - "Federal Aviation Administration (FAA)" → ["Federal", "FAA"]
      - "Acme Corp" → ["Acme"]
      - "pytest" → ["pytest"]

    Args:
        facet: Optional facet name. If provided, loads from facets/{facet}/entities.jsonl
                If None, loads from ALL facets using load_all_attached_entities().
        spoken: If True, returns list of shortened forms for speech recognition.
                If False, returns semicolon-delimited string of full names.

    Returns:
        When spoken=False: Semicolon-delimited string of entity names with aka values in parentheses
                          (e.g., "John Smith (Johnny); Acme Corp (ACME, AcmeCo)"),
                          or None if no entities found.
        When spoken=True: List of shortened entity names for speech, or None if no entities found.
    """
    # Load entities using existing utilities
    if facet is None:
        # Load from ALL facets with deduplication
        entities = load_all_attached_entities()
    else:
        # Load from specific facet
        entities = load_entities(facet)

    if not entities:
        return None

    # Transform entity dicts into desired format
    if not spoken:
        # Non-spoken mode: semicolon-delimited string of full names with aka in parentheses
        entity_names = []
        for entity in entities:
            name = entity.get("name", "")
            if name and name not in entity_names:
                # Check for aka values and append in parentheses
                aka_list = entity.get("aka", [])
                if isinstance(aka_list, list) and aka_list:
                    # Format: "Name (aka1, aka2, aka3)"
                    aka_str = ", ".join(aka_list)
                    formatted_name = f"{name} ({aka_str})"
                else:
                    formatted_name = name
                entity_names.append(formatted_name)
        return "; ".join(entity_names) if entity_names else None
    else:
        # Spoken mode: list of shortened forms
        spoken_names = _extract_spoken_names(entities)
        return spoken_names if spoken_names else None


def load_recent_entity_names(*, limit: int = 20) -> list[str] | None:
    """Load recently active entity names for transcription context.

    Returns spoken-form names from the most recently seen entities across all
    facets. Caller is responsible for formatting the list as needed.

    Args:
        limit: Maximum number of entities to include (default 20)

    Returns:
        List of spoken-form entity names, or None if no entities found.

    Example:
        >>> load_recent_entity_names(limit=5)
        ["Alice", "Bob", "R2", "Acme", "FAA"]
    """
    # Get most recently seen entities
    entities = load_all_attached_entities(sort_by="last_seen", limit=limit)
    if not entities:
        return None

    # Extract spoken names
    spoken_names = _extract_spoken_names(entities)
    if not spoken_names:
        return None

    return spoken_names


def validate_aka_uniqueness(
    aka: str,
    entities: list[dict[str, Any]],
    exclude_entity_name: str | None = None,
    fuzzy_threshold: int = 90,
) -> str | None:
    """Check if an aka collides with another entity's name or aka.

    Uses the same fuzzy matching logic as find_matching_attached_entity to
    catch collisions that would cause ambiguous lookups.

    Args:
        aka: The alias to validate
        entities: List of entity dicts to check against
        exclude_entity_name: Entity name to exclude from checks (the entity
                            being updated). Case-sensitive exact match.
        fuzzy_threshold: Minimum score for fuzzy matching (default: 90)

    Returns:
        Name of conflicting entity if collision found, None if ok

    Example:
        >>> entities = [{"name": "CTT", ...}, {"name": "Other", ...}]
        >>> validate_aka_uniqueness("CTT", entities, exclude_entity_name="Other")
        "CTT"  # Conflicts with entity named "CTT"
        >>> validate_aka_uniqueness("ctt", entities, exclude_entity_name="CTT")
        None  # Ok, adding to CTT's own akas
    """
    # Filter out the entity being updated
    check_entities = [
        e
        for e in entities
        if e.get("name") != exclude_entity_name and not e.get("detached")
    ]

    if not check_entities:
        return None

    # Use the existing matching function to detect collisions
    match = find_matching_attached_entity(aka, check_entities, fuzzy_threshold)
    if match:
        return match.get("name")

    return None


def find_matching_attached_entity(
    detected_name: str,
    attached_entities: list[dict[str, Any]],
    fuzzy_threshold: int = 90,
) -> dict[str, Any] | None:
    """Find an attached entity matching a detected name.

    Uses tiered matching strategy (in order of precedence):
    1. Exact name, id, or aka match
    2. Case-insensitive name, id, or aka match
    3. Slugified query match against id
    4. First-word match (unambiguous only, min 3 chars)
    5. Fuzzy match using rapidfuzz (score >= threshold)

    Args:
        detected_name: Name, id (slug), or aka to search for
        attached_entities: List of attached entity dicts to search
        fuzzy_threshold: Minimum score (0-100) for fuzzy matching (default: 90)

    Returns:
        Matched entity dict, or None if no match found

    Example:
        >>> attached = [{"id": "robert_johnson", "name": "Robert Johnson", "aka": ["Bob", "Bobby"]}]
        >>> find_matching_attached_entity("Bob", attached)
        {"id": "robert_johnson", "name": "Robert Johnson", "aka": ["Bob", "Bobby"]}
        >>> find_matching_attached_entity("robert_johnson", attached)
        {"id": "robert_johnson", "name": "Robert Johnson", "aka": ["Bob", "Bobby"]}
    """
    if not detected_name or not attached_entities:
        return None

    detected_lower = detected_name.lower()
    detected_slug = entity_slug(detected_name)

    # Build lookup structures for efficient matching
    # Maps exact name/id/aka -> entity
    exact_map: dict[str, dict[str, Any]] = {}
    # Maps id -> entity for slug matching
    id_map: dict[str, dict[str, Any]] = {}
    # Maps lowercase first word -> list of entities (for ambiguity detection)
    first_word_map: dict[str, list[dict[str, Any]]] = {}
    # All candidate strings for fuzzy matching -> entity
    fuzzy_candidates: dict[str, dict[str, Any]] = {}

    for entity in attached_entities:
        name = entity.get("name", "")
        entity_id = entity.get("id", "")
        if not name:
            continue

        name_lower = name.lower()

        # Tier 1 & 2: Exact and case-insensitive for name
        exact_map[name] = entity
        exact_map[name_lower] = entity

        # Also add id to exact map (compute from name if not present)
        if entity_id:
            exact_map[entity_id] = entity
            id_map[entity_id] = entity
        else:
            # Compute slug from name for entities without id
            name_slug = entity_slug(name)
            if name_slug:
                id_map[name_slug] = entity

        # Also add akas
        aka_list = entity.get("aka", [])
        if isinstance(aka_list, list):
            for aka in aka_list:
                if aka:
                    exact_map[aka] = entity
                    exact_map[aka.lower()] = entity

        # Tier 4: First word
        first_word = name.split()[0].lower() if name else ""
        if first_word and len(first_word) >= 3:
            if first_word not in first_word_map:
                first_word_map[first_word] = []
            first_word_map[first_word].append(entity)

        # Tier 5: Fuzzy candidates (name and akas)
        fuzzy_candidates[name] = entity
        if isinstance(aka_list, list):
            for aka in aka_list:
                if aka:
                    fuzzy_candidates[aka] = entity

    # Tier 1: Exact match (name, id, or aka)
    if detected_name in exact_map:
        return exact_map[detected_name]

    # Tier 2: Case-insensitive match
    if detected_lower in exact_map:
        return exact_map[detected_lower]

    # Tier 3: Slugified query match against id
    if detected_slug and detected_slug in id_map:
        return id_map[detected_slug]

    # Tier 4: First-word match (only if unambiguous)
    if len(detected_name) >= 3:
        matches = first_word_map.get(detected_lower, [])
        if len(matches) == 1:
            return matches[0]

    # Tier 5: Fuzzy match
    if len(detected_name) >= 4 and fuzzy_candidates:
        try:
            from rapidfuzz import fuzz, process

            result = process.extractOne(
                detected_name,
                fuzzy_candidates.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=fuzzy_threshold,
            )
            if result:
                matched_str, _score, _index = result
                return fuzzy_candidates[matched_str]
        except ImportError:
            # rapidfuzz not available, skip fuzzy matching
            pass

    return None


def resolve_entity(
    facet: str,
    query: str,
    fuzzy_threshold: int = 90,
    include_detached: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Resolve an entity query to a single attached entity.

    This is the primary entry point for MCP tools to look up entities.
    Accepts any form of entity reference (name, id/slug, aka) and resolves
    to a single unambiguous entity.

    Uses tiered matching strategy:
    1. Exact name, id, or aka match
    2. Case-insensitive match
    3. Slugified query match against id
    4. First-word match (only if unambiguous)
    5. Fuzzy match (if single result above threshold)

    Args:
        facet: Facet name (e.g., "personal", "work")
        query: Name, id (slug), or aka to search for
        fuzzy_threshold: Minimum score (0-100) for fuzzy matching (default: 90)
        include_detached: If True, also search detached entities (default: False)

    Returns:
        Tuple of (entity, candidates):
        - If found: (entity_dict, None)
        - If not found: (None, list of closest candidates)
        - If ambiguous: (None, list of matching candidates)

    Examples:
        >>> entity, _ = resolve_entity("work", "Alice Johnson")
        >>> entity, _ = resolve_entity("work", "alice_johnson")  # by id
        >>> entity, _ = resolve_entity("work", "Ali")  # by aka
        >>> _, candidates = resolve_entity("work", "unknown")  # not found
    """
    if not query or not query.strip():
        return None, []

    # Load attached entities
    entities = load_entities(facet, day=None, include_detached=include_detached)
    if not entities:
        return None, []

    # Try to find a match
    match = find_matching_attached_entity(query, entities, fuzzy_threshold)
    if match:
        return match, None

    # No match found - find closest candidates for error message
    # Get top fuzzy matches as suggestions
    candidates: list[dict[str, Any]] = []

    try:
        from rapidfuzz import fuzz, process

        # Build candidate strings
        fuzzy_candidates: dict[str, dict[str, Any]] = {}
        for entity in entities:
            name = entity.get("name", "")
            if name:
                fuzzy_candidates[name] = entity
            aka_list = entity.get("aka", [])
            if isinstance(aka_list, list):
                for aka in aka_list:
                    if aka:
                        fuzzy_candidates[aka] = entity

        # Get top 3 matches regardless of threshold
        results = process.extract(
            query,
            fuzzy_candidates.keys(),
            scorer=fuzz.token_sort_ratio,
            limit=3,
        )
        seen_names: set[str] = set()
        for matched_str, _score, _index in results:
            entity = fuzzy_candidates[matched_str]
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                candidates.append(entity)
    except ImportError:
        # rapidfuzz not available, return first few entities as candidates
        candidates = entities[:3]

    return None, candidates


def touch_entity(facet: str, name: str, day: str) -> str:
    """Update last_seen timestamp on an attached entity.

    Sets the last_seen field to the provided day if the entity exists
    and either has no last_seen or the new day is more recent.

    Args:
        facet: Facet name
        name: Exact name of the attached entity to touch
        day: Day string in YYYYMMDD format

    Returns:
        "updated" if entity was found and last_seen was updated,
        "skipped" if entity was found but day is not more recent,
        "not_found" if entity was not found

    Example:
        >>> touch_entity("work", "Alice Johnson", "20250115")
        "updated"
    """
    # Load ALL attached entities including detached to avoid data loss on save
    entities = load_entities(facet, day=None, include_detached=True)

    for entity in entities:
        # Skip detached entities
        if entity.get("detached"):
            continue
        if entity.get("name") == name:
            current_last_seen = entity.get("last_seen", "")
            # Only update if new day is more recent (or no existing last_seen)
            if not current_last_seen or day > current_last_seen:
                entity["last_seen"] = day
                save_entities(facet, entities, day=None)
                return "updated"
            # Entity found but day is not more recent
            return "skipped"

    return "not_found"


def parse_knowledge_graph_entities(day: str) -> list[str]:
    """Parse entity names from a day's knowledge graph.

    Extracts entity names from markdown tables in the knowledge graph insight.
    Entity names appear in bold (**Name**) in the first column of tables.

    Args:
        day: Day string in YYYYMMDD format

    Returns:
        List of unique entity names found in the knowledge graph.
        Returns empty list if KG doesn't exist or can't be parsed.

    Example:
        >>> parse_knowledge_graph_entities("20260108")
        ["Jeremie Miller (Jer)", "Neal Satterfield", "Flightline", ...]
    """
    journal = get_journal()
    kg_path = Path(journal) / day / "insights" / "knowledge_graph.md"

    if not kg_path.exists():
        return []

    try:
        content = kg_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    # Extract bold names from first column of markdown tables
    # Pattern matches: | **Name** | ... (first column of table rows)
    # Also matches relationship mapping tables: | **Name** | **Target** | ...
    entity_names: set[str] = set()

    # Match table rows with bold text in first or second column
    # Format: | **Entity Name** | Type | ... or | **Source** | **Target** | ...
    table_row_pattern = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\|", re.MULTILINE)

    for match in table_row_pattern.finditer(content):
        name = match.group(1).strip()
        if name:
            entity_names.add(name)

    # Also extract targets from relationship mapping (second column)
    # Format: | **Source** | **Target** | Relationship | ...
    relationship_pattern = re.compile(
        r"^\|\s*\*\*.+?\*\*\s*\|\s*\*\*(.+?)\*\*\s*\|", re.MULTILINE
    )

    for match in relationship_pattern.finditer(content):
        name = match.group(1).strip()
        if name:
            entity_names.add(name)

    return list(entity_names)


def touch_entities_from_activity(
    facet: str, names: list[str], day: str
) -> dict[str, Any]:
    """Update last_seen for attached entities matching activity names.

    For each name in the activity list, attempts to find a matching
    attached entity using fuzzy matching and updates its last_seen field.

    Args:
        facet: Facet name
        names: List of entity names from activity (e.g., knowledge graph)
        day: Day string in YYYYMMDD format

    Returns:
        Summary dict with:
        - matched: List of (activity_name, attached_name) tuples for matches found
        - updated: List of attached entity names that were updated
        - skipped: List of attached entity names already up-to-date

    Example:
        >>> touch_entities_from_activity("work", ["Bob", "FAA"], "20260108")
        {"matched": [("Bob", "Robert Johnson"), ("FAA", "Federal Aviation Administration")],
         "updated": ["Robert Johnson", "Federal Aviation Administration"],
         "skipped": []}
    """
    if not names:
        return {"matched": [], "updated": [], "skipped": []}

    # Load attached entities (excluding detached)
    attached = load_entities(facet, day=None, include_detached=False)
    if not attached:
        return {"matched": [], "updated": [], "skipped": []}

    # Track matches and which entities need updating
    matched: list[tuple[str, str]] = []
    needs_update: dict[str, str] = {}  # attached_name -> most_recent_day

    for activity_name in names:
        entity = find_matching_attached_entity(activity_name, attached)
        if entity:
            attached_name = entity.get("name", "")
            if attached_name:
                matched.append((activity_name, attached_name))
                # Track the day for this entity (may be touched multiple times)
                current = needs_update.get(attached_name, "")
                if not current or day > current:
                    needs_update[attached_name] = day

    # Now batch the updates
    updated: list[str] = []
    skipped: list[str] = []

    for attached_name, update_day in needs_update.items():
        result = touch_entity(facet, attached_name, update_day)
        if result == "updated":
            updated.append(attached_name)
        else:
            # "skipped" (already up-to-date) or "not_found"
            skipped.append(attached_name)

    return {"matched": matched, "updated": updated, "skipped": skipped}


def load_detected_entities_recent(facet: str, days: int = 30) -> list[dict[str, Any]]:
    """Load detected entities from last N days, excluding those matching attached entities.

    Scans detected entity files in reverse chronological order (newest first),
    aggregating by (type, name) to provide count and last_seen tracking.

    Uses fuzzy matching to exclude detected entities that match attached entities
    by name, aka, normalized form, first word, or fuzzy similarity.

    Args:
        facet: Facet name
        days: Number of days to look back (default: 30)

    Returns:
        List of detected entity dictionaries with aggregation data:
        - type: Entity type
        - name: Entity name
        - description: Description from most recent detection
        - count: Number of days entity was detected
        - last_seen: Most recent day (YYYYMMDD) entity was detected

        Entities are excluded if they match an attached entity via fuzzy matching.

    Example:
        >>> load_detected_entities_recent("personal", days=30)
        [{"type": "Person", "name": "Charlie", "description": "Met at coffee shop",
          "count": 3, "last_seen": "20250115"}]
    """
    from datetime import datetime, timedelta

    journal = get_journal()

    # Load attached entities (excluding detached) for fuzzy matching
    # Detached entities should appear in detected list again
    attached = load_entities(facet, include_detached=False)

    # Cache for already-checked names to avoid repeated fuzzy matching
    # Maps detected name -> True (excluded) or False (not excluded)
    exclusion_cache: dict[str, bool] = {}

    def is_excluded(name: str) -> bool:
        """Check if a detected name matches any attached entity."""
        if name in exclusion_cache:
            return exclusion_cache[name]
        match = find_matching_attached_entity(name, attached)
        excluded = match is not None
        exclusion_cache[name] = excluded
        return excluded

    # Calculate date range cutoff
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff_date.strftime("%Y%m%d")

    # Get entities directory and find all day files
    entities_dir = Path(journal) / "facets" / facet / "entities"
    if not entities_dir.exists():
        return []

    # Glob day files and sort descending (newest first)
    day_files = sorted(entities_dir.glob("*.jsonl"), reverse=True)

    # Aggregate entities by (type, name)
    # Key: (type, name) -> {entity data with count, last_seen}
    detected_map: dict[tuple[str, str], dict[str, Any]] = {}

    for day_file in day_files:
        day = day_file.stem  # YYYYMMDD

        # Skip files outside date range
        if day < cutoff_str:
            continue

        # Parse entities from this day
        day_entities = parse_entity_file(str(day_file))

        for entity in day_entities:
            etype = entity.get("type", "")
            name = entity.get("name", "")

            # Skip if matches attached entity (using fuzzy matching)
            if is_excluded(name):
                continue

            key = (etype, name)

            if key not in detected_map:
                # First occurrence (most recent day) - store full entity
                detected_map[key] = {
                    "type": etype,
                    "name": name,
                    "description": entity.get("description", ""),
                    "count": 1,
                    "last_seen": day,
                }
            else:
                # Subsequent occurrence - just increment count
                detected_map[key]["count"] += 1

    return list(detected_map.values())


def format_entities(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format entity JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.
    Works for both attached entities (facets/*/entities.jsonl) and
    detected entities (facets/*/entities/*.jsonl).

    Args:
        entries: Raw JSONL entries (one entity per line)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting facet name and type)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (unix ms)
                - markdown: str
                - source: dict (original entity entry)
            - meta: Dict with optional "header" and "error" keys
    """
    from datetime import datetime

    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []

    # Determine if attached or detected, extract facet name and day
    facet_name = "unknown"
    is_detected = False
    day_str: str | None = None
    file_mtime_ms = 0

    if file_path:
        file_path = Path(file_path)

        # Get file modification time as fallback timestamp (in milliseconds)
        try:
            file_mtime_ms = int(file_path.stat().st_mtime * 1000)
        except (OSError, ValueError):
            pass

        # Extract facet name from path
        # Pattern: facets/{facet}/entities.jsonl or facets/{facet}/entities/{day}.jsonl
        path_str = str(file_path)
        facet_match = re.search(r"facets/([^/]+)/entities", path_str)
        if facet_match:
            facet_name = facet_match.group(1)

        # Check if detected (has day in filename)
        if file_path.parent.name == "entities" and file_path.stem.isdigit():
            is_detected = True
            day_str = file_path.stem

    # Build header
    if is_detected and day_str:
        # Format day as YYYY-MM-DD for readability
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        header_title = f"# Detected Entities: {facet_name} ({formatted_day})\n"
    else:
        header_title = f"# Attached Entities: {facet_name}\n"

    entity_count = len(entries)
    meta["header"] = f"{header_title}\n{entity_count} entities"

    # Calculate base timestamp for detected entities (midnight of that day)
    detected_base_ts = 0
    if is_detected and day_str:
        try:
            dt = datetime.strptime(day_str, "%Y%m%d")
            detected_base_ts = int(dt.timestamp() * 1000)
        except ValueError:
            pass

    # Format each entity as a chunk
    for entity in entries:
        etype = entity.get("type", "Unknown")
        name = entity.get("name", "Unnamed")
        description = entity.get("description", "")

        # Determine timestamp
        if is_detected:
            ts = detected_base_ts
        else:
            # Attached: use activity timestamp (full fallback chain)
            ts = entity_last_active_ts(entity)

        # Build markdown for this entity
        lines = [
            f"### {etype}: {name}\n",
            "",
        ]

        # Description or placeholder
        if description:
            lines.append(description)
        else:
            lines.append("*(No description available)*")
        lines.append("")

        # Additional fields (skip core fields, timestamp fields, id, and detached flag)
        skip_fields = {
            "id",
            "type",
            "name",
            "description",
            "updated_at",
            "attached_at",
            "last_seen",
            "detached",
        }

        # Handle tags specially
        tags = entity.get("tags")
        if tags and isinstance(tags, list):
            lines.append(f"**Tags:** {', '.join(tags)}")

        # Handle aka specially
        aka = entity.get("aka")
        if aka and isinstance(aka, list):
            lines.append(f"**Also known as:** {', '.join(aka)}")

        # Other custom fields
        for key, value in entity.items():
            if key in skip_fields or key in ("tags", "aka"):
                continue
            # Format value appropriately
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            # Capitalize first letter of key for display
            display_key = key.replace("_", " ").title()
            lines.append(f"**{display_key}:** {value_str}")

        lines.append("")

        chunks.append(
            {
                "timestamp": ts,
                "markdown": "\n".join(lines),
                "source": entity,
            }
        )

    # Indexer metadata - topic depends on attached vs detected
    topic = "entity:detected" if is_detected else "entity:attached"
    meta["indexer"] = {"topic": topic}

    return chunks, meta


# -----------------------------------------------------------------------------
# Entity Observations
# -----------------------------------------------------------------------------


class ObservationNumberError(Exception):
    """Raised when observation_number doesn't match expected value."""

    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Observation number mismatch: expected {expected}, got {actual}"
        )


def observations_file_path(facet: str, name: str) -> Path:
    """Return path to observations file for an entity.

    Observations are stored in the entity's memory folder:
    facets/{facet}/entities/{entity_slug}/observations.jsonl

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name (will be slugified)

    Returns:
        Path to observations.jsonl file

    Raises:
        ValueError: If name slugifies to empty string
    """
    folder = entity_memory_path(facet, name)
    return folder / "observations.jsonl"


def load_observations(facet: str, name: str) -> list[dict[str, Any]]:
    """Load observations for an entity.

    Args:
        facet: Facet name
        name: Entity name

    Returns:
        List of observation dictionaries with content, observed_at, source_day keys.
        Returns empty list if file doesn't exist.

    Example:
        >>> load_observations("work", "Alice Johnson")
        [{"content": "Prefers async communication", "observed_at": 1736784000000, "source_day": "20250113"}]
    """
    path = observations_file_path(facet, name)

    if not path.exists():
        return []

    observations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                observations.append(data)
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    return observations


def save_observations(
    facet: str, name: str, observations: list[dict[str, Any]]
) -> None:
    """Save observations to entity's observations file using atomic write.

    Args:
        facet: Facet name
        name: Entity name
        observations: List of observation dictionaries
    """
    path = observations_file_path(facet, name)

    # Create parent directory (entity memory folder) if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Format observations as JSONL
    lines = []
    for obs in observations:
        lines.append(json.dumps(obs, ensure_ascii=False) + "\n")

    # Atomic write using temp file + rename
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".observations_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def add_observation(
    facet: str,
    name: str,
    content: str,
    observation_number: int,
    source_day: str | None = None,
) -> dict[str, Any]:
    """Add an observation to an entity with guard validation.

    Similar to todo_add, requires the caller to provide the expected next
    observation number (current count + 1) to prevent stale writes.

    Args:
        facet: Facet name
        name: Entity name
        content: The observation text
        observation_number: Expected next number; must be current_count + 1
        source_day: Optional day (YYYYMMDD) when observation was made

    Returns:
        Dictionary with updated observations list and count

    Raises:
        ObservationNumberError: If observation_number doesn't match expected
        ValueError: If content is empty

    Example:
        >>> add_observation("work", "Alice", "Prefers morning meetings", 1, "20250113")
        {"observations": [...], "count": 1}
    """
    content = content.strip()
    if not content:
        raise ValueError("Observation content cannot be empty")

    observations = load_observations(facet, name)
    expected = len(observations) + 1

    if observation_number != expected:
        raise ObservationNumberError(expected, observation_number)

    # Create new observation
    observation = {
        "content": content,
        "observed_at": int(time.time() * 1000),
    }
    if source_day:
        observation["source_day"] = source_day

    observations.append(observation)
    save_observations(facet, name, observations)

    return {"observations": observations, "count": len(observations)}
