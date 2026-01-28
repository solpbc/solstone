# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity loading functions.

This module handles loading entities from storage:
- load_entities: Load attached or detected entities for a facet
- load_all_attached_entities: Load from all facets with deduplication
- load_entity_names / load_recent_entity_names: For transcription context
"""

import json
import os
import re
from pathlib import Path

from think.entities.core import (
    EntityDict,
    entity_last_active_ts,
    entity_slug,
    is_valid_entity_type,
)
from think.entities.journal import load_all_journal_entities
from think.entities.relationships import (
    enrich_relationship_with_journal,
    load_facet_relationship,
    scan_facet_relationships,
)
from think.utils import get_journal


def detected_entities_path(facet: str, day: str) -> Path:
    """Return path to detected entities file for a facet and day.

    Args:
        facet: Facet name (e.g., "personal", "work")
        day: Day in YYYYMMDD format

    Returns:
        Path to facets/{facet}/entities/{day}.jsonl
    """
    return Path(get_journal()) / "facets" / facet / "entities" / f"{day}.jsonl"


def parse_entity_file(
    file_path: str, *, validate_types: bool = True
) -> list[EntityDict]:
    """Parse entities from a JSONL file.

    This is the low-level file parsing function used for detected entity files.
    Each line in the file should be a JSON object with type, name, and description fields.

    Generates `id` field (slug) for entities that don't have one.

    Args:
        file_path: Absolute path to entities JSONL file
        validate_types: If True, filters out invalid entity types (default: True)

    Returns:
        List of entity dictionaries with id, type, name, and description keys

    Example:
        >>> parse_entity_file("/path/to/20250101.jsonl")
        [{"id": "john_smith", "type": "Person", "name": "John Smith", "description": "Friend"}]
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

                # Generate id from name if not present
                entity_id = data.get("id") or entity_slug(name)

                # Preserve all fields from JSON, ensuring core fields exist
                # Put id first for readability in JSONL output
                entity: EntityDict = {
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


def _load_entities_from_relationships(
    facet: str, *, include_detached: bool = False, include_blocked: bool = False
) -> list[EntityDict]:
    """Load attached entities from facet relationships + journal entities.

    Args:
        facet: Facet name
        include_detached: If True, includes detached entities
        include_blocked: If True, includes blocked entities (journal-level block)

    Returns:
        List of enriched entity dicts
    """
    entity_ids = scan_facet_relationships(facet)
    if not entity_ids:
        return []

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
        enriched = enrich_relationship_with_journal(relationship, journal_entity)

        # Skip blocked if not requested (blocked is set from journal entity)
        if not include_blocked and enriched.get("blocked"):
            continue

        entities.append(enriched)

    return entities


def load_entities(
    facet: str,
    day: str | None = None,
    *,
    include_detached: bool = False,
    include_blocked: bool = False,
) -> list[EntityDict]:
    """Load entities from facet.

    For attached entities (day=None), loads from facet relationships
    enriched with journal entity data.

    For detected entities (day provided), loads from day-specific JSONL files.

    Args:
        facet: Facet name
        day: Optional day in YYYYMMDD format for detected entities
        include_detached: If True, includes entities with detached=True.
                         Default False excludes detached entities.
                         Only applies to attached entities (day=None).
        include_blocked: If True, includes entities with blocked=True (journal-level).
                        Default False excludes blocked entities.
                        Only applies to attached entities (day=None).

    Returns:
        List of entity dictionaries with id, type, name, description, and other fields.

    Example:
        >>> load_entities("personal")
        [{"id": "john_smith", "type": "Person", "name": "John Smith", "description": "Friend"}]
    """
    # For detected entities, use day-specific files
    if day is not None:
        path = detected_entities_path(facet, day)
        return parse_entity_file(str(path))

    # For attached entities, load from relationships
    return _load_entities_from_relationships(
        facet, include_detached=include_detached, include_blocked=include_blocked
    )


def load_all_attached_entities(
    *,
    sort_by: str | None = None,
    limit: int | None = None,
) -> list[EntityDict]:
    """Load all attached entities from all facets with deduplication.

    Iterates facets in sorted (alphabetical) order. When the same entity
    ID appears in multiple facets, keeps the first occurrence.

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
    all_entities: list[EntityDict] = []

    # Process facets in sorted order for deterministic results
    for facet_path in sorted(facets_dir.iterdir()):
        if not facet_path.is_dir():
            continue

        facet_name = facet_path.name

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


def _is_speakable(name: str) -> bool:
    """Check if a name is suitable for speech recognition vocabularies.

    Allows letters, digits, spaces, periods, hyphens, and apostrophes.
    Must contain at least one letter (Rev.ai requirement).
    Rejects underscores and other programming symbols.

    Args:
        name: The name to check

    Returns:
        True if the name is speakable (has a letter, no underscores/symbols)
    """
    # Must have at least one letter, only allowed chars, no underscores
    return bool(re.fullmatch(r"[a-zA-Z0-9\s.\-']+", name)) and any(
        c.isalpha() for c in name
    )


def _extract_spoken_names(entities: list[EntityDict]) -> list[str]:
    """Extract spoken-form names from entity list.

    Extracts shortened forms optimized for audio transcription:
    - First word from base name (without parentheses)
    - All items from within parentheses (comma-separated)
    - Filters out names with underscores or no letters (not speakable)

    Examples:
        - "Ryan Reed (R2)" → ["Ryan", "R2"]
        - "Federal Aviation Administration (FAA)" → ["Federal", "FAA"]
        - "Acme Corp" → ["Acme"]
        - "send2trash" → ["send2trash"] (allowed: has letters)
        - "entity_registry" → [] (filtered: contains underscore)

    Args:
        entities: List of entity dictionaries with "name" and optional "aka" fields

    Returns:
        List of unique spoken names, preserving insertion order
    """
    spoken_names: list[str] = []

    def add_if_speakable(name: str) -> None:
        """Add name to spoken_names if it's speakable and not already present."""
        if name and name not in spoken_names and _is_speakable(name):
            spoken_names.append(name)

    def add_name_variants(name: str) -> None:
        """Extract and add first word + parenthetical items from a name."""
        if not name:
            return

        # Get base name (without parens) and extract first word
        base_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
        first_word = base_name.split()[0] if base_name else None

        # Add first word if speakable
        add_if_speakable(first_word)

        # Extract and add all items from parens (comma-separated)
        paren_match = re.search(r"\(([^)]+)\)", name)
        if paren_match:
            paren_items = [item.strip() for item in paren_match.group(1).split(",")]
            for item in paren_items:
                add_if_speakable(item)

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
    """Load entity names from entities for AI transcription context.

    This function extracts just the entity names (no types or descriptions) from
    entity files. When spoken=False (default), returns them as a
    semicolon-delimited string. When spoken=True, returns a list of shortened forms
    optimized for audio transcription.

    When facet is None, loads and merges entities from ALL facets with
    deduplication (first occurrence wins when same name appears in multiple facets).

    When spoken=True, uses uniform processing for all entity types:
    - Extracts first word from base name (without parentheses)
    - Extracts all items from within parentheses (comma-separated)
    - Examples:
      - "Ryan Reed (R2)" → ["Ryan", "R2"]
      - "Federal Aviation Administration (FAA)" → ["Federal", "FAA"]
      - "Acme Corp" → ["Acme"]
      - "pytest" → ["pytest"]

    Args:
        facet: Optional facet name. If provided, loads from that facet only.
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
