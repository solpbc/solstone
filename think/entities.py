"""Facet-scoped entity utilities for detected and attached entities."""

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


def is_valid_entity_type(etype: str) -> bool:
    """Validate entity type: alphanumeric and spaces only, at least 3 characters."""
    if not etype or len(etype.strip()) < 3:
        return False
    # Must contain only alphanumeric and spaces, and at least one alphanumeric character
    return bool(
        re.match(r"^[A-Za-z0-9 ]+$", etype) and re.search(r"[A-Za-z0-9]", etype)
    )


def parse_entity_file(
    file_path: str, *, validate_types: bool = True
) -> list[dict[str, Any]]:
    """Parse entities from a JSONL file.

    This is the low-level file parsing function used by all entity loading code.
    Each line in the file should be a JSON object with type, name, and description fields.

    Args:
        file_path: Absolute path to entities.jsonl file
        validate_types: If True, filters out invalid entity types (default: True)

    Returns:
        List of entity dictionaries with type, name, and description keys

    Example:
        >>> parse_entity_file("/path/to/entities.jsonl")
        [{"type": "Person", "name": "John Smith", "description": "Friend from college"}]
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

                # Preserve all fields from JSON, ensuring core fields exist
                entity = {"type": etype, "name": name, "description": desc}
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

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    facet_path = Path(journal) / "facets" / facet

    if day is None:
        # Attached entities
        return facet_path / "entities.jsonl"
    else:
        # Detected entities for specific day
        return facet_path / "entities" / f"{day}.jsonl"


def load_entities(facet: str, day: Optional[str] = None) -> list[dict[str, Any]]:
    """Load entities from facet entity file.

    Args:
        facet: Facet name
        day: Optional day in YYYYMMDD format for detected entities

    Returns:
        List of entity dictionaries with type, name, and description keys

    Example:
        >>> load_entities("personal")
        [{"type": "Person", "name": "John Smith", "description": "Friend from college"}]
    """
    path = entity_file_path(facet, day)
    return parse_entity_file(str(path))


def save_entities(
    facet: str, entities: list[dict[str, Any]], day: Optional[str] = None
) -> None:
    """Save entities to facet entity file using atomic write.

    Args:
        facet: Facet name
        entities: List of entity dictionaries (must have type, name, description keys;
                  attached entities may also have attached_at, updated_at timestamps)
        day: Optional day in YYYYMMDD format for detected entities

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    path = entity_file_path(facet, day)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort entities by type, then name for consistency
    sorted_entities = sorted(
        entities, key=lambda e: (e.get("type", ""), e.get("name", ""))
    )

    # Format entities as JSONL
    lines = []
    for entity in sorted_entities:
        lines.append(json.dumps(entity, ensure_ascii=False) + "\n")

    # Atomic write using temp file + rename
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".entities_", suffix=".tmp"
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


def update_entity(
    facet: str,
    type: str,
    name: str,
    old_description: str,
    new_description: str,
    day: Optional[str] = None,
) -> None:
    """Update an entity's description after validating current state.

    Sets updated_at timestamp to current time on successful update.

    Args:
        facet: Facet name
        type: Entity type to match
        name: Entity name to match
        old_description: Current description (guard - must match)
        new_description: New description to set
        day: Optional day for detected entities

    Raises:
        ValueError: If entity not found or guard mismatch
        RuntimeError: If JOURNAL_PATH is not set
    """
    entities = load_entities(facet, day)

    for entity in entities:
        if entity.get("type") == type and entity.get("name") == name:
            current_desc = entity.get("description", "")
            if current_desc != old_description:
                raise ValueError(
                    f"Description mismatch for '{name}': expected '{old_description}', "
                    f"found '{current_desc}'"
                )
            entity["description"] = new_description
            entity["updated_at"] = int(time.time() * 1000)
            save_entities(facet, entities, day)
            return

    raise ValueError(f"Entity '{name}' of type '{type}' not found in facet '{facet}'")


def load_all_attached_entities() -> list[dict[str, Any]]:
    """Load all attached entities from all facets with deduplication.

    Iterates facets in sorted (alphabetical) order. When the same entity
    name appears in multiple facets, keeps the first occurrence.

    Returns:
        List of entity dictionaries, deduplicated by name

    Example:
        >>> load_all_attached_entities()
        [{"type": "Person", "name": "John Smith", "description": "Friend from college"}, ...]

    Note:
        Used for agent context loading. Provides deterministic behavior
        despite allowing independent entity descriptions across facets.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    facets_dir = Path(journal) / "facets"
    if not facets_dir.exists():
        return []

    # Track seen names for deduplication
    seen_names = set()
    all_entities = []

    # Process facets in sorted order for deterministic results
    for facet_path in sorted(facets_dir.iterdir()):
        if not facet_path.is_dir():
            continue

        entities_file = facet_path / "entities.jsonl"
        if not entities_file.exists():
            continue

        # Use parse_entity_file for consistency
        for entity in parse_entity_file(str(entities_file)):
            name = entity.get("name", "")
            # Keep first occurrence only
            if name and name not in seen_names:
                seen_names.add(name)
                all_entities.append(entity)

    return all_entities


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
    try:
        if facet is None:
            # Load from ALL facets with deduplication
            entities = load_all_attached_entities()
        else:
            # Load from specific facet
            entities = load_entities(facet)
    except RuntimeError:
        # JOURNAL_PATH not set
        return None

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
        spoken_names = []

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

        return spoken_names if spoken_names else None


def load_detected_entities_recent(facet: str, days: int = 30) -> list[dict[str, Any]]:
    """Load detected entities from last N days, excluding attached entity names/akas.

    Scans detected entity files in reverse chronological order (newest first),
    aggregating by (type, name) to provide count and last_seen tracking.

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

        Entities are excluded if their name matches an attached entity name or aka.

    Example:
        >>> load_detected_entities_recent("personal", days=30)
        [{"type": "Person", "name": "Charlie", "description": "Met at coffee shop",
          "count": 3, "last_seen": "20250115"}]
    """
    from datetime import datetime, timedelta

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    # Load attached entities and build exclusion set
    attached = load_entities(facet)
    excluded_names: set[str] = set()
    for entity in attached:
        name = entity.get("name", "")
        if name:
            excluded_names.add(name)
        # Also exclude aka values
        aka_list = entity.get("aka", [])
        if isinstance(aka_list, list):
            excluded_names.update(aka_list)

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

            # Skip if matches attached/aka
            if name in excluded_names:
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
