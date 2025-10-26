"""Domain-scoped entity utilities for detected and attached entities."""

import json
import os
import re
import tempfile
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


def entity_file_path(domain: str, day: Optional[str] = None) -> Path:
    """Return path to entity file for a domain.

    Args:
        domain: Domain name (e.g., "personal", "work")
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

    domain_path = Path(journal) / "domains" / domain

    if day is None:
        # Attached entities
        return domain_path / "entities.jsonl"
    else:
        # Detected entities for specific day
        return domain_path / "entities" / f"{day}.jsonl"


def load_entities(domain: str, day: Optional[str] = None) -> list[dict[str, Any]]:
    """Load entities from domain entity file.

    Args:
        domain: Domain name
        day: Optional day in YYYYMMDD format for detected entities

    Returns:
        List of entity dictionaries with type, name, and description keys

    Example:
        >>> load_entities("personal")
        [{"type": "Person", "name": "John Smith", "description": "Friend from college"}]
    """
    path = entity_file_path(domain, day)
    return parse_entity_file(str(path))


def save_entities(
    domain: str, entities: list[dict[str, Any]], day: Optional[str] = None
) -> None:
    """Save entities to domain entity file using atomic write.

    Args:
        domain: Domain name
        entities: List of entity dictionaries (must have type, name, description keys)
        day: Optional day in YYYYMMDD format for detected entities

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    path = entity_file_path(domain, day)

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
    domain: str,
    type: str,
    name: str,
    old_description: str,
    new_description: str,
    day: Optional[str] = None,
) -> None:
    """Update an entity's description after validating current state.

    Args:
        domain: Domain name
        type: Entity type to match
        name: Entity name to match
        old_description: Current description (guard - must match)
        new_description: New description to set
        day: Optional day for detected entities

    Raises:
        ValueError: If entity not found or guard mismatch
        RuntimeError: If JOURNAL_PATH is not set
    """
    entities = load_entities(domain, day)

    for entity in entities:
        if entity.get("type") == type and entity.get("name") == name:
            current_desc = entity.get("description", "")
            if current_desc != old_description:
                raise ValueError(
                    f"Description mismatch for '{name}': expected '{old_description}', "
                    f"found '{current_desc}'"
                )
            entity["description"] = new_description
            save_entities(domain, entities, day)
            return

    raise ValueError(f"Entity '{name}' of type '{type}' not found in domain '{domain}'")


def load_all_attached_entities() -> list[dict[str, Any]]:
    """Load all attached entities from all domains with deduplication.

    Iterates domains in sorted (alphabetical) order. When the same entity
    name appears in multiple domains, keeps the first occurrence.

    Returns:
        List of entity dictionaries, deduplicated by name

    Example:
        >>> load_all_attached_entities()
        [{"type": "Person", "name": "John Smith", "description": "Friend from college"}, ...]

    Note:
        Used for agent context loading. Provides deterministic behavior
        despite allowing independent entity descriptions across domains.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    domains_dir = Path(journal) / "domains"
    if not domains_dir.exists():
        return []

    # Track seen names for deduplication
    seen_names = set()
    all_entities = []

    # Process domains in sorted order for deterministic results
    for domain_path in sorted(domains_dir.iterdir()):
        if not domain_path.is_dir():
            continue

        entities_file = domain_path / "entities.jsonl"
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
    domain: str | None = None,
    spoken: bool = False,
) -> str | list[str] | None:
    """Load entity names from entities.md for AI transcription context.

    This function extracts just the entity names (no types or descriptions) from
    entity files. When spoken=False (default), returns them as a
    semicolon-delimited string. When spoken=True, returns a list of shortened forms
    optimized for audio transcription.

    When domain is None, loads and merges entities from ALL domains with
    deduplication (first occurrence wins when same name appears in multiple domains).
    Falls back to top-level entities.md if no domains exist.

    When spoken=True, uses uniform processing for all entity types:
    - Extracts first word from base name (without parentheses)
    - Extracts all items from within parentheses (comma-separated)
    - Examples:
      - "Ryan Reed (R2)" → ["Ryan", "R2"]
      - "Federal Aviation Administration (FAA)" → ["Federal", "FAA"]
      - "Acme Corp" → ["Acme"]
      - "pytest" → ["pytest"]

    Args:
        domain: Optional domain name. If provided, loads from domains/{domain}/entities.md
                If None, loads from ALL domains using load_all_attached_entities(),
                with fallback to top-level entities.md for backward compatibility.
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
        if domain is None:
            # Load from ALL domains with deduplication
            entities = load_all_attached_entities()

            # Fallback to top-level entities.jsonl if no domain entities found
            if not entities:
                from dotenv import load_dotenv

                load_dotenv()
                journal = os.getenv("JOURNAL_PATH")
                if journal:
                    from pathlib import Path

                    entities_path = Path(journal) / "entities.jsonl"
                    if entities_path.is_file():
                        entities = parse_entity_file(str(entities_path))
        else:
            # Load from specific domain
            entities = load_entities(domain)
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


def load_detected_entities_recent(domain: str, days: int = 30) -> list[dict[str, Any]]:
    """Load detected entities from last N days, excluding attached entity names/akas.

    Args:
        domain: Domain name
        days: Number of days to look back (default: 30)

    Returns:
        List of detected entity dictionaries, deduplicated by name,
        excluding any names that match attached entities or their aka values

    Example:
        >>> load_detected_entities_recent("personal", days=30)
        [{"type": "Person", "name": "Charlie", "description": "Met at coffee shop"}]
    """
    from datetime import datetime, timedelta

    # Load attached entities and build exclusion set
    attached = load_entities(domain)
    excluded_names = set()
    for entity in attached:
        name = entity.get("name", "")
        if name:
            excluded_names.add(name)
        # Also exclude aka values
        aka_list = entity.get("aka", [])
        if isinstance(aka_list, list):
            excluded_names.update(aka_list)

    # Calculate date range (last N days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Collect detected entities from date range
    detected = []
    seen_names = set()

    current = start_date
    while current <= end_date:
        day_str = current.strftime("%Y%m%d")
        try:
            day_entities = load_entities(domain, day_str)
            for entity in day_entities:
                name = entity.get("name", "")
                # Skip if already seen, or if matches attached/aka
                if name and name not in seen_names and name not in excluded_names:
                    seen_names.add(name)
                    detected.append(entity)
        except RuntimeError:
            # File doesn't exist for this day, skip
            pass

        current += timedelta(days=1)

    return detected
