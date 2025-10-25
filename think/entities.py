"""Domain-scoped entity utilities for detected and attached entities."""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from think.indexer.entities import parse_entity_line


def entity_file_path(domain: str, day: Optional[str] = None) -> Path:
    """Return path to entity file for a domain.

    Args:
        domain: Domain name (e.g., "personal", "work")
        day: Optional day in YYYYMMDD format for detected entities

    Returns:
        Path to entities.md (attached) or entities/YYYYMMDD.md (detected)

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
        return domain_path / "entities.md"
    else:
        # Detected entities for specific day
        return domain_path / "entities" / f"{day}.md"


def load_entities(domain: str, day: Optional[str] = None) -> list[tuple[str, str, str]]:
    """Load entities from domain entity file.

    Args:
        domain: Domain name
        day: Optional day in YYYYMMDD format for detected entities

    Returns:
        List of (type, name, description) tuples

    Example:
        >>> load_entities("personal")
        [("Person", "John Smith", "Friend from college")]
    """
    path = entity_file_path(domain, day)

    if not path.exists():
        return []

    entities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_entity_line(line)
            if parsed:
                entities.append(parsed)

    return entities


def save_entities(
    domain: str, entities: list[tuple[str, str, str]], day: Optional[str] = None
) -> None:
    """Save entities to domain entity file using atomic write.

    Args:
        domain: Domain name
        entities: List of (type, name, description) tuples
        day: Optional day in YYYYMMDD format for detected entities

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    path = entity_file_path(domain, day)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort entities by type, then name for consistency
    sorted_entities = sorted(entities, key=lambda e: (e[0], e[1]))

    # Format entities as markdown
    lines = []
    for etype, name, desc in sorted_entities:
        if desc:
            lines.append(f"- **{etype}**: {name} - {desc}\n")
        else:
            lines.append(f"- **{etype}**: {name}\n")

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

    for i, (etype, ename, desc) in enumerate(entities):
        if etype == type and ename == name:
            if desc != old_description:
                raise ValueError(
                    f"Description mismatch for '{name}': expected '{old_description}', "
                    f"found '{desc}'"
                )
            entities[i] = (type, name, new_description)
            save_entities(domain, entities, day)
            return

    raise ValueError(f"Entity '{name}' of type '{type}' not found in domain '{domain}'")


def load_all_attached_entities() -> list[tuple[str, str, str]]:
    """Load all attached entities from all domains with deduplication.

    Iterates domains in sorted (alphabetical) order. When the same entity
    name appears in multiple domains, keeps the first occurrence.

    Returns:
        List of (type, name, description) tuples, deduplicated by name

    Example:
        >>> load_all_attached_entities()
        [("Person", "John Smith", "Friend from college"), ...]

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

        entities_file = domain_path / "entities.md"
        if not entities_file.exists():
            continue

        with open(entities_file, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_entity_line(line)
                if parsed:
                    etype, name, desc = parsed
                    # Keep first occurrence only
                    if name not in seen_names:
                        seen_names.add(name)
                        all_entities.append((etype, name, desc))

    return all_entities


def load_entity_names(
    *,
    domain: str | None = None,
    spoken: bool = False,
) -> str | list[str] | None:
    """Load entity names from entities.md for AI transcription context.

    This function extracts just the entity names (no types or descriptions) from
    entity files. When spoken=False (default), returns them as a
    comma-delimited string. When spoken=True, returns a list of shortened forms
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
                If False, returns comma-delimited string of full names.

    Returns:
        When spoken=False: Comma-delimited string of entity names (e.g., "John Smith, Acme Corp"),
                          or None if no entities found.
        When spoken=True: List of shortened entity names for speech, or None if no entities found.
    """
    # Load entities using existing utilities
    try:
        if domain is None:
            # Load from ALL domains with deduplication
            entities = load_all_attached_entities()

            # Fallback to top-level entities.md if no domain entities found
            if not entities:
                from dotenv import load_dotenv

                load_dotenv()
                journal = os.getenv("JOURNAL_PATH")
                if journal:
                    from pathlib import Path
                    entities_path = Path(journal) / "entities.md"
                    if entities_path.is_file():
                        with open(entities_path, "r", encoding="utf-8") as f:
                            for line in f:
                                parsed = parse_entity_line(line)
                                if parsed:
                                    entities.append(parsed)
        else:
            # Load from specific domain
            entities = load_entities(domain)
    except RuntimeError:
        # JOURNAL_PATH not set
        return None

    if not entities:
        return None

    # Transform (type, name, desc) tuples into desired format
    if not spoken:
        # Non-spoken mode: comma-delimited string of full names
        entity_names = []
        for _, name, _ in entities:
            if name and name not in entity_names:
                entity_names.append(name)
        return ", ".join(entity_names) if entity_names else None
    else:
        # Spoken mode: list of shortened forms
        spoken_names = []
        for _, name, _ in entities:
            # Get base name (without parens) and extract first word
            base_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
            first_word = base_name.split()[0] if base_name else None

            # Add first word
            if first_word and first_word not in spoken_names:
                spoken_names.append(first_word)

            # Extract and add all items from parens (comma-separated)
            paren_match = re.search(r"\(([^)]+)\)", name)
            if paren_match:
                paren_items = [
                    item.strip() for item in paren_match.group(1).split(",")
                ]
                for item in paren_items:
                    if item and item not in spoken_names:
                        spoken_names.append(item)

        return spoken_names if spoken_names else None
