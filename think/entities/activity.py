# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity activity tracking and detected entity management.

This module handles:
- Updating last_seen timestamps on entities
- Parsing knowledge graph for entity names
- Loading detected entities with aggregation
"""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from think.entities.core import EntityDict
from think.entities.loading import load_entities, parse_entity_file
from think.entities.matching import find_matching_entity
from think.entities.saving import save_entities
from think.utils import get_journal


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
    # Load ALL attached entities including detached/blocked to avoid data loss on save
    entities = load_entities(
        facet, day=None, include_detached=True, include_blocked=True
    )

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

    Extracts entity names from markdown tables in the knowledge graph output.
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
    kg_path = Path(journal) / day / "agents" / "knowledge_graph.md"

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
        entity = find_matching_entity(activity_name, attached)
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


def load_detected_entities_recent(facet: str, days: int = 30) -> list[EntityDict]:
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
        match = find_matching_entity(name, attached)
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
    detected_map: dict[tuple[str, str], EntityDict] = {}

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
