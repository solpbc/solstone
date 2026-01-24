# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity matching and resolution.

This module provides entity lookup functions:
- find_matching_attached_entity: Low-level fuzzy matching
- resolve_entity: High-level resolution with candidates
- validate_aka_uniqueness: Check for aka collisions
"""

from think.entities.core import EntityDict, entity_slug
from think.entities.loading import load_entities


def validate_aka_uniqueness(
    aka: str,
    entities: list[EntityDict],
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
    attached_entities: list[EntityDict],
    fuzzy_threshold: int = 90,
) -> EntityDict | None:
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
    exact_map: dict[str, EntityDict] = {}
    # Maps id -> entity for slug matching
    id_map: dict[str, EntityDict] = {}
    # Maps lowercase first word -> list of entities (for ambiguity detection)
    first_word_map: dict[str, list[EntityDict]] = {}
    # All candidate strings for fuzzy matching -> entity
    fuzzy_candidates: dict[str, EntityDict] = {}

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
) -> tuple[EntityDict | None, list[EntityDict] | None]:
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
    candidates: list[EntityDict] = []

    try:
        from rapidfuzz import fuzz, process

        # Build candidate strings
        fuzzy_candidates: dict[str, EntityDict] = {}
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
