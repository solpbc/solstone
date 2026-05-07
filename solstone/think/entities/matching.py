# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity matching and resolution.

This module provides entity lookup functions:
- find_matching_entity: Low-level fuzzy matching per name (returns MatchResult)
- build_name_resolution_map: Batch name-to-entity-id resolution
- resolve_entity: High-level resolution with candidates
- validate_aka_uniqueness: Check for aka collisions
"""

import logging
from enum import IntEnum

from solstone.think.entities.core import EntityDict, entity_slug
from solstone.think.entities.loading import load_entities

logger = logging.getLogger(__name__)


class MatchTier(IntEnum):
    """Confidence tier for entity matches, ordered from highest to lowest."""

    EXACT = 1  # exact name, id, or aka
    CASE_INSENSITIVE = 2  # case-insensitive name, id, or aka
    EMAIL = 3  # email address match
    SLUG = 4  # slugified query match against id
    FIRST_WORD = 5  # first-word match (bidirectional)
    TOKEN_SUBSET = 6  # token-subset match
    PREFIX = 7  # prefix-token match
    FUZZY = 8  # fuzzy match via rapidfuzz


class MatchResult(dict):
    """Entity match result with confidence tier.

    Behaves like an EntityDict (dict) so existing callers work unchanged.
    Also exposes .tier for callers that need confidence information.
    """

    tier: MatchTier

    def __init__(self, entity: EntityDict, tier: MatchTier):
        super().__init__(entity)
        self.tier = tier

    @property
    def is_high_confidence(self) -> bool:
        """True for tiers 1-4 (exact, case-insensitive, email, slug)."""
        return self.tier <= MatchTier.SLUG


def _token_subset_match(name_a_lower: str, name_b_lower: str) -> bool:
    """True if all tokens of the shorter name appear in the longer (min 2 tokens in shorter)."""
    tokens_a = set(name_a_lower.split())
    tokens_b = set(name_b_lower.split())
    shorter, longer = sorted([tokens_a, tokens_b], key=len)
    return len(shorter) >= 2 and shorter <= longer


def _prefix_token_match(name_a_lower: str, name_b_lower: str) -> bool:
    """True if sorted tokens are pairwise equal or ≥4-char prefixes of each other."""
    sorted_a = sorted(name_a_lower.split())
    sorted_b = sorted(name_b_lower.split())
    if len(sorted_a) != len(sorted_b):
        return False
    return all(
        a == b or (len(a) >= 4 and b.startswith(a)) or (len(b) >= 4 and a.startswith(b))
        for a, b in zip(sorted_a, sorted_b)
    )


def is_name_variant_match(name_a: str, name_b: str) -> bool:
    """Check if two names are plausible variants of each other.

    Uses three strategies:
    - First-word: one name equals the first word of the other
    - Token-subset: all tokens of the shorter name appear in the longer (min 2 tokens)
    - Prefix-token: same token count, pairwise equal or ≥4-char prefix match
    """
    a_lower = name_a.strip().lower()
    b_lower = name_b.strip().lower()
    if not a_lower or not b_lower:
        return False

    a_words = a_lower.split()
    b_words = b_lower.split()
    if a_lower == b_words[0] or b_lower == a_words[0]:
        return True

    if _token_subset_match(a_lower, b_lower):
        return True

    if _prefix_token_match(a_lower, b_lower):
        return True

    return False


def validate_aka_uniqueness(
    aka: str,
    entities: list[EntityDict],
    exclude_entity_name: str | None = None,
    fuzzy_threshold: int = 90,
) -> str | None:
    """Check if an aka collides with another entity's name or aka.

    Uses the same fuzzy matching logic as find_matching_entity to
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
    # Filter out the entity being updated, detached entities, and blocked entities
    check_entities = [
        e
        for e in entities
        if e.get("name") != exclude_entity_name
        and not e.get("detached")
        and not e.get("blocked")
    ]

    if not check_entities:
        return None

    # Use the existing matching function to detect collisions
    match = find_matching_entity(aka, check_entities, fuzzy_threshold)
    if match:
        return match.get("name")

    return None


def find_matching_entity(
    detected_name: str,
    entities: list[EntityDict],
    fuzzy_threshold: int = 90,
) -> MatchResult | None:
    """Find an entity matching a detected name.

    Works with any list of entity dicts (journal-level or facet-attached).

    Uses tiered matching strategy (in order of precedence):
    1. Exact name, id, or aka match
    2. Case-insensitive name, id, or aka match
    3. Email match
    4. Slugified query match against id
    5. First-word match (bidirectional, unambiguous only, min 3 chars)
    6. Token-subset match (unambiguous only, min 2 tokens in shorter)
    7. Prefix-token match (unambiguous only, ≥4-char prefix)
    8. Fuzzy match using rapidfuzz (score >= threshold)

    Returns a MatchResult (dict subclass with .tier and .is_high_confidence)
    so existing callers that treat the result as a dict work unchanged.
    Tiers 1-4 are high confidence (safe for auto-merge); 5+ are lower.

    Args:
        detected_name: Name, id (slug), or aka to search for
        entities: List of entity dicts to search
        fuzzy_threshold: Minimum score (0-100) for fuzzy matching (default: 90)

    Returns:
        MatchResult with entity data and confidence tier, or None if no match

    Example:
        >>> entities = [{"id": "robert_johnson", "name": "Robert Johnson", "aka": ["Bob", "Bobby"]}]
        >>> result = find_matching_entity("Bob", entities)
        >>> result["id"]
        'robert_johnson'
        >>> result.tier
        <MatchTier.EXACT: 1>
        >>> result.is_high_confidence
        True
    """
    if not detected_name or not entities:
        return None

    detected_lower = detected_name.lower()
    detected_slug = entity_slug(detected_name)

    # Build lookup structures for efficient matching
    # Maps exact-case name/id/aka -> entity (tier 1: exact)
    exact_case_map: dict[str, EntityDict] = {}
    # Maps lowered name/id/aka -> entity (tier 2: case-insensitive)
    lower_map: dict[str, EntityDict] = {}
    # Maps id -> entity for slug matching
    id_map: dict[str, EntityDict] = {}
    # Maps lowercase first word -> list of entities (for ambiguity detection)
    first_word_map: dict[str, list[EntityDict]] = {}
    # All candidate strings for fuzzy matching -> entity
    fuzzy_candidates: dict[str, EntityDict] = {}
    # Maps lowercase email -> entity
    email_map: dict[str, EntityDict] = {}

    for entity in entities:
        name = entity.get("name", "")
        entity_id = entity.get("id", "")
        if not name:
            continue

        name_lower = name.lower()

        # Tier 1: Exact-case name
        exact_case_map[name] = entity
        # Tier 2: Case-insensitive name
        lower_map[name_lower] = entity

        # Also add id (compute from name if not present)
        if entity_id:
            exact_case_map[entity_id] = entity
            lower_map[entity_id.lower()] = entity
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
                    exact_case_map[aka] = entity
                    lower_map[aka.lower()] = entity

        # Build email lookup
        entity_emails = entity.get("emails", [])
        if isinstance(entity_emails, list):
            for email in entity_emails:
                if email:
                    email_map[email.lower()] = entity

        # Tier 5: First word
        first_word = name.split()[0].lower() if name else ""
        if first_word and len(first_word) >= 3:
            if first_word not in first_word_map:
                first_word_map[first_word] = []
            first_word_map[first_word].append(entity)

        # Tier 8: Fuzzy candidates (name and akas)
        fuzzy_candidates[name] = entity
        if isinstance(aka_list, list):
            for aka in aka_list:
                if aka:
                    fuzzy_candidates[aka] = entity

    # Tier 1: Exact match (name, id, or aka — case-sensitive)
    if detected_name in exact_case_map:
        return MatchResult(exact_case_map[detected_name], MatchTier.EXACT)

    # Tier 2: Case-insensitive match
    if detected_lower in lower_map:
        return MatchResult(lower_map[detected_lower], MatchTier.CASE_INSENSITIVE)

    # Tier 3: Email match
    if "@" in detected_name:
        email_match = email_map.get(detected_lower)
        if email_match:
            return MatchResult(email_match, MatchTier.EMAIL)

    # Tier 4: Slugified query match against id
    if detected_slug and detected_slug in id_map:
        return MatchResult(id_map[detected_slug], MatchTier.SLUG)

    # Tier 5: First-word match (bidirectional, only if unambiguous)
    if len(detected_name) >= 3:
        # Short→long: detected name IS a first word of an entity
        matches = first_word_map.get(detected_lower, [])
        if len(matches) == 1:
            return MatchResult(matches[0], MatchTier.FIRST_WORD)

        # Long→short: detected name's first word matches a single-token entity
        detected_first = detected_name.split()[0].lower()
        if detected_first != detected_lower and len(detected_first) >= 3:
            fw_matches = first_word_map.get(detected_first, [])
            if len(fw_matches) == 1:
                matched_name = fw_matches[0].get("name", "")
                # Only match when the entity is a single-token name (e.g.,
                # "Javier Garcia" → "Javier"). Reject when both names are
                # multi-token and merely share a first word (e.g., "Person B"
                # should NOT match "Person A").
                if len(matched_name.split()) == 1:
                    return MatchResult(fw_matches[0], MatchTier.FIRST_WORD)

    # Tier 6: Token-subset match (unambiguous only)
    subset_matches = [
        e
        for e in entities
        if e.get("name") and _token_subset_match(detected_lower, e["name"].lower())
    ]
    if len(subset_matches) == 1:
        return MatchResult(subset_matches[0], MatchTier.TOKEN_SUBSET)

    # Tier 7: Prefix-token match (unambiguous only)
    prefix_matches = [
        e
        for e in entities
        if e.get("name") and _prefix_token_match(detected_lower, e["name"].lower())
    ]
    if len(prefix_matches) == 1:
        return MatchResult(prefix_matches[0], MatchTier.PREFIX)

    # Tier 8: Fuzzy match
    if len(detected_name) >= 4 and fuzzy_candidates:
        try:
            from rapidfuzz import fuzz, process

            fuzz_result = process.extractOne(
                detected_name,
                fuzzy_candidates.keys(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=fuzzy_threshold,
            )
            if fuzz_result:
                matched_str, _score, _index = fuzz_result
                return MatchResult(fuzzy_candidates[matched_str], MatchTier.FUZZY)
        except ImportError:
            # rapidfuzz not available, skip fuzzy matching
            pass

    return None


def build_name_resolution_map(
    signal_names: list[str],
    entities: list[EntityDict],
    fuzzy_threshold: int = 90,
) -> dict[str, str]:
    """Map names to entity IDs using consistent tiered matching.

    Batch version of find_matching_entity() — builds lookup structures once
    and resolves all names, ensuring every call site gets the same resolution
    for the same name.

    Uses tiered matching (in precedence order):
    1. Exact name, id, or aka match
    2. Case-insensitive name, id, or aka match
    3. Slugified query match against id
    4. First-word match (bidirectional, unambiguous only, min 3 chars)
    4b. Token-subset match (unambiguous only, min 2 tokens in shorter)
    4c. Prefix-token match (unambiguous only, ≥4-char prefix)
    5. Fuzzy match via rapidfuzz (score >= threshold)

    Logs when ambiguous first-word matches prevent tier-4 resolution.

    Args:
        signal_names: Names to resolve (e.g., signal entity_name values)
        entities: Entity dicts with at least "id" and "name" (optionally "aka")
        fuzzy_threshold: Minimum fuzzy score (default: 90)

    Returns:
        Dict mapping resolved name → entity_id. Unresolved names are omitted.
    """
    if not entities or not signal_names:
        return {}

    # Pre-build lookup structures once (mirrors find_matching_entity tiers)
    exact_map: dict[str, str] = {}  # name/id/aka/lowered → entity_id
    id_set: set[str] = set()  # all entity IDs for slug matching
    first_word_map: dict[str, list[str]] = {}  # lowercase first word → [entity_ids]
    fuzzy_candidates: dict[str, str] = {}  # candidate string → entity_id
    entity_name_info: list[
        tuple[str, str]
    ] = []  # (entity_id, name_lower) for new tiers

    for entity in entities:
        name = entity.get("name", "")
        entity_id = entity.get("id", "")
        if not name or not entity_id:
            continue

        name_lower = name.lower()
        id_set.add(entity_id)
        entity_name_info.append((entity_id, name_lower))

        # Tiers 1 & 2: exact and case-insensitive for name, id, akas
        exact_map[name] = entity_id
        exact_map[name_lower] = entity_id
        exact_map[entity_id] = entity_id

        aka_list = entity.get("aka", [])
        if isinstance(aka_list, list):
            for aka in aka_list:
                if aka:
                    exact_map[aka] = entity_id
                    exact_map[aka.lower()] = entity_id

        # Tier 4: first-word lookup
        first_word = name.split()[0].lower() if name else ""
        if first_word and len(first_word) >= 3:
            first_word_map.setdefault(first_word, []).append(entity_id)

        # Tier 5: fuzzy candidates
        fuzzy_candidates[name] = entity_id
        if isinstance(aka_list, list):
            for aka in aka_list:
                if aka:
                    fuzzy_candidates[aka] = entity_id

    result: dict[str, str] = {}
    unresolved: list[str] = []

    for sname in signal_names:
        if not sname:
            continue

        sname_lower = sname.lower()
        sname_slug = entity_slug(sname)

        # Tier 1: exact match
        if sname in exact_map:
            result[sname] = exact_map[sname]
            continue

        # Tier 2: case-insensitive match
        if sname_lower in exact_map:
            result[sname] = exact_map[sname_lower]
            continue

        # Tier 3: slugified match against entity IDs
        if sname_slug and sname_slug in id_set:
            result[sname] = sname_slug
            continue

        # Tier 4: first-word match (bidirectional, unambiguous only)
        if len(sname) >= 3:
            matches = first_word_map.get(sname_lower, [])
            if len(matches) == 1:
                result[sname] = matches[0]
                continue
            if len(matches) > 1:
                logger.info(
                    "Ambiguous first-word match for %r: %d candidates %s — skipped",
                    sname,
                    len(matches),
                    matches,
                )

            # Long→short: first word of sname in first_word_map
            sname_first = sname.split()[0].lower()
            if sname_first != sname_lower and len(sname_first) >= 3:
                fw_matches = first_word_map.get(sname_first, [])
                if len(fw_matches) == 1:
                    result[sname] = fw_matches[0]
                    continue

        # Tier 4b: Token-subset match (unambiguous only)
        subset_matches = [
            eid
            for eid, ename in entity_name_info
            if _token_subset_match(sname_lower, ename)
        ]
        if len(subset_matches) == 1:
            result[sname] = subset_matches[0]
            continue

        # Tier 4c: Prefix-token match (unambiguous only)
        prefix_matches = [
            eid
            for eid, ename in entity_name_info
            if _prefix_token_match(sname_lower, ename)
        ]
        if len(prefix_matches) == 1:
            result[sname] = prefix_matches[0]
            continue

        # Defer to fuzzy matching
        unresolved.append(sname)

    # Tier 5: fuzzy matching for remaining names
    if unresolved and fuzzy_candidates:
        try:
            from rapidfuzz import fuzz, process

            for sname in unresolved:
                if len(sname) < 4:
                    continue
                match_result = process.extractOne(
                    sname,
                    fuzzy_candidates.keys(),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=fuzzy_threshold,
                )
                if match_result:
                    matched_str, _score, _index = match_result
                    result[sname] = fuzzy_candidates[matched_str]
        except ImportError:
            pass

    return result


def find_entity_by_email(
    email: str,
    entities: list[EntityDict],
) -> EntityDict | None:
    """Find an entity by email address.

    Args:
        email: Email address to search for
        entities: List of entity dicts to search

    Returns:
        Matched entity dict, or None if no match
    """
    if not email:
        return None
    email_lower = email.lower()
    for entity in entities:
        entity_emails = entity.get("emails", [])
        if isinstance(entity_emails, list):
            if email_lower in [e.lower() for e in entity_emails]:
                return entity
    return None


def resolve_entity(
    facet: str,
    query: str,
    fuzzy_threshold: int = 90,
    include_detached: bool = False,
    include_blocked: bool = False,
) -> tuple[EntityDict | None, list[EntityDict] | None]:
    """Resolve an entity query to a single attached entity.

    This is the primary entry point for tool functions to look up entities.
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
        include_blocked: If True, also search blocked entities (default: False)

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
    entities = load_entities(
        facet,
        day=None,
        include_detached=include_detached,
        include_blocked=include_blocked,
    )
    if not entities:
        return None, []

    # Try to find a match
    match = find_matching_entity(query, entities, fuzzy_threshold)
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
