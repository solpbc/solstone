# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entity seeding functions.

This module handles seeding entities from structured imports:
- seed_entities: Match or create entities and add optional observations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from think.entities.core import EntityDict


def seed_entities(
    facet: str,
    day: str,
    entities: list[dict],
) -> list[EntityDict]:
    """Seed entities from structured imports.

    Each dict should have: name (required), type (default "Person"),
    email (optional), context (optional), observations (optional list of strings).

    Matches by email first, then name. Creates new entities for non-matches.
    If observations are provided, adds them via add_observation() with dedup.

    Args:
        facet: Facet name for entity context
        day: Day string YYYYMMDD for activity tracking
        entities: List of entity dicts to seed

    Returns:
        List of resolved/created entity dicts
    """
    from think.entities.core import entity_slug
    from think.entities.journal import (
        create_journal_entity,
        load_all_journal_entities,
        load_journal_entity,
        save_journal_entity,
    )
    from think.entities.matching import find_entity_by_email, find_matching_entity
    from think.entities.observations import add_observation, load_observations

    # Load all journal entities for matching
    all_entities = load_all_journal_entities()
    entity_list = list(all_entities.values())

    resolved: list[EntityDict] = []

    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue

        entity_type = ent.get("type", "Person")
        email = ent.get("email", "")

        matched = None

        # Try email match first
        if email:
            matched = find_entity_by_email(email, entity_list)

        # Fall back to name match
        if not matched:
            matched = find_matching_entity(name, entity_list)

        if matched:
            # Merge email into existing entity if new
            if email:
                existing_emails = set(e.lower() for e in matched.get("emails", []))
                if email.lower() not in existing_emails:
                    matched["emails"] = sorted(existing_emails | {email.lower()})
                    save_journal_entity(matched)
            resolved.append(matched)
            resolved_name = matched.get("name", name)
        else:
            # Create new entity
            eid = entity_slug(name)
            emails = [email.lower()] if email else None
            new_entity = load_journal_entity(eid) or create_journal_entity(
                entity_id=eid,
                name=name,
                entity_type=entity_type,
                emails=emails,
            )
            entity_list.append(new_entity)  # Add to list for future matches
            resolved.append(new_entity)
            resolved_name = new_entity.get("name", name)

        # Add observations if provided, with dedup
        observations = ent.get("observations", [])
        if observations:
            existing_obs = load_observations(facet, resolved_name)
            existing_contents = {o["content"] for o in existing_obs}
            for obs_content in observations:
                if obs_content not in existing_contents:
                    add_observation(facet, resolved_name, obs_content, source_day=day)
                    existing_contents.add(obs_content)

    return resolved
