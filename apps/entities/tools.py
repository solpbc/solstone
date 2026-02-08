# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tool functions for entity management.

This module provides callable entity tool functions for the entities app.
"""

import re
from typing import Any

from fastmcp import Context

from think.entities import (
    ObservationNumberError,
    add_observation,
    is_valid_entity_type,
    load_entities,
    load_observations,
    resolve_entity,
    save_entities,
    update_entity_description,
    validate_aka_uniqueness,
)
from think.facets import log_tool_action
from think.utils import now_ms

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_or_error(
    facet: str, entity: str
) -> tuple[dict[str, Any] | None, dict | None]:
    """Resolve entity identifier to entity dict, or return error response.

    Uses resolve_entity() to find an entity by id, name, or aka.
    Returns (entity, None) on success, or (None, error_dict) on failure.

    Also checks for blocked entities - if the query matches a blocked entity,
    returns an error telling the agent to ignore this entity.

    Args:
        facet: Facet name
        entity: Entity identifier - can be entity_id (e.g., "alice_johnson"),
                full name (e.g., "Alice Johnson"), or alias/aka (e.g., "Ali")

    Returns:
        Tuple of (entity, error_response):
        - If found: (entity_dict with id field, None)
        - If blocked: (None, error_dict indicating entity is blocked)
        - If not found: (None, error_dict with candidates for disambiguation)
    """
    # First try to resolve normally (excludes blocked entities)
    resolved, candidates = resolve_entity(facet, entity)

    if resolved:
        return resolved, None

    # Check if the query matches a blocked entity
    blocked_match, _ = resolve_entity(facet, entity, include_blocked=True)
    if blocked_match and blocked_match.get("blocked"):
        return None, {
            "error": f"Entity '{blocked_match.get('name', entity)}' is blocked",
            "suggestion": (
                "This entity has been blocked and should be ignored. "
                "Do not attempt to use or reference this entity."
            ),
        }

    # Build helpful error message with candidates for disambiguation
    if candidates:
        candidate_list = [
            {
                "id": c.get("id", ""),
                "name": c.get("name", ""),
                "type": c.get("type", ""),
            }
            for c in candidates[:3]
        ]
        return None, {
            "error": f"Entity '{entity}' not found in attached entities",
            "candidates": candidate_list,
            "suggestion": "Use entity_id or full name from candidates above",
        }
    else:
        return None, {
            "error": f"Entity '{entity}' not found in attached entities",
            "suggestion": "Verify the entity exists in the facet",
        }


# -----------------------------------------------------------------------------
# Tool functions
# -----------------------------------------------------------------------------


def entity_list(facet: str, day: str | None = None) -> dict[str, Any]:
    """List entities for a facet.

    Args:
        facet: Facet name (e.g., "personal", "work")
        day: Optional day in YYYYMMDD format. If None, returns attached (permanent)
             entities. If provided, returns detected entities for that specific day.

    Returns:
        Dictionary containing:
        - facet: The facet name
        - day: The day (or None for attached entities)
        - count: Number of entities found
        - entities: List of entity objects with id, type, name, description
                    (attached entities also have attached_at, updated_at, last_seen)

    Examples:
        - entity_list("personal")  # List attached entities
        - entity_list("personal", "20250101")  # List detected entities for a day
    """
    try:
        entities = load_entities(facet, day)

        return {
            "facet": facet,
            "day": day,
            "count": len(entities),
            "entities": entities,
        }
    except Exception as exc:
        return {
            "error": f"Failed to list entities: {exc}",
            "suggestion": "Check that the facet exists",
        }


def entity_detect(
    day: str,
    facet: str,
    type: str,
    entity: str,
    description: str,
    context: Context | None = None,
) -> dict[str, Any]:
    """Record a detected entity for a specific day in a facet.

    This tool adds an entity to the daily detected entities file. Detected entities
    are ephemeral observations from a specific day that can later be promoted to
    attached entities if they appear frequently.

    Resolution behavior:
    - If entity matches an attached entity (by id, name, or aka), uses the
      attached entity's canonical name for the detection record.
    - If no match found, uses the provided value as the entity name.

    Args:
        day: Day in YYYYMMDD format when entity was detected
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (e.g., Person, Company, Project, Tool)
        entity: Entity identifier - can be entity_id (e.g., "alice_johnson"),
                full name (e.g., "Alice Johnson"), or alias/aka. If matches an
                attached entity, uses its canonical name. Otherwise uses this
                value as the detection name.
        description: Day-specific description of the entity

    Returns:
        Dictionary containing:
        - facet: The facet name
        - day: The day
        - message: Success message
        - entity: The detected entity details (includes resolved name if matched)
        - resolved_from: The attached entity if one was matched (optional)

    Examples:
        - entity_detect("20250101", "personal", "Person", "Alice", "Met at conference")
        - entity_detect("20250101", "work", "Company", "acme_corp", "Discussed contract")
    """
    try:
        # Validate entity type
        if not is_valid_entity_type(type):
            return {
                "error": f"Invalid entity type '{type}'",
                "suggestion": "Must be alphanumeric with spaces only, at least 3 characters long",
            }

        # Try to resolve to an attached entity
        resolved, _candidates = resolve_entity(facet, entity)

        # Check if query matches a blocked entity
        if not resolved:
            blocked_match, _ = resolve_entity(facet, entity, include_blocked=True)
            if blocked_match and blocked_match.get("blocked"):
                return {
                    "error": f"Entity '{blocked_match.get('name', entity)}' is blocked",
                    "suggestion": (
                        "This entity has been blocked and should be ignored. "
                        "Do not record detections for this entity."
                    ),
                }

        # Use resolved name if found, otherwise use provided value
        if resolved:
            name = resolved.get("name", entity)
        else:
            name = entity

        # Load existing entities for the day
        existing = load_entities(facet, day)

        # Check for duplicate by name (case-insensitive)
        name_lower = name.lower()
        for e in existing:
            if e.get("name", "").lower() == name_lower:
                return {
                    "error": f"Entity '{name}' already detected for {day}",
                    "suggestion": "Entity already exists in detected list for this day",
                }

        # Add new entity
        new_entity = {"type": type, "name": name, "description": description}
        existing.append(new_entity)
        save_entities(facet, existing, day)

        log_tool_action(
            facet=facet,
            action="entity_detect",
            params={
                "type": type,
                "entity": entity,
                "name": name,
                "description": description,
            },
            context=context,
            day=day,
        )

        result = {
            "facet": facet,
            "day": day,
            "message": f"Entity '{name}' detected successfully",
            "entity": new_entity,
        }

        # Include resolved entity info if we matched one
        if resolved:
            result["resolved_from"] = resolved

        return result
    except Exception as exc:
        return {
            "error": f"Failed to detect entity: {exc}",
            "suggestion": "Check that the facet exists",
        }


def entity_attach(
    facet: str, type: str, entity: str, description: str, context: Context | None = None
) -> dict[str, Any]:
    """Attach an entity permanently to a facet.

    This tool adds an entity to the persistent attached entities. Attached entities
    are long-term tracked entities that appear in facet summaries and agent context.

    Resolution behavior:
    - If entity matches an existing attached entity (by id, name, or aka), returns
      that entity without creating a duplicate.
    - If no match found, creates a new entity using the provided value as the name.

    If the entity was previously detached (removed by the user), this tool
    will return an error - the user intentionally removed it, so agents
    should not re-attach it automatically.

    Args:
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (e.g., Person, Company, Project, Tool)
        entity: Entity identifier - can be entity_id (e.g., "alice_johnson"),
                full name (e.g., "Alice Johnson"), or alias/aka. If no match
                found, this value becomes the new entity's name.
        description: Persistent description of the entity

    Returns:
        Dictionary containing:
        - facet: The facet name
        - message: Success message (indicates if found existing or created new)
        - entity: The entity details (includes entity_id)
        - created: Boolean indicating if a new entity was created

    Examples:
        - entity_attach("personal", "Person", "Alice Johnson", "Close friend from college")
        - entity_attach("work", "Company", "acme_corp", "Primary client")  # by id
    """
    try:
        # Validate entity type
        if not is_valid_entity_type(type):
            return {
                "error": f"Invalid entity type '{type}'",
                "suggestion": "Must be alphanumeric with spaces only, at least 3 characters long",
            }

        # Try to resolve to existing entity first (include detached and blocked)
        resolved, _candidates = resolve_entity(
            facet, entity, include_detached=True, include_blocked=True
        )

        if resolved:
            # Found existing entity - check for blocked first (takes precedence)
            if resolved.get("blocked"):
                return {
                    "error": f"Entity '{resolved.get('name', entity)}' is blocked",
                    "suggestion": (
                        "This entity has been blocked and should be ignored. "
                        "Do not attempt to attach or use this entity."
                    ),
                }
            # Check for detached
            if resolved.get("detached"):
                # User intentionally removed this entity - don't re-attach
                return {
                    "error": f"Entity '{resolved.get('name', entity)}' was previously removed by the user",
                    "suggestion": (
                        "The user intentionally detached this entity from the "
                        f"'{facet}' facet. Do not attempt to re-attach it."
                    ),
                }
            # Already attached - return it
            return {
                "facet": facet,
                "message": f"Entity '{resolved.get('name')}' already attached",
                "entity": resolved,
                "created": False,
            }

        # No match found - create new entity using provided value as name
        name = entity  # Use the provided entity value as the new name

        # Load ALL attached entities including detached/blocked for saving
        existing = load_entities(
            facet, day=None, include_detached=True, include_blocked=True
        )

        # Add new entity with timestamps (id will be generated by save_entities)
        now = now_ms()
        new_entity = {
            "type": type,
            "name": name,
            "description": description,
            "attached_at": now,
            "updated_at": now,
        }
        existing.append(new_entity)
        save_entities(facet, existing, day=None)

        # Reload to get the generated id
        saved_entities = load_entities(facet, day=None)
        saved_entity = next(
            (e for e in saved_entities if e.get("name") == name), new_entity
        )

        log_tool_action(
            facet=facet,
            action="entity_attach",
            params={
                "type": type,
                "entity": entity,
                "name": name,
                "description": description,
            },
            context=context,
        )

        return {
            "facet": facet,
            "message": f"Entity '{name}' attached successfully",
            "entity": saved_entity,
            "created": True,
        }
    except Exception as exc:
        return {
            "error": f"Failed to attach entity: {exc}",
            "suggestion": "Check that the facet exists",
        }


def entity_update(
    facet: str,
    entity: str,
    old_description: str,
    new_description: str,
    day: str | None = None,
    context: Context | None = None,
) -> dict[str, Any]:
    """Update an existing entity's description using guard-based validation.

    This tool modifies the description of an entity that's already tracked
    in a facet. To prevent accidental overwrites, you must provide the current
    description as a guard value. If the guard doesn't match the current state,
    the operation fails with an error showing the actual current description.

    Args:
        facet: Facet name (e.g., "personal", "work")
        entity: Entity identifier - can be entity_id (e.g., "alice_johnson"),
                full name (e.g., "Alice Johnson"), or alias/aka (e.g., "Ali")
        old_description: Current description (must match for safety)
        new_description: New description to replace it with
        day: Optional day in YYYYMMDD format for detected entities.
             If None, updates attached entities.
             If provided, updates detected entities for that day.

    Returns:
        Dictionary containing:
        - facet: The facet name
        - day: The day (or None for attached entities)
        - message: Success message
        - entity: The updated entity details (includes entity_id)

    Examples:
        - entity_update("personal", "alice_johnson", "Met at conference", "Close colleague")
        - entity_update("work", "Acme Corp", "New client", "Key client since Q1 2025")
        - entity_update("personal", "Bob", "Friend", "College roommate", day="20250101")
    """
    try:
        # For attached entities, resolve entity to find exact entity name
        if day is None:
            resolved, error = _resolve_or_error(facet, entity)
            if error:
                return error
            # Use the resolved entity's canonical name
            resolved_name = resolved.get("name", entity)
        else:
            # For detected entities, use entity directly (no resolution)
            resolved_name = entity

        updated = update_entity_description(
            facet, resolved_name, old_description, new_description, day
        )

        log_tool_action(
            facet=facet,
            action="entity_update",
            params={
                "entity": entity,
                "name": resolved_name,
                "old_description": old_description,
                "new_description": new_description,
            },
            context=context,
            day=day,
        )

        return {
            "facet": facet,
            "day": day,
            "message": f"Entity '{resolved_name}' updated successfully",
            "entity": updated,
        }
    except ValueError as exc:
        error_msg = str(exc)
        # Extract actual description if it's a guard mismatch
        if "Description mismatch" in error_msg:
            return {
                "error": "Guard mismatch - description has changed",
                "suggestion": error_msg,
            }
        return {
            "error": error_msg,
            "suggestion": "Verify the entity exists and try again",
        }
    except Exception as exc:
        return {
            "error": f"Failed to update entity: {exc}",
            "suggestion": "Check that the facet exists",
        }


def entity_add_aka(
    facet: str, entity: str, aka: str, context: Context | None = None
) -> dict[str, Any]:
    """Add an alias (aka) to an attached entity.

    This tool adds an alternative name, acronym, or nickname to an attached entity's
    aka list. The aka field is used to improve entity recognition in audio transcription
    and search. Duplicates are automatically prevented - if the alias already exists,
    the operation succeeds with a notification message.

    Special handling:
    - First-word aliases are automatically skipped (e.g., "Jeremie" for "Jeremie Miller")
    - This avoids redundancy since first words are already extracted for transcription
    - Only meaningful aliases like nicknames, acronyms, or abbreviations are added

    Args:
        facet: Facet name (e.g., "personal", "work")
        entity: Entity identifier - can be entity_id (e.g., "jeremie_miller"),
                full name (e.g., "Jeremie Miller"), or alias/aka (e.g., "Jer")
        aka: Alias or acronym to add (e.g., "PG" for "PostgreSQL", "Jer" for "Jeremie Miller")

    Returns:
        Dictionary containing:
        - facet: The facet name
        - message: Success message indicating if aka was added, already existed, or was skipped
        - entity: The updated entity details (includes entity_id and aka list)

    Examples:
        - entity_add_aka("work", "postgresql", "PG")  # By id
        - entity_add_aka("work", "PostgreSQL", "Postgres")  # By name
        - entity_add_aka("personal", "Jer", "Jeremy")  # By existing aka
    """
    try:
        # Resolve entity by id, name, or aka
        resolved, error = _resolve_or_error(facet, entity)
        if error:
            return error

        resolved_name = resolved.get("name", "")

        # Check if aka is just the first word of the entity name (silently ignore)
        base_name = re.sub(r"\s*\([^)]+\)", "", resolved_name).strip()
        first_word = base_name.split()[0] if base_name else None
        if first_word and aka.lower() == first_word.lower():
            return {
                "facet": facet,
                "message": f"Alias '{aka}' is already the first word of '{resolved_name}' (skipped)",
                "entity": resolved,
            }

        # Get or initialize aka list
        aka_list = resolved.get("aka", [])
        if not isinstance(aka_list, list):
            aka_list = []

        # Check if already present (dedup)
        if aka in aka_list:
            return {
                "facet": facet,
                "message": f"Alias '{aka}' already exists for entity '{resolved_name}'",
                "entity": resolved,
            }

        # Load all entities for validation and saving (include blocked to preserve on save)
        entities = load_entities(
            facet, day=None, include_detached=True, include_blocked=True
        )

        # Check if aka conflicts with another entity's name or aka
        conflict = validate_aka_uniqueness(
            aka, entities, exclude_entity_name=resolved_name
        )
        if conflict:
            return {
                "error": f"Alias '{aka}' conflicts with existing entity '{conflict}'",
                "suggestion": "Choose a different alias or merge the entities",
            }
        updated_entity = None
        for e in entities:
            if e.get("name") == resolved_name:
                aka_list.append(aka)
                e["aka"] = aka_list
                e["updated_at"] = now_ms()
                updated_entity = e
                break

        # Save back atomically
        save_entities(facet, entities, day=None)

        # Log to today's log since attached entities aren't day-scoped
        log_tool_action(
            facet=facet,
            action="entity_add_aka",
            params={"entity": entity, "name": resolved_name, "aka": aka},
            context=context,
        )

        return {
            "facet": facet,
            "message": f"Added alias '{aka}' to entity '{resolved_name}'",
            "entity": updated_entity,
        }

    except Exception as exc:
        return {
            "error": f"Failed to add aka: {exc}",
            "suggestion": "Check that the facet exists and is accessible",
        }


def entity_observations(facet: str, entity: str) -> dict[str, Any]:
    """List observations for an attached entity.

    Observations are durable factoids about an entity that accumulate over time.
    They capture useful information like preferences, expertise, relationships,
    and biographical facts that help with future interactions.

    IMPORTANT: You must call this tool before using entity_observe() to add
    new observations. The count returned is required for the guard validation.

    Args:
        facet: Facet name (e.g., "personal", "work")
        entity: Entity identifier - can be entity_id (e.g., "alice_johnson"),
                full name (e.g., "Alice Johnson"), or alias/aka (e.g., "Ali")

    Returns:
        Dictionary containing:
        - facet: The facet name
        - entity: The resolved entity details (includes entity_id)
        - count: Number of observations (use count+1 as observation_number in entity_observe)
        - observations: List of observation objects with content, observed_at, and optional source_day

    Examples:
        - entity_observations("work", "alice_johnson")  # by id
        - entity_observations("work", "Alice Johnson")  # by name
        - entity_observations("personal", "Ali")  # by aka
    """
    try:
        # Resolve entity by id, name, or aka
        resolved, error = _resolve_or_error(facet, entity)
        if error:
            return error

        resolved_name = resolved.get("name", "")
        observations = load_observations(facet, resolved_name)

        return {
            "facet": facet,
            "entity": resolved,
            "count": len(observations),
            "observations": observations,
        }
    except Exception as exc:
        return {
            "error": f"Failed to list observations: {exc}",
            "suggestion": "Check that the facet and entity exist",
        }


def entity_observe(
    facet: str,
    entity: str,
    content: str,
    observation_number: int,
    source_day: str | None = None,
    context: Context | None = None,
) -> dict[str, Any]:
    """Add an observation to an attached entity with guard validation.

    Observations are durable factoids about entities - preferences, expertise,
    relationships, schedules, biographical facts, etc. They should be useful
    for future interactions and NOT be day-specific activity logs.

    Good observations:
    - "Prefers async communication over meetings"
    - "Works PST timezone, typically available after 10am"
    - "Has deep expertise in distributed systems and Rust"
    - "Reports to Sarah Chen on the platform team"

    Bad observations (use entity_detect for these):
    - "Discussed API migration today" (day-specific activity)
    - "Sent contract for review" (ephemeral action)

    IMPORTANT: You must call entity_observations() first to get the current
    count. The observation_number must equal count + 1 to prevent stale writes.

    Args:
        facet: Facet name (e.g., "personal", "work")
        entity: Entity identifier - can be entity_id (e.g., "alice_johnson"),
                full name (e.g., "Alice Johnson"), or alias/aka (e.g., "Ali")
        content: The observation text (should be a durable factoid)
        observation_number: Expected next number; must be current count + 1
        source_day: Optional day (YYYYMMDD) when this was observed

    Returns:
        Dictionary containing:
        - facet: The facet name
        - entity: The resolved entity details (includes entity_id)
        - message: Success message
        - count: Updated observation count
        - observations: Updated list of observations

    Examples:
        - entity_observe("work", "alice_johnson", "Prefers morning meetings", 1)
        - entity_observe("work", "Alice Johnson", "Expert in Kubernetes", 2, "20250113")
    """
    try:
        # Resolve entity by id, name, or aka
        resolved, error = _resolve_or_error(facet, entity)
        if error:
            return error

        resolved_name = resolved.get("name", "")

        result = add_observation(
            facet, resolved_name, content, observation_number, source_day
        )

        log_tool_action(
            facet=facet,
            action="entity_observe",
            params={
                "entity": entity,
                "name": resolved_name,
                "content": content,
                "observation_number": observation_number,
            },
            context=context,
        )

        return {
            "facet": facet,
            "entity": resolved,
            "message": f"Observation added to '{resolved_name}'",
            "count": result["count"],
            "observations": result["observations"],
        }
    except ObservationNumberError as exc:
        return {
            "error": str(exc),
            "suggestion": f"Call entity_observations() first, then retry with observation_number={exc.expected}",
        }
    except ValueError as exc:
        return {
            "error": str(exc),
            "suggestion": "Provide a non-empty observation",
        }
    except Exception as exc:
        return {
            "error": f"Failed to add observation: {exc}",
            "suggestion": "Check that the facet and entity exist",
        }
