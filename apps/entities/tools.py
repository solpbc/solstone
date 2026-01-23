# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP tools for entity management.

This module provides the entity MCP tools for the entities app.
Tools are auto-discovered and registered via the @register_tool decorator.
"""

import re
import time
from typing import Any

from fastmcp import Context

from muse.mcp import HINTS, register_tool
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

# Declare tool pack - creates the "entities" pack with all entity tools
TOOL_PACKS = {
    "entities": [
        "entity_list",
        "entity_detect",
        "entity_attach",
        "entity_update",
        "entity_add_aka",
        "entity_observations",
        "entity_observe",
    ],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_or_error(
    facet: str, name: str
) -> tuple[dict[str, Any] | None, dict | None]:
    """Resolve entity name to entity dict, or return error response.

    Uses resolve_entity() to find an entity by name, id, or aka.
    Returns (entity, None) on success, or (None, error_dict) on failure.

    Args:
        facet: Facet name
        name: Entity name, id (slug), or aka to search for

    Returns:
        Tuple of (entity, error_response):
        - If found: (entity_dict, None)
        - If not found: (None, error_dict with "error" and "suggestion" keys)
    """
    entity, candidates = resolve_entity(facet, name)

    if entity:
        return entity, None

    # Build helpful error message with candidates
    if candidates:
        names = [c.get("name", "") for c in candidates[:3]]
        suggestion = f"Did you mean: {', '.join(names)}?"
    else:
        suggestion = "verify the entity exists in the facet"

    return None, {
        "error": f"Entity '{name}' not found in attached entities",
        "suggestion": suggestion,
    }


# -----------------------------------------------------------------------------
# MCP Tools
# -----------------------------------------------------------------------------


@register_tool(annotations=HINTS)
def entity_list(facet: str, day: str | None = None) -> dict[str, Any]:
    """List entities for a facet.

    Args:
        facet: Facet name (e.g., "personal", "work")
        day: Optional day in YYYYMMDD format. If None, returns attached entities
             from entities.jsonl. If provided, returns detected entities from
             entities/YYYYMMDD.jsonl

    Returns:
        Dictionary containing:
        - facet: The facet name
        - day: The day (or None for attached entities)
        - count: Number of entities found
        - entities: List of entity objects with type, name, description
                    (attached entities may also have attached_at, updated_at)

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
            "suggestion": "check that the facet exists",
        }


@register_tool(annotations=HINTS)
def entity_detect(
    day: str,
    facet: str,
    type: str,
    name: str,
    description: str,
    context: Context | None = None,
) -> dict[str, Any]:
    """Record a detected entity for a specific day in a facet.

    This tool adds an entity to the daily detected entities file at
    facets/{facet}/entities/{day}.jsonl. Detected entities are ephemeral
    observations from a specific day that can later be promoted to attached
    entities if they appear frequently.

    Args:
        day: Day in YYYYMMDD format when entity was detected
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (e.g., Person, Organization, Project, Tool, Location, Event)
        name: Entity name (e.g., "John Smith", "Acme Corp")
        description: Day-specific description of the entity

    Returns:
        Dictionary containing:
        - facet: The facet name
        - day: The day
        - message: Success message
        - entity: The added entity details

    Examples:
        - entity_detect("20250101", "personal", "Person", "Alice", "Met at conference")
        - entity_detect("20250101", "work", "Company", "Acme", "New client prospect")
    """
    try:
        # Validate entity type
        if not is_valid_entity_type(type):
            return {
                "error": f"Invalid entity type '{type}'",
                "suggestion": "must be alphanumeric with spaces only, at least 3 characters long",
            }

        # Load existing entities for the day
        existing = load_entities(facet, day)

        # Check for duplicate by name (case-insensitive, consistent with save validation)
        name_lower = name.lower()
        for entity in existing:
            if entity.get("name", "").lower() == name_lower:
                return {
                    "error": f"Entity '{name}' already detected for {day}",
                    "suggestion": "entity already exists in detected list for this day",
                }

        # Add new entity
        existing.append({"type": type, "name": name, "description": description})
        save_entities(facet, existing, day)
        log_tool_action(
            facet=facet,
            action="entity_detect",
            params={"type": type, "name": name, "description": description},
            context=context,
            day=day,
        )

        return {
            "facet": facet,
            "day": day,
            "message": f"Entity '{name}' detected successfully",
            "entity": {"type": type, "name": name, "description": description},
        }
    except Exception as exc:
        return {
            "error": f"Failed to detect entity: {exc}",
            "suggestion": "check that the facet exists",
        }


@register_tool(annotations=HINTS)
def entity_attach(
    facet: str, type: str, name: str, description: str, context: Context | None = None
) -> dict[str, Any]:
    """Attach an entity permanently to a facet.

    This tool adds an entity to the persistent attached entities file at
    facets/{facet}/entities.jsonl. Attached entities are long-term tracked
    entities that appear in facet summaries and agent context.

    Entity names must be unique within a facet (regardless of type).

    If the entity was previously detached (removed by the user), this tool
    will return an error - the user intentionally removed it, so agents
    should not re-attach it automatically. Users can re-attach manually
    via the web UI if they change their mind.

    Sets attached_at and updated_at timestamps on the new entity.

    Args:
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (e.g., Person, Organization, Project, Tool, Location, Event)
        name: Entity name (e.g., "John Smith", "Acme Corp") - must be unique in facet
        description: Persistent description of the entity

    Returns:
        Dictionary containing:
        - facet: The facet name
        - message: Success message
        - entity: The attached entity details (id, type, name, description)

    Examples:
        - entity_attach("personal", "Person", "Alice", "Close friend from college")
        - entity_attach("work", "Company", "Acme", "Primary client")
    """
    try:
        # Validate entity type
        if not is_valid_entity_type(type):
            return {
                "error": f"Invalid entity type '{type}'",
                "suggestion": "must be alphanumeric with spaces only, at least 3 characters long",
            }

        # Load ALL attached entities including detached ones
        existing = load_entities(facet, day=None, include_detached=True)

        # Check for existing entity by name (case-insensitive, active or detached)
        name_lower = name.lower()
        for entity in existing:
            if entity.get("name", "").lower() == name_lower:
                if entity.get("detached"):
                    # User intentionally removed this entity - don't re-attach
                    return {
                        "error": f"Entity '{name}' was previously removed by the user",
                        "suggestion": (
                            "The user intentionally detached this entity from the "
                            f"'{facet}' facet. Either it's the wrong facet for this "
                            "entity, or it's not important to them. Do not attempt "
                            "to re-attach it."
                        ),
                    }
                else:
                    return {
                        "error": f"Entity '{name}' already attached to facet",
                        "suggestion": "entity names must be unique within a facet",
                    }

        # Add new entity with timestamps (id will be generated by save_entities)
        now = int(time.time() * 1000)
        existing.append(
            {
                "type": type,
                "name": name,
                "description": description,
                "attached_at": now,
                "updated_at": now,
            }
        )
        save_entities(facet, existing, day=None)

        # Log to today's log since attached entities aren't day-scoped
        log_tool_action(
            facet=facet,
            action="entity_attach",
            params={"type": type, "name": name, "description": description},
            context=context,
        )

        return {
            "facet": facet,
            "message": f"Entity '{name}' attached successfully",
            "entity": {"type": type, "name": name, "description": description},
        }
    except Exception as exc:
        return {
            "error": f"Failed to attach entity: {exc}",
            "suggestion": "check that the facet exists",
        }


@register_tool(annotations=HINTS)
def entity_update(
    facet: str,
    name: str,
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

    The name parameter accepts any form of entity reference:
    - Exact formal name: "Alice Johnson"
    - Entity id (slug): "alice_johnson"
    - Alias (aka): "Ali"

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name, id, or aka to update
        old_description: Current description (must match for safety)
        new_description: New description to replace it with
        day: Optional day in YYYYMMDD format for detected entities.
             If None, updates attached entities in entities.jsonl.
             If provided, updates detected entities in entities/YYYYMMDD.jsonl.

    Returns:
        Dictionary containing:
        - facet: The facet name
        - day: The day (or None for attached entities)
        - message: Success message
        - entity: The updated entity details

    Examples:
        - entity_update("personal", "Alice", "Met at conference", "Close colleague from tech conference")
        - entity_update("work", "acme_corp", "New client", "Key client since Q1 2025")
        - entity_update("personal", "Bob", "Friend", "College roommate", day="20250101")
    """
    try:
        # For attached entities, resolve name to find exact entity name
        if day is None:
            entity, error = _resolve_or_error(facet, name)
            if error:
                return error
            # Use the resolved entity's canonical name
            resolved_name = entity.get("name", name)
        else:
            # For detected entities, use name directly (no resolution)
            resolved_name = name

        updated = update_entity_description(
            facet, resolved_name, old_description, new_description, day
        )

        log_tool_action(
            facet=facet,
            action="entity_update",
            params={
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
            "suggestion": "verify the entity exists and try again",
        }
    except Exception as exc:
        return {
            "error": f"Failed to update entity: {exc}",
            "suggestion": "check that the facet exists",
        }


@register_tool(annotations=HINTS)
def entity_add_aka(
    facet: str, name: str, aka: str, context: Context | None = None
) -> dict[str, Any]:
    """Add an alias (aka) to an attached entity.

    This tool adds an alternative name, acronym, or nickname to an attached entity's
    aka list. The aka field is used to improve entity recognition in audio transcription
    and search. Duplicates are automatically prevented - if the alias already exists,
    the operation succeeds with a notification message.

    The name parameter accepts any form of entity reference:
    - Exact formal name: "Jeremie Miller"
    - Entity id (slug): "jeremie_miller"
    - Existing alias (aka): "Jer"

    Special handling:
    - First-word aliases are automatically skipped (e.g., "Jeremie" for "Jeremie Miller")
    - This avoids redundancy since first words are already extracted for transcription
    - Only meaningful aliases like nicknames, acronyms, or abbreviations are added

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name, id, or aka to update
        aka: Alias or acronym to add (e.g., "PG" for "PostgreSQL", "Jer" for "Jeremie Miller")

    Returns:
        Dictionary containing:
        - facet: The facet name
        - message: Success message indicating if aka was added, already existed, or was skipped
        - entity: The updated entity details including the aka list

    Examples:
        - entity_add_aka("work", "PostgreSQL", "Postgres")  # Added
        - entity_add_aka("work", "postgresql", "PG")  # By id, added
        - entity_add_aka("personal", "Jeremie Miller", "Jer")  # Added
        - entity_add_aka("personal", "jeremie_miller", "Jeremie")  # Skipped (first word)
    """
    try:
        # Resolve entity by name, id, or aka
        entity, error = _resolve_or_error(facet, name)
        if error:
            return error

        resolved_name = entity.get("name", "")

        # Check if aka is just the first word of the entity name (silently ignore)
        base_name = re.sub(r"\s*\([^)]+\)", "", resolved_name).strip()
        first_word = base_name.split()[0] if base_name else None
        if first_word and aka.lower() == first_word.lower():
            return {
                "facet": facet,
                "message": f"Alias '{aka}' is already the first word of '{resolved_name}' (skipped)",
                "entity": entity,
            }

        # Get or initialize aka list
        aka_list = entity.get("aka", [])
        if not isinstance(aka_list, list):
            aka_list = []

        # Check if already present (dedup)
        if aka in aka_list:
            return {
                "facet": facet,
                "message": f"Alias '{aka}' already exists for entity '{resolved_name}'",
                "entity": entity,
            }

        # Load all entities for validation and saving
        entities = load_entities(facet, day=None, include_detached=True)

        # Check if aka conflicts with another entity's name or aka
        conflict = validate_aka_uniqueness(aka, entities, exclude_entity_name=resolved_name)
        if conflict:
            return {
                "error": f"Alias '{aka}' conflicts with existing entity '{conflict}'",
                "suggestion": "choose a different alias or merge the entities",
            }
        updated_entity = None
        for e in entities:
            if e.get("name") == resolved_name:
                aka_list.append(aka)
                e["aka"] = aka_list
                e["updated_at"] = int(time.time() * 1000)
                updated_entity = e
                break

        # Save back atomically
        save_entities(facet, entities, day=None)

        # Log to today's log since attached entities aren't day-scoped
        log_tool_action(
            facet=facet,
            action="entity_add_aka",
            params={"name": resolved_name, "aka": aka},
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
            "suggestion": "check that the facet exists and is accessible",
        }


@register_tool(annotations=HINTS)
def entity_observations(facet: str, name: str) -> dict[str, Any]:
    """List observations for an attached entity.

    Observations are durable factoids about an entity that accumulate over time.
    They capture useful information like preferences, expertise, relationships,
    and biographical facts that help with future interactions.

    IMPORTANT: You must call this tool before using entity_observe() to add
    new observations. The count returned is required for the guard validation.

    The name parameter accepts any form of entity reference:
    - Exact formal name: "Alice Johnson"
    - Entity id (slug): "alice_johnson"
    - Alias (aka): "Ali"

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name, id, or aka to get observations for

    Returns:
        Dictionary containing:
        - facet: The facet name
        - entity: The resolved entity details
        - count: Number of observations (use count+1 as observation_number in entity_observe)
        - observations: List of observation objects with content, observed_at, and optional source_day

    Examples:
        - entity_observations("work", "Alice Johnson")
        - entity_observations("personal", "postgresql")  # by id
    """
    try:
        # Resolve entity by name, id, or aka
        entity, error = _resolve_or_error(facet, name)
        if error:
            return error

        resolved_name = entity.get("name", "")
        observations = load_observations(facet, resolved_name)

        return {
            "facet": facet,
            "entity": entity,
            "count": len(observations),
            "observations": observations,
        }
    except Exception as exc:
        return {
            "error": f"Failed to list observations: {exc}",
            "suggestion": "check that the facet and entity exist",
        }


@register_tool(annotations=HINTS)
def entity_observe(
    facet: str,
    name: str,
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

    The name parameter accepts any form of entity reference:
    - Exact formal name: "Alice Johnson"
    - Entity id (slug): "alice_johnson"
    - Alias (aka): "Ali"

    Args:
        facet: Facet name (e.g., "personal", "work")
        name: Entity name, id, or aka to add observation to
        content: The observation text (should be a durable factoid)
        observation_number: Expected next number; must be current count + 1
        source_day: Optional day (YYYYMMDD) when this was observed

    Returns:
        Dictionary containing:
        - facet: The facet name
        - entity: The resolved entity details
        - message: Success message
        - count: Updated observation count
        - observations: Updated list of observations

    Examples:
        - entity_observe("work", "Alice", "Prefers morning meetings", 1)
        - entity_observe("work", "alice_johnson", "Expert in Kubernetes", 2, "20250113")
    """
    try:
        # Resolve entity by name, id, or aka
        entity, error = _resolve_or_error(facet, name)
        if error:
            return error

        resolved_name = entity.get("name", "")

        result = add_observation(
            facet, resolved_name, content, observation_number, source_day
        )

        log_tool_action(
            facet=facet,
            action="entity_observe",
            params={
                "name": resolved_name,
                "content": content,
                "observation_number": observation_number,
            },
            context=context,
        )

        return {
            "facet": facet,
            "entity": entity,
            "message": f"Observation added to '{resolved_name}'",
            "count": result["count"],
            "observations": result["observations"],
        }
    except ObservationNumberError as exc:
        return {
            "error": str(exc),
            "suggestion": f"call entity_observations() first, then retry with observation_number={exc.expected}",
        }
    except ValueError as exc:
        return {
            "error": str(exc),
            "suggestion": "provide a non-empty observation",
        }
    except Exception as exc:
        return {
            "error": f"Failed to add observation: {exc}",
            "suggestion": "check that the facet and entity exist",
        }
