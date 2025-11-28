"""MCP tools for entity management.

Note: These functions are registered as MCP tools by muse/mcp.py
They can also be imported and called directly for testing or internal use.
"""

import re
from datetime import datetime
from typing import Any

from fastmcp import Context

from think.entities import (
    is_valid_entity_type,
    load_entities,
    save_entities,
    update_entity,
)
from think.facets import log_tool_action


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
        - entities: List of entity objects with type, name, and description

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
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to list entities: {exc}",
            "suggestion": "check that the facet exists",
        }


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
    facets/{facet}/entities/{day}.md. Detected entities are ephemeral
    observations from a specific day that can later be promoted to attached
    entities if they appear frequently.

    Args:
        day: Day in YYYYMMDD format when entity was detected
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (Person, Company, Project, or Tool)
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

        # Check for duplicate
        for entity in existing:
            if entity.get("type") == type and entity.get("name") == name:
                return {
                    "error": f"Entity '{name}' of type '{type}' already detected for {day}",
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
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to detect entity: {exc}",
            "suggestion": "check that the facet exists",
        }


def entity_attach(
    facet: str, type: str, name: str, description: str, context: Context | None = None
) -> dict[str, Any]:
    """Attach an entity permanently to a facet.

    This tool adds an entity to the persistent attached entities file at
    facets/{facet}/entities.jsonl. Attached entities are long-term tracked
    entities that appear in facet summaries and agent context.

    Args:
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (Person, Company, Project, or Tool)
        name: Entity name (e.g., "John Smith", "Acme Corp")
        description: Persistent description of the entity

    Returns:
        Dictionary containing:
        - facet: The facet name
        - message: Success message
        - entity: The attached entity details

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

        # Load existing attached entities
        existing = load_entities(facet, day=None)

        # Check for duplicate
        for entity in existing:
            if entity.get("type") == type and entity.get("name") == name:
                return {
                    "error": f"Entity '{name}' of type '{type}' already attached",
                    "suggestion": "entity already exists in attached list for this facet",
                }

        # Add new entity
        existing.append({"type": type, "name": name, "description": description})
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
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to attach entity: {exc}",
            "suggestion": "check that the facet exists",
        }


def entity_update(
    facet: str,
    type: str,
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

    Args:
        facet: Facet name (e.g., "personal", "work")
        type: Entity type (Person, Company, Project, or Tool)
        name: Entity name to update
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
        - entity_update("personal", "Person", "Alice", "Met at conference", "Close colleague from tech conference")
        - entity_update("work", "Company", "Acme", "New client", "Key client since Q1 2025")
        - entity_update("personal", "Person", "Bob", "Friend", "College roommate", day="20250101")
    """
    try:
        # Validate entity type
        if not is_valid_entity_type(type):
            return {
                "error": f"Invalid entity type '{type}'",
                "suggestion": "must be alphanumeric with spaces only, at least 3 characters long",
            }

        update_entity(facet, type, name, old_description, new_description, day)

        log_tool_action(
            facet=facet,
            action="entity_update",
            params={
                "type": type,
                "name": name,
                "old_description": old_description,
                "new_description": new_description,
            },
            context=context,
            day=day,
        )

        return {
            "facet": facet,
            "day": day,
            "message": f"Entity '{name}' updated successfully",
            "entity": {"type": type, "name": name, "description": new_description},
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
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to update entity: {exc}",
            "suggestion": "check that the facet exists",
        }


def entity_add_aka(
    facet: str, type: str, name: str, aka: str, context: Context | None = None
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
        type: Entity type (Person, Company, Project, Tool, etc.)
        name: Entity name to update
        aka: Alias or acronym to add (e.g., "PG" for "PostgreSQL", "Jer" for "Jeremie Miller")

    Returns:
        Dictionary containing:
        - facet: The facet name
        - message: Success message indicating if aka was added, already existed, or was skipped
        - entity: The updated entity details including the aka list

    Examples:
        - entity_add_aka("work", "Tool", "PostgreSQL", "Postgres")  # Added
        - entity_add_aka("work", "Tool", "PostgreSQL", "PG")  # Added
        - entity_add_aka("personal", "Person", "Jeremie Miller", "Jer")  # Added
        - entity_add_aka("personal", "Person", "Jeremie Miller", "Jeremie")  # Skipped (first word)
        - entity_add_aka("work", "Organization", "Anthropic PBC", "Anthropic")  # Skipped (first word)
    """
    try:
        # Validate entity type
        if not is_valid_entity_type(type):
            return {
                "error": f"Invalid entity type '{type}'",
                "suggestion": "must be alphanumeric with spaces only, at least 3 characters long",
            }

        # Load attached entities only
        entities = load_entities(facet, day=None)

        # Find and update the entity
        for entity in entities:
            if entity.get("type") == type and entity.get("name") == name:
                # Check if aka is just the first word of the entity name (silently ignore)
                base_name = re.sub(r"\s*\([^)]+\)", "", name).strip()
                first_word = base_name.split()[0] if base_name else None
                if first_word and aka.lower() == first_word.lower():
                    return {
                        "facet": facet,
                        "message": f"Alias '{aka}' is already the first word of '{name}' (skipped)",
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
                        "message": f"Alias '{aka}' already exists for entity '{name}'",
                        "entity": entity,
                    }

                # Add the new aka
                aka_list.append(aka)
                entity["aka"] = aka_list

                # Save back atomically
                save_entities(facet, entities, day=None)

                # Log to today's log since attached entities aren't day-scoped
                log_tool_action(
                    facet=facet,
                    action="entity_add_aka",
                    params={"type": type, "name": name, "aka": aka},
                    context=context,
                )

                return {
                    "facet": facet,
                    "message": f"Added alias '{aka}' to entity '{name}'",
                    "entity": entity,
                }

        # Entity not found
        return {
            "error": f"Entity '{name}' of type '{type}' not found in attached entities",
            "suggestion": "verify the entity exists in the facet (only attached entities supported, not detected)",
        }

    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to add aka: {exc}",
            "suggestion": "check that the facet exists and is accessible",
        }
