# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Entities app routes - facet entity management."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

from apps.utils import log_app_action
from convey import state
from think.entities import (
    entity_last_active_ts,
    entity_memory_path,
    entity_slug,
    is_valid_entity_type,
    load_detected_entities_recent,
    load_entities,
    load_facet_relationship,
    load_journal_entity,
    load_observations,
    rename_entity_memory,
    save_entities,
    save_journal_entity,
    scan_facet_relationships,
    scan_journal_entities,
    validate_aka_uniqueness,
)
from think.facets import get_facets

entities_bp = Blueprint(
    "app:entities",
    __name__,
    url_prefix="/app/entities",
)


def _get_entity_metadata(facet_name: str, entity_name: str) -> dict:
    """Get observation count and voiceprint status for an entity.

    Args:
        facet_name: The facet name
        entity_name: The entity name

    Returns:
        dict with observation_count and has_voiceprint keys
    """
    try:
        folder = entity_memory_path(facet_name, entity_name)
    except ValueError:
        return {"observation_count": 0, "has_voiceprint": False}

    # Count observations
    obs_file = folder / "observations.jsonl"
    obs_count = 0
    if obs_file.exists():
        try:
            with open(obs_file, "r", encoding="utf-8") as f:
                obs_count = sum(1 for line in f if line.strip())
        except OSError:
            pass  # File read error, default to 0

    # Check for voiceprint
    has_voiceprint = (folder / "voiceprints.npz").exists()

    return {"observation_count": obs_count, "has_voiceprint": has_voiceprint}


def get_facet_entities_data(facet_name: str) -> dict:
    """Get entity data for a facet: attached and detected entities.

    Returns:
        dict with keys:
            - attached: list of entity dicts with type, name, description,
                        attached_at, updated_at, last_seen timestamps,
                        plus observation_count, has_voiceprint, and last_active_ts
            - detected: list of {"type": str, "name": str, "description": str, "count": int, "last_seen": str}
    """
    # Load attached entities (already returns list of dicts)
    attached = load_entities(facet_name)

    # Enrich attached entities with metadata
    for entity in attached:
        name = entity.get("name", "")
        if name:
            metadata = _get_entity_metadata(facet_name, name)
            entity["observation_count"] = metadata["observation_count"]
            entity["has_voiceprint"] = metadata["has_voiceprint"]
        # Add computed activity timestamp for frontend sorting/display
        entity["last_active_ts"] = entity_last_active_ts(entity)

    # Load detected entities directly from files (excludes attached names/akas)
    detected = load_detected_entities_recent(facet_name)

    return {"attached": attached, "detected": detected}


@entities_bp.route("/api/<facet_name>")
def get_entities(facet_name: str) -> Any:
    """Get entities for a specific facet (attached and detected)."""
    try:
        data = get_facet_entities_data(facet_name)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to get entities: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/entity/<entity_id>")
def get_entity(facet_name: str, entity_id: str) -> Any:
    """Get a single entity by id.

    Uses exact id matching only. URL fragments always contain the entity id,
    so fuzzy matching is not needed here (it's used by MCP tools instead).
    Includes detached entities so they can be viewed and re-attached.
    """
    try:
        # Load all entities including detached, find by exact id match
        entities = load_entities(facet_name, include_detached=True)
        entity = next((e for e in entities if e.get("id") == entity_id), None)

        if entity is None:
            return jsonify({"error": f"Entity '{entity_id}' not found"}), 404

        entity_name = entity.get("name", "")
        entity = entity.copy()

        # Add metadata
        metadata = _get_entity_metadata(facet_name, entity_name)
        entity["observation_count"] = metadata["observation_count"]
        entity["has_voiceprint"] = metadata["has_voiceprint"]
        # Add computed activity timestamp for frontend display
        entity["last_active_ts"] = entity_last_active_ts(entity)

        # Ensure id is set
        if "id" not in entity:
            entity["id"] = entity_slug(entity_name)

        # Load observations
        observations = load_observations(facet_name, entity_name)

        return jsonify({"entity": entity, "observations": observations})

    except Exception as e:
        return jsonify({"error": f"Failed to get entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>", methods=["POST"])
def add_entity(facet_name: str) -> Any:
    """Add/attach an entity to a facet.

    Entity names must be unique within a facet (regardless of type).
    If a previously detached entity with the same name exists,
    re-activates it instead of creating a duplicate.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    etype = data.get("type", "").strip()
    name = data.get("name", "").strip()
    desc = data.get("description", "").strip()

    if not etype or not name:
        return jsonify({"error": "Type and name are required"}), 400

    # Validate entity type
    if not is_valid_entity_type(etype):
        return jsonify({"error": f"Invalid entity type '{etype}'"}), 400

    try:
        # Load ALL attached entities including detached ones
        entities = load_entities(facet_name, include_detached=True)

        # Check for existing entity by name (case-insensitive, active or detached)
        name_lower = name.lower()
        for entity in entities:
            if entity.get("name", "").lower() == name_lower:
                if entity.get("detached"):
                    # Re-activate detached entity
                    entity.pop("detached", None)
                    entity["updated_at"] = int(time.time() * 1000)
                    # Update type and description if provided
                    entity["type"] = etype
                    if desc:
                        entity["description"] = desc
                    save_entities(facet_name, entities)

                    log_app_action(
                        app="entities",
                        facet=facet_name,
                        action="entity_reattach",
                        params={
                            "type": etype,
                            "name": name,
                            "description": entity.get("description", ""),
                        },
                    )
                    return jsonify({"success": True, "reattached": True})
                else:
                    return (
                        jsonify(
                            {"error": "Entity with this name already exists in facet"}
                        ),
                        409,
                    )

        # Add new entity with timestamps (id will be generated by save_entities)
        now = int(time.time() * 1000)
        entities.append(
            {
                "type": etype,
                "name": name,
                "description": desc,
                "attached_at": now,
                "updated_at": now,
            }
        )

        # Save back
        save_entities(facet_name, entities)

        log_app_action(
            app="entities",
            facet=facet_name,
            action="entity_attach",
            params={"type": etype, "name": name, "description": desc},
        )

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to add entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>", methods=["DELETE"])
def detach_entity(facet_name: str) -> Any:
    """Detach an entity from a facet (soft delete).

    Sets detached=True instead of removing the entity, preserving
    all metadata for potential re-attachment later.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()

    if not name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        # Load ALL attached entities including detached ones
        entities = load_entities(facet_name, include_detached=True)

        # Find the entity to detach by name
        target_entity = None
        for e in entities:
            if e.get("name") == name:
                if not e.get("detached"):
                    target_entity = e
                break

        if not target_entity:
            return jsonify({"error": "Entity not found in facet"}), 404

        # Soft delete: set detached flag and update timestamp
        target_entity["detached"] = True
        target_entity["updated_at"] = int(time.time() * 1000)

        # Save updated list (entity remains in file with detached=True)
        save_entities(facet_name, entities)

        log_app_action(
            app="entities",
            facet=facet_name,
            action="entity_detach",
            params={
                "type": target_entity.get("type", ""),
                "name": name,
                "description": target_entity.get("description", ""),
                "aka": target_entity.get("aka", []),
            },
        )

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to detach entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/update", methods=["PUT"])
def update_entity(facet_name: str) -> Any:
    """Update entity name, type, and AKA list."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    new_type = data.get("type", "").strip()
    aka_list_str = data.get("aka_list", "").strip()

    if not old_name or not new_name:
        return jsonify({"error": "old_name and new_name are required"}), 400

    try:
        # Parse comma-delimited aka list
        if aka_list_str:
            aka_list = [
                item.strip() for item in aka_list_str.split(",") if item.strip()
            ]
        else:
            aka_list = []

        # Load ALL attached entities including detached to avoid data loss on save
        entities = load_entities(facet_name, include_detached=True)

        # Find target entity by name (only search active entities)
        target = None
        target_index = -1
        for i, entity in enumerate(entities):
            if entity.get("detached"):
                continue  # Skip detached entities
            if entity.get("name") == old_name:
                target = entity
                target_index = i
                break

        if not target:
            return jsonify({"error": "Entity not found"}), 404

        # Capture old values before modification
        old_aka = target.get("aka", [])
        old_type = target.get("type", "")

        # Check if new name conflicts with existing active entities (excluding current)
        # Use case-insensitive comparison to match save_entities validation
        if new_name.lower() != old_name.lower():
            new_name_lower = new_name.lower()
            for i, entity in enumerate(entities):
                if entity.get("detached"):
                    continue  # Skip detached entities in conflict check
                if (
                    i != target_index
                    and entity.get("name", "").lower() == new_name_lower
                ):
                    return (
                        jsonify({"error": f"Entity '{new_name}' already exists"}),
                        409,
                    )

        # Validate akas don't conflict with other entities
        for aka in aka_list:
            conflict = validate_aka_uniqueness(
                aka, entities, exclude_entity_name=old_name
            )
            if conflict:
                return (
                    jsonify(
                        {"error": f"Alias '{aka}' conflicts with entity '{conflict}'"}
                    ),
                    409,
                )

        # Update entity
        target["name"] = new_name
        if new_type:
            target["type"] = new_type
        if aka_list:
            target["aka"] = aka_list
        else:
            target.pop("aka", None)
        target["updated_at"] = int(time.time() * 1000)

        # Save updated entities (id will be regenerated by save_entities)
        save_entities(facet_name, entities)

        # Rename entity memory folder if name changed
        if new_name != old_name:
            try:
                rename_entity_memory(facet_name, old_name, new_name)
            except OSError as e:
                # Log but don't fail - folder rename is best-effort
                logger.warning(
                    f"Failed to rename entity memory folder for '{old_name}' -> '{new_name}': {e}"
                )

        log_app_action(
            app="entities",
            facet=facet_name,
            action="entity_update",
            params={
                "old_type": old_type,
                "new_type": new_type or old_type,
                "old_name": old_name,
                "new_name": new_name,
                "old_aka": old_aka,
                "new_aka": aka_list,
            },
        )

        return jsonify({"success": True, "entity": target})

    except Exception as e:
        return jsonify({"error": f"Failed to update entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/description", methods=["PUT"])
def update_description(facet_name: str) -> Any:
    """Update an entity's description."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_name = data.get("name", "").strip()
    new_description = data.get("description", "").strip()

    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        # Load ALL attached entities including detached to avoid data loss on save
        entities = load_entities(facet_name, include_detached=True)

        # Find and update the entity by name (active only), capturing old description
        updated = False
        old_description = ""
        entity_type = ""
        for entity in entities:
            if entity.get("detached"):
                continue  # Skip detached entities
            if entity.get("name") == entity_name:
                old_description = entity.get("description", "")
                entity_type = entity.get("type", "")
                entity["description"] = new_description
                entity["updated_at"] = int(time.time() * 1000)
                updated = True
                break

        if not updated:
            return jsonify({"error": "Entity not found in facet"}), 404

        # Save updated list
        save_entities(facet_name, entities)

        log_app_action(
            app="entities",
            facet=facet_name,
            action="entity_update_description",
            params={
                "type": entity_type,
                "name": entity_name,
                "old_description": old_description,
                "new_description": new_description,
            },
        )

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to update entity description: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/generate-description", methods=["POST"])
def generate_description(facet_name: str) -> Any:
    """Generate a description for an entity using AI agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_type = data.get("type", "").strip()
    entity_name = data.get("name", "").strip()
    current_description = data.get("current_description", "")

    if not entity_type or not entity_name:
        return jsonify({"error": "Type and name are required"}), 400

    # Check for Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY not set"}), 500

    try:
        from convey.utils import spawn_agent

        # Build concise prompt - agent has detailed instructions
        current_desc = current_description or "(none)"
        prompt = (
            f"Entity Type: {entity_type}\n"
            f"Entity Name: {entity_name}\n"
            f"Facet: {facet_name}\n"
            f"Current Description: {current_desc}"
        )

        agent_id = spawn_agent(
            prompt=prompt,
            persona="entities:entity_describe",
            provider="google",
        )

        return jsonify({"success": True, "agent_id": agent_id})

    except Exception as e:
        return (
            jsonify({"error": f"Failed to generate entity description: {str(e)}"}),
            500,
        )


@entities_bp.route("/api/<facet_name>/assist", methods=["POST"])
def assist_add(facet_name: str) -> Any:
    """Use entity_assist agent to quickly add an entity with AI-generated details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        from convey.utils import spawn_agent

        # Format prompt as specified by entity_assist agent
        prompt = f"For the '{facet_name}' facet, this is the user's request to attach a new entity: {name}"

        # Create agent request - entity_assist persona already has provider configured
        agent_id = spawn_agent(
            prompt=prompt,
            persona="entities:entity_assist",
        )

        return jsonify({"success": True, "agent_id": agent_id})

    except Exception as e:
        return jsonify({"error": f"Failed to start entity assistant: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/detected/preview")
def preview_delete(facet_name: str) -> Any:
    """Preview which days contain a detected entity before deletion."""
    entity_name = request.args.get("name", "").strip()
    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        entities_dir = Path(state.journal_root) / "facets" / facet_name / "entities"
        if not entities_dir.exists():
            return jsonify({"success": True, "days": []})

        # Scan all day files for this entity
        found_days = []
        for day_file in sorted(entities_dir.glob("*.jsonl")):
            day = day_file.stem
            entities = load_entities(facet_name, day)

            # Find all occurrences of this entity name (any type)
            for entity in entities:
                if entity.get("name") == entity_name:
                    found_days.append(
                        {
                            "day": day,
                            "type": entity.get("type", ""),
                            "description": entity.get("description", ""),
                        }
                    )

        return jsonify({"success": True, "days": found_days})

    except Exception as e:
        return jsonify({"error": f"Failed to preview entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/detected", methods=["DELETE"])
def delete_detected(facet_name: str) -> Any:
    """Delete a detected entity from all day files."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_name = data.get("name", "").strip()
    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        entities_dir = Path(state.journal_root) / "facets" / facet_name / "entities"
        if not entities_dir.exists():
            return jsonify({"success": True, "days_modified": []})

        # Iterate through all day files and remove the entity
        days_modified = []
        deleted_entries = []
        for day_file in sorted(entities_dir.glob("*.jsonl")):
            day = day_file.stem
            entities = load_entities(facet_name, day)

            # Capture entities being removed before filtering
            for e in entities:
                if e.get("name") == entity_name:
                    deleted_entries.append(
                        {
                            "day": day,
                            "type": e.get("type", ""),
                            "description": e.get("description", ""),
                        }
                    )

            # Filter out entities matching this name (any type)
            original_count = len(entities)
            filtered_entities = [e for e in entities if e.get("name") != entity_name]

            # Only save if we actually removed something
            if len(filtered_entities) < original_count:
                save_entities(facet_name, filtered_entities, day)
                days_modified.append(day)

        if deleted_entries:
            log_app_action(
                app="entities",
                facet=facet_name,
                action="entity_delete_detected",
                params={
                    "name": entity_name,
                    "deleted_entries": deleted_entries,
                },
            )

        return jsonify({"success": True, "days_modified": days_modified})

    except Exception as e:
        return jsonify({"error": f"Failed to delete entity: {str(e)}"}), 500


# =============================================================================
# Journal-wide entity endpoints (all-facet mode)
# =============================================================================


def _build_facet_relationships(
    entity_id: str, entity_name: str, facets_config: dict
) -> tuple[list, int, int]:
    """Build facet relationships list for a journal entity.

    Args:
        entity_id: The entity id
        entity_name: The entity name
        facets_config: Dict of facet configs from get_facets()

    Returns:
        Tuple of (facet_relationships list, total_observation_count, latest_active_ts)
    """
    facet_relationships = []
    total_observation_count = 0
    latest_active_ts = 0

    for facet_name in facets_config:
        relationship = load_facet_relationship(facet_name, entity_id)
        if not relationship:
            continue
        # Skip detached relationships
        if relationship.get("detached"):
            continue

        facet_config = facets_config.get(facet_name, {})
        metadata = _get_entity_metadata(facet_name, entity_name)

        facet_rel = {
            "name": facet_name,
            "title": facet_config.get("title", facet_name),
            "color": facet_config.get("color", "#888"),
            "emoji": facet_config.get("emoji", ""),
            "description": relationship.get("description", ""),
            "last_seen": relationship.get("last_seen"),
            "attached_at": relationship.get("attached_at"),
            "updated_at": relationship.get("updated_at"),
            "observation_count": metadata["observation_count"],
            "has_voiceprint": metadata["has_voiceprint"],
        }

        # Compute last_active_ts for this relationship
        rel_active_ts = entity_last_active_ts(relationship)
        facet_rel["last_active_ts"] = rel_active_ts

        total_observation_count += metadata["observation_count"]
        if rel_active_ts > latest_active_ts:
            latest_active_ts = rel_active_ts

        facet_relationships.append(facet_rel)

    # Sort facet relationships by last_active_ts (most recent first)
    facet_relationships.sort(key=lambda r: r.get("last_active_ts", 0), reverse=True)

    return facet_relationships, total_observation_count, latest_active_ts


def get_journal_entities_data() -> dict:
    """Get all journal entities with facet relationship data.

    Returns:
        dict with:
            - entities: list of journal entities enriched with facet info
    """
    facets_config = get_facets()
    entity_ids = scan_journal_entities()

    entities = []
    for entity_id in entity_ids:
        journal_entity = load_journal_entity(entity_id)
        if not journal_entity:
            continue

        entity_name = journal_entity.get("name", "")

        # Build facet relationships
        facet_relationships, total_observation_count, latest_active_ts = (
            _build_facet_relationships(entity_id, entity_name, facets_config)
        )

        # Build enriched entity
        enriched = {
            "id": entity_id,
            "name": entity_name,
            "type": journal_entity.get("type", ""),
            "aka": journal_entity.get("aka", []),
            "is_principal": journal_entity.get("is_principal", False),
            "facets": facet_relationships,
            "total_observation_count": total_observation_count,
            "last_active_ts": latest_active_ts,
        }

        entities.append(enriched)

    # Sort by last_active_ts (most recent first)
    entities.sort(key=lambda e: e.get("last_active_ts", 0), reverse=True)

    return {"entities": entities}


@entities_bp.route("/api/journal")
def get_journal_entities() -> Any:
    """Get all journal entities with facet relationship summaries."""
    try:
        data = get_journal_entities_data()
        return jsonify(data)
    except Exception as e:
        logger.exception("Failed to get journal entities")
        return jsonify({"error": f"Failed to get journal entities: {str(e)}"}), 500


@entities_bp.route("/api/journal/entity/<entity_id>")
def get_journal_entity(entity_id: str) -> Any:
    """Get a single journal entity by id with full facet relationship details."""
    try:
        journal_entity = load_journal_entity(entity_id)
        if not journal_entity:
            return jsonify({"error": f"Entity '{entity_id}' not found"}), 404

        entity_name = journal_entity.get("name", "")
        facets_config = get_facets()

        # Build facet relationships
        facet_relationships, total_observation_count, latest_active_ts = (
            _build_facet_relationships(entity_id, entity_name, facets_config)
        )

        # Build enriched entity
        enriched = {
            "id": entity_id,
            "name": entity_name,
            "type": journal_entity.get("type", ""),
            "aka": journal_entity.get("aka", []),
            "is_principal": journal_entity.get("is_principal", False),
            "facets": facet_relationships,
            "total_observation_count": total_observation_count,
            "last_active_ts": latest_active_ts,
        }

        return jsonify({"entity": enriched})

    except Exception as e:
        logger.exception("Failed to get journal entity")
        return jsonify({"error": f"Failed to get journal entity: {str(e)}"}), 500


@entities_bp.route("/api/journal/entity/<entity_id>", methods=["PUT"])
def update_journal_entity(entity_id: str) -> Any:
    """Update a journal entity's name, type, and/or akas."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Load existing entity
        journal_entity = load_journal_entity(entity_id)
        if not journal_entity:
            return jsonify({"error": f"Entity '{entity_id}' not found"}), 404

        # Track what changed for logging
        changes = {}

        # Update name if provided
        new_name = data.get("name", "").strip()
        if new_name and new_name != journal_entity.get("name", ""):
            changes["name"] = {"old": journal_entity.get("name"), "new": new_name}
            journal_entity["name"] = new_name

        # Update type if provided
        new_type = data.get("type", "").strip()
        if new_type:
            if not is_valid_entity_type(new_type):
                return jsonify({"error": f"Invalid entity type: {new_type}"}), 400
            if new_type != journal_entity.get("type", ""):
                changes["type"] = {"old": journal_entity.get("type"), "new": new_type}
                journal_entity["type"] = new_type

        # Update akas if provided
        if "aka" in data:
            new_akas = data["aka"]
            if isinstance(new_akas, str):
                # Parse comma-separated string
                new_akas = [a.strip() for a in new_akas.split(",") if a.strip()]
            elif not isinstance(new_akas, list):
                new_akas = []

            old_akas = journal_entity.get("aka", [])
            if set(new_akas) != set(old_akas):
                changes["aka"] = {"old": old_akas, "new": new_akas}
                journal_entity["aka"] = new_akas

        if not changes:
            return jsonify({"success": True, "message": "No changes made"})

        # Update timestamp
        journal_entity["updated_at"] = int(time.time() * 1000)

        # Save the updated entity
        save_journal_entity(journal_entity)

        # Log the action
        log_app_action(
            app="entities",
            facet=None,  # Journal-level action
            action="journal_entity_update",
            params={
                "entity_id": entity_id,
                "changes": changes,
            },
        )

        return jsonify({"success": True, "entity": journal_entity})

    except Exception as e:
        logger.exception("Failed to update journal entity")
        return jsonify({"error": f"Failed to update journal entity: {str(e)}"}), 500
