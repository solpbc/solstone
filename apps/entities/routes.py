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
    entity_memory_path,
    entity_slug,
    is_valid_entity_type,
    load_detected_entities_recent,
    load_entities,
    load_observations,
    rename_entity_memory,
    resolve_entity,
    save_entities,
)

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
                        plus observation_count and has_voiceprint
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
    """Get a single entity with observations.

    Accepts entity lookup by id (slug), name, or aka.
    """
    try:
        # Try to resolve the entity_id to an actual entity
        entity, candidates = resolve_entity(facet_name, entity_id)
        if entity is None:
            if candidates:
                suggestions = [c.get("name", "") for c in candidates[:3]]
                return (
                    jsonify(
                        {
                            "error": f"Entity '{entity_id}' not found. Did you mean: {', '.join(suggestions)}?"
                        }
                    ),
                    404,
                )
            return jsonify({"error": f"Entity '{entity_id}' not found"}), 404

        entity_name = entity.get("name", "")
        entity = entity.copy()

        # Add metadata
        metadata = _get_entity_metadata(facet_name, entity_name)
        entity["observation_count"] = metadata["observation_count"]
        entity["has_voiceprint"] = metadata["has_voiceprint"]

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

        # Check for existing entity by name (active or detached)
        for entity in entities:
            if entity.get("name") == name:
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
def remove_entity(facet_name: str) -> Any:
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
        return jsonify({"error": f"Failed to remove entity: {str(e)}"}), 500


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
        if new_name != old_name:
            for i, entity in enumerate(entities):
                if entity.get("detached"):
                    continue  # Skip detached entities in conflict check
                if i != target_index and entity.get("name") == new_name:
                    return (
                        jsonify({"error": f"Entity '{new_name}' already exists"}),
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
