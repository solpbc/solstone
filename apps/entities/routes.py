"""Entities app routes - facet entity management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from convey import state
from think.entities import load_entities, save_entities
from think.indexer import search_entities

entities_bp = Blueprint(
    "app:entities",
    __name__,
    url_prefix="/app/entities",
)


def get_facet_entities_data(facet_name: str) -> dict:
    """Get entity data for a facet: attached and detected entities.

    Returns:
        dict with keys:
            - attached: list of {"type": str, "name": str, "description": str}
            - detected: list of {"type": str, "name": str, "description": str, "count": int, "last_seen": str}
    """
    # Load attached entities (already returns list of dicts)
    attached = load_entities(facet_name)

    # Query detected entities from indexer
    _, detected_results = search_entities(
        "",
        limit=1000,  # Get all detected entities
        facet=facet_name,
        attached=False,
        order="day",  # Most recent first
    )

    # Aggregate detected entities by (type, name)
    detected_map = {}
    for result in detected_results:
        meta = result["metadata"]
        key = (meta["type"], meta["name"])

        if key not in detected_map:
            detected_map[key] = {
                "type": meta["type"],
                "name": meta["name"],
                "description": result["text"],  # Most recent day's description
                "count": 1,
                "last_seen": meta["day"],
            }
        else:
            detected_map[key]["count"] += 1

    detected = list(detected_map.values())

    return {"attached": attached, "detected": detected}


@entities_bp.route("/api/<facet_name>")
def get_entities(facet_name: str) -> Any:
    """Get entities for a specific facet (attached and detected)."""
    try:
        data = get_facet_entities_data(facet_name)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to get entities: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>", methods=["POST"])
def add_entity(facet_name: str) -> Any:
    """Add/attach an entity to a facet."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    etype = data.get("type", "").strip()
    name = data.get("name", "").strip()
    # Support both "desc" and "description" for backwards compatibility
    desc = data.get("desc", "") or data.get("description", "")
    desc = desc.strip()

    if not etype or not name:
        return jsonify({"error": "Type and name are required"}), 400

    try:
        # Load existing attached entities
        entities = load_entities(facet_name)

        # Check for duplicates
        for entity in entities:
            if entity.get("type") == etype and entity.get("name") == name:
                return jsonify({"error": "Entity already exists in facet"}), 409

        # Add new entity
        entities.append({"type": etype, "name": name, "description": desc})

        # Save back
        save_entities(facet_name, entities)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to add entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>", methods=["DELETE"])
def remove_entity(facet_name: str) -> Any:
    """Remove/detach an entity from a facet."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    etype = data.get("type", "").strip()
    name = data.get("name", "").strip()

    if not etype or not name:
        return jsonify({"error": "Type and name are required"}), 400

    try:
        # Load existing attached entities
        entities = load_entities(facet_name)

        # Filter out the entity to remove
        filtered = [
            e
            for e in entities
            if not (e.get("type") == etype and e.get("name") == name)
        ]

        # Check if anything was removed
        if len(filtered) == len(entities):
            return jsonify({"error": "Entity not found in facet"}), 404

        # Save filtered list
        save_entities(facet_name, filtered)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to remove entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/update", methods=["PUT"])
def update_entity(facet_name: str) -> Any:
    """Update entity name and AKA list."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_type = data.get("type", "").strip()
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    aka_list_str = data.get("aka_list", "").strip()

    if not entity_type or not old_name or not new_name:
        return jsonify({"error": "Type, old_name, and new_name are required"}), 400

    try:
        # Parse comma-delimited aka list
        if aka_list_str:
            aka_list = [
                item.strip() for item in aka_list_str.split(",") if item.strip()
            ]
        else:
            aka_list = []

        # Load attached entities
        entities = load_entities(facet_name)

        # Find target entity
        target = None
        target_index = -1
        for i, entity in enumerate(entities):
            if entity.get("type") == entity_type and entity.get("name") == old_name:
                target = entity
                target_index = i
                break

        if not target:
            return jsonify({"error": "Entity not found"}), 404

        # Check if new name conflicts with existing entities (excluding current)
        if new_name != old_name:
            for i, entity in enumerate(entities):
                if (
                    i != target_index
                    and entity.get("type") == entity_type
                    and entity.get("name") == new_name
                ):
                    return (
                        jsonify({"error": f"Entity '{new_name}' already exists"}),
                        409,
                    )

        # Update entity
        target["name"] = new_name
        if aka_list:
            target["aka"] = aka_list
        else:
            target.pop("aka", None)

        # Save updated entities
        save_entities(facet_name, entities)

        return jsonify({"success": True, "entity": target})

    except Exception as e:
        return jsonify({"error": f"Failed to update entity: {str(e)}"}), 500


@entities_bp.route("/api/<facet_name>/description", methods=["PUT"])
def update_description(facet_name: str) -> Any:
    """Update an entity's description."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_type = data.get("type", "").strip()
    entity_name = data.get("name", "").strip()
    new_description = data.get("description", "").strip()

    if not entity_type or not entity_name:
        return jsonify({"error": "Type and name are required"}), 400

    try:
        # Load existing attached entities
        entities = load_entities(facet_name)

        # Find and update the entity
        updated = False
        for entity in entities:
            if entity.get("type") == entity_type and entity.get("name") == entity_name:
                entity["description"] = new_description
                updated = True
                break

        if not updated:
            return jsonify({"error": "Entity not found in facet"}), 404

        # Save updated list
        save_entities(facet_name, entities)

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
        # Build context for the agent
        context_parts = [
            f"Entity Type: {entity_type}",
            f"Entity Name: {entity_name}",
            f"Facet: {facet_name}",
        ]

        if current_description:
            context_parts.append(f"Current Description: {current_description}")
        else:
            context_parts.append("Current Description: (none)")

        context = "\n".join(context_parts)

        prompt = f"""Please generate a compelling, informative description for this entity based on the following context:

{context}

Generate a clear, concise description (1-2 sentences) that captures what this {entity_type.lower()} is and why it's relevant. The description should be appropriate for a personal knowledge management system and help users understand the entity's significance or role."""

        # Create agent request - events will be broadcast by shared watcher
        from convey.utils import spawn_agent

        agent_id = spawn_agent(
            prompt=prompt,
            persona="facet_describe",
            backend="google",
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

        # Create agent request - entity_assist persona already has backend configured
        agent_id = spawn_agent(
            prompt=prompt,
            persona="entity_assist",
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
        for day_file in sorted(entities_dir.glob("*.jsonl")):
            day = day_file.stem
            entities = load_entities(facet_name, day)

            # Filter out entities matching this name (any type)
            original_count = len(entities)
            filtered_entities = [e for e in entities if e.get("name") != entity_name]

            # Only save if we actually removed something
            if len(filtered_entities) < original_count:
                save_entities(facet_name, filtered_entities, day)
                days_modified.append(day)

        return jsonify({"success": True, "days_modified": days_modified})

    except Exception as e:
        return jsonify({"error": f"Failed to delete entity: {str(e)}"}), 500
