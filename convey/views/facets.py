from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.entities import (
    load_detected_entities_recent,
    load_entities,
    save_entities,
)
from think.facets import get_facet_news, get_facets, set_facet_disabled
from think.indexer import search_entities, search_events
from think.utils import get_topics

from .. import state
from ..utils import DATE_RE, adjacent_days, format_date

bp = Blueprint("facets", __name__, template_folder="../templates")


@bp.route("/facets")
def facets_page() -> str:
    return render_template("facets.html", active="facets")


@bp.route("/api/facets")
def facets_list() -> Any:
    """Return available facets with their metadata."""
    return jsonify(get_facets())


@bp.route("/api/facets", methods=["POST"])
def create_facet() -> Any:
    """Create a new facet with the provided metadata."""
    from ..utils import error_response, success_response

    data = request.get_json()
    if not data:
        return error_response("No data provided")

    facet_name = data.get("name", "").strip()
    if not facet_name:
        return error_response("Facet name is required")

    # Validate facet name (basic alphanumeric + hyphens/underscores)
    if not facet_name.replace("-", "").replace("_", "").isalnum():
        return error_response(
            "Facet name must be alphanumeric with optional hyphens or underscores"
        )

    facet_path = Path(state.journal_root) / "facets" / facet_name

    # Check if facet already exists
    if facet_path.exists():
        return error_response("Facet already exists", 409)

    try:
        # Create facet directory
        facet_path.mkdir(parents=True, exist_ok=True)

        # Create facet.json
        facet_data = {
            "title": data.get("title", facet_name),
            "description": data.get("description", ""),
        }

        if data.get("color"):
            facet_data["color"] = data["color"]
        if data.get("emoji"):
            facet_data["emoji"] = data["emoji"]

        from ..utils import save_json

        facet_json = facet_path / "facet.json"
        save_json(facet_json, facet_data)

        # Create empty entities.jsonl
        entities_jsonl = facet_path / "entities.jsonl"
        entities_jsonl.write_text("", encoding="utf-8")

        return success_response({"facet": facet_name})

    except Exception as e:
        return error_response(f"Failed to create facet: {str(e)}", 500)


@bp.route("/facets/<facet_name>")
def facet_detail(facet_name: str) -> str:
    """Display detailed view for a specific facet."""
    facets = get_facets()
    if facet_name not in facets:
        return render_template("404.html"), 404

    facet_data = facets[facet_name]
    today = date.today().strftime("%Y%m%d")
    return render_template(
        "facet_detail.html",
        facet_name=facet_name,
        facet_data=facet_data,
        today=today,
        active="facets",
    )


@bp.route("/api/facets/<facet_name>", methods=["PUT"])
def update_facet(facet_name: str) -> Any:
    """Update an existing facet's metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    facet_path = Path(state.journal_root) / "facets" / facet_name
    facet_json = facet_path / "facet.json"

    if not facet_json.exists():
        return jsonify({"error": "Facet not found"}), 404

    try:
        from ..utils import load_json, save_json

        # Read existing facet.json
        existing_data = load_json(facet_json)
        if not existing_data:
            return jsonify({"error": "Failed to load facet data"}), 500

        # Update only provided fields
        if "title" in data:
            existing_data["title"] = data["title"]
        if "description" in data:
            existing_data["description"] = data["description"]
        if "color" in data:
            if data["color"]:
                existing_data["color"] = data["color"]
            else:
                existing_data.pop("color", None)
        if "emoji" in data:
            if data["emoji"]:
                existing_data["emoji"] = data["emoji"]
            else:
                existing_data.pop("emoji", None)

        # Write updated facet.json
        if not save_json(facet_json, existing_data):
            return jsonify({"error": "Failed to save facet data"}), 500

        return jsonify({"success": True, "facet": facet_name})

    except Exception as e:
        return jsonify({"error": f"Failed to update facet: {str(e)}"}), 500


@bp.route("/api/facets/<facet_name>/toggle", methods=["POST"])
def toggle_facet_state(facet_name: str) -> Any:
    """Toggle facet enabled/disabled state for automated agent runs."""
    facets = get_facets()
    if facet_name not in facets:
        return jsonify({"error": "Facet not found"}), 404

    try:
        # Get current state
        current_state = facets[facet_name].get("disabled", False)
        new_state = not current_state

        # Update the state
        set_facet_disabled(facet_name, new_state)

        return jsonify(
            {
                "success": True,
                "facet": facet_name,
                "disabled": new_state,
                "message": f"Facet {'disabled' if new_state else 'enabled'}",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to toggle facet state: {str(e)}"}), 500


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


@bp.route("/api/facets/<facet_name>/entities")
def get_facet_entities(facet_name: str) -> Any:
    """Get entities for a specific facet (attached and detected)."""
    try:
        data = get_facet_entities_data(facet_name)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to get entities: {str(e)}"}), 500


@bp.route("/api/facets/<facet_name>/entities", methods=["POST"])
def add_facet_entity(facet_name: str) -> Any:
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


@bp.route("/api/facets/<facet_name>/entities", methods=["DELETE"])
def remove_facet_entity(facet_name: str) -> Any:
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


@bp.route("/api/facets/<facet_name>/generate-description", methods=["POST"])
def generate_facet_description(facet_name: str) -> Any:
    """Generate a description for a facet using AI agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    current_description = data.get("current_description", "")

    # Check for Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY not set"}), 500

    facet_path = Path(state.journal_root) / "facets" / facet_name
    if not facet_path.exists():
        return jsonify({"error": "Facet not found"}), 404

    try:
        # Get facet metadata
        facets = get_facets()
        facet_data = facets.get(facet_name, {})

        # Build context for the agent
        context_parts = [
            f"Facet Name: {facet_name}",
            f"Facet Title: {facet_data.get('title', facet_name)}",
        ]

        if current_description:
            context_parts.append(f"Current Description: {current_description}")
        else:
            context_parts.append("Current Description: (none)")

        # Check if facet has entities using load_entity_names
        from think.entities import load_entity_names

        try:
            entity_names = load_entity_names(facet=facet_name)
            if entity_names:
                context_parts.append(f"Facet Entities: {entity_names}")
        except Exception:
            pass

        context = "\n".join(context_parts)

        prompt = f"""Please generate a compelling, informative description for this facet based on the following context:

{context}

Generate a clear, engaging 1-2 sentence description that captures the essence and purpose of this facet. The description should help users understand what they'll find in this facet and be appropriate for a personal knowledge management system."""

        # Create agent request - events will be broadcast by shared watcher
        from ..utils import spawn_agent

        agent_id = spawn_agent(
            prompt=prompt,
            persona="facet_describe",
            backend="google",
        )

        return jsonify({"success": True, "agent_id": agent_id})

    except Exception as e:
        return jsonify({"error": f"Failed to generate description: {str(e)}"}), 500


@bp.route("/api/facets/<facet_name>/entities/description", methods=["PUT"])
def update_entity_description(facet_name: str) -> Any:
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


@bp.route("/api/facets/<facet_name>/entities/generate-description", methods=["POST"])
def generate_entity_description(facet_name: str) -> Any:
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
        from ..utils import spawn_agent

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


@bp.route("/api/facets/<facet_name>/entities/assist", methods=["POST"])
def assist_entity_add(facet_name: str) -> Any:
    """Use entity_assist agent to quickly add an entity with AI-generated details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        from ..utils import spawn_agent

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


@bp.route("/api/facets/<facet_name>/news")
def get_facet_news_feed(facet_name: str) -> Any:
    """Return paginated news entries for a facet."""

    cursor = request.args.get("cursor")
    day = request.args.get("day")
    # Default to 5 newsletters for initial load, 5 more for "load more"
    limit = request.args.get("days", default=5, type=int) or 5
    if limit < 0:
        limit = 5

    try:
        news_payload = get_facet_news(facet_name, cursor=cursor, limit=limit, day=day)
        return jsonify(news_payload)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to load news: {str(exc)}"}), 500


@bp.route("/facets/<facet_name>/entities/manage")
def entity_manager(facet_name: str) -> str:
    """Display entity management page for a facet."""
    facets = get_facets()
    if facet_name not in facets:
        return render_template("404.html"), 404

    try:
        # Load attached entities
        attached_entities = load_entities(facet_name)

        # Load recent detected entities (last 30 days, excluding attached names/akas)
        detected_entities = load_detected_entities_recent(facet_name, days=30)

        return render_template(
            "entity_manager.html",
            facet_name=facet_name,
            attached_entities=attached_entities,
            detected_entities=detected_entities,
            active="facets",
        )
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


@bp.route("/facets/<facet_name>/calendar/<day>")
def facet_day(facet_name: str, day: str) -> str:
    """Display calendar day view for a specific facet."""
    # Validate date format
    if not re.fullmatch(DATE_RE.pattern, day):
        return "", 404

    # Validate facet exists
    facets = get_facets()
    if facet_name not in facets:
        return render_template("404.html"), 404

    facet_data = facets[facet_name]

    # Get navigation dates
    prev_day, next_day = adjacent_days(state.journal_root, day)
    today_day = date.today().strftime("%Y%m%d")
    title = format_date(day)

    # Load facet-filtered occurrences for this day
    topics = get_topics()
    _, results = search_events(query="", facet=facet_name, day=day, limit=1000)

    # Transform search results into timeline format
    occurrences = []
    for result in results:
        event = result.get("event", {})
        metadata = result.get("metadata", {})
        topic = metadata.get("topic", "other")

        # Add topic color
        topic_color = topics.get(topic, {}).get("color", "#6c757d")

        occurrence = {
            "title": event.get("title", ""),
            "summary": event.get("summary", ""),
            "subject": event.get("subject", ""),
            "details": event.get("details", event.get("description", "")),
            "participants": event.get("participants", []),
            "topic": topic,
            "color": topic_color,
        }

        # Convert time strings to ISO timestamps
        if event.get("start"):
            occurrence["startTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['start']}"
        if event.get("end"):
            occurrence["endTime"] = f"{day[:4]}-{day[4:6]}-{day[6:]}T{event['end']}"

        occurrences.append(occurrence)

    return render_template(
        "facet_day.html",
        facet_name=facet_name,
        facet_data=facet_data,
        day=day,
        title=title,
        prev_day=prev_day,
        next_day=next_day,
        today_day=today_day,
        occurrences=occurrences,
        active="facets",
    )


@bp.route("/api/facets/<facet_name>/entities/manage/add-aka", methods=["POST"])
def add_aka_from_detected(facet_name: str) -> Any:
    """Add a detected entity name to an attached entity's aka list."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    target_entity = data.get("target_entity", "").strip()
    source_entity = data.get("source_entity", "").strip()

    if not target_entity or not source_entity:
        return jsonify({"error": "Both target and source entities are required"}), 400

    try:
        # Load attached entities
        entities = load_entities(facet_name)

        # Find target entity
        target = None
        for entity in entities:
            if entity.get("name") == target_entity:
                target = entity
                break

        if not target:
            return jsonify({"error": "Target entity not found"}), 404

        # Add source to aka list (create if doesn't exist)
        aka_list = target.get("aka", [])
        if not isinstance(aka_list, list):
            aka_list = []

        # Don't add duplicates
        if source_entity not in aka_list:
            aka_list.append(source_entity)
            target["aka"] = aka_list

            # Save updated entities
            save_entities(facet_name, entities)

            return jsonify({"success": True})
        else:
            return jsonify({"error": "This name is already in the aka list"}), 409

    except Exception as e:
        return jsonify({"error": f"Failed to add aka: {str(e)}"}), 500


@bp.route("/api/facets/<facet_name>/entities/manage/update-aka", methods=["POST"])
def update_aka_list(facet_name: str) -> Any:
    """Update an attached entity's aka list directly."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_name = data.get("entity_name", "").strip()
    aka_list_str = data.get("aka_list", "").strip()

    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

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

        # Find and update target entity
        target = None
        for entity in entities:
            if entity.get("name") == entity_name:
                target = entity
                break

        if not target:
            return jsonify({"error": "Entity not found"}), 404

        # Update aka list (or remove field if empty)
        if aka_list:
            target["aka"] = aka_list
        else:
            target.pop("aka", None)

        # Save updated entities
        save_entities(facet_name, entities)

        return jsonify({"success": True, "aka": aka_list})

    except Exception as e:
        return jsonify({"error": f"Failed to update aka list: {str(e)}"}), 500


@bp.route("/api/facets/<facet_name>/entities/update", methods=["PUT"])
def update_entity_full(facet_name: str) -> Any:
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


@bp.route("/api/facets/<facet_name>/entities/detected/preview")
def preview_detected_entity_delete(facet_name: str) -> Any:
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


@bp.route("/api/facets/<facet_name>/entities/detected", methods=["DELETE"])
def delete_detected_entity(facet_name: str) -> Any:
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
