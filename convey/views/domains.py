from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Blueprint, jsonify, render_template, request

from think.domains import get_domain_news, get_domains
from think.entities import (
    load_detected_entities_recent,
    load_entities,
    save_entities,
)
from think.indexer import search_entities

bp = Blueprint("domains", __name__, template_folder="../templates")


@bp.route("/domains")
def domains_page() -> str:
    return render_template("domains.html", active="domains")


@bp.route("/api/domains")
def domains_list() -> Any:
    """Return available domains with their metadata."""
    return jsonify(get_domains())


@bp.route("/api/domains", methods=["POST"])
def create_domain() -> Any:
    """Create a new domain with the provided metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    domain_name = data.get("name", "").strip()
    if not domain_name:
        return jsonify({"error": "Domain name is required"}), 400

    # Validate domain name (basic alphanumeric + hyphens/underscores)
    if not domain_name.replace("-", "").replace("_", "").isalnum():
        return (
            jsonify(
                {
                    "error": "Domain name must be alphanumeric with optional hyphens or underscores"
                }
            ),
            400,
        )

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name

    # Check if domain already exists
    if domain_path.exists():
        return jsonify({"error": "Domain already exists"}), 409

    try:
        # Create domain directory
        domain_path.mkdir(parents=True, exist_ok=True)

        # Create domain.json
        domain_data = {
            "title": data.get("title", domain_name),
            "description": data.get("description", ""),
        }

        if data.get("color"):
            domain_data["color"] = data["color"]
        if data.get("emoji"):
            domain_data["emoji"] = data["emoji"]

        domain_json = domain_path / "domain.json"
        with open(domain_json, "w", encoding="utf-8") as f:
            json.dump(domain_data, f, indent=2, ensure_ascii=False)

        # Create empty entities.jsonl
        entities_jsonl = domain_path / "entities.jsonl"
        entities_jsonl.write_text("", encoding="utf-8")

        return jsonify({"success": True, "domain": domain_name})

    except Exception as e:
        return jsonify({"error": f"Failed to create domain: {str(e)}"}), 500


@bp.route("/domains/<domain_name>")
def domain_detail(domain_name: str) -> str:
    """Display detailed view for a specific domain."""
    domains = get_domains()
    if domain_name not in domains:
        return render_template("404.html"), 404

    domain_data = domains[domain_name]
    return render_template(
        "domain_detail.html",
        domain_name=domain_name,
        domain_data=domain_data,
        active="domains",
    )


@bp.route("/api/domains/<domain_name>", methods=["PUT"])
def update_domain(domain_name: str) -> Any:
    """Update an existing domain's metadata."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    domain_json = domain_path / "domain.json"

    if not domain_json.exists():
        return jsonify({"error": "Domain not found"}), 404

    try:
        # Read existing domain.json
        with open(domain_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

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

        # Write updated domain.json
        with open(domain_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        return jsonify({"success": True, "domain": domain_name})

    except Exception as e:
        return jsonify({"error": f"Failed to update domain: {str(e)}"}), 500


def get_domain_entities_data(domain_name: str) -> dict:
    """Get entity data for a domain: attached and detected entities.

    Returns:
        dict with keys:
            - attached: list of {"type": str, "name": str, "description": str}
            - detected: list of {"type": str, "name": str, "description": str, "count": int, "last_seen": str}
    """
    # Load attached entities (already returns list of dicts)
    attached = load_entities(domain_name)

    # Query detected entities from indexer
    _, detected_results = search_entities(
        "",
        limit=1000,  # Get all detected entities
        domain=domain_name,
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


@bp.route("/api/domains/<domain_name>/entities")
def get_domain_entities(domain_name: str) -> Any:
    """Get entities for a specific domain (attached and detected)."""
    try:
        data = get_domain_entities_data(domain_name)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Failed to get entities: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities", methods=["POST"])
def add_domain_entity(domain_name: str) -> Any:
    """Add/attach an entity to a domain."""
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
        entities = load_entities(domain_name)

        # Check for duplicates
        for entity in entities:
            if entity.get("type") == etype and entity.get("name") == name:
                return jsonify({"error": "Entity already exists in domain"}), 409

        # Add new entity
        entities.append({"type": etype, "name": name, "description": desc})

        # Save back
        save_entities(domain_name, entities)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to add entity: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities", methods=["DELETE"])
def remove_domain_entity(domain_name: str) -> Any:
    """Remove/detach an entity from a domain."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    etype = data.get("type", "").strip()
    name = data.get("name", "").strip()

    if not etype or not name:
        return jsonify({"error": "Type and name are required"}), 400

    try:
        # Load existing attached entities
        entities = load_entities(domain_name)

        # Filter out the entity to remove
        filtered = [
            e
            for e in entities
            if not (e.get("type") == etype and e.get("name") == name)
        ]

        # Check if anything was removed
        if len(filtered) == len(entities):
            return jsonify({"error": "Entity not found in domain"}), 404

        # Save filtered list
        save_entities(domain_name, filtered)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to remove entity: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/generate-description", methods=["POST"])
def generate_domain_description(domain_name: str) -> Any:
    """Generate a description for a domain using AI agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    current_description = data.get("current_description", "")

    # Check for Google API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY not set"}), 500

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    domain_path = Path(journal) / "domains" / domain_name
    if not domain_path.exists():
        return jsonify({"error": "Domain not found"}), 404

    try:
        # Get domain metadata
        domains = get_domains()
        domain_data = domains.get(domain_name, {})

        # Build context for the agent
        context_parts = [
            f"Domain Name: {domain_name}",
            f"Domain Title: {domain_data.get('title', domain_name)}",
        ]

        if current_description:
            context_parts.append(f"Current Description: {current_description}")
        else:
            context_parts.append("Current Description: (none)")

        # Check if domain has entities using load_entity_names
        from think.entities import load_entity_names

        try:
            entity_names = load_entity_names(domain=domain_name)
            if entity_names:
                context_parts.append(f"Domain Entities: {entity_names}")
        except Exception:
            pass

        context = "\n".join(context_parts)

        prompt = f"""Please generate a compelling, informative description for this domain based on the following context:

{context}

Generate a clear, engaging 1-2 sentence description that captures the essence and purpose of this domain. The description should help users understand what they'll find in this domain and be appropriate for a personal knowledge management system."""

        # Create agent request - events will be broadcast by shared watcher
        from muse.cortex_client import cortex_request
        from pathlib import Path

        agent_file = cortex_request(
            prompt=prompt,
            persona="domain_describe",
            backend="google",
        )

        # Extract agent_id from the filename
        agent_id = Path(agent_file).stem.replace("_active", "")

        return jsonify({"success": True, "agent_id": agent_id})

    except Exception as e:
        return jsonify({"error": f"Failed to generate description: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities/description", methods=["PUT"])
def update_entity_description(domain_name: str) -> Any:
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
        entities = load_entities(domain_name)

        # Find and update the entity
        updated = False
        for entity in entities:
            if entity.get("type") == entity_type and entity.get("name") == entity_name:
                entity["description"] = new_description
                updated = True
                break

        if not updated:
            return jsonify({"error": "Entity not found in domain"}), 404

        # Save updated list
        save_entities(domain_name, entities)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"error": f"Failed to update entity description: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities/generate-description", methods=["POST"])
def generate_entity_description(domain_name: str) -> Any:
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
            f"Domain: {domain_name}",
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
        from muse.cortex_client import cortex_request
        from pathlib import Path

        agent_file = cortex_request(
            prompt=prompt,
            persona="domain_describe",
            backend="google",
        )

        # Extract agent_id from the filename
        agent_id = Path(agent_file).stem.replace("_active", "")

        return jsonify({"success": True, "agent_id": agent_id})

    except Exception as e:
        return (
            jsonify({"error": f"Failed to generate entity description: {str(e)}"}),
            500,
        )


@bp.route("/api/domains/<domain_name>/entities/assist", methods=["POST"])
def assist_entity_add(domain_name: str) -> Any:
    """Use entity_assist agent to quickly add an entity with AI-generated details."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        # Import cortex request function
        from muse.cortex_client import cortex_request

        # Format prompt as specified by entity_assist agent
        prompt = f"For the '{domain_name}' domain, this is the user's request to attach a new entity: {name}"

        # Create agent request - entity_assist persona already has backend configured
        agent_file = cortex_request(
            prompt=prompt,
            persona="entity_assist",
        )

        # Extract agent_id from the filename
        agent_id = Path(agent_file).stem.replace("_active", "")

        return jsonify({"success": True, "agent_id": agent_id})

    except Exception as e:
        return jsonify({"error": f"Failed to start entity assistant: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/news")
def get_domain_news_feed(domain_name: str) -> Any:
    """Return paginated news entries for a domain."""

    cursor = request.args.get("cursor")
    day = request.args.get("day")
    # Default to 5 newsletters for initial load, 5 more for "load more"
    limit = request.args.get("days", default=5, type=int) or 5
    if limit < 0:
        limit = 5

    try:
        news_payload = get_domain_news(domain_name, cursor=cursor, limit=limit, day=day)
        return jsonify(news_payload)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to load news: {str(exc)}"}), 500


@bp.route("/domains/<domain_name>/entities/manage")
def entity_manager(domain_name: str) -> str:
    """Display entity management page for a domain."""
    domains = get_domains()
    if domain_name not in domains:
        return render_template("404.html"), 404

    try:
        # Load attached entities
        attached_entities = load_entities(domain_name)

        # Load recent detected entities (last 30 days, excluding attached names/akas)
        detected_entities = load_detected_entities_recent(domain_name, days=30)

        return render_template(
            "entity_manager.html",
            domain_name=domain_name,
            attached_entities=attached_entities,
            detected_entities=detected_entities,
            active="domains",
        )
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


@bp.route("/api/domains/<domain_name>/entities/manage/add-aka", methods=["POST"])
def add_aka_from_detected(domain_name: str) -> Any:
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
        entities = load_entities(domain_name)

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
            save_entities(domain_name, entities)

            return jsonify({"success": True})
        else:
            return jsonify({"error": "This name is already in the aka list"}), 409

    except Exception as e:
        return jsonify({"error": f"Failed to add aka: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities/manage/update-aka", methods=["POST"])
def update_aka_list(domain_name: str) -> Any:
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
        entities = load_entities(domain_name)

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
        save_entities(domain_name, entities)

        return jsonify({"success": True, "aka": aka_list})

    except Exception as e:
        return jsonify({"error": f"Failed to update aka list: {str(e)}"}), 500


@bp.route("/api/domains/<domain_name>/entities/detected/preview")
def preview_detected_entity_delete(domain_name: str) -> Any:
    """Preview which days contain a detected entity before deletion."""
    entity_name = request.args.get("name", "").strip()
    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        load_dotenv()
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            return jsonify({"error": "JOURNAL_PATH not set"}), 500

        entities_dir = Path(journal) / "domains" / domain_name / "entities"
        if not entities_dir.exists():
            return jsonify({"success": True, "days": []})

        # Scan all day files for this entity
        found_days = []
        for day_file in sorted(entities_dir.glob("*.jsonl")):
            day = day_file.stem
            entities = load_entities(domain_name, day)

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


@bp.route("/api/domains/<domain_name>/entities/detected", methods=["DELETE"])
def delete_detected_entity(domain_name: str) -> Any:
    """Delete a detected entity from all day files."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    entity_name = data.get("name", "").strip()
    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

    try:
        load_dotenv()
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            return jsonify({"error": "JOURNAL_PATH not set"}), 500

        entities_dir = Path(journal) / "domains" / domain_name / "entities"
        if not entities_dir.exists():
            return jsonify({"success": True, "days_modified": []})

        # Iterate through all day files and remove the entity
        days_modified = []
        for day_file in sorted(entities_dir.glob("*.jsonl")):
            day = day_file.stem
            entities = load_entities(domain_name, day)

            # Filter out entities matching this name (any type)
            original_count = len(entities)
            filtered_entities = [e for e in entities if e.get("name") != entity_name]

            # Only save if we actually removed something
            if len(filtered_entities) < original_count:
                save_entities(domain_name, filtered_entities, day)
                days_modified.append(day)

        return jsonify({"success": True, "days_modified": days_modified})

    except Exception as e:
        return jsonify({"error": f"Failed to delete entity: {str(e)}"}), 500
