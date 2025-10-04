from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Blueprint, jsonify, render_template, request

from think.indexer import scan_entities, search_entities

from .. import state
from ..utils import (
    format_date,
    generate_top_summary,
    modify_entity_file,
    modify_entity_in_file,
    parse_entity_line,
    update_top_entry,
)

bp = Blueprint("entities", __name__, template_folder="../templates")


def reload_entities() -> None:
    """Rescan entity files and rebuild the search index."""
    scan_entities(state.journal_root)


@bp.route("/entities")
def entities() -> str:
    return render_template("entities.html", active="entities")


@bp.route("/entities/api/types")
def entities_types() -> Any:
    """Return available entity types and their counts."""
    types = ["Person", "Company", "Project", "Tool"]
    data: Dict[str, int] = {}
    for t in types:
        total, _ = search_entities("", limit=0, etype=t)
        data[t] = total
    return jsonify(data)


@bp.route("/entities/api/list")
def entities_list() -> Any:
    """Return entities for a specific type ordered by count."""
    etype = request.args.get("type")
    if not etype:
        return jsonify([])
    _total_top, top_results = search_entities(
        "",
        limit=500,
        etype=etype,
        top=True,
        order="count",
    )
    _total_other, other_results = search_entities(
        "",
        limit=500,
        etype=etype,
        top=False,
        order="count",
    )
    results = []
    for r in top_results + other_results:
        meta = r["metadata"]
        results.append(
            {
                "name": meta["name"],
                "desc": r["text"],
                "top": meta.get("top", False),
                "count": meta.get("days", 0),
            }
        )
    return jsonify(results)


@bp.route("/entities/api/details")
def entities_details() -> Any:
    """Return detailed info for a single entity."""
    etype = request.args.get("type")
    name = request.args.get("name")
    if not etype or not name:
        return jsonify({})
    _total, results = search_entities(
        "",
        limit=1000,
        etype=etype,
        name=name,
        order="day",
    )
    if not results:
        return jsonify({})
    aggregated = results[0]
    top_flag = aggregated["metadata"].get("top", False)
    desc = aggregated["text"]
    descriptions: Dict[str, str] = {}
    raw_dates: List[str] = []
    for r in results[1:]:
        meta = r["metadata"]
        day = meta.get("day")
        if day:
            if day not in raw_dates:
                raw_dates.append(day)
            if r["text"]:
                descriptions[format_date(day)] = r["text"]
    raw_dates.sort()
    return jsonify(
        {
            "name": name,
            "top": top_flag,
            "desc": desc,
            "dates": [format_date(d) for d in raw_dates],
            "raw_dates": raw_dates,
            "descriptions": descriptions,
        }
    )


@bp.route("/entities/api/top_generate", methods=["POST"])
def api_top_generate() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    # Get all entity appearances to collect descriptions
    _total, results = search_entities(
        "", limit=1000, etype=etype, name=name, order="day"
    )
    if not results:
        return ("", 400)

    # Build info structure compatible with generate_top_summary
    descriptions = {}
    primary = ""
    for result in results:
        meta = result["metadata"]
        day = meta.get("day")
        text = result.get("text", "")
        if day and text and text != name:  # Skip empty descriptions and just the name
            descriptions[day] = text
        elif not day and text and text != name:  # Top-level entity
            primary = text

    # If no primary description from top-level, use the most recent daily description
    if not primary and descriptions:
        latest_day = max(descriptions.keys())
        primary = descriptions[latest_day]

    info = {"descriptions": descriptions, "primary": primary}

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ("", 400)
    try:
        desc = generate_top_summary(info, api_key)
        return jsonify({"desc": desc})
    except Exception as e:  # pragma: no cover - network errors
        return jsonify({"error": str(e)}), 500


@bp.route("/entities/api/top_update", methods=["POST"])
def api_top_update() -> Any:
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name")
    desc = (payload.get("desc") or "").replace("\n", " ").replace("\r", " ").strip()
    update_top_entry(state.journal_root, etype, name, desc)
    reload_entities()
    return jsonify({"status": "ok"})


@bp.route("/entities/api/create", methods=["POST"])
def api_create_entity() -> Any:
    """Create a new top-level entity."""
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name", "").strip()
    description = payload.get("description", "").strip()

    if not etype or not name:
        return (
            jsonify({"success": False, "error": "Entity type and name are required"}),
            400,
        )

    # Validate entity type
    valid_types = ["Person", "Company", "Project", "Tool"]
    if etype not in valid_types:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Invalid entity type. Must be one of: {', '.join(valid_types)}",
                }
            ),
            400,
        )

    try:
        # Check if entity already exists at top level
        _total, existing = search_entities("", limit=1, etype=etype, name=name)
        if existing:
            # Check if any existing entity is a top-level entity
            for result in existing:
                if result["metadata"].get("top", False):
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": f"A top-level {etype} named '{name}' already exists",
                            }
                        ),
                        409,
                    )

        # Create the entity by adding it to the top-level entities.md file
        if description:
            update_top_entry(state.journal_root, etype, name, description)
        else:
            update_top_entry(
                state.journal_root, etype, name, name
            )  # Use name as default description

        # Reload entities index
        reload_entities()

        return jsonify({"success": True})

    except Exception as e:
        logging.error(f"Error creating entity '{etype}: {name}': {str(e)}")
        return (
            jsonify(
                {"success": False, "error": "An error occurred creating the entity"}
            ),
            500,
        )


@bp.route("/entities/api/merge", methods=["POST"])
def api_merge_entities() -> Any:
    """Merge source entity into target entity by renaming all occurrences."""
    payload = request.get_json(force=True)
    source_type = payload.get("source_type")
    source_name = payload.get("source_name")
    target_type = payload.get("target_type")
    target_name = payload.get("target_name")
    selected_days = payload.get("days")  # Optional: specific days to merge

    if not all([source_type, source_name, target_type, target_name]):
        return jsonify({"error": "Missing required fields"}), 400

    if source_name == target_name and source_type == target_type:
        return jsonify({"error": "Cannot merge entity with itself"}), 400

    # Always get the source entity results for later use
    _total, source_results = search_entities(
        "", limit=1000, etype=source_type, name=source_name, order="day"
    )

    if not source_results:
        return (
            jsonify(
                {"error": f"Source entity '{source_type}: {source_name}' not found"}
            ),
            404,
        )

    # If specific days are provided, use those; otherwise get all days
    if selected_days:
        days_to_update = selected_days
    else:
        # Collect all days where source appears (excluding aggregated entity)
        days_to_update = []
        for result in source_results:
            meta = result.get("metadata", {})
            day = meta.get("day")
            if day:  # Only process day-specific occurrences
                days_to_update.append(day)

    successful_days = []
    failed_days = []

    # Process each day's entities.md file to merge source into target
    for day in days_to_update:
        entities_file = os.path.join(state.journal_root, day, "entities.md")
        if not os.path.exists(entities_file):
            failed_days.append(day)
            continue

        try:
            with open(entities_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for line in lines:
                # Parse entity line to check if it matches source
                parsed = parse_entity_line(line)
                if parsed:
                    etype, name, desc = parsed
                    if etype == source_type and name == source_name:
                        # Replace with target entity, preserving description if present
                        new_line = f"* {target_type}: {target_name}"
                        if desc:
                            new_line += f" - {desc}"
                        new_line += "\n" if line.endswith("\n") else ""
                        new_lines.append(new_line)
                        modified = True
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            if modified:
                with open(entities_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                successful_days.append(day)
            else:
                # No source entity found in this day
                failed_days.append(day)
        except Exception as e:
            logging.error(f"Error processing entities.md for day {day}: {e}")
            failed_days.append(day)

    # Handle top-level entity merge
    top_file = os.path.join(state.journal_root, "entities.md")

    # First remove source from top-level if it exists
    modify_entity_in_file(
        top_file, source_type, source_name, None, "remove", require_match=False
    )

    # Ensure target exists in top-level (if source was top-level, promote target)
    _total, target_results = search_entities(
        "", limit=1, etype=target_type, name=target_name
    )
    if target_results and not target_results[0]["metadata"].get("top", False):
        # Target exists but is not top-level, check if source was top-level
        source_was_top = any(r["metadata"].get("top", False) for r in source_results)
        if source_was_top:
            # Promote target to top-level with source's description if available
            source_desc = next(
                (r["text"] for r in source_results if r["metadata"].get("top", False)),
                target_name,
            )
            update_top_entry(state.journal_root, target_type, target_name, source_desc)

    if failed_days:
        logging.info(
            f"Entity merge '{source_type}: {source_name}' -> '{target_type}: {target_name}' "
            f"failed for days: {failed_days}"
        )

    reload_entities()

    return jsonify(
        {
            "status": "ok",
            "successful_days": successful_days,
            "failed_days": failed_days,
            "message": f"Merged {len(successful_days)} occurrences",
        }
    )


@bp.route("/entities/api/remove", methods=["POST"])
@bp.route("/entities/api/rename", methods=["POST"])
def api_modify_entity() -> Any:
    payload = request.get_json(force=True)
    action = "remove" if request.path.endswith("remove") else "rename"
    days = payload.get("days", [])
    etype = payload.get("type")
    name = payload.get("name")
    new_name = payload.get("new_name") if action == "rename" else None

    successful_days = []
    failed_days = []

    for day in days:
        success = modify_entity_file(
            state.journal_root, day, etype, name, new_name, action
        )
        if success:
            successful_days.append(day)
        else:
            failed_days.append(day)

    if action == "rename" and new_name:
        top_file = os.path.join(state.journal_root, "entities.md")
        success = modify_entity_in_file(
            top_file, etype, name, new_name, "rename", require_match=False
        )
        if not success:
            logging.info(
                f"Entity '{etype}: {name}' not found in top-level entities.md, skipping top-level rename"
            )

    if failed_days:
        logging.info(
            f"Entity '{etype}: {name}' operation '{action}' failed for days: {failed_days}"
        )

    reload_entities()
    return jsonify(
        {"status": "ok", "successful_days": successful_days, "failed_days": failed_days}
    )


@bp.route("/entities/api/generate-description", methods=["POST"])
def api_generate_entity_description() -> Any:
    """Generate a description for a new entity using AI agent."""
    payload = request.get_json(force=True)
    etype = payload.get("type")
    name = payload.get("name", "").strip()

    if not etype or not name:
        return (
            jsonify({"success": False, "error": "Entity type and name are required"}),
            400,
        )

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return jsonify({"error": "JOURNAL_PATH not set"}), 500

    try:
        # Import the run_agent_via_cortex function
        from ..cortex_utils import run_agent_via_cortex

        # Build context for the agent
        prompt = f"""Generate a single complete sentence description for this entity:

Entity Type: {etype}
Entity Name: {name}

Use the search tools to research this entity in the journal, then create a precise, informative single sentence that captures the entity's role, purpose, and significance."""

        # Use the entity_describe persona with cortex
        description = run_agent_via_cortex(prompt=prompt, persona="entity_describe")

        if description:
            # Clean up the description - remove any quotes or extra formatting
            description = description.strip()
            if description.startswith('"') and description.endswith('"'):
                description = description[1:-1]

            return jsonify({"success": True, "description": description})
        else:
            return (
                jsonify({"success": False, "error": "Failed to generate description"}),
                500,
            )

    except Exception as e:
        logging.error(f"Error generating entity description: {e}")
        return (
            jsonify(
                {"success": False, "error": f"Failed to generate description: {str(e)}"}
            ),
            500,
        )
