# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from apps.utils import log_app_action
from convey import state
from think.utils import get_config as get_journal_config

settings_bp = Blueprint(
    "app:settings",
    __name__,
    url_prefix="/app/settings",
)


@settings_bp.route("/api/config")
def get_config() -> Any:
    """Return the journal configuration."""
    try:
        config = get_journal_config()
        return jsonify(config)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/config", methods=["PUT"])
def update_config() -> Any:
    """Update the journal configuration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        config_dir = Path(state.journal_root) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / "journal.json"

        # Load existing config using shared utility
        config = get_journal_config()

        # Update the identity section with provided data
        if "identity" in data:
            config["identity"].update(data["identity"])

        # Write back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return jsonify({"success": True, "config": config})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet", methods=["POST"])
def create_facet() -> Any:
    """Create a new facet.

    Accepts JSON with:
        title: Display title (required)

    The facet name (slug) is auto-generated from the title.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        title = data.get("title", "").strip()
        if not title:
            return jsonify({"error": "Title is required"}), 400

        # Generate slug from title: lowercase, replace spaces/special chars with hyphens
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
        slug = slug.strip("-")  # Remove leading/trailing hyphens

        if not slug:
            return (
                jsonify({"error": "Title must contain at least one letter or number"}),
                400,
            )

        # Check for conflicts with existing facets
        from think.facets import get_facets

        existing = get_facets()
        if slug in existing:
            return jsonify({"error": f"Facet '{slug}' already exists"}), 409

        # Create facet directory and config
        facet_path = Path(state.journal_root) / "facets" / slug
        facet_path.mkdir(parents=True, exist_ok=True)

        config = {
            "title": title,
            "description": "",
            "color": "#667eea",
            "emoji": "📦",
        }

        config_file = facet_path / "facet.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log the creation
        log_app_action(
            app="settings",
            facet=slug,
            action="facet_create",
            params={"title": title},
        )

        return jsonify({"success": True, "facet": slug, "config": config}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet/<facet_name>")
def get_facet_config(facet_name: str) -> Any:
    """Get configuration for a specific facet."""
    try:
        from think.facets import get_facets

        facets = get_facets()
        if facet_name not in facets:
            return jsonify({"error": "Facet not found"}), 404

        return jsonify({"facet": facet_name, "config": facets[facet_name]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet/<facet_name>", methods=["PUT"])
def update_facet_config(facet_name: str) -> Any:
    """Update configuration for a specific facet."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Build path to facet config file
        facet_path = Path(state.journal_root) / "facets" / facet_name
        config_file = facet_path / "facet.json"

        if not facet_path.exists():
            return jsonify({"error": "Facet not found"}), 404

        # Read existing config or create new one
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        # Track changes for logging
        changed_fields = {}
        allowed_fields = ["title", "description", "color", "emoji", "muted"]
        for field in allowed_fields:
            if field in data:
                old_value = config.get(field)
                new_value = data[field]
                if old_value != new_value:
                    changed_fields[field] = {"old": old_value, "new": new_value}
                config[field] = new_value

        # Write back to file
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Log only if something actually changed
        if changed_fields:
            log_app_action(
                app="settings",
                facet=facet_name,
                action="facet_update",
                params={"changed_fields": changed_fields},
            )

        return jsonify({"success": True, "facet": facet_name, "config": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/api/facet/<facet_name>/logs")
def get_facet_logs(facet_name: str) -> Any:
    """Get action logs for a facet, one day at a time.

    Query params:
        cursor: Optional YYYYMMDD - load the day before this date

    Returns:
        {day, entries, next_cursor} where next_cursor is null if no more days
    """
    logs_dir = Path(state.journal_root) / "facets" / facet_name / "logs"

    if not logs_dir.exists():
        return jsonify({"day": None, "entries": [], "next_cursor": None})

    # Find all log files sorted newest first
    log_files = sorted(
        [f for f in logs_dir.iterdir() if re.fullmatch(r"\d{8}\.jsonl", f.name)],
        key=lambda f: f.stem,
        reverse=True,
    )

    if not log_files:
        return jsonify({"day": None, "entries": [], "next_cursor": None})

    # Apply cursor filter if provided
    cursor = request.args.get("cursor")
    if cursor:
        log_files = [f for f in log_files if f.stem < cursor]

    if not log_files:
        return jsonify({"day": None, "entries": [], "next_cursor": None})

    # Load the first (newest) day
    target_file = log_files[0]
    day = target_file.stem
    entries = []

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        pass

    # Reverse to show newest first within the day
    entries.reverse()

    # Determine next cursor
    next_cursor = log_files[1].stem if len(log_files) > 1 else None

    return jsonify({"day": day, "entries": entries, "next_cursor": next_cursor})
