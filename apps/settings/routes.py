from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

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

        # Update allowed fields only
        allowed_fields = ["title", "description", "color", "emoji", "muted"]
        for field in allowed_fields:
            if field in data:
                config[field] = data[field]

        # Write back to file
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return jsonify({"success": True, "facet": facet_name, "config": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
