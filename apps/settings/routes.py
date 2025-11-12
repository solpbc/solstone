from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.utils import get_config as get_journal_config

from convey import state

settings_bp = Blueprint(
    "settings",
    __name__,
    url_prefix="/app/settings",
)


@settings_bp.route("/")
def index() -> str:
    """Render the settings view."""
    return render_template("app.html", app="settings")


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
