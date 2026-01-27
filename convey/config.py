# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Convey configuration management and API routes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Blueprint, request

from . import state
from .utils import error_response, load_json, save_json, success_response

logger = logging.getLogger(__name__)

bp = Blueprint("config", __name__, url_prefix="/api/config")


def _get_config_path() -> Path:
    """Get path to config/convey.json in journal root."""
    return Path(state.journal_root) / "config" / "convey.json"


def load_convey_config() -> dict[str, Any]:
    """Load config/convey.json from journal root.

    Returns:
        Config dict with optional fields:
        - facets.order: list of facet names
        - facets.selected: currently selected facet name or null
        - apps.order: list of app names
        Empty dict if file doesn't exist or can't be parsed.
    """
    config_path = _get_config_path()
    data = load_json(config_path)
    return data if isinstance(data, dict) else {}


def save_convey_config(config: dict[str, Any]) -> bool:
    """Save config/convey.json atomically.

    Args:
        config: Configuration dict to save

    Returns:
        True if successful, False otherwise
    """
    config_path = _get_config_path()

    # Ensure config directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    return save_json(config_path, config, indent=2)


def get_selected_facet() -> str | None:
    """Get selected facet from config.

    Returns:
        Selected facet name, or None if not set
    """
    config = load_convey_config()
    facets_config = config.get("facets", {})
    return facets_config.get("selected")


def set_selected_facet(facet: str | None) -> None:
    """Update selected facet in config.

    Args:
        facet: Facet name to select, or None to clear selection
    """
    config = load_convey_config()

    # Ensure facets section exists
    if "facets" not in config:
        config["facets"] = {}

    # Update selected field
    config["facets"]["selected"] = facet

    # Save config (async safe - doesn't block if write fails)
    success = save_convey_config(config)
    if not success:
        logger.warning(f"Failed to save selected facet: {facet}")


def apply_facet_order(facets: list[dict], config: dict) -> list[dict]:
    """Apply custom ordering from config to facet list.

    Args:
        facets: List of facet dicts with 'name' field
        config: Config dict with optional facets.order field

    Returns:
        Reordered facet list (ordered items first, then alphabetical remainder)
    """
    order = config.get("facets", {}).get("order", [])
    if not order:
        return facets

    # Create lookup by name
    facet_map = {f["name"]: f for f in facets}

    # Ordered items first (if they exist)
    ordered = [facet_map[name] for name in order if name in facet_map]

    # Remaining items alphabetically
    ordered_names = set(order)
    remaining = sorted(
        [f for f in facets if f["name"] not in ordered_names],
        key=lambda f: f["name"],
    )

    return ordered + remaining


def apply_app_order(apps: dict[str, Any], config: dict) -> dict[str, Any]:
    """Apply custom ordering from config to app dict.

    Groups apps by starred status, then applies ordering within each group.
    Starred apps appear first, followed by unstarred apps.

    Args:
        apps: Dict mapping app name to app data
        config: Config dict with optional apps.order and apps.starred fields

    Returns:
        Reordered dict (starred apps first in order, then unstarred apps in order)
    """
    order = config.get("apps", {}).get("order", [])
    starred = set(config.get("apps", {}).get("starred", []))

    # Separate apps into starred and unstarred
    starred_apps = {}
    unstarred_apps = {}

    for name, data in apps.items():
        if name in starred:
            starred_apps[name] = data
        else:
            unstarred_apps[name] = data

    # Helper to order a subset of apps
    def order_apps(app_dict: dict[str, Any], app_order: list[str]) -> dict[str, Any]:
        ordered = {}
        # Ordered items first (if they exist)
        for name in app_order:
            if name in app_dict:
                ordered[name] = app_dict[name]
        # Remaining items alphabetically
        ordered_names = set(app_order)
        for name in sorted(app_dict.keys()):
            if name not in ordered_names:
                ordered[name] = app_dict[name]
        return ordered

    # Order each group
    ordered_starred = order_apps(starred_apps, order) if starred_apps else {}
    ordered_unstarred = order_apps(unstarred_apps, order) if unstarred_apps else {}

    # Combine: starred first, then unstarred
    result = {}
    result.update(ordered_starred)
    result.update(ordered_unstarred)

    return result


def validate_config(config: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate config structure and values.

    Args:
        config: Config dict to validate

    Returns:
        (is_valid, error_message) tuple
    """
    # Check top-level structure
    if not isinstance(config, dict):
        return False, "Config must be a JSON object"

    # Validate facets section if present
    if "facets" in config:
        facets_config = config["facets"]
        if not isinstance(facets_config, dict):
            return False, "facets must be an object"

        # Validate facets.order
        if "order" in facets_config:
            order = facets_config["order"]
            if not isinstance(order, list):
                return False, "facets.order must be an array"
            if not all(isinstance(name, str) for name in order):
                return False, "facets.order must contain only strings"

        # Validate facets.selected
        if "selected" in facets_config:
            selected = facets_config["selected"]
            if selected is not None and not isinstance(selected, str):
                return False, "facets.selected must be a string or null"

    # Validate apps section if present
    if "apps" in config:
        apps_config = config["apps"]
        if not isinstance(apps_config, dict):
            return False, "apps must be an object"

        # Validate apps.order
        if "order" in apps_config:
            order = apps_config["order"]
            if not isinstance(order, list):
                return False, "apps.order must be an array"
            if not all(isinstance(name, str) for name in order):
                return False, "apps.order must contain only strings"

        # Validate apps.starred
        if "starred" in apps_config:
            starred = apps_config["starred"]
            if not isinstance(starred, list):
                return False, "apps.starred must be an array"
            if not all(isinstance(name, str) for name in starred):
                return False, "apps.starred must contain only strings"

    return True, None


# API Routes


@bp.route("/convey")
def get_config() -> tuple[Any, int]:
    """GET /api/config/convey - Return current convey configuration.

    Returns:
        JSON response with config data
    """
    try:
        config = load_convey_config()
        return success_response({"config": config})
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        return error_response("Failed to load configuration", 500)


@bp.route("/convey", methods=["POST"])
def update_config() -> tuple[Any, int]:
    """POST /api/config/convey - Update convey configuration.

    Request body: Full or partial config object
    {
        "facets": {
            "order": ["work", "personal"],
            "selected": "work"
        },
        "apps": {
            "order": ["home", "calendar"]
        }
    }

    Returns:
        JSON success/error response
    """
    try:
        # Parse request
        new_config = request.get_json()
        if not new_config:
            return error_response("Request body must be JSON", 400)

        # Validate structure
        valid, error_msg = validate_config(new_config)
        if not valid:
            return error_response(f"Invalid config: {error_msg}", 400)

        # Merge with existing config (partial updates supported)
        current_config = load_convey_config()

        # Deep merge facets section
        if "facets" in new_config:
            if "facets" not in current_config:
                current_config["facets"] = {}
            current_config["facets"].update(new_config["facets"])

        # Deep merge apps section
        if "apps" in new_config:
            if "apps" not in current_config:
                current_config["apps"] = {}
            current_config["apps"].update(new_config["apps"])

        # Save updated config
        success = save_convey_config(current_config)
        if not success:
            return error_response("Failed to save configuration", 500)

        return success_response({"config": current_config})

    except Exception as e:
        logger.error(f"Failed to update config: {e}", exc_info=True)
        return error_response("Failed to update configuration", 500)


@bp.route("/facets/order", methods=["POST"])
def update_facet_order() -> tuple[Any, int]:
    """POST /api/config/facets/order - Update facet ordering.

    Request body: {"order": ["work", "personal", "research"]}

    Returns:
        JSON success/error response
    """
    try:
        data = request.get_json()
        if not data or "order" not in data:
            return error_response("Request must include 'order' array", 400)

        order = data["order"]
        if not isinstance(order, list):
            return error_response("'order' must be an array", 400)

        if not all(isinstance(name, str) for name in order):
            return error_response("'order' must contain only strings", 400)

        # Load config and update facets.order
        config = load_convey_config()
        if "facets" not in config:
            config["facets"] = {}
        config["facets"]["order"] = order

        # Save
        success = save_convey_config(config)
        if not success:
            return error_response("Failed to save facet order", 500)

        return success_response({"order": order})

    except Exception as e:
        logger.error(f"Failed to update facet order: {e}", exc_info=True)
        return error_response("Failed to update facet order", 500)


@bp.route("/apps/order", methods=["POST"])
def update_app_order() -> tuple[Any, int]:
    """POST /api/config/apps/order - Update app ordering.

    Request body: {"order": ["home", "calendar", "todos"]}

    Returns:
        JSON success/error response
    """
    try:
        data = request.get_json()
        if not data or "order" not in data:
            return error_response("Request must include 'order' array", 400)

        order = data["order"]
        if not isinstance(order, list):
            return error_response("'order' must be an array", 400)

        if not all(isinstance(name, str) for name in order):
            return error_response("'order' must contain only strings", 400)

        # Load config and update apps.order
        config = load_convey_config()
        if "apps" not in config:
            config["apps"] = {}
        config["apps"]["order"] = order

        # Save
        success = save_convey_config(config)
        if not success:
            return error_response("Failed to save app order", 500)

        return success_response({"order": order})

    except Exception as e:
        logger.error(f"Failed to update app order: {e}", exc_info=True)
        return error_response("Failed to update app order", 500)


@bp.route("/apps/star", methods=["POST"])
def toggle_app_star() -> tuple[Any, int]:
    """POST /api/config/apps/star - Toggle starred status of an app.

    Request body: {"app": "calendar", "starred": true}

    Returns:
        JSON success/error response
    """
    try:
        data = request.get_json()
        if not data or "app" not in data or "starred" not in data:
            return error_response(
                "Request must include 'app' and 'starred' fields", 400
            )

        app_name = data["app"]
        starred = data["starred"]

        if not isinstance(app_name, str):
            return error_response("'app' must be a string", 400)

        if not isinstance(starred, bool):
            return error_response("'starred' must be a boolean", 400)

        # Load config and update apps.starred
        config = load_convey_config()
        if "apps" not in config:
            config["apps"] = {}

        starred_apps = set(config["apps"].get("starred", []))

        if starred:
            starred_apps.add(app_name)
        else:
            starred_apps.discard(app_name)

        config["apps"]["starred"] = sorted(starred_apps)

        # Save
        success = save_convey_config(config)
        if not success:
            return error_response("Failed to save app starred status", 500)

        return success_response({"app": app_name, "starred": starred})

    except Exception as e:
        logger.error(f"Failed to toggle app star: {e}", exc_info=True)
        return error_response("Failed to toggle app starred status", 500)


@bp.route("/facets/select", methods=["POST"])
def select_facet() -> tuple[Any, int]:
    """POST /api/config/facets/select - Update selected facet.

    Request body: {"facet": "work"} or {"facet": null}

    Returns:
        JSON success/error response
    """
    try:
        data = request.get_json()
        if data is None:
            return error_response("Request body must be JSON", 400)

        if "facet" not in data:
            return error_response("Request must include 'facet' field", 400)

        facet = data["facet"]
        if facet is not None and not isinstance(facet, str):
            return error_response("'facet' must be a string or null", 400)

        # Update config
        set_selected_facet(facet)

        return success_response({"facet": facet})

    except Exception as e:
        logger.error(f"Failed to update selected facet: {e}", exc_info=True)
        return error_response("Failed to update selected facet", 500)
