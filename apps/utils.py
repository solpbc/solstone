"""Utility functions for Convey app storage in journal."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from convey import state

# Compiled pattern for app name validation
APP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def get_app_storage_path(
    app_name: str,
    *sub_dirs: str,
    ensure_exists: bool = True,
) -> Path:
    """
    Get path to app storage directory in journal.

    Args:
        app_name: App name (must match [a-z][a-z0-9_]*)
        *sub_dirs: Optional subdirectory components
        ensure_exists: Create directory if it doesn't exist (default: True)

    Returns:
        Path to <journal>/apps/<app_name>/<sub_dirs>/

    Raises:
        ValueError: If app_name contains invalid characters

    Examples:
        get_app_storage_path("search")  # → Path("<journal>/apps/search")
        get_app_storage_path("search", "cache")  # → Path("<journal>/apps/search/cache")
    """
    # Validate app_name to prevent path traversal
    if not APP_NAME_PATTERN.match(app_name):
        raise ValueError(f"Invalid app name: {app_name}")

    # Build path
    path = Path(state.journal_root) / "apps" / app_name
    for sub_dir in sub_dirs:
        path = path / sub_dir

    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)

    return path


def load_app_config(
    app_name: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Load app configuration from <journal>/apps/<app_name>/config.json.

    Args:
        app_name: App name
        default: Default value if config doesn't exist (default: None)

    Returns:
        Loaded JSON dict or default value if file doesn't exist

    Examples:
        config = load_app_config("my_app")  # Returns None if missing
        config = load_app_config("my_app", {})  # Returns {} if missing
    """
    from convey.utils import load_json

    storage_path = get_app_storage_path(app_name, ensure_exists=False)
    config_path = storage_path / "config.json"
    return load_json(config_path) or default


def save_app_config(
    app_name: str,
    config: dict[str, Any],
) -> bool:
    """
    Save app configuration to <journal>/apps/<app_name>/config.json.

    Args:
        app_name: App name
        config: Configuration dict to save

    Returns:
        True if successful, False otherwise
    """
    from convey.utils import save_json

    storage_path = get_app_storage_path(app_name, ensure_exists=True)
    config_path = storage_path / "config.json"
    return save_json(config_path, config)


def log_app_action(
    app: str,
    facet: str,
    action: str,
    params: dict[str, Any],
    day: str | None = None,
) -> None:
    """Log a user-initiated action from a Convey app.

    Creates a JSONL log entry in facets/{facet}/logs/{day}.jsonl for tracking
    user actions made through the web UI.

    Args:
        app: App name where action originated (e.g., "entities", "todos")
        facet: Facet where action occurred
        action: Action type (e.g., "entity_add", "todo_complete")
        params: Action-specific parameters to record
        day: Day in YYYYMMDD format (defaults to today)

    Example:
        log_app_action(
            app="entities",
            facet="work",
            action="entity_add",
            params={"type": "Person", "name": "Alice"},
        )
    """
    from think.facets import _write_action_log

    if day is None:
        day = datetime.now().strftime("%Y%m%d")

    _write_action_log(
        facet=facet,
        action=action,
        params=params,
        source="app",
        actor=app,
        day=day,
    )
