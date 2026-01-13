# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DATE_RE = re.compile(r"\d{8}")


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to 'Wednesday April 2nd' format."""
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        day = date_obj.day
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return date_obj.strftime(f"%A %B {day}{suffix}")
    except ValueError:
        return date_str


def format_date_short(date_str: str) -> str:
    """Convert YYYYMMDD to smart relative/short format.

    Returns:
        - "Today", "Yesterday", "Tomorrow" for those days
        - Day name (e.g., "Wednesday") for dates within the past 6 days
        - "Sat Nov 29" for other dates in current/recent year
        - "Sat Nov 29 '24" for dates >6 months ago in a different year
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        date_normalized = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        delta_days = (date_normalized - today).days

        # Today, Yesterday, Tomorrow
        if delta_days == 0:
            return "Today"
        elif delta_days == -1:
            return "Yesterday"
        elif delta_days == 1:
            return "Tomorrow"
        # Within past 6 days - use day name
        elif -6 <= delta_days < 0:
            return date_obj.strftime("%A")
        # Default short format
        else:
            short = date_obj.strftime("%a %b %-d")
            # Add year suffix if >6 months ago AND different year
            months_ago = (today.year - date_obj.year) * 12 + (
                today.month - date_obj.month
            )
            if months_ago > 6 and date_obj.year != today.year:
                short += date_obj.strftime(" '%y")
            return short
    except ValueError:
        return date_str


def time_since(epoch: int) -> str:
    """Return short human readable age for ``epoch`` seconds."""
    seconds = int(time.time() - epoch)
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    weeks = days // 7
    return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def spawn_agent(
    prompt: str,
    persona: str,
    provider: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Spawn a Cortex agent and return the agent_id.

    Thin wrapper around cortex_request that ensures imports are handled
    and returns the agent_id directly.

    Args:
        prompt: The task or question for the agent
        persona: Agent persona - system (e.g., "default") or app-qualified (e.g., "entities:entity_assist")
        provider: Optional provider override (openai, google, anthropic)
        config: Additional configuration (max_tokens, facet, continue_from, etc.)

    Returns:
        agent_id string (timestamp-based)

    Raises:
        ValueError: If config is invalid
        Exception: On agent spawn failure
    """
    from muse.cortex_client import cortex_request

    return cortex_request(
        prompt=prompt,
        persona=persona,
        provider=provider,
        config=config,
    )


def parse_pagination_params(
    default_limit: int = 20,
    max_limit: int = 100,
    min_limit: int = 1,
) -> tuple[int, int]:
    """Parse and validate pagination parameters from request.args.

    Extracts limit and offset from Flask request.args, validates them,
    and enforces bounds to prevent API abuse.

    Args:
        default_limit: Default value for limit if not provided or invalid
        max_limit: Maximum allowed value for limit
        min_limit: Minimum allowed value for limit

    Returns:
        (limit, offset) tuple with validated integers

    Example:
        limit, offset = parse_pagination_params(default_limit=20, max_limit=100)
    """
    from flask import request

    # Parse limit with error handling
    try:
        limit = int(request.args.get("limit", default_limit))
    except (ValueError, TypeError):
        limit = default_limit

    # Parse offset with error handling
    try:
        offset = int(request.args.get("offset", 0))
    except (ValueError, TypeError):
        offset = 0

    # Enforce bounds
    limit = max(min_limit, min(limit, max_limit))
    offset = max(0, offset)

    return limit, offset


def load_json(path: str | Path) -> dict | list | None:
    """Load JSON file with consistent error handling.

    Args:
        path: Path to JSON file (string or Path object)

    Returns:
        Parsed JSON data (dict or list), or None if file doesn't exist or can't be parsed

    Example:
        data = load_json("config.json")
        if data:
            print(data.get("key"))
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_json(
    path: str | Path,
    data: dict | list,
    indent: int = 2,
    add_newline: bool = True,
) -> bool:
    """Save JSON file with consistent formatting.

    Args:
        path: Path to JSON file (string or Path object)
        data: Data to serialize (dict or list)
        indent: Indentation level (default: 2)
        add_newline: Whether to add trailing newline for readability (default: True)

    Returns:
        True if successful, False otherwise

    Example:
        success = save_json("config.json", {"key": "value"})
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            if add_newline:
                f.write("\n")
        return True
    except (OSError, TypeError):
        return False


def error_response(message: str, code: int = 400) -> tuple[Any, int]:
    """Create a standard JSON error response.

    Provides consistent error response format across all API endpoints.

    Args:
        message: Error message to return to client
        code: HTTP status code (default: 400 Bad Request)

    Returns:
        Tuple of (jsonify response, status_code) ready for Flask return

    Example:
        return error_response("Invalid input", 400)
        return error_response("Not found", 404)
    """
    from flask import jsonify

    return jsonify({"error": message}), code


def success_response(
    data: dict[str, Any] | None = None, code: int = 200
) -> tuple[Any, int]:
    """Create a standard JSON success response.

    Provides consistent success response format across all API endpoints.

    Args:
        data: Optional dict of additional data to include in response
        code: HTTP status code (default: 200 OK)

    Returns:
        Tuple of (jsonify response, status_code) ready for Flask return

    Example:
        return success_response()  # Returns {"success": True}
        return success_response({"agent_id": "123"})  # Returns {"success": True, "agent_id": "123"}
    """
    from flask import jsonify

    response_data = {"success": True}
    if data:
        response_data.update(data)
    return jsonify(response_data), code
