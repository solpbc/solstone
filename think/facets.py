# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Facet-specific utilities and tooling for the think module."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastmcp import Context
from fastmcp.server.dependencies import get_http_headers

from think.entities import get_identity_names
from think.utils import get_journal


def _get_principal_display_name() -> str | None:
    """Get the display name for the principal from identity config.

    Returns the first identity name (preferred if set, else full name).
    Returns None if identity is not configured.
    """
    names = get_identity_names()
    return names[0] if names else None


def _format_principal_role(
    entities: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract principal entity and format role line if all info available.

    Args:
        entities: List of entity dicts from load_entities()

    Returns:
        Tuple of (role_line, filtered_entities) where:
        - role_line is markdown like "**Jer's Role**: Description" or None if incomplete
        - filtered_entities excludes the principal entity
    """
    # Find principal entity
    principal = None
    other_entities = []
    for entity in entities:
        if entity.get("is_principal"):
            principal = entity
        else:
            other_entities.append(entity)

    if not principal:
        return None, entities

    # Get display name and description
    display_name = _get_principal_display_name()
    description = principal.get("description", "").strip()

    # Only format if we have both name and description
    if not display_name or not description:
        return None, entities

    role_line = f"**{display_name}'s Role**: {description}"
    return role_line, other_entities


def _format_entity_name_with_aka(entity: dict[str, Any]) -> str:
    """Format entity name, appending aka values in parentheses if present.

    Args:
        entity: Entity dict with 'name' and optional 'aka' list

    Returns:
        Formatted name string, e.g. "John Smith" or "John Smith (JS, Johnny)"
    """
    name = entity.get("name", "")
    aka_list = entity.get("aka", [])
    if isinstance(aka_list, list) and aka_list:
        aka_str = ", ".join(aka_list)
        return f"{name} ({aka_str})"
    return name


def _format_activity_line(activity: dict[str, Any], *, bold_name: bool = False) -> str:
    """Format a single activity as a markdown list item.

    Args:
        activity: Activity dict with 'name'/'id', 'description', 'priority'
        bold_name: If True, wraps name in **bold**

    Returns:
        Formatted string like "**Meetings** (high): Description" or "Meetings (high): Description"
    """
    name = activity.get("name", activity.get("id", ""))
    desc = activity.get("description", "")
    priority = activity.get("priority", "normal")

    # Format with priority tag if non-normal
    if priority == "high":
        priority_suffix = " (high)"
    elif priority == "low":
        priority_suffix = " (low)"
    else:
        priority_suffix = ""

    if bold_name:
        name_part = f"**{name}**{priority_suffix}"
    else:
        name_part = f"{name}{priority_suffix}"

    if desc:
        return f"{name_part}: {desc}"
    return name_part


def _get_actor_info(context: Context | None = None) -> tuple[str, str | None]:
    """Extract actor (name) and agent_id from meta or HTTP headers.

    Priority: meta (stdio/anthropic/google) > HTTP headers (openai)

    Args:
        context: Optional FastMCP context with request metadata

    Returns:
        Tuple of (actor, agent_id) where actor defaults to "mcp"
        and agent_id may be None.
    """
    # First try meta from context (stdio transport)
    if context is not None:
        try:
            meta = context.request_context.meta
            if meta:
                # Convert Pydantic model to dict and filter out None values
                meta_dict = {
                    k: v for k, v in meta.model_dump().items() if v is not None
                }
                name = meta_dict.get("name")
                agent_id = meta_dict.get("agent_id")
                if name or agent_id:
                    actor = name if name else "mcp"
                    return actor, agent_id
        except Exception:
            pass

    # Fallback to HTTP headers (HTTP transport)
    try:
        headers = get_http_headers(include_all=True)
        # Normalize headers to lowercase for case-insensitive lookup
        headers_lower = {k.lower(): v for k, v in headers.items()}

        name = headers_lower.get("x-agent-name")
        agent_id = headers_lower.get("x-agent-id")

        actor = name if name else "mcp"
        return actor, agent_id
    except Exception:
        # Not in HTTP context (stdio, tests)
        return "mcp", None


def _write_action_log(
    facet: str | None,
    action: str,
    params: dict[str, Any],
    source: str,
    actor: str,
    day: str | None = None,
    agent_id: str | None = None,
) -> None:
    """Write action to the daily audit log.

    Internal function that writes JSONL log entries. When facet is provided,
    writes to facets/{facet}/logs/{day}.jsonl. When facet is None, writes to
    config/actions/{day}.jsonl for journal-level actions.

    Use log_tool_action() for MCP tools or log_app_action() for web apps.

    Args:
        facet: Facet name where the action occurred, or None for journal-level
        action: Action type (e.g., "todo_add", "entity_attach")
        params: Dictionary of action-specific parameters
        source: Origin type - "tool" for MCP agents, "app" for web UI
        actor: For tools: agent name. For apps: app name
        day: Day in YYYYMMDD format (defaults to today)
        agent_id: Optional agent ID (only for tool actions)
    """
    journal = get_journal()

    if day is None:
        day = datetime.now().strftime("%Y%m%d")

    # Build log file path based on whether facet is provided
    if facet is not None:
        log_path = Path(journal) / "facets" / facet / "logs" / f"{day}.jsonl"
    else:
        log_path = Path(journal) / "config" / "actions" / f"{day}.jsonl"

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create log entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "actor": actor,
        "action": action,
        "params": params,
    }

    # Add facet only if provided
    if facet is not None:
        entry["facet"] = facet

    # Add agent_id only if available
    if agent_id is not None:
        entry["agent_id"] = agent_id

    # Append to log file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_tool_action(
    facet: str | None,
    action: str,
    params: dict[str, Any],
    context: Context | None = None,
    day: str | None = None,
) -> None:
    """Log an agent-initiated action from an MCP tool.

    Creates a JSONL log entry for tracking successful modifications made via
    MCP tools. Automatically extracts actor identity (agent name) from FastMCP
    context.

    When facet is provided, writes to facets/{facet}/logs/{day}.jsonl.
    When facet is None, writes to config/actions/{day}.jsonl for journal-level
    actions (settings changes, system operations, etc.).

    Args:
        facet: Facet name where the action occurred, or None for journal-level
        action: Action type (e.g., "todo_add", "entity_attach")
        params: Dictionary of action-specific parameters
        context: Optional FastMCP context for extracting agent name/agent_id
        day: Day in YYYYMMDD format (defaults to today)
    """
    actor, agent_id = _get_actor_info(context)
    _write_action_log(
        facet=facet,
        action=action,
        params=params,
        source="tool",
        actor=actor,
        day=day,
        agent_id=agent_id,
    )


def get_facets() -> dict[str, dict[str, object]]:
    """Return available facets with metadata.

    Each key is the facet name. The value contains the facet metadata
    from facet.json including title, description, and the facet path.
    """
    facets_dir = Path(get_journal()) / "facets"
    facets: dict[str, dict[str, object]] = {}

    if not facets_dir.exists():
        return facets

    for facet_path in sorted(facets_dir.iterdir()):
        if not facet_path.is_dir():
            continue

        facet_name = facet_path.name
        facet_json = facet_path / "facet.json"

        if not facet_json.exists():
            continue

        try:
            with open(facet_json, "r", encoding="utf-8") as f:
                facet_data = json.load(f)

            if isinstance(facet_data, dict):
                facet_info = {
                    "path": str(facet_path),
                    "title": facet_data.get("title", facet_name),
                    "description": facet_data.get("description", ""),
                    "color": facet_data.get("color", ""),
                    "emoji": facet_data.get("emoji", ""),
                    "muted": facet_data.get("muted", False),
                }

                facets[facet_name] = facet_info
        except Exception as exc:  # pragma: no cover - metadata optional
            logging.debug("Error reading %s: %s", facet_json, exc)

    return facets


def get_enabled_facets() -> dict[str, dict[str, object]]:
    """Return non-muted facets only.

    Convenience wrapper around get_facets() that filters out muted facets.
    Used by scheduled agents to skip processing for muted facets.
    """
    return {k: v for k, v in get_facets().items() if not v.get("muted", False)}


def facet_summary(facet: str) -> str:
    """Generate a nicely formatted markdown summary of a facet.

    Args:
        facet: The facet name to summarize

    Returns:
        Formatted markdown string with facet title, description, entities,
        and activities

    Raises:
        FileNotFoundError: If the facet doesn't exist
    """
    from think.activities import get_facet_activities
    from think.entities import load_entities

    facet_path = Path(get_journal()) / "facets" / facet
    if not facet_path.exists():
        raise FileNotFoundError(f"Facet '{facet}' not found at {facet_path}")

    # Load facet metadata
    facet_json_path = facet_path / "facet.json"
    if not facet_json_path.exists():
        raise FileNotFoundError(f"facet.json not found for facet '{facet}'")

    with open(facet_json_path, "r", encoding="utf-8") as f:
        facet_data = json.load(f)

    # Extract metadata
    title = facet_data.get("title", facet)
    description = facet_data.get("description", "")
    color = facet_data.get("color", "")

    # Build markdown summary
    lines = []

    # Title without emoji
    lines.append(f"# {title}")

    # Add color as a badge if available
    if color:
        lines.append(f"![Color]({color})")
        lines.append("")

    # Description
    if description:
        lines.append(f"**Description:** {description}")
        lines.append("")

    # Load entities if available
    entities = load_entities(facet)
    if entities:
        # Extract principal role line and filter principal from list
        role_line, display_entities = _format_principal_role(entities)

        if role_line:
            lines.append(role_line)
            lines.append("")

        if display_entities:
            lines.append("## Entities")
            lines.append("")
            for entity in display_entities:
                entity_type = entity.get("type", "")
                formatted_name = _format_entity_name_with_aka(entity)
                desc = entity.get("description", "")

                if desc:
                    lines.append(f"- **{entity_type}**: {formatted_name} - {desc}")
                else:
                    lines.append(f"- **{entity_type}**: {formatted_name}")
            lines.append("")

    # Load activities if available
    activities = get_facet_activities(facet)
    if activities:
        lines.append("## Activities")
        lines.append("")
        for activity in activities:
            lines.append(f"- {_format_activity_line(activity, bold_name=True)}")
        lines.append("")

    return "\n".join(lines)


def get_facet_news(
    facet: str,
    *,
    cursor: Optional[str] = None,
    limit: int = 1,
    day: Optional[str] = None,
) -> dict[str, Any]:
    """Return facet news entries grouped by day, newest first.

    Parameters
    ----------
    facet:
        Facet name containing the news directory.
    cursor:
        Optional date string (``YYYYMMDD``). When provided, only news files with
        a date strictly earlier than the cursor are returned. This supports
        pagination in the UI where older entries are fetched on demand.
    limit:
        Maximum number of news days to return. Defaults to one day per request.
    day:
        Optional specific day (``YYYYMMDD``) to return. When provided, returns
        only news for that specific day if it exists. Overrides cursor and limit.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``days`` (list of news day payloads), ``next_cursor``
        (date string for subsequent requests) and ``has_more`` boolean flag.
    """
    news_dir = Path(get_journal()) / "facets" / facet / "news"
    if not news_dir.exists():
        return {"days": [], "next_cursor": None, "has_more": False}

    # If specific day requested, check for that file directly
    if day:
        news_path = news_dir / f"{day}.md"
        if news_path.exists() and news_path.is_file():
            selected = [news_path]
        else:
            return {"days": [], "next_cursor": None, "has_more": False}
    else:
        news_files = [
            path
            for path in news_dir.iterdir()
            if path.is_file() and re.fullmatch(r"\d{8}\.md", path.name)
        ]

        # Sort newest first by file name (YYYYMMDD.md)
        news_files.sort(key=lambda p: p.stem, reverse=True)

        if cursor:
            news_files = [path for path in news_files if path.stem < cursor]

        if limit is not None and limit > 0:
            selected = news_files[:limit]
        else:
            selected = news_files

    days: list[dict[str, Any]] = []

    for news_path in selected:
        date_key = news_path.stem

        # Read the raw markdown content
        raw_content = ""
        try:
            raw_content = news_path.read_text(encoding="utf-8")
        except Exception:
            pass

        days.append(
            {
                "date": date_key,
                "raw_content": raw_content,
            }
        )

    # When specific day requested, no pagination
    if day:
        has_more = False
        next_cursor = None
    else:
        has_more = len(news_files) > len(selected)
        next_cursor = selected[-1].stem if has_more and selected else None

    return {"days": days, "next_cursor": next_cursor, "has_more": has_more}


def is_facet_muted(facet: str) -> bool:
    """Check if a facet is currently muted.

    Args:
        facet: Facet name to check

    Returns:
        True if facet is muted, False if unmuted or facet doesn't exist
    """
    facets = get_facets()
    if facet not in facets:
        return False
    return bool(facets[facet].get("muted", False))


def get_active_facets(day: str) -> set[str]:
    """Return facets that had activity on a given day.

    Activity is determined by the presence of occurrence events (not anticipations)
    in the facet's events file for that day.

    Args:
        day: Day in YYYYMMDD format

    Returns:
        Set of facet names that had at least one occurrence event on that day
    """
    facets_dir = Path(get_journal()) / "facets"
    active: set[str] = set()

    if not facets_dir.exists():
        return active

    for facet_path in facets_dir.iterdir():
        if not facet_path.is_dir():
            continue

        facet_name = facet_path.name
        events_file = facet_path / "events" / f"{day}.jsonl"

        if not events_file.exists():
            continue

        # Check for at least one occurrence (occurred=true)
        try:
            with open(events_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        # Only count occurrences, not anticipations
                        if event.get("occurred", True):
                            active.add(facet_name)
                            break  # Found one, no need to check more
                    except json.JSONDecodeError:
                        continue
        except (OSError, IOError):
            continue

    return active


def set_facet_muted(facet: str, muted: bool) -> None:
    """Mute or unmute a facet by updating facet.json.

    Creates an audit log entry when the state changes.

    Args:
        facet: Facet name to modify
        muted: True to mute, False to unmute

    Raises:
        FileNotFoundError: If facet doesn't exist
    """
    facet_path = Path(get_journal()) / "facets" / facet
    if not facet_path.exists():
        raise FileNotFoundError(f"Facet '{facet}' not found at {facet_path}")

    facet_json_path = facet_path / "facet.json"
    if not facet_json_path.exists():
        raise FileNotFoundError(f"facet.json not found for facet '{facet}'")

    # Load current config
    with open(facet_json_path, "r", encoding="utf-8") as f:
        facet_data = json.load(f)

    # Check if state is actually changing
    current_state = bool(facet_data.get("muted", False))
    if current_state == muted:
        # No change needed
        return

    # Update muted field
    if muted:
        facet_data["muted"] = True
    else:
        # Remove the field when unmuting (cleaner for default case)
        facet_data.pop("muted", None)

    # Write back atomically
    import tempfile

    temp_fd, temp_path = tempfile.mkstemp(
        dir=facet_json_path.parent, suffix=".json", text=True
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(facet_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_path, facet_json_path)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise

    # Log the change
    action = "facet_mute" if muted else "facet_unmute"
    log_tool_action(
        facet=facet,
        action=action,
        params={"muted": muted},
    )


def facet_summaries(*, detailed: bool = False) -> str:
    """Generate a formatted list summary of all facets for use in agent prompts.

    Returns a markdown-formatted string with each facet as a list item including:
    - Facet name and hashtag ID
    - Description
    - Entity names (if available)
    - Activity names (if available)

    Parameters
    ----------
    detailed:
        If True, includes full entity and activity details (name: description).
        If False (default), includes only names as semicolon-separated lists.

    Returns
    -------
    str
        Formatted markdown string with all facets, entities, and activities
    """
    from think.activities import get_facet_activities
    from think.entities import load_entities

    facets = get_facets()
    if not facets:
        return "No facets found."

    lines = []
    lines.append("## Available Facets\n")

    for facet_name, facet_info in sorted(facets.items()):
        # Build facet header with name in parentheses
        title = facet_info.get("title", facet_name)
        description = facet_info.get("description", "")

        # Main list item for facet
        lines.append(f"- **{title}** (`{facet_name}`)")

        if description:
            lines.append(f"  {description}")

        # Load entities for this facet
        try:
            if detailed:
                entities = load_entities(facet_name)
                if entities:
                    # Extract principal role and filter from list
                    role_line, display_entities = _format_principal_role(entities)

                    if role_line:
                        lines.append(f"  - {role_line}")

                    if display_entities:
                        lines.append(f"  - **{title} Entities**:")
                        for entity in display_entities:
                            formatted_name = _format_entity_name_with_aka(entity)
                            desc = entity.get("description", "")

                            if desc:
                                lines.append(f"    - {formatted_name}: {desc}")
                            else:
                                lines.append(f"    - {formatted_name}")
            else:
                # Simple mode: load entities, filter principal, show names only
                entities = load_entities(facet_name)
                if entities:
                    role_line, display_entities = _format_principal_role(entities)

                    if role_line:
                        lines.append(f"  - {role_line}")

                    if display_entities:
                        # Build semicolon-separated names list
                        entity_names = "; ".join(
                            e.get("name", "") for e in display_entities
                        )
                        lines.append(f"  - **{title} Entities**: {entity_names}")
        except Exception:
            # No entities file or error loading - that's fine, skip it
            pass

        # Load activities for this facet
        try:
            activities = get_facet_activities(facet_name)
            if activities:
                if detailed:
                    lines.append(f"  - **{title} Activities**:")
                    for activity in activities:
                        lines.append(
                            f"    - {_format_activity_line(activity, bold_name=False)}"
                        )
                else:
                    # Simple mode: activity names only
                    activity_names = "; ".join(
                        a.get("name", a.get("id", "")) for a in activities
                    )
                    lines.append(f"  - **{title} Activities**: {activity_names}")
        except Exception:
            # No activities file or error loading - that's fine, skip it
            pass

        lines.append("")  # Empty line between facets

    return "\n".join(lines).strip()


def format_logs(
    entries: list[dict],
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    """Format action log JSONL entries to markdown chunks.

    This is the formatter function used by the formatters registry.
    Handles both facet-scoped logs (facets/{facet}/logs/) and journal-level
    logs (config/actions/).

    Args:
        entries: Raw JSONL entries (one action log per line)
        context: Optional context with:
            - file_path: Path to JSONL file (for extracting facet name and day)

    Returns:
        Tuple of (chunks, meta) where:
            - chunks: List of dicts with keys:
                - timestamp: int (unix ms)
                - markdown: str
                - source: dict (original log entry)
            - meta: Dict with optional "header" and "error" keys
    """
    ctx = context or {}
    file_path = ctx.get("file_path")
    meta: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []
    skipped_count = 0

    # Extract facet name and day from path
    facet_name: str | None = None
    day_str: str | None = None
    is_journal_level = False

    if file_path:
        file_path = Path(file_path)
        path_str = str(file_path)

        # Check for journal-level logs: config/actions/YYYYMMDD.jsonl
        if "config/actions" in path_str or "config\\actions" in path_str:
            is_journal_level = True
        else:
            # Extract facet name from path: facets/{facet}/logs/YYYYMMDD.jsonl
            facet_match = re.search(r"facets/([^/]+)/logs", path_str)
            if facet_match:
                facet_name = facet_match.group(1)

        # Extract day from filename
        if file_path.stem.isdigit() and len(file_path.stem) == 8:
            day_str = file_path.stem

    # Build header
    if day_str:
        formatted_day = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:8]}"
        if is_journal_level:
            meta["header"] = f"# Journal Action Log ({formatted_day})"
        elif facet_name:
            meta["header"] = f"# Action Log: {facet_name} ({formatted_day})"
        else:
            meta["header"] = f"# Action Log ({formatted_day})"
    else:
        if is_journal_level:
            meta["header"] = "# Journal Action Log"
        elif facet_name:
            meta["header"] = f"# Action Log: {facet_name}"
        else:
            meta["header"] = "# Action Log"

    # Format each log entry as a chunk
    for entry in entries:
        # Skip entries without action field
        action = entry.get("action")
        if not action:
            skipped_count += 1
            continue

        # Parse timestamp
        ts = 0
        timestamp_str = entry.get("timestamp", "")
        time_display = ""
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str)
                ts = int(dt.timestamp() * 1000)
                time_display = dt.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                pass

        # Extract fields
        source = entry.get("source", "unknown")
        actor = entry.get("actor", "unknown")
        params = entry.get("params", {})
        agent_id = entry.get("agent_id")

        # Format action name for display (e.g., "todo_add" -> "Todo Add")
        action_display = action.replace("_", " ").title()

        # Build markdown
        lines = [f"### {action_display} by {actor}", ""]

        # Metadata line
        meta_parts = [f"**Source:** {source}"]
        if time_display:
            meta_parts.append(f"**Time:** {time_display}")
        lines.append(" | ".join(meta_parts))

        # Agent link if present
        if agent_id:
            lines.append(f"**Agent:** [{agent_id}](/app/agents/{agent_id})")

        lines.append("")

        # Parameters
        if params and isinstance(params, dict):
            lines.append("**Parameters:**")
            for key, value in params.items():
                # Format value - truncate long strings
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                lines.append(f"- {key}: {value}")
            lines.append("")

        chunks.append(
            {
                "timestamp": ts,
                "markdown": "\n".join(lines),
                "source": entry,
            }
        )

    # Report skipped entries
    if skipped_count > 0:
        error_msg = f"Skipped {skipped_count} entries missing 'action' field"
        meta["error"] = error_msg
        logging.info(error_msg)

    # Indexer metadata - topic is "action" for action logs
    meta["indexer"] = {"topic": "action"}

    return chunks, meta
