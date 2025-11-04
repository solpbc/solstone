"""Facet-specific utilities and tooling for the think module."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastmcp import Context
from fastmcp.server.dependencies import get_http_headers


def _get_actor_info(context: Context | None = None) -> tuple[str, str | None]:
    """Extract actor (persona) and agent_id from meta or HTTP headers.

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
                persona = meta_dict.get("persona")
                agent_id = meta_dict.get("agent_id")
                if persona or agent_id:
                    actor = persona if persona else "mcp"
                    return actor, agent_id
        except Exception:
            pass

    # Fallback to HTTP headers (HTTP transport)
    try:
        headers = get_http_headers(include_all=True)
        # Normalize headers to lowercase for case-insensitive lookup
        headers_lower = {k.lower(): v for k, v in headers.items()}

        persona = headers_lower.get("x-agent-persona")
        agent_id = headers_lower.get("x-agent-id")

        actor = persona if persona else "mcp"
        return actor, agent_id
    except Exception:
        # Not in HTTP context (stdio, tests)
        return "mcp", None


def log_action(
    facet: str,
    day: str,
    action: str,
    params: dict[str, Any],
    context: Context | None = None,
) -> None:
    """Log an MCP tool action to the facet's daily audit log.

    Creates a JSONL log entry in facets/{facet}/logs/{day}.jsonl for tracking
    successful todo and entity modifications made via MCP tools. Automatically
    extracts actor identity from meta (stdio) or HTTP headers (HTTP transport).

    Args:
        facet: Facet name where the action occurred
        day: Day in YYYYMMDD format when the action occurred
        action: Action type (e.g., "todo_add", "entity_attach")
        params: Dictionary of action-specific parameters
        context: Optional FastMCP context for extracting meta

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    # Build log file path
    log_path = Path(journal) / "facets" / facet / "logs" / f"{day}.jsonl"

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get actor info from meta or HTTP headers
    actor, agent_id = _get_actor_info(context)

    # Create log entry
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "params": params,
    }

    # Add agent_id only if available
    if agent_id is not None:
        entry["agent_id"] = agent_id

    # Append to log file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_facets() -> dict[str, dict[str, object]]:
    """Return available facets with metadata.

    Each key is the facet name. The value contains the facet metadata
    from facet.json including title, description, and the facet path.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    facets_dir = Path(journal) / "facets"
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
                    "disabled": facet_data.get("disabled", False),
                }

                facets[facet_name] = facet_info
        except Exception as exc:  # pragma: no cover - metadata optional
            logging.debug("Error reading %s: %s", facet_json, exc)

    return facets


def facet_summary(facet: str) -> str:
    """Generate a nicely formatted markdown summary of a facet.

    Args:
        facet: The facet name to summarize

    Returns:
        Formatted markdown string with facet title, description, and entities

    Raises:
        FileNotFoundError: If the facet doesn't exist
        RuntimeError: If JOURNAL_PATH is not set
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal or journal == "":
        raise RuntimeError("JOURNAL_PATH not set")

    facet_path = Path(journal) / "facets" / facet
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

    # Load entities if available using load_entities
    from think.entities import load_entities

    entities = load_entities(facet)
    if entities:
        lines.append("## Entities")
        lines.append("")
        for entity in entities:
            entity_type = entity.get("type", "")
            name = entity.get("name", "")
            desc = entity.get("description", "")
            # Include aka values in parentheses after name
            aka_list = entity.get("aka", [])
            if isinstance(aka_list, list) and aka_list:
                aka_str = ", ".join(aka_list)
                formatted_name = f"{name} ({aka_str})"
            else:
                formatted_name = name

            if desc:
                lines.append(f"- **{entity_type}**: {formatted_name} - {desc}")
            else:
                lines.append(f"- **{entity_type}**: {formatted_name}")
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

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    news_dir = Path(journal) / "facets" / facet / "news"
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


def is_facet_disabled(facet: str) -> bool:
    """Check if a facet is currently disabled.

    Args:
        facet: Facet name to check

    Returns:
        True if facet is disabled, False if enabled or facet doesn't exist
    """
    facets = get_facets()
    if facet not in facets:
        return False
    return bool(facets[facet].get("disabled", False))


def set_facet_disabled(facet: str, disabled: bool) -> None:
    """Enable or disable a facet by updating facet.json.

    Creates an audit log entry when the state changes.

    Args:
        facet: Facet name to modify
        disabled: True to disable, False to enable

    Raises:
        FileNotFoundError: If facet doesn't exist
        RuntimeError: If JOURNAL_PATH not set
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    facet_path = Path(journal) / "facets" / facet
    if not facet_path.exists():
        raise FileNotFoundError(f"Facet '{facet}' not found at {facet_path}")

    facet_json_path = facet_path / "facet.json"
    if not facet_json_path.exists():
        raise FileNotFoundError(f"facet.json not found for facet '{facet}'")

    # Load current config
    with open(facet_json_path, "r", encoding="utf-8") as f:
        facet_data = json.load(f)

    # Check if state is actually changing
    current_state = bool(facet_data.get("disabled", False))
    if current_state == disabled:
        # No change needed
        return

    # Update disabled field
    if disabled:
        facet_data["disabled"] = True
    else:
        # Remove the field when enabling (cleaner for default case)
        facet_data.pop("disabled", None)

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
    today = datetime.now().strftime("%Y%m%d")
    action = "facet_disable" if disabled else "facet_enable"
    log_action(
        facet=facet,
        day=today,
        action=action,
        params={"disabled": disabled},
        context=None,
    )


def facet_summaries(*, detailed_entities: bool = False) -> str:
    """Generate a formatted list summary of all facets for use in agent prompts.

    Returns a markdown-formatted string with each facet as a list item including:
    - Facet name and hashtag ID
    - Description
    - Entity names (if available)

    Parameters
    ----------
    detailed_entities:
        If True, includes full entity details (name: description).
        If False (default), includes only entity names as a comma-separated list.

    Returns
    -------
    str
        Formatted markdown string with all facets and their entities

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH is not set
    """
    from think.entities import load_entities, load_entity_names

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
            if detailed_entities:
                entities = load_entities(facet_name)
                if entities:
                    lines.append(f"  - **{title} Entities**:")
                    for entity in entities:
                        name = entity.get("name", "")
                        desc = entity.get("description", "")
                        # Include aka values in parentheses after name
                        aka_list = entity.get("aka", [])
                        if isinstance(aka_list, list) and aka_list:
                            aka_str = ", ".join(aka_list)
                            formatted_name = f"{name} ({aka_str})"
                        else:
                            formatted_name = name

                        if desc:
                            lines.append(f"    - {formatted_name}: {desc}")
                        else:
                            lines.append(f"    - {formatted_name}")
            else:
                entity_names = load_entity_names(facet=facet_name)
                if entity_names:
                    lines.append(f"  - **{title} Entities**: {entity_names}")
        except Exception:
            # No entities file or error loading - that's fine, skip it
            pass

        lines.append("")  # Empty line between facets

    return "\n".join(lines).strip()
