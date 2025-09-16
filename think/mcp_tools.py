#!/usr/bin/env python3
"""MCP tools for the Sunstone journal assistant."""

import base64
import os
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.resources import FileResource, TextResource

from think.cluster import cluster_range
from think.domains import domain_summary
from think.indexer import search_events as search_events_impl
from think.indexer import search_summaries as search_summaries_impl
from think.indexer import search_transcripts as search_transcripts_impl
from think.messages import send_message as send_message_impl
from think.utils import get_raw_file

# Create the MCP server
mcp = FastMCP("sunstone")

# Add annotation hints for all MCP tools
HINTS = {"readOnlyHint": True, "openWorldHint": False}

# Tool packs - logical groupings of tools
TOOL_PACKS = {
    "journal": [
        "search_summaries",
        "search_transcripts",
        "search_events",
        "get_domain",
        "send_message",
        "get_resource",
    ],
    "todo": [
        "todo_list",
        "todo_add",
        "todo_remove",
        "todo_done",
    ],
}


_TODO_ENTRY_RE = re.compile(r"^- \[( |x|X)\]\s?(.*)$")


def _todo_file_path(day: str) -> Path:
    """Return the absolute path to the todo checklist for ``day``."""

    journal = os.getenv("JOURNAL_PATH", "journal")
    return Path(journal) / day / "todos" / "today.md"


def _load_todo_entries(day: str) -> tuple[Path, list[str]]:
    """Load todo entries for ``day`` as raw markdown lines."""

    path = _todo_file_path(day)
    day_dir = path.parents[1]
    if not day_dir.is_dir():
        raise FileNotFoundError(f"day folder '{day}' not found")

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, []

    text = path.read_text(encoding="utf-8")
    entries = [line.rstrip("\n") for line in text.splitlines() if line.strip()]
    return path, entries


def _write_todo_entries(path: Path, entries: list[str]) -> None:
    """Persist todo ``entries`` back to ``path`` ensuring a newline terminator."""

    content = "\n".join(entries)
    if entries:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _format_numbered(entries: list[str]) -> str:
    """Return todo ``entries`` formatted with ``1:`` style numbering."""

    if not entries:
        return "0: (no todos)"

    return "\n".join(f"{idx}: {line}" for idx, line in enumerate(entries, start=1))


def _parse_entry(entry: str) -> tuple[bool, str]:
    """Return completion flag and body text for a todo entry."""

    match = _TODO_ENTRY_RE.match(entry)
    if not match:
        raise ValueError("entry is not a markdown checklist item")
    completed = match.group(1).lower() == "x"
    text = match.group(2)
    return completed, text


def _validate_line_number(line_number: int, max_line: int) -> None:
    """Ensure ``line_number`` is within the inclusive range ``[1, max_line]``."""

    if line_number < 1 or line_number > max_line:
        raise IndexError(
            f"line number {line_number} is out of range (1..{max_line})"
        )


@mcp.tool(annotations=HINTS)
def todo_list(day: str) -> dict[str, Any]:
    """Return the numbered markdown checklist for ``day``'s todos.

    Args:
        day: Journal day in ``YYYYMMDD`` format.

    Returns:
        Dictionary containing the formatted ``markdown`` view with ``N:`` line
        prefixes, or an error payload when the journal day is missing.
    """

    try:
        _, entries = _load_todo_entries(day)
        return {"day": day, "markdown": _format_numbered(entries)}
    except FileNotFoundError:
        return {
            "error": f"Day '{day}' not found",
            "suggestion": "verify JOURNAL_PATH and the requested day folder",
        }
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to list todos: {exc}"}


@mcp.tool(annotations=HINTS)
def todo_add(day: str, line_number: int, text: str) -> dict[str, Any]:
    """Append a new unchecked todo entry using the next sequential line number.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        line_number: Expected next line value; must be ``current_count + 1``.
        text: Body of the todo item (stored after the ``- [ ]`` prefix).

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """

    try:
        path, entries = _load_todo_entries(day)
        expected_line = len(entries) + 1
        if line_number != expected_line:
            return {
                "error": (
                    f"line number {line_number} must match the next available line"
                ),
                "suggestion": f"retry with {expected_line}",
            }

        body = text.strip()
        if not body:
            return {
                "error": "todo text cannot be empty",
                "suggestion": "provide a short description of the task",
            }

        entries.append(f"- [ ] {body}")
        _write_todo_entries(path, entries)
        return {"day": day, "markdown": _format_numbered(entries)}
    except FileNotFoundError:
        return {
            "error": f"Day '{day}' not found",
            "suggestion": "create the day directory before adding todos",
        }
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to add todo: {exc}"}


@mcp.tool(annotations=HINTS)
def todo_remove(day: str, line_number: int, guard: str) -> dict[str, Any]:
    """Delete an existing todo entry after verifying its current text.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        line_number: 1-based index of the entry to remove.
        guard: Full todo line (e.g., ``- [ ] Review logs``) expected on the numbered line.

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """

    try:
        path, entries = _load_todo_entries(day)
        _validate_line_number(line_number, len(entries))

        entry = entries[line_number - 1]
        _parse_entry(entry)
        if guard != entry:
            return {
                "error": "guard text does not match current todo",
                "suggestion": f"expected '{entry}'",
            }

        del entries[line_number - 1]
        _write_todo_entries(path, entries)
        return {"day": day, "markdown": _format_numbered(entries)}
    except FileNotFoundError:
        return {
            "error": f"Day '{day}' not found",
            "suggestion": "verify the day folder exists before removing todos",
        }
    except IndexError as exc:
        return {"error": str(exc), "suggestion": "refresh the todo list"}
    except ValueError as exc:
        return {
            "error": f"Malformed todo entry: {exc}",
            "suggestion": "recreate the todo manually if needed",
        }
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to remove todo: {exc}"}


@mcp.tool(annotations=HINTS)
def todo_done(day: str, line_number: int, guard: str) -> dict[str, Any]:
    """Mark a todo entry as completed by switching its checkbox to ``[x]``.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        line_number: 1-based index of the entry to mark as done.
        guard: Full todo line (e.g., ``- [ ] Review logs``) expected on the numbered line.

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """

    try:
        path, entries = _load_todo_entries(day)
        _validate_line_number(line_number, len(entries))

        entry = entries[line_number - 1]
        _, body = _parse_entry(entry)
        if guard != entry:
            return {
                "error": "guard text does not match current todo",
                "suggestion": f"expected '{entry}'",
            }

        entries[line_number - 1] = f"- [x] {body}"
        _write_todo_entries(path, entries)
        return {"day": day, "markdown": _format_numbered(entries)}
    except FileNotFoundError:
        return {
            "error": f"Day '{day}' not found",
            "suggestion": "verify the day folder exists before updating todos",
        }
    except IndexError as exc:
        return {"error": str(exc), "suggestion": "refresh the todo list"}
    except ValueError as exc:
        return {
            "error": f"Malformed todo entry: {exc}",
            "suggestion": "recreate the todo manually if needed",
        }
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to complete todo: {exc}"}


@mcp.tool(annotations=HINTS)
def search_summaries(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    topic: str | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Search across journal topic summaries using semantic full-text search.

    This tool searches through pre-processed topic summaries that represent
    key themes and subjects from your journal entries. Use this when looking
    for high-level concepts, themes, or when you need an overview of topics
    discussed over time.

    Args:
        query: Natural language search query (e.g., "meetings product launch")
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)
        topic: Optional topic name to filter results by
        day: Optional day to filter results by in ``YYYYMMDD`` format

    Returns:
        Dictionary containing:
        - total: Total number of matching topics
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of matching topics with day, topic, and text excerpt, ordered by text relevance

    Examples:
        - search_summaries("machine learning projects")
        - search_summaries("team retrospectives", limit=10)
        - search_summaries("planning", topic="standup")
        - search_summaries("meetings", day="20240101")
    """
    try:
        kwargs = {}
        if topic is not None:
            kwargs["topic"] = topic
        if day is not None:
            kwargs["day"] = day
        total, results = search_summaries_impl(query, limit, offset, **kwargs)

        items = []
        for r in results:
            meta = r.get("metadata", {})
            topic = meta.get("topic", "")
            items.append(
                {
                    "day": meta.get("day", ""),
                    "topic": topic,
                    "text": r.get("text", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search topics: {exc}",
            "suggestion": "try adjusting the query or ensure indexes exist",
        }


@mcp.tool(annotations=HINTS)
def search_transcripts(
    query: str,
    day: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 5,
    offset: int = 0,
) -> dict[str, Any]:
    """Search raw transcripts and screen diffs for a specific day or date range.

    This tool scans raw audio transcripts (``*_audio.json``) and screenshot
    diffs (``*_diff.json``) produced throughout the day(s). Use it when you need
    to recall exact wording, short snippets, or visual context from given dates.

    Args:
        query: Natural language search query (e.g., "error message")
        day: Optional specific day to search in ``YYYYMMDD`` format
        start_date: Optional start date for range search in ``YYYYMMDD`` format
        end_date: Optional end date for range search in ``YYYYMMDD`` format
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)

    Returns:
        Dictionary containing:
        - total: Total number of matching raw entries
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of entries with day, time, type, and text snippet

    Examples:
        - search_transcripts("error message", day="20240101")
        - search_transcripts("feature flag", start_date="20240101", end_date="20240107")
        - search_transcripts("bug fix")  # searches all days
    """
    try:
        total, results = search_transcripts_impl(
            query,
            limit=limit,
            offset=offset,
            day=day,
            start_date=start_date,
            end_date=end_date,
        )

        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "time": meta.get("time", ""),
                    "type": meta.get("type", ""),
                    "text": r.get("text", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search raw data: {exc}",
            "suggestion": "verify the day parameter or adjust the query",
        }


@mcp.tool(annotations=HINTS)
def search_events(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    day: str | None = None,
    topic: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Search structured events extracted from journal summaries.

    This tool searches JSON event data generated from your daily summaries.
    Use it to find meetings, tasks, or other notable activities. Results may
    be filtered by day, topic, or a time range.

    Args:
        query: Natural language search query (e.g., "team standup")
        limit: Optional maximum number of events to return (default: 5)
        offset: Optional number of results to skip for pagination (default: 0)
        day: Optional ``YYYYMMDD`` day to filter results
        topic: Optional topic name to filter by
        start: Optional start time to filter events starting on or after this ``HH:MM:SS`` time
        end: Optional end time to filter events ending on or before this ``HH:MM:SS`` time

    Returns:
        Dictionary with ``limit``, ``offset`` and ``results`` list containing day, topic,
        start/end times and short event summaries.
        Ordered by day and start time (most recent first).

    Examples:
        - search_events("sprint review")
        - search_events("planning", day="20240101", limit=10)
        - search_events("standup", limit=5, offset=10)
    """

    try:
        total, rows = search_events_impl(
            query,
            limit=limit,
            offset=offset,
            day=day,
            start=start,
            end=end,
            topic=topic,
        )

        items = []
        for r in rows:
            meta = r.get("metadata", {})
            occ = r.get("occurrence", {})
            items.append(
                {
                    "day": meta.get("day", ""),
                    "topic": meta.get("topic", ""),
                    "start": meta.get("start", ""),
                    "end": meta.get("end", ""),
                    "title": occ.get("title") or r.get("text", ""),
                    "summary": occ.get("summary", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search events: {exc}",
            "suggestion": "try adjusting the query or filters",
        }


@mcp.tool(annotations=HINTS)
def get_domain(domain: str) -> dict[str, Any]:
    """Get a comprehensive summary of a domain including its metadata, entities, and matters.

    This tool generates a formatted markdown summary for a specified domain in the journal.
    The summary includes the domain's title, description, tracked entities,
    and all matters organized by their status (active, archived).
    Use this when you need an overview of a domain's current state, its associated entities,
    and the matters being tracked within it.

    Args:
        domain: The domain name to retrieve the summary for

    Returns:
        Dictionary containing:
        - domain: The domain name that was queried
        - summary: Formatted markdown text with the complete domain summary including:
            - Domain title
            - Domain description
            - List of tracked entities
            - Matters grouped by status with priority indicators

    Examples:
        - get_domain("personal")
        - get_domain("work_projects")
        - get_domain("research")

    Raises:
        If the domain doesn't exist or JOURNAL_PATH is not set, returns an error dictionary
        with an error message and suggestion for resolution.
    """
    try:
        # Get the domain summary markdown
        summary_text = domain_summary(domain)
        return {"domain": domain, "summary": summary_text}
    except FileNotFoundError:
        return {
            "error": f"Domain '{domain}' not found",
            "suggestion": "verify the domain name exists or check JOURNAL_PATH is set correctly",
        }
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to get domain summary: {exc}",
            "suggestion": "check that the domain exists and has valid metadata",
        }


@mcp.tool(annotations=HINTS)
def send_message(body: str) -> dict[str, Any]:
    """Send a message to the user's inbox for asynchronous communication.

    This tool allows MCP agents and tools to leave messages in the user's inbox
    that can be reviewed later through the web interface. Messages appear as unread
    notifications and can be archived after review. Use this for:
    - Alerting about things or issues that need attention
    - Leaving reminders or follow-up items
    - Anything concerning you encounter that should be raised

    Args:
        body: The message content to send. Can be plain text or markdown formatted.
              Keep messages concise but informative. Include relevant context or
              action items if applicable.

    Returns:
        Dictionary containing either:
        - success: True and message_id if the message was sent successfully
        - error: Error message if sending failed

    Examples:
        - send_message("While analysing I found a potential security vulnerability")
        - send_message("Daily summary ready for review in domain 'work_projects'")
        - send_message("Failed to process transcript for 20240115 - file corrupted")
        - send_message("Reminder: Review the pending PRs in the dashboard")
    """
    try:
        # Send the message with MCP tool identification
        message_id = send_message_impl(body=body, from_type="agent", from_id="mcp_tool")

        return {
            "success": True,
            "message_id": message_id,
            "message": f"Message sent successfully to inbox (ID: {message_id})",
        }
    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to send message: {exc}",
            "suggestion": "check journal directory permissions and structure",
        }


@mcp.tool(annotations=HINTS)
async def get_resource(uri: str) -> object:
    """Return the contents of a journal resource.

    Many MCP clients cannot read ``journal://`` resources directly. This tool
    acts as a wrapper around the server resources so they can be fetched via a
    normal tool call.

    The following resource types are supported:

    - ``journal://summary/{day}/{topic}`` — markdown topic summaries
    - ``journal://raw/{day}/{time}/{length}`` — raw transcripts for a time range
    - ``journal://media/{day}/{name}`` — raw FLAC or PNG media files
    - ``journal://todo/{day}`` — daily TODO.md task tracking file

    Args:
        uri: Resource URI to fetch.

    Returns:
        ``Image`` or ``Audio`` objects for binary media, or a plain string for
        text resources.
    """

    try:
        resource = await mcp._resource_manager.get_resource(uri)
        data = await resource.read()

        if isinstance(data, bytes):
            # Return base64 encoded data for binary content
            return base64.b64encode(data).decode("utf-8")

        # text content
        return str(data)
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to fetch resource: {exc}"}


@mcp.resource("journal://summary/{day}/{topic}")
def get_summary(day: str, topic: str) -> TextResource:
    """Return the markdown summary for a topic."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    md_path = Path(journal) / day / "topics" / f"{topic}.md"

    if not md_path.is_file():
        text = f"Topic '{topic}' not found for day {day}"
    else:
        text = md_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://summary/{day}/{topic}",
        name=f"Summary: {topic} ({day})",
        description=f"Summary of {topic} topic from {day}",
        mime_type="text/markdown",
        text=text,
    )


@mcp.resource("journal://raw/{day}/{time}/{length}")
def get_transcripts(day: str, time: str, length: str) -> TextResource:
    """Return raw audio and screen transcripts for a specific time range.

    This resource provides raw audio and screen transcripts for a given
    time range. The data is organized into 5-minute intervals and formatted
    as markdown. Each 5 minute segment could potentially be very large if there was a lot of activity, so it is recommended to use this with a specific minimum time range.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    try:
        # Parse the length as minutes and convert to end time
        length_minutes = int(length)
        from datetime import datetime, timedelta

        # Parse start time
        start_dt = datetime.strptime(f"{day}{time}", "%Y%m%d%H%M%S")
        # Calculate end time
        end_dt = start_dt + timedelta(minutes=length_minutes)
        end_time = end_dt.strftime("%H%M%S")

        # Use cluster_range with raw screen data
        markdown_content = cluster_range(
            day=day, start=time, end=end_time, screen="raw"
        )

        return TextResource(
            uri=f"journal://raw/{day}/{time}/{length}",
            name=f"Transcripts: {day} {time} ({length}min)",
            description=f"Raw screen activity from {day} starting at {time} for {length} minutes",
            mime_type="text/markdown",
            text=markdown_content,
        )

    except Exception as e:
        error_content = f"# Error\n\nFailed to generate transcripts for {day} {time} ({length}min): {str(e)}"
        return TextResource(
            uri=f"journal://raw/{day}/{time}/{length}",
            name=f"Transcripts Error: {day} {time} ({length}min)",
            description="Error generating raw screen transcripts",
            mime_type="text/markdown",
            text=error_content,
        )


@mcp.resource("journal://media/{day}/{name}")
def get_media(day: str, name: str) -> FileResource:
    """Return a raw FLAC or PNG file referenced by a transcript.

    Parameters
    ----------
    day:
        Day folder in ``YYYYMMDD`` format.
    name:
        Transcript JSON filename such as ``HHMMSS_audio.json`` or
        ``HHMMSS_monitor_1_diff.json``.

    Returns
    -------
    FileResource
        Resource pointing to the raw media file referenced by ``name``.
    """

    rel_path, mime, _ = get_raw_file(day, name)
    journal = os.getenv("JOURNAL_PATH", "journal")
    abs_path = Path(journal) / day / rel_path
    return FileResource(
        uri=f"journal://media/{day}/{name}",
        name=f"Media: {name}",
        description=f"Raw media file from {day}",
        mime_type=mime,
        path=abs_path,
    )


@mcp.resource("journal://todo/{day}")
def get_todo(day: str) -> TextResource:
    """Return the TODO.md file for a specific day.

    This resource provides access to the daily task tracking file which contains
    two sections: "Today" for current day tasks with timestamps, and "Future"
    for upcoming tasks with target dates. Tasks can be marked as completed [x],
    uncompleted [ ], or cancelled with strikethrough formatting.

    Args:
        day: Day in YYYYMMDD format

    Returns:
        TextResource containing the TODO.md content for the specified day,
        or a message indicating the file doesn't exist.
    """
    journal = os.getenv("JOURNAL_PATH", "journal")
    todo_path = Path(journal) / day / "TODO.md"

    if not todo_path.is_file():
        # Check if the day folder exists
        day_path = Path(journal) / day
        if not day_path.is_dir():
            text = f"# TODO for {day}\n\nDay folder {day} does not exist."
        else:
            # Return empty TODO template if file doesn't exist
            text = f"# TODO for {day}\n\n## Today\n\n(No tasks yet)\n\n## Future\n\n(No future tasks)"
    else:
        text = todo_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://todo/{day}",
        name=f"TODO: {day}",
        description=f"Task tracking for {day}",
        mime_type="text/markdown",
        text=text,
    )


def get_tools(pack: str = "default") -> list[str]:
    """Get list of tool names for a given pack.

    Args:
        pack: Name of the tool pack (default: "default" which maps to "journal")

    Returns:
        List of tool names in the pack

    Raises:
        KeyError: If pack doesn't exist
    """
    # "default" is an alias for "journal"
    if pack == "default":
        pack = "journal"

    if pack not in TOOL_PACKS:
        raise KeyError(
            f"Unknown tool pack '{pack}'. Available: {list(TOOL_PACKS.keys())}"
        )
    return TOOL_PACKS[pack]


def main() -> None:
    """Run the MCP server using the requested transport."""
    import argparse

    from think.utils import setup_cli

    parser = argparse.ArgumentParser(description="Sunstone MCP Tools Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport method: stdio (default) or http",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6270,
        help="Port to bind to for HTTP transport (default: 6270)",
    )
    parser.add_argument(
        "--path", default="/mcp", help="HTTP path for MCP endpoints (default: /mcp)"
    )

    args = setup_cli(parser)

    if args.transport == "http":
        # Write URI file for service discovery
        journal = os.getenv("JOURNAL_PATH")
        if journal:
            from pathlib import Path

            uri_file = Path(journal) / "agents" / "mcp.uri"
            uri_file.parent.mkdir(parents=True, exist_ok=True)
            mcp_uri = f"http://127.0.0.1:{args.port}{args.path}"
            uri_file.write_text(mcp_uri)
            print(f"MCP Tools URI written to {uri_file}")

        mcp.run(
            transport="http",
            host="127.0.0.1",
            port=args.port,
            path=args.path,
            show_banner=False,
        )
    else:
        # default stdio transport
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
