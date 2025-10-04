#!/usr/bin/env python3
"""MCP tools for the Sunstone journal assistant."""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from fastmcp import FastMCP
from fastmcp.resources import FileResource, TextResource

from think import todo
from think.cluster import cluster_range
from think.domains import domain_summary
from think.indexer import search_events as search_events_impl
from think.indexer import search_news as search_news_impl
from think.indexer import search_summaries as search_summaries_impl
from think.indexer import search_transcripts as search_transcripts_impl
from think.messages import send_message as send_message_impl
from think.utils import get_raw_file

# Create the MCP server
mcp = FastMCP("sunstone")

# Add annotation hints for all MCP tools
HINTS = {"readOnlyHint": True, "openWorldHint": False}

F = TypeVar("F", bound=Callable[..., Any])


def register_tool(*tool_args: Any, **tool_kwargs: Any) -> Callable[[F], F]:
    """Register ``func`` as an MCP tool while keeping it directly callable."""

    def decorator(func: F) -> F:
        tool_obj = mcp.tool(*tool_args, **tool_kwargs)(func)
        # Preserve FastMCP metadata so tests can call ``tool.fn`` while the
        # module keeps the plain callable available.
        setattr(func, "fn", getattr(tool_obj, "fn", func))
        return func

    return decorator


# Tool packs - logical groupings of tools
TOOL_PACKS = {
    "journal": [
        "search_summaries",
        "search_transcripts",
        "search_events",
        "search_news",
        "get_domain",
        "send_message",
        "get_resource",
    ],
    "todo": [
        "todo_list",
        "todo_add",
        "todo_remove",
        "todo_done",
        "todo_upcoming",
    ],
    "domains": [
        "domain_news",
    ],
}


@register_tool(annotations=HINTS)
def todo_list(day: str) -> dict[str, Any]:
    """Return the numbered markdown checklist for ``day``'s todos.

    Args:
        day: Journal day in ``YYYYMMDD`` format.

    Returns:
        Dictionary containing the formatted ``markdown`` view with ``N:`` line
        prefixes, or an error payload when the journal day is missing.
    """

    try:
        checklist = todo.TodoChecklist.load(day)
        return {"day": day, "markdown": checklist.numbered()}
    except FileNotFoundError:
        return {"error": f"Day '{day}' has no entries"}
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to list todos: {exc}"}


@register_tool(annotations=HINTS)
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
        # Validate that the day is not in the past
        try:
            todo_date = datetime.strptime(day, "%Y%m%d").date()
            today = datetime.now().date()
            if todo_date < today:
                today_str = today.strftime("%Y%m%d")
                return {
                    "error": f"Cannot add todo to past date {day}",
                    "suggestion": f"todos can only be added to today ({today_str}) or future days",
                }
        except ValueError:
            return {
                "error": f"Invalid day format '{day}'",
                "suggestion": "use YYYYMMDD format (e.g., 20250104)",
            }

        checklist = todo.TodoChecklist.load(day, ensure_day=True)
        checklist.add_entry(line_number, text)
        return {"day": day, "markdown": checklist.numbered()}
    except RuntimeError as exc:
        return {"error": str(exc)}
    except todo.TodoLineNumberError as exc:
        return {
            "error": str(exc),
            "suggestion": f"retry with {exc.expected}",
        }
    except todo.TodoEmptyTextError as exc:
        return {
            "error": str(exc),
            "suggestion": "provide a short description of the task",
        }
    except todo.TodoDomainError as exc:
        valid_domains_str = ", ".join(exc.valid_domains)
        return {
            "error": str(exc),
            "suggestion": f"use one of the valid domains: {valid_domains_str}",
        }
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to add todo: {exc}"}


@register_tool(annotations=HINTS)
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
        checklist = todo.TodoChecklist.load(day)
        checklist.remove_entry(line_number, guard)
        return {"day": day, "markdown": checklist.numbered()}
    except FileNotFoundError:
        return {
            "error": f"Day '{day}' not found",
            "suggestion": "verify the day folder exists before removing todos",
        }
    except todo.TodoGuardMismatchError as exc:
        return {
            "error": str(exc),
            "suggestion": f"expected '{exc.expected}'",
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


@register_tool(annotations=HINTS)
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
        checklist = todo.TodoChecklist.load(day)
        checklist.mark_done(line_number, guard)
        return {"day": day, "markdown": checklist.numbered()}
    except FileNotFoundError:
        return {
            "error": f"Day '{day}' not found",
            "suggestion": "verify the day folder exists before updating todos",
        }
    except todo.TodoGuardMismatchError as exc:
        return {
            "error": str(exc),
            "suggestion": f"expected '{exc.expected}'",
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


@register_tool(annotations=HINTS)
def todo_upcoming(limit: int = 20) -> dict[str, Any]:
    """Return upcoming todos across future days as markdown sections.

    This tool retrieves todos from future journal days, organized by date.
    Use this before adding any todo with a scope beyond today to check if
    it has already been scheduled for another upcoming day, avoiding duplicates
    and ensuring proper task organization across the timeline.

    Args:
        limit: Maximum number of todos to return (default: 20)

    Returns:
        Dictionary containing:
        - limit: The limit value used for this query
        - markdown: Formatted markdown with todos grouped by day, each section
                   showing the day's date and its todo items
        - error: Error message if the operation fails (only on exception)

    Examples:
        - todo_upcoming()  # Return up to 20 upcoming todos
        - todo_upcoming(limit=10)  # Return up to 10 upcoming todos
        - todo_upcoming(limit=50)  # Return up to 50 upcoming todos
    """

    try:
        markdown = todo.upcoming(limit=limit)
        return {"limit": limit, "markdown": markdown}
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to load upcoming todos: {exc}"}


@register_tool(annotations=HINTS)
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


@register_tool(annotations=HINTS)
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


@register_tool(annotations=HINTS)
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


@register_tool(annotations=HINTS)
def search_news(
    query: str,
    limit: int = 5,
    offset: int = 0,
    *,
    domain: str | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Search domain news content using full-text search.

    This tool searches through news markdown files stored in domain-specific
    news directories (domains/<domain>/news/YYYYMMDD.md). Use this when looking
    for news items, announcements, or domain-specific updates that have been
    captured in the journal.

    Args:
        query: Natural language search query (e.g., "product launch", "security update")
        limit: Optional maximum number of results to return (default: 5, max: 20)
        offset: Optional number of results to skip for pagination (default: 0)
        domain: Optional domain name to filter results by (e.g., "ml_research", "work")
        day: Optional day to filter results by in YYYYMMDD format

    Returns:
        Dictionary containing:
        - total: Total number of matching news items
        - limit: Current limit value used for this query
        - offset: Current offset value used for this query
        - results: List of matching news items with domain, day, and text snippet,
                  ordered by relevance

    Examples:
        - search_news("product announcement")
        - search_news("security", domain="work", limit=10)
        - search_news("ai breakthrough", day="20250118")
        - search_news("quarterly results", limit=5, offset=5)
    """
    try:
        kwargs = {}
        if domain is not None:
            kwargs["domain"] = domain
        if day is not None:
            kwargs["day"] = day

        total, results = search_news_impl(query, limit, offset, **kwargs)

        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append(
                {
                    "domain": meta.get("domain", ""),
                    "day": meta.get("day", ""),
                    "text": r.get("text", ""),
                    "path": meta.get("path", ""),
                }
            )

        return {"total": total, "limit": limit, "offset": offset, "results": items}
    except Exception as exc:
        return {
            "error": f"Failed to search news: {exc}",
            "suggestion": "try adjusting the query or ensure news indexes exist",
        }


@register_tool(annotations=HINTS)
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


@register_tool(annotations=HINTS)
def domain_news(domain: str, day: str, markdown: str | None = None) -> dict[str, Any]:
    """Read or write news for a specific domain and day.

    This tool manages domain-specific news stored in markdown files organized by date.
    When markdown content is provided, it writes/updates the news file for that day.
    When markdown is not provided, it reads and returns the existing news for that day.
    News files are stored as `domains/<domain>/news/YYYYMMDD.md`.

    Args:
        domain: The domain name to manage news for
        day: The day in YYYYMMDD format
        markdown: Optional markdown content to write. If not provided, reads existing news.
                 Should follow the format with dated header and news entries with source/time.

    Returns:
        Dictionary containing either:
        - domain, day, and news content when reading
        - domain, day, and success message when writing
        - error and suggestion if operation fails

    Examples:
        - domain_news("ml_research", "20250118")  # Read news for the day
        - domain_news("work", "20250118", "# 2025-01-18 News...")  # Write news
    """
    try:
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            raise RuntimeError("JOURNAL_PATH not set")

        journal_path = Path(journal)
        domain_path = journal_path / "domains" / domain

        # Check if domain exists
        if not domain_path.exists():
            return {
                "error": f"Domain '{domain}' not found",
                "suggestion": "Create the domain first or check the domain name",
            }

        # Ensure news directory exists
        news_dir = domain_path / "news"
        news_dir.mkdir(exist_ok=True)

        # Path to the specific day's news file
        news_file = news_dir / f"{day}.md"

        if markdown is not None:
            # Write mode - save the markdown content
            news_file.write_text(markdown, encoding="utf-8")
            return {
                "domain": domain,
                "day": day,
                "message": f"News for {day} saved successfully in domain '{domain}'",
            }
        else:
            # Read mode - return existing news or empty message
            if news_file.exists():
                news_content = news_file.read_text(encoding="utf-8")
                return {"domain": domain, "day": day, "news": news_content}
            else:
                return {
                    "domain": domain,
                    "day": day,
                    "news": None,
                    "message": f"No news recorded for {day} in domain '{domain}'",
                }

    except RuntimeError as exc:
        return {
            "error": str(exc),
            "suggestion": "ensure JOURNAL_PATH environment variable is set",
        }
    except Exception as exc:
        return {
            "error": f"Failed to process domain news: {exc}",
            "suggestion": "check domain exists and has proper permissions",
        }


@register_tool(annotations=HINTS)
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


@register_tool(annotations=HINTS)
async def get_resource(uri: str) -> object:
    """Return the contents of a journal resource.

    Many MCP clients cannot read ``journal://`` resources directly. This tool
    acts as a wrapper around the server resources so they can be fetched via a
    normal tool call.

    The following resource types are supported:

    - ``journal://summary/{day}/{topic}`` — markdown topic summaries
    - ``journal://transcripts/full/{day}/{time}/{length}`` — full transcripts (audio + raw screen)
    - ``journal://transcripts/audio/{day}/{time}/{length}`` — audio transcripts only
    - ``journal://transcripts/screen/{day}/{time}/{length}`` — screen summaries only
    - ``journal://media/{day}/{name}`` — raw FLAC or PNG media files
    - ``journal://todo/{day}`` — daily ``todos/today.md`` checklist file
    - ``journal://news/{domain}/{day}`` — domain news markdown for a specific day

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


def _get_transcript_resource(
    mode: str, day: str, time: str, length: str
) -> TextResource:
    """Shared handler for all transcript resource modes.

    Args:
        mode: Transcript mode - "full", "audio", or "screen"
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    try:
        # Parse the length as minutes and convert to end time
        length_minutes = int(length)

        # Validate maximum length to prevent context overload
        if length_minutes > 120:
            error_content = f"# Error\n\nRequested {length_minutes} minutes exceeds the maximum of 120 minutes per call to minimize context overload. Please request a shorter time range."
            return TextResource(
                uri=f"journal://transcripts/{mode}/{day}/{time}/{length}",
                name=f"Transcripts Error ({mode}): {day} {time} ({length}min)",
                description=f"Error: Requested length exceeds maximum",
                mime_type="text/markdown",
                text=error_content,
            )

        from datetime import datetime, timedelta

        # Parse start time
        start_dt = datetime.strptime(f"{day}{time}", "%Y%m%d%H%M%S")
        # Calculate end time
        end_dt = start_dt + timedelta(minutes=length_minutes)
        end_time = end_dt.strftime("%H%M%S")

        # Configure cluster_range based on mode
        if mode == "full":
            markdown_content = cluster_range(
                day=day, start=time, end=end_time, audio=True, screen="raw"
            )
            description = f"Full transcripts (audio + raw screen) from {day} starting at {time} for {length} minutes"
        elif mode == "audio":
            markdown_content = cluster_range(
                day=day, start=time, end=end_time, audio=True, screen=None
            )
            description = f"Audio transcripts only from {day} starting at {time} for {length} minutes"
        elif mode == "screen":
            markdown_content = cluster_range(
                day=day, start=time, end=end_time, audio=False, screen="summary"
            )
            description = f"Screen summaries only from {day} starting at {time} for {length} minutes"
        else:
            raise ValueError(f"Invalid transcript mode: {mode}")

        return TextResource(
            uri=f"journal://transcripts/{mode}/{day}/{time}/{length}",
            name=f"Transcripts ({mode}): {day} {time} ({length}min)",
            description=description,
            mime_type="text/markdown",
            text=markdown_content,
        )

    except Exception as e:
        error_content = f"# Error\n\nFailed to generate {mode} transcripts for {day} {time} ({length}min): {str(e)}"
        return TextResource(
            uri=f"journal://transcripts/{mode}/{day}/{time}/{length}",
            name=f"Transcripts Error ({mode}): {day} {time} ({length}min)",
            description=f"Error generating {mode} transcripts",
            mime_type="text/markdown",
            text=error_content,
        )


@mcp.resource("journal://transcripts/full/{day}/{time}/{length}")
def get_transcripts_full(day: str, time: str, length: str) -> TextResource:
    """Return full audio and raw screen transcripts for a specific time range.

    This resource provides both audio transcripts and raw screen diffs for a given
    time range. The data is organized into 5-minute intervals and formatted
    as markdown. Each 5 minute segment could potentially be very large if there was a lot of activity.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("full", day, time, length)


@mcp.resource("journal://transcripts/audio/{day}/{time}/{length}")
def get_transcripts_audio(day: str, time: str, length: str) -> TextResource:
    """Return audio transcripts only for a specific time range.

    This resource provides audio transcripts without screen data for a given
    time range. Useful when you only need verbal/audio content.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("audio", day, time, length)


@mcp.resource("journal://transcripts/screen/{day}/{time}/{length}")
def get_transcripts_screen(day: str, time: str, length: str) -> TextResource:
    """Return screen summaries only for a specific time range.

    This resource provides processed screen summaries without audio data for a given
    time range. Useful when you only need visual activity summaries.

    Args:
        day: Day in YYYYMMDD format
        time: Start time in HHMMSS format
        length: Length in minutes for the time range
    """
    return _get_transcript_resource("screen", day, time, length)


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
    """Return the ``todos/today.md`` checklist for ``day``."""

    todo_path = todo.todo_file_path(day)

    if not todo_path.is_file():
        day_path = todo_path.parents[1]
        if not day_path.is_dir():
            text = f"No journal entries for {day}."
        else:
            text = "(No todos recorded.)"
    else:
        text = todo_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://todo/{day}",
        name=f"Todos: {day}",
        description=f"Checklist entries for {day}",
        mime_type="text/markdown",
        text=text,
    )


@mcp.resource("journal://news/{domain}/{day}")
def get_news_content(domain: str, day: str) -> TextResource:
    """Return the news markdown content for a specific domain and day."""
    journal = os.getenv("JOURNAL_PATH", "journal")
    news_path = Path(journal) / "domains" / domain / "news" / f"{day}.md"

    if not news_path.is_file():
        domain_path = news_path.parents[1]
        if not domain_path.is_dir():
            text = f"Domain '{domain}' not found."
        else:
            text = f"No news recorded for {day} in domain '{domain}'."
    else:
        text = news_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://news/{domain}/{day}",
        name=f"News: {domain}/{day}",
        description=f"News content for domain '{domain}' on {day}",
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
