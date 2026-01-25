# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP tools and resources for todo management.

This module provides the todo MCP tools and resource handler for the todos app.
Tools are auto-discovered and registered via the @register_tool decorator.
The resource is registered via @mcp.resource decorator.
"""

from datetime import datetime
from typing import Any

from fastmcp import Context
from fastmcp.resources import TextResource

from apps.todos import todo
from think.facets import log_tool_action
from think.mcp import HINTS, mcp, register_tool

# Declare tool pack - creates the "todo" pack with all todo tools
TOOL_PACKS = {
    "todo": ["todo_list", "todo_add", "todo_cancel", "todo_done", "todo_upcoming"],
}


# -----------------------------------------------------------------------------
# MCP Tools
# -----------------------------------------------------------------------------


@register_tool(annotations=HINTS)
def todo_list(day: str, facet: str, day_to: str | None = None) -> dict[str, Any]:
    """Return the numbered todo checklist for a day or date range in a specific facet.

    Args:
        day: Journal day in ``YYYYMMDD`` format (start of range if day_to provided).
        facet: Facet name (e.g., "personal", "work").
        day_to: Optional end day in ``YYYYMMDD`` format for range queries (inclusive).

    Returns:
        Dictionary containing the formatted ``markdown`` view with ``N:`` line
        prefixes. Cancelled items are shown with strikethrough to maintain
        sequential line numbering for ``todo_add`` operations.

        For range queries, includes ``day_to`` in response and markdown is grouped
        by day with ``### YYYYMMDD`` headers. Days with no todos are omitted.

    Examples:
        - todo_list("20250101", "work")  # Single day
        - todo_list("20250101", "work", "20250107")  # Week range
    """
    try:
        # Validate day format
        try:
            datetime.strptime(day, "%Y%m%d")
        except ValueError:
            return {
                "error": f"Invalid day format '{day}'",
                "suggestion": "use YYYYMMDD format (e.g., 20250104)",
            }

        # Single day mode (no day_to)
        if day_to is None:
            checklist = todo.TodoChecklist.load(day, facet)
            return {"day": day, "facet": facet, "markdown": checklist.display()}

        # Validate day_to format
        try:
            datetime.strptime(day_to, "%Y%m%d")
        except ValueError:
            return {
                "error": f"Invalid day_to format '{day_to}'",
                "suggestion": "use YYYYMMDD format (e.g., 20250107)",
            }

        # Validate range order
        if day > day_to:
            return {
                "error": f"day '{day}' must be before or equal to day_to '{day_to}'",
                "suggestion": f"swap the values: todo_list('{day_to}', '{facet}', '{day}')",
            }

        # Same day treated as single day (no headers)
        if day == day_to:
            checklist = todo.TodoChecklist.load(day, facet)
            return {"day": day, "facet": facet, "markdown": checklist.display()}

        # Range mode: find all todo files in range
        days_with_todos = todo.get_todo_days_in_range(facet, day, day_to)

        if not days_with_todos:
            return {
                "day": day,
                "day_to": day_to,
                "facet": facet,
                "markdown": "No todos in range.",
            }

        # Build markdown with day headers
        sections: list[str] = []
        for day_str in days_with_todos:
            checklist = todo.TodoChecklist.load(day_str, facet)
            if checklist.items:
                section = f"### {day_str}\n{checklist.display()}"
                sections.append(section)

        markdown = "\n\n".join(sections) if sections else "No todos in range."

        return {
            "day": day,
            "day_to": day_to,
            "facet": facet,
            "markdown": markdown,
        }

    except FileNotFoundError:
        return {"error": f"No todos found for facet '{facet}' on day '{day}'"}
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to list todos: {exc}"}


@register_tool(annotations=HINTS)
def todo_add(
    day: str, facet: str, line_number: int, text: str, context: Context | None = None
) -> dict[str, Any]:
    """Append a new unchecked todo entry using the next sequential line number.

    Args:
        day: The day this item is due on in ``YYYYMMDD`` format, must always be today or in the future.
        facet: Facet name (e.g., "personal", "work").
        line_number: Expected next line value; must be ``current_count + 1``.
        text: Body of the todo item. Time can be included as ``(HH:MM)`` suffix.

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

        checklist = todo.TodoChecklist.load(day, facet)
        item = checklist.add_entry(line_number, text)
        log_tool_action(
            facet=facet,
            action="todo_add",
            params={"line_number": line_number, "text": item.text},
            context=context,
            day=day,
        )
        return {"day": day, "facet": facet, "markdown": checklist.display()}
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
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to add todo: {exc}"}


@register_tool(annotations=HINTS)
def todo_cancel(
    day: str, facet: str, line_number: int, context: Context | None = None
) -> dict[str, Any]:
    """Cancel a todo entry (soft delete). The entry remains in the file but is hidden from view.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").
        line_number: 1-based index of the entry to cancel.

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """
    try:
        checklist = todo.TodoChecklist.load(day, facet)
        item = checklist.cancel_entry(line_number)
        log_tool_action(
            facet=facet,
            action="todo_cancel",
            params={"line_number": line_number, "text": item.text},
            context=context,
            day=day,
        )
        return {"day": day, "facet": facet, "markdown": checklist.display()}
    except FileNotFoundError:
        return {
            "error": f"No todos found for facet '{facet}' on day '{day}'",
            "suggestion": "verify the facet and day exist before cancelling todos",
        }
    except IndexError as exc:
        return {"error": str(exc), "suggestion": "refresh the todo list"}
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to cancel todo: {exc}"}


@register_tool(annotations=HINTS)
def todo_done(
    day: str, facet: str, line_number: int, context: Context | None = None
) -> dict[str, Any]:
    """Mark a todo entry as completed.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").
        line_number: 1-based index of the entry to mark as done.

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """
    try:
        checklist = todo.TodoChecklist.load(day, facet)
        item = checklist.mark_done(line_number)
        log_tool_action(
            facet=facet,
            action="todo_done",
            params={"line_number": line_number, "text": item.text},
            context=context,
            day=day,
        )
        return {"day": day, "facet": facet, "markdown": checklist.display()}
    except FileNotFoundError:
        return {
            "error": f"No todos found for facet '{facet}' on day '{day}'",
            "suggestion": "verify the facet and day exist before updating todos",
        }
    except IndexError as exc:
        return {"error": str(exc), "suggestion": "refresh the todo list"}
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to complete todo: {exc}"}


@register_tool(annotations=HINTS)
def todo_upcoming(limit: int = 20, facet: str | None = None) -> dict[str, Any]:
    """Return upcoming todos across future days as markdown sections.

    This tool retrieves todos from future journal days, organized by facet and date.
    Use this before adding any todo with a scope beyond today to check if
    it has already been scheduled for another upcoming day, avoiding duplicates
    and ensuring proper task organization across the timeline.

    Args:
        limit: Maximum number of todos to return (default: 20)
        facet: Optional facet filter. When None, aggregates todos from all facets.
                When specified, only returns todos for that facet.

    Returns:
        Dictionary containing:
        - limit: The limit value used for this query
        - facet: The facet filter used (or None for all facets)
        - markdown: Formatted markdown with todos grouped by facet and day, each section
                   showing "Facet Title: YYYYMMDD" and its todo items
        - error: Error message if the operation fails (only on exception)

    Examples:
        - todo_upcoming()  # Return up to 20 upcoming todos from all facets
        - todo_upcoming(limit=10)  # Return up to 10 upcoming todos from all facets
        - todo_upcoming(facet="personal")  # Return personal facet todos only
        - todo_upcoming(limit=50, facet="work")  # Return up to 50 work todos
    """
    try:
        markdown = todo.upcoming(limit=limit, facet=facet)
        return {"limit": limit, "facet": facet, "markdown": markdown}
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to load upcoming todos: {exc}"}


# -----------------------------------------------------------------------------
# MCP Resource
# -----------------------------------------------------------------------------


@mcp.resource("journal://todo/{facet}/{day}")
def get_todo(facet: str, day: str) -> TextResource:
    """Return the facet-scoped todo checklist for a specific day."""
    checklist = todo.TodoChecklist.load(day, facet)

    if not checklist.exists:
        facet_path = checklist.path.parents[1]  # facets/{facet}/todos
        if not facet_path.is_dir():
            text = f"No todos folder for facet '{facet}'."
        else:
            text = f"(No todos recorded for {day} in facet '{facet}'.)"
    else:
        text = checklist.display()

    return TextResource(
        uri=f"journal://todo/{facet}/{day}",
        name=f"Todos: {facet}/{day}",
        description=f"Checklist entries for facet '{facet}' on {day}",
        mime_type="text/plain",
        text=text,
    )
