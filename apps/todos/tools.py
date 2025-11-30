"""MCP tools and resources for todo management.

This module provides the todo MCP tools and resource handler for the todos app.
Tools are auto-discovered and registered via the @register_tool decorator.
The resource is registered via @mcp.resource decorator.
"""

from datetime import datetime
from typing import Any

from fastmcp import Context
from fastmcp.resources import TextResource

from muse.mcp import HINTS, mcp, register_tool
from think import todo
from think.facets import log_tool_action

# Declare tool pack - creates the "todo" pack with all todo tools
TOOL_PACKS = {
    "todo": ["todo_list", "todo_add", "todo_remove", "todo_done", "todo_upcoming"],
}


# -----------------------------------------------------------------------------
# MCP Tools
# -----------------------------------------------------------------------------


@register_tool(annotations=HINTS)
def todo_list(day: str, facet: str) -> dict[str, Any]:
    """Return the numbered markdown checklist for ``day``'s todos in a specific facet.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").

    Returns:
        Dictionary containing the formatted ``markdown`` view with ``N:`` line
        prefixes, or an error payload when the journal day is missing.
    """

    try:
        checklist = todo.TodoChecklist.load(day, facet)
        return {"day": day, "facet": facet, "markdown": checklist.numbered()}
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

        checklist = todo.TodoChecklist.load(day, facet)
        checklist.add_entry(line_number, text)
        log_tool_action(
            facet=facet,
            action="todo_add",
            params={"line_number": line_number, "text": text},
            context=context,
            day=day,
        )
        return {"day": day, "facet": facet, "markdown": checklist.numbered()}
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
def todo_remove(
    day: str, facet: str, line_number: int, guard: str, context: Context | None = None
) -> dict[str, Any]:
    """Delete an existing todo entry after verifying its current text.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").
        line_number: 1-based index of the entry to remove.
        guard: Full todo line (e.g., ``- [ ] Review logs``) expected on the numbered line.

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """

    try:
        checklist = todo.TodoChecklist.load(day, facet)
        checklist.remove_entry(line_number, guard)
        log_tool_action(
            facet=facet,
            action="todo_remove",
            params={"line_number": line_number, "text": guard},
            context=context,
            day=day,
        )
        return {"day": day, "facet": facet, "markdown": checklist.numbered()}
    except FileNotFoundError:
        return {
            "error": f"No todos found for facet '{facet}' on day '{day}'",
            "suggestion": "verify the facet and day exist before removing todos",
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
def todo_done(
    day: str, facet: str, line_number: int, guard: str, context: Context | None = None
) -> dict[str, Any]:
    """Mark a todo entry as completed by switching its checkbox to ``[x]``.

    Args:
        day: Journal day in ``YYYYMMDD`` format.
        facet: Facet name (e.g., "personal", "work").
        line_number: 1-based index of the entry to mark as done.
        guard: Full todo line (e.g., ``- [ ] Review logs``) expected on the numbered line.

    Returns:
        Dictionary with the updated ``markdown`` checklist including numbering,
        or an error payload if validation fails.
    """

    try:
        checklist = todo.TodoChecklist.load(day, facet)
        checklist.mark_done(line_number, guard)
        log_tool_action(
            facet=facet,
            action="todo_done",
            params={"line_number": line_number, "text": guard},
            context=context,
            day=day,
        )
        return {"day": day, "facet": facet, "markdown": checklist.numbered()}
    except FileNotFoundError:
        return {
            "error": f"No todos found for facet '{facet}' on day '{day}'",
            "suggestion": "verify the facet and day exist before updating todos",
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

    todo_path = todo.todo_file_path(day, facet)

    if not todo_path.is_file():
        facet_path = todo_path.parents[1]  # facets/{facet}/todos
        if not facet_path.is_dir():
            text = f"No todos folder for facet '{facet}'."
        else:
            text = f"(No todos recorded for {day} in facet '{facet}'.)"
    else:
        text = todo_path.read_text(encoding="utf-8")

    return TextResource(
        uri=f"journal://todo/{facet}/{day}",
        name=f"Todos: {facet}/{day}",
        description=f"Checklist entries for facet '{facet}' on {day}",
        mime_type="text/markdown",
        text=text,
    )
