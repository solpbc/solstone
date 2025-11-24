"""MCP resource handlers for todos."""

from fastmcp.resources import TextResource

from muse.mcp import mcp
from think import todo


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
