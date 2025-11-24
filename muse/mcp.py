#!/usr/bin/env python3
"""MCP server for Sunstone journal assistant.

This module creates the FastMCP server instance and registers all tools and resources
from the muse/tools/ and muse/resources/ directories.

Tool modules are located in muse/tools/ and contain plain functions.
Resource handlers are located in muse/resources/ and use @mcp.resource decorators.
"""

from typing import Any, Callable, TypeVar

from fastmcp import FastMCP

# Create the MCP server instance
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
        "search_insights",
        "search_transcripts",
        "search_events",
        "get_facet",
        "get_resource",
    ],
    "todo": [
        "todo_list",
        "todo_add",
        "todo_remove",
        "todo_done",
        "todo_upcoming",
    ],
    "facets": [
        "facet_news",
    ],
    "entities": [
        "entity_list",
        "entity_detect",
        "entity_attach",
        "entity_update",
        "entity_add_aka",
    ],
}


# Import and register tool modules
# These imports trigger the registration of tools via the @register_tool decorator
from muse.tools import entities, facets, search, todo

# Register todo tools
todo_list = register_tool(annotations=HINTS)(todo.todo_list)
todo_add = register_tool(annotations=HINTS)(todo.todo_add)
todo_remove = register_tool(annotations=HINTS)(todo.todo_remove)
todo_done = register_tool(annotations=HINTS)(todo.todo_done)
todo_upcoming = register_tool(annotations=HINTS)(todo.todo_upcoming)

# Register search tools
search_insights = register_tool(annotations=HINTS)(search.search_insights)
search_transcripts = register_tool(annotations=HINTS)(search.search_transcripts)
search_events = register_tool(annotations=HINTS)(search.search_events)

# Register entity tools
entity_list = register_tool(annotations=HINTS)(entities.entity_list)
entity_detect = register_tool(annotations=HINTS)(entities.entity_detect)
entity_attach = register_tool(annotations=HINTS)(entities.entity_attach)
entity_update = register_tool(annotations=HINTS)(entities.entity_update)
entity_add_aka = register_tool(annotations=HINTS)(entities.entity_add_aka)

# Register facet tools
get_facet = register_tool(annotations=HINTS)(facets.get_facet)
facet_news = register_tool(annotations=HINTS)(facets.facet_news)

# Register resource tool (get_resource moved from messaging)
from muse.tools.messaging import get_resource as get_resource_impl

get_resource = register_tool(annotations=HINTS)(get_resource_impl)

# Import resource modules - these self-register via @mcp.resource decorators
from muse.resources import insights, media, todos, transcripts  # noqa: F401


# Phase 2: App-level tool discovery
def _discover_app_tools():
    """Discover and load tools from apps/*/tools.py.

    This function scans the apps/ directory for tools.py files and
    dynamically registers any tools found there. This allows individual
    apps to contribute their own MCP tools without modifying core code.

    Apps can define tools using the @register_tool decorator:

        # apps/myapp/tools.py
        from muse.mcp import register_tool, HINTS

        @register_tool(annotations=HINTS)
        def my_tool(arg: str) -> dict:
            return {"result": "..."}

    Tools are registered when the module is imported. If an app's tools.py
    file fails to import, the error is logged but doesn't prevent other
    apps from loading or the server from starting.
    """
    import importlib
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    apps_dir = Path(__file__).parent.parent / "apps"

    if not apps_dir.exists():
        logger.debug("No apps/ directory found, skipping app tool discovery")
        return

    discovered_count = 0

    for app_dir in sorted(apps_dir.iterdir()):
        # Skip non-directories and private directories
        if not app_dir.is_dir() or app_dir.name.startswith("_"):
            continue

        tools_file = app_dir / "tools.py"
        if not tools_file.exists():
            continue

        app_name = app_dir.name

        try:
            # Import triggers @register_tool decorators
            module_name = f"apps.{app_name}.tools"
            importlib.import_module(module_name)
            discovered_count += 1
            logger.info(f"Loaded MCP tools from app: {app_name}")
        except Exception as e:
            # Gracefully handle errors - don't break server startup
            logger.error(
                f"Failed to load tools from app '{app_name}': {e}", exc_info=True
            )

    if discovered_count > 0:
        logger.info(f"Discovered tools from {discovered_count} app(s)")


_discover_app_tools()


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
