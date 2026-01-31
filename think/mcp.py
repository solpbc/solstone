#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP server for solstone journal assistant.

This module creates the FastMCP server instance and registers all tools and resources
from the think/tools/ and think/resources/ directories.

Tool modules are located in think/tools/ and contain plain functions.
Resource handlers are located in think/resources/ and use @mcp.resource decorators.
"""

from typing import Any, Callable, TypeVar

from fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP("solstone")

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
# Apps can extend existing packs or create new ones via TOOL_PACKS in their tools.py
TOOL_PACKS: dict[str, list[str]] = {
    "journal": [
        "search_journal",
        "get_events",
        "get_facet",
        "get_resource",
    ],
    "facets": [
        "facet_news",
    ],
    "apps": [],  # Auto-populated with all app-discovered tools
}


# Import and register tool modules
# These imports trigger the registration of tools via the @register_tool decorator
from think.tools import facets, search

# Register search tools
search_journal = register_tool(annotations=HINTS)(search.search_journal)
get_events = register_tool(annotations=HINTS)(search.get_events)

# Register facet tools
get_facet = register_tool(annotations=HINTS)(facets.get_facet)
facet_news = register_tool(annotations=HINTS)(facets.facet_news)

# Register resource tool (get_resource moved from messaging)
from think.tools.messaging import get_resource as get_resource_impl

get_resource = register_tool(annotations=HINTS)(get_resource_impl)

# Import resource modules - these self-register via @mcp.resource decorators
from think.resources import media, outputs, transcripts  # noqa: F401


# Phase 2: App-level tool discovery
def _discover_app_tools():
    """Discover and load tools from apps/*/tools.py.

    This function scans the apps/ directory for tools.py files and
    dynamically registers any tools found there. This allows individual
    apps to contribute their own MCP tools without modifying core code.

    Apps can define tools using the @register_tool decorator:

        # apps/myapp/tools.py
        from think.mcp import register_tool, HINTS

        @register_tool(annotations=HINTS)
        def my_tool(arg: str) -> dict:
            return {"result": "..."}

    Apps can also declare pack membership via a module-level TOOL_PACKS dict:

        # apps/myapp/tools.py
        TOOL_PACKS = {
            "journal": ["my_tool"],      # Add to existing pack
            "myapp": ["my_tool"],         # Create new pack
        }

    All app tools are automatically added to the "apps" pack. Tools are
    registered when the module is imported. If an app's tools.py file
    fails to import, the error is logged but doesn't prevent other apps
    from loading or the server from starting.
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
    total_tools = 0

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
            module = importlib.import_module(module_name)

            # Collect tool names (functions decorated with @register_tool have .fn)
            app_tools = [
                name
                for name, obj in vars(module).items()
                if callable(obj) and hasattr(obj, "fn") and not name.startswith("_")
            ]

            # Add all app tools to the "apps" pack
            TOOL_PACKS["apps"].extend(app_tools)
            total_tools += len(app_tools)

            # Merge app-declared pack memberships
            if hasattr(module, "TOOL_PACKS"):
                for pack, tools in module.TOOL_PACKS.items():
                    if pack not in TOOL_PACKS:
                        TOOL_PACKS[pack] = []
                    TOOL_PACKS[pack].extend(tools)
                    logger.debug(f"App '{app_name}' added {tools} to pack '{pack}'")

            discovered_count += 1
            logger.info(f"Loaded {len(app_tools)} MCP tool(s) from app: {app_name}")
        except Exception as e:
            # Gracefully handle errors - don't break server startup
            logger.error(
                f"Failed to load tools from app '{app_name}': {e}", exc_info=True
            )

    if discovered_count > 0:
        logger.info(f"Discovered {total_tools} tool(s) from {discovered_count} app(s)")


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

    parser = argparse.ArgumentParser(description="solstone MCP Tools Server")
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
        help="Port to bind to for HTTP transport (default: 6270; cortex uses dynamic port)",
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
