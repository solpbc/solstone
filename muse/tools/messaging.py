"""MCP tools for resource operations.

Note: These functions are registered as MCP tools by muse/mcp.py
They can also be imported and called directly for testing or internal use.
"""

import base64
from typing import Any


async def get_resource(uri: str, mcp: Any) -> object:
    """Return the contents of a journal resource.

    Many MCP clients cannot read ``journal://`` resources directly. This tool
    acts as a wrapper around the server resources so they can be fetched via a
    normal tool call.

    The following resource types are supported:

    - ``journal://insight/{day}/{topic}`` — markdown topic insights
    - ``journal://transcripts/full/{day}/{time}/{length}`` — full transcripts (audio + raw screen)
    - ``journal://transcripts/audio/{day}/{time}/{length}`` — audio transcripts only
    - ``journal://transcripts/screen/{day}/{time}/{length}`` — screen summaries only
    - ``journal://media/{day}/{name}`` — raw FLAC or PNG media files
    - ``journal://todo/{facet}/{day}`` — facet-scoped todo checklist file

    Args:
        uri: Resource URI to fetch.
        mcp: The MCP server instance (passed from muse/mcp.py).

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
