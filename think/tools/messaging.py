# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP tools for resource operations.

Note: These functions are registered as MCP tools by think/mcp.py
They can also be imported and called directly for testing or internal use.
"""

import base64


async def get_resource(uri: str) -> object:
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

    Returns:
        Base64-encoded string for binary media, or a plain string for
        text resources.
    """
    # Import here to avoid circular import at module load time
    from think.mcp import mcp

    try:
        # Use the resource manager directly - bypasses Context initialization issues
        result = await mcp._resource_manager.read_resource(uri)

        # read_resource returns str for text, bytes for binary
        if isinstance(result, bytes):
            return base64.b64encode(result).decode("ascii")
        return result
    except Exception as exc:
        return {"error": f"Failed to fetch resource: {exc}"}
