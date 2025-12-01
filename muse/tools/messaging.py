"""MCP tools for resource operations.

Note: These functions are registered as MCP tools by muse/mcp.py
They can also be imported and called directly for testing or internal use.
"""

from fastmcp import Context


async def get_resource(uri: str, ctx: Context) -> object:
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

    try:
        content_list = await ctx.read_resource(uri)

        # read_resource returns a list of content items
        if not content_list:
            return {"error": f"Resource not found: {uri}"}

        content = content_list[0]

        # Check if content is binary (blob) or text
        # FastMCP returns BlobResourceContents for binary, TextResourceContents for text
        if hasattr(content, "blob") and content.blob is not None:
            # Binary content - already base64 encoded by FastMCP
            return content.blob
        elif hasattr(content, "text"):
            return content.text
        else:
            # Fallback for unexpected content types
            return str(content)
    except Exception as exc:  # pragma: no cover - unexpected failure
        return {"error": f"Failed to fetch resource: {exc}"}
