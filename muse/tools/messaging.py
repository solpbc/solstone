"""MCP tools for messaging and resource operations.

Note: These functions are registered as MCP tools by muse/mcp.py
They can also be imported and called directly for testing or internal use.
"""

import base64
from typing import Any

from think.messages import send_message as send_message_impl


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
        - send_message("Daily summary ready for review in facet 'work_projects'")
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


async def get_resource(uri: str, mcp: Any) -> object:
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
