"""MCP tools for the chat app."""

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from muse.mcp import HINTS, register_tool


@register_tool(annotations=HINTS)
def send_message(body: str) -> dict[str, Any]:
    """Send a message to the user's inbox for asynchronous communication.

    This tool allows MCP agents and tools to leave messages in the user's inbox
    that can be reviewed later through the chat app interface. Messages appear as
    unread notifications and can be archived after review. Use this for:
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
        load_dotenv()
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            return {
                "error": "JOURNAL_PATH not set",
                "suggestion": "ensure JOURNAL_PATH environment variable is configured",
            }

        timestamp = int(time.time() * 1000)
        message_id = f"msg_{timestamp}"

        message = {
            "id": message_id,
            "timestamp": timestamp,
            "from": {"type": "agent", "id": "mcp_tool"},
            "body": body,
            "status": "unread",
        }

        # Create inbox directory if it doesn't exist
        inbox_dir = Path(journal) / "apps" / "chat" / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        # Write message file
        message_path = inbox_dir / f"{message_id}.json"
        with open(message_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)

        return {
            "success": True,
            "message_id": message_id,
            "message": f"Message sent successfully to inbox (ID: {message_id})",
        }
    except Exception as exc:
        return {
            "error": f"Failed to send message: {exc}",
            "suggestion": "check journal directory permissions and structure",
        }
