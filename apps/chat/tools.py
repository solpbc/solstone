"""MCP tools for the chat app."""

import json
import os
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
        - success: True and agent_id if the message was sent successfully
        - error: Error message if sending failed

    Examples:
        - send_message("While analysing I found a potential security vulnerability")
        - send_message("Daily summary ready for review in facet 'work_projects'")
        - send_message("Failed to process transcript for 20240115 - file corrupted")
        - send_message("Reminder: Review the pending PRs in the dashboard")
    """
    try:
        load_dotenv()
        journal_path = os.getenv("JOURNAL_PATH")
        if not journal_path:
            return {
                "error": "JOURNAL_PATH not set",
                "suggestion": "ensure JOURNAL_PATH environment variable is configured",
            }

        # Create a synthetic agent (just the agent JSONL file)
        from muse.cortex_client import create_synthetic_agent

        agent_id = create_synthetic_agent(result=body)

        # Create chat metadata for this message
        # Extract title from first line (up to 50 chars)
        title = body.split("\n")[0][:50].strip()
        if len(body.split("\n")[0]) > 50:
            title += "..."

        chat_record = {
            "agent_id": agent_id,
            "ts": int(agent_id),  # agent_id is already the timestamp
            "from": {"type": "agent", "id": "mcp_tool"},
            "title": title,
            "unread": True,
        }

        # Save chat metadata
        chats_dir = Path(journal_path) / "apps" / "chat" / "chats"
        chats_dir.mkdir(parents=True, exist_ok=True)
        chat_file = chats_dir / f"{agent_id}.json"

        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(chat_record, f, indent=2)

        return {
            "success": True,
            "agent_id": agent_id,
            "message": f"Message sent successfully to inbox (agent_id: {agent_id})",
        }
    except Exception as exc:
        return {
            "error": f"Failed to send message: {exc}",
            "suggestion": "check journal directory permissions and structure",
        }
