# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP tools for the chat app."""

import json
from pathlib import Path
from typing import Any

from fastmcp import Context

from think.facets import _get_actor_info
from think.mcp import HINTS, register_tool
from think.utils import get_journal

# Declare pack membership - add send_message to journal pack
TOOL_PACKS = {
    "journal": ["send_message"],
}


@register_tool(annotations=HINTS)
def send_message(
    body: str, facet: str | None = None, context: Context | None = None
) -> dict[str, Any]:
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
        facet: Optional facet to associate the message with (e.g., "work", "personal").
               Always provide a facet if the context implies one or if you know which
               facet is relevant. This enables proper filtering in the inbox UI.

    Returns:
        Dictionary containing either:
        - success: True and agent_id if the message was sent successfully
        - error: Error message if sending failed

    Examples:
        - send_message("Security vulnerability found in auth module", facet="work")
        - send_message("Daily summary ready for review", facet="acme_project")
        - send_message("Failed to process transcript for 20240115 - file corrupted")
        - send_message("Reminder: Review the pending PRs", facet="opensource")
    """
    try:
        journal_path = get_journal()

        # Create a synthetic agent (just the agent JSONL file)
        from think.cortex_client import create_synthetic_agent

        agent_id = create_synthetic_agent(result=body)

        # Create chat metadata for this message
        # Generate title using Gemini (same as user-initiated chats)
        from apps.chat.routes import generate_chat_title

        title = generate_chat_title(body)

        # Extract caller's agent identity from context
        actor, caller_agent_id = _get_actor_info(context)

        chat_record = {
            "ts": int(agent_id),  # agent_id is already the timestamp
            "from": {"type": "agent", "id": caller_agent_id or "mcp_tool"},
            "title": title,
            "unread": True,
            "facet": facet,
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
