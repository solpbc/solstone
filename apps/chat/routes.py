from __future__ import annotations

import json
import os
import time
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from apps.utils import get_app_storage_path
from convey.config import get_selected_facet
from convey.utils import load_json, save_json
from think.models import GEMINI_LITE, gemini_generate

chat_bp = Blueprint(
    "app:chat",
    __name__,
    url_prefix="/app/chat",
)


@chat_bp.route("/")
def index():
    """Chat app index - computes unread data for background template."""
    # Load unread chats for badge and submenu
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)

    unread_chats = []
    unread_count = 0

    if chats_dir.exists():
        for chat_file in chats_dir.glob("*.json"):
            chat_data = load_json(chat_file)
            if chat_data and chat_data.get("unread"):
                unread_chats.append(chat_data)
                unread_count += 1

    # Sort by timestamp desc, take top 10
    unread_chats.sort(key=lambda c: c.get("ts", 0), reverse=True)
    unread_chats = unread_chats[:10]

    return render_template(
        "app.html",
        app="chat",
        unread_chats=unread_chats,
        unread_count=unread_count,
    )


TITLE_SYSTEM_INSTRUCTION = (
    "Take the user provided text and come up with a three word title that "
    "concisely but uniquely identifies the user's request for quick reference "
    "and recall. Output only the three word title, nothing else."
)


def generate_chat_title(message: str) -> str:
    """Generate a short title for a chat message using Gemini Flash Lite."""
    try:
        title = gemini_generate(
            message,
            model=GEMINI_LITE,
            system_instruction=TITLE_SYSTEM_INSTRUCTION,
            max_output_tokens=50,
            timeout=10000,
        ).strip()
        return title if title else message.split("\n")[0][:30]
    except Exception:
        return message.split("\n")[0][:30]


@chat_bp.route("/api/send", methods=["POST"])
def send_message() -> Any:
    from convey import state

    payload = request.get_json(force=True)
    message = payload.get("message", "")
    attachments = payload.get("attachments", [])
    backend = payload.get("backend", state.chat_backend)
    continue_from = payload.get("continue_from")

    config: dict[str, Any] = {}
    if continue_from:
        config["continue_from"] = continue_from

    if backend == "openai":
        key_name = "OPENAI_API_KEY"
    elif backend == "anthropic":
        key_name = "ANTHROPIC_API_KEY"
    else:
        key_name = "GOOGLE_API_KEY"

    if not os.getenv(key_name):
        resp = jsonify({"error": f"{key_name} not set"})
        resp.status_code = 500
        return resp

    try:
        from convey.utils import spawn_agent

        # Prepare the full prompt with attachments
        if attachments:
            full_prompt = "\n".join([message] + attachments)
        else:
            full_prompt = message

        # Get selected facet for context and chat record
        facet = get_selected_facet()

        # Enhance prompt with facet focus if selected
        if facet:
            full_prompt = (
                f"Focus on the `{facet}` facet and answer this request: {full_prompt}"
            )

        # Create agent request - events will be broadcast by shared watcher
        agent_id = spawn_agent(
            prompt=full_prompt,
            persona="default",
            backend=backend,
            config=config,
        )

        # Save chat metadata to app storage
        ts = int(time.time() * 1000)

        # Generate title for new chats only (not continuations)
        if continue_agent_id:
            title = None  # Don't update title for continuations
        else:
            title = generate_chat_title(message)

        chat_record = {
            "agent_id": agent_id,
            "ts": ts,
            "facet": facet,
            "title": title,
        }
        chats_dir = get_app_storage_path("chat", "chats")
        chat_file = chats_dir / f"{agent_id}.json"
        save_json(chat_file, chat_record)

        return jsonify(agent_id=agent_id)
    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


@chat_bp.route("/api/chats")
def list_chats() -> Any:
    """Return all saved chat metadata, sorted by timestamp descending."""
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)

    chats = []
    unread_count = 0

    if chats_dir.exists():
        for chat_file in chats_dir.glob("*.json"):
            chat_data = load_json(chat_file)
            if chat_data:
                chats.append(chat_data)
                if chat_data.get("unread", False):
                    unread_count += 1

    # Sort by timestamp descending (most recent first)
    chats.sort(key=lambda c: c.get("ts", 0), reverse=True)

    return jsonify(chats=chats, unread_count=unread_count)


@chat_bp.route("/api/agent/<agent_id>")
def agent_events(agent_id: str) -> Any:
    """Return events from an agent run.

    Returns all events written to disk so far. For active agents, client
    should subscribe to WebSocket for real-time updates.
    """
    from muse.cortex_client import read_agent_events

    try:
        events = read_agent_events(agent_id)

        # Check if agent is complete (last event is finish or error)
        is_complete = False
        if events:
            last_event = events[-1]
            is_complete = last_event.get("event") in ("finish", "error")

        return jsonify(events=events, is_complete=is_complete)

    except FileNotFoundError:
        # Agent file doesn't exist yet - return empty
        return jsonify(events=[], is_complete=False)


@chat_bp.route("/api/clear", methods=["POST"])
def clear_history() -> Any:
    """No-op since we use one-shot pattern with no persistent state."""

    return jsonify(ok=True)


@chat_bp.route("/api/chat/<agent_id>/read", methods=["POST"])
def mark_chat_read(agent_id: str) -> Any:
    """Mark a chat as read by updating its metadata.

    Args:
        agent_id: The agent/chat ID

    Returns:
        Success status or error
    """
    try:
        chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
        chat_file = chats_dir / f"{agent_id}.json"

        if not chat_file.exists():
            resp = jsonify({"error": f"Chat not found: {agent_id}"})
            resp.status_code = 404
            return resp

        # Load, update, and save
        chat_data = load_json(chat_file)
        if chat_data:
            chat_data["unread"] = False
            save_json(chat_file, chat_data)

        return jsonify({"success": True})

    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


