from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from apps.utils import get_app_storage_path
from convey.config import get_selected_facet
from convey.utils import load_json, save_json
from think.models import GEMINI_LITE, gemini_generate

logger = logging.getLogger(__name__)


def normalize_chat(chat_data: dict, fallback_id: str | None = None) -> dict:
    """Normalize a chat record to ensure required fields exist.

    Handles legacy records that may be missing chat_id or thread.

    Args:
        chat_data: Raw chat record dict
        fallback_id: ID to use if chat_id/agent_id missing (e.g., filename stem)

    Returns:
        Normalized chat dict with chat_id and thread guaranteed
    """
    # Ensure chat_id exists
    if "chat_id" not in chat_data:
        chat_data["chat_id"] = chat_data.get("agent_id", fallback_id)

    # Ensure thread exists
    if "thread" not in chat_data:
        legacy_id = chat_data.get("agent_id", chat_data.get("chat_id"))
        if legacy_id:
            chat_data["thread"] = [legacy_id]

    return chat_data


chat_bp = Blueprint(
    "app:chat",
    __name__,
    url_prefix="/app/chat",
)


def _load_all_chats() -> tuple[list[dict], int]:
    """Load and normalize all chat records.

    Returns:
        Tuple of (chats list, unread count). Each chat dict has chat_id normalized.
    """
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    chats = []
    unread_count = 0

    if chats_dir.exists():
        for chat_file in chats_dir.glob("*.json"):
            chat_data = load_json(chat_file)
            if chat_data:
                normalize_chat(chat_data, chat_file.stem)
                chats.append(chat_data)
                if chat_data.get("unread"):
                    unread_count += 1

    return chats, unread_count


def get_chat_background_data() -> dict:
    """Load chat data for background template (submenu, badges)."""
    all_chats, unread_count = _load_all_chats()

    # Sort: unread first, then by timestamp desc
    all_chats.sort(key=lambda c: (not c.get("unread", False), -c.get("ts", 0)))

    return {
        "recent_chats": all_chats[:10],
        "unread_count": unread_count,
    }


@chat_bp.app_context_processor
def inject_chat_data():
    """Inject chat data into all templates for background service."""
    return get_chat_background_data()


@chat_bp.route("/")
def index():
    """Chat app index - context processor provides recent_chats and unread_count."""
    return render_template("app.html")


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
            timeout_s=10,
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
    continue_chat = payload.get("continue_chat")  # chat_id to continue

    # For continuation, we need to find the last agent in the thread
    config: dict[str, Any] = {}
    chats_dir = get_app_storage_path("chat", "chats")
    existing_chat = None

    if continue_chat:
        chat_file = chats_dir / f"{continue_chat}.json"
        if chat_file.exists():
            existing_chat = load_json(chat_file)
            if existing_chat:
                normalize_chat(existing_chat, continue_chat)
                # Continue from the last agent in the thread
                last_agent = existing_chat["thread"][-1]
                config["continue_from"] = last_agent

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

        # Pass facet through config for system prompt enhancement
        if facet:
            config["facet"] = facet

        # Create agent request - events will be broadcast by shared watcher
        agent_id = spawn_agent(
            prompt=full_prompt,
            persona="default",
            backend=backend,
            config=config,
        )

        ts = int(time.time() * 1000)

        if existing_chat:
            # Continuation: append new agent to existing chat's thread
            existing_chat["thread"].append(agent_id)
            existing_chat["updated_ts"] = ts
            chat_file = chats_dir / f"{continue_chat}.json"
            save_json(chat_file, existing_chat)
            chat_id = continue_chat
        else:
            # New chat: create record with thread array
            chat_id = agent_id
            title = generate_chat_title(message)
            chat_record = {
                "chat_id": chat_id,
                "thread": [agent_id],
                "ts": ts,
                "facet": facet,
                "title": title,
            }
            chat_file = chats_dir / f"{chat_id}.json"
            save_json(chat_file, chat_record)

        return jsonify(chat_id=chat_id, agent_id=agent_id)
    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


@chat_bp.route("/api/chats")
def list_chats() -> Any:
    """Return all saved chat metadata, sorted by timestamp descending."""
    chats, unread_count = _load_all_chats()

    # Sort by timestamp descending (most recent first)
    chats.sort(key=lambda c: c.get("ts", 0), reverse=True)

    return jsonify(chats=chats, unread_count=unread_count)


@chat_bp.route("/api/chat/<chat_id>/events")
def chat_events(chat_id: str) -> Any:
    """Return all events from a chat thread.

    Loads the chat record, then hydrates events from all agents in the thread.
    For active chats, client should subscribe to WebSocket for real-time updates.
    """
    from muse.cortex_client import get_agent_status, read_agent_events

    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    chat_file = chats_dir / f"{chat_id}.json"

    if not chat_file.exists():
        resp = jsonify({"error": f"Chat not found: {chat_id}"})
        resp.status_code = 404
        return resp

    chat = load_json(chat_file)
    if not chat:
        resp = jsonify({"error": f"Failed to load chat: {chat_id}"})
        resp.status_code = 500
        return resp

    normalize_chat(chat, chat_id)
    thread = chat["thread"]

    # Hydrate events from all agents in the thread
    all_events = []
    for agent_id in thread:
        try:
            events = read_agent_events(agent_id)
            all_events.extend(events)
        except FileNotFoundError:
            # Agent file might not exist yet for very new agents
            pass

    # Check if the last agent in the thread is complete
    is_complete = False
    if thread:
        is_complete = get_agent_status(thread[-1]) == "completed"

    # Find the backend used by the last agent in the thread
    last_backend = None
    if thread:
        last_agent_id = thread[-1]
        for event in reversed(all_events):
            if event.get("agent_id") == last_agent_id and event.get("event") == "start":
                last_backend = event.get("backend")
                break

    return jsonify(
        events=all_events,
        chat=chat,
        thread=thread,
        is_complete=is_complete,
        last_backend=last_backend,
    )


@chat_bp.route("/api/chat/<chat_id>")
def get_chat(chat_id: str) -> Any:
    """Get chat metadata by chat_id.

    Args:
        chat_id: The chat ID

    Returns:
        Chat metadata JSON or 404 if not found
    """
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    chat_file = chats_dir / f"{chat_id}.json"

    if not chat_file.exists():
        resp = jsonify({"error": f"Chat not found: {chat_id}"})
        resp.status_code = 404
        return resp

    chat_data = load_json(chat_file)
    if not chat_data:
        resp = jsonify({"error": f"Failed to load chat: {chat_id}"})
        resp.status_code = 500
        return resp

    return jsonify(chat_data)


@chat_bp.route("/api/agent/<agent_id>/chat")
def find_chat_by_agent(agent_id: str) -> Any:
    """Find the chat that contains a given agent_id in its thread.

    This is used by the background service to find which chat a completion
    event belongs to, since continuation agents have different IDs than
    the chat itself.

    Args:
        agent_id: The agent ID to search for

    Returns:
        Chat metadata JSON or 404 if not found in any thread
    """
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)

    if not chats_dir.exists():
        resp = jsonify({"error": "No chats found"})
        resp.status_code = 404
        return resp

    # Search all chats for one containing this agent_id in its thread
    for chat_file in chats_dir.glob("*.json"):
        chat_data = load_json(chat_file)
        if chat_data:
            normalize_chat(chat_data, chat_file.stem)
            if agent_id in chat_data.get("thread", []):
                return jsonify(chat_data)

    resp = jsonify({"error": f"No chat found containing agent: {agent_id}"})
    resp.status_code = 404
    return resp


@chat_bp.route("/api/chat/<chat_id>/read", methods=["POST"])
def mark_chat_read(chat_id: str) -> Any:
    """Mark a chat as read by updating its metadata.

    Args:
        chat_id: The chat ID

    Returns:
        Success status or error
    """
    try:
        chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
        chat_file = chats_dir / f"{chat_id}.json"

        if not chat_file.exists():
            resp = jsonify({"error": f"Chat not found: {chat_id}"})
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


@chat_bp.route("/api/chat/<chat_id>/bookmark", methods=["POST"])
def toggle_bookmark(chat_id: str) -> Any:
    """Toggle bookmark status for a chat.

    Creates/removes a symlink in the bookmarks directory and updates
    the chat's 'bookmarked' field with a timestamp.

    Args:
        chat_id: The chat ID

    Returns:
        JSON with 'bookmarked' timestamp (or null if unbookmarked)
    """
    try:
        chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
        bookmarks_dir = get_app_storage_path("chat", "bookmarks")

        chat_file = chats_dir / f"{chat_id}.json"
        bookmark_link = bookmarks_dir / f"{chat_id}.json"

        if not chat_file.exists():
            resp = jsonify({"error": f"Chat not found: {chat_id}"})
            resp.status_code = 404
            return resp

        chat_data = load_json(chat_file)
        if not chat_data:
            resp = jsonify({"error": f"Failed to load chat: {chat_id}"})
            resp.status_code = 500
            return resp

        if chat_data.get("bookmarked"):
            # Unbookmark: remove field and symlink
            del chat_data["bookmarked"]
            if bookmark_link.is_symlink() or bookmark_link.exists():
                bookmark_link.unlink()
            bookmarked = None
            logger.info(f"Unbookmarked chat {chat_id}")
        else:
            # Bookmark: add timestamp and create symlink
            bookmarked = int(time.time() * 1000)
            chat_data["bookmarked"] = bookmarked
            # Create relative symlink
            if not bookmark_link.exists():
                bookmark_link.symlink_to(f"../chats/{chat_id}.json")
            logger.info(f"Bookmarked chat {chat_id}")

        save_json(chat_file, chat_data)
        return jsonify({"bookmarked": bookmarked})

    except Exception as e:
        logger.exception(f"Error toggling bookmark for chat {chat_id}")
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


@chat_bp.route("/api/bookmarks")
def list_bookmarks() -> Any:
    """Return all bookmarked chats, optionally filtered by facet.

    Query params:
        facet: Optional facet name to filter by

    Returns:
        JSON with 'bookmarks' list sorted by bookmarked timestamp (newest first)
    """
    bookmarks_dir = get_app_storage_path("chat", "bookmarks", ensure_exists=False)
    facet = request.args.get("facet")

    bookmarks = []

    if bookmarks_dir.exists():
        for link in bookmarks_dir.glob("*.json"):
            if link.is_symlink():
                target = link.resolve()
                if target.exists():
                    chat_data = load_json(target)
                    if chat_data:
                        # Filter by facet if specified
                        if facet is not None and chat_data.get("facet") != facet:
                            continue
                        normalize_chat(chat_data, link.stem)
                        bookmarks.append(chat_data)

    # Sort by bookmarked timestamp, newest first
    bookmarks.sort(key=lambda c: c.get("bookmarked", 0), reverse=True)

    return jsonify({"bookmarks": bookmarks})
