# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from apps.utils import get_app_storage_path
from convey.config import get_selected_facet
from convey.utils import load_json, save_json
from think.models import generate

logger = logging.getLogger(__name__)


def _load_chat(chat_id: str) -> dict | None:
    """Load a chat record by ID, injecting chat_id from filename.

    Args:
        chat_id: The chat ID (filename stem)

    Returns:
        Chat data dict with chat_id injected, or None if not found
    """
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    chat_file = chats_dir / f"{chat_id}.json"

    if not chat_file.exists():
        return None

    chat_data = load_json(chat_file)
    if chat_data:
        chat_data["chat_id"] = chat_id
    return chat_data


chat_bp = Blueprint(
    "app:chat",
    __name__,
    url_prefix="/app/chat",
)


def _load_all_chats() -> tuple[list[dict], int]:
    """Load all chat records.

    Returns:
        Tuple of (chats list, unread count). Each chat dict has chat_id injected.
    """
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    chats = []
    unread_count = 0

    if chats_dir.exists():
        for chat_file in chats_dir.glob("*.json"):
            chat_data = load_json(chat_file)
            if chat_data:
                chat_data["chat_id"] = chat_file.stem
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


def _get_backend_api_key(backend: str) -> str | None:
    """Get the API key name for a backend and check if it's set.

    Args:
        backend: The backend name (openai, anthropic, google)

    Returns:
        The API key value if set, None otherwise
    """
    key_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    key_name = key_names.get(backend, "GOOGLE_API_KEY")
    return os.getenv(key_name)


def _get_backend_key_name(backend: str) -> str:
    """Get the environment variable name for a backend's API key.

    Args:
        backend: The backend name (openai, anthropic, google)

    Returns:
        The environment variable name
    """
    key_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return key_names.get(backend, "GOOGLE_API_KEY")


def generate_chat_title(message: str) -> str:
    """Generate a short title for a chat message using configured provider."""
    try:
        title = generate(
            contents=message,
            context="app.chat.title",
            system_instruction=TITLE_SYSTEM_INSTRUCTION,
            max_output_tokens=50,
            timeout_s=10,
        ).strip()
        return title if title else message.split("\n")[0][:30]
    except Exception:
        return message.split("\n")[0][:30]


@chat_bp.route("/api/send", methods=["POST"])
def send_message() -> Any:
    payload = request.get_json(force=True)
    message = payload.get("message", "")
    attachments = payload.get("attachments", [])
    backend = payload.get("backend")
    continue_chat = payload.get("continue_chat")  # chat_id to continue

    if not backend:
        resp = jsonify({"error": "backend is required"})
        resp.status_code = 400
        return resp

    # For continuation, derive thread to find last agent
    config: dict[str, Any] = {}
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    is_continuation = False

    if continue_chat and (chats_dir / f"{continue_chat}.json").exists():
        from muse.cortex_client import get_agent_thread

        # Derive thread from chat_id (which equals first agent_id)
        try:
            thread = get_agent_thread(continue_chat)
            config["continue_from"] = thread[-1]
            is_continuation = True
        except FileNotFoundError:
            pass  # Chat exists but agent file missing - treat as new

    if not _get_backend_api_key(backend):
        key_name = _get_backend_key_name(backend)
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

        if is_continuation:
            # Continuation: update timestamp only (thread is derived from agents)
            chat_data = load_json(chats_dir / f"{continue_chat}.json")
            if chat_data:
                chat_data["updated_ts"] = ts
                save_json(chats_dir / f"{continue_chat}.json", chat_data)
            chat_id = continue_chat
        else:
            # New chat: create metadata record (no thread stored)
            chat_id = agent_id
            title = generate_chat_title(message)
            chat_record = {
                "ts": ts,
                "facet": facet,
                "title": title,
            }
            # Ensure chats directory exists for new chats
            chats_dir = get_app_storage_path("chat", "chats")
            save_json(chats_dir / f"{chat_id}.json", chat_record)

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

    Derives thread from agent files, then hydrates events from all agents.
    For active chats, client should subscribe to WebSocket for real-time updates.
    """
    from muse.cortex_client import (
        get_agent_end_state,
        get_agent_status,
        get_agent_thread,
        read_agent_events,
    )

    chat = _load_chat(chat_id)
    if not chat:
        resp = jsonify({"error": f"Chat not found: {chat_id}"})
        resp.status_code = 404
        return resp

    # Derive thread from agent files (chat_id = first agent_id)
    try:
        thread = get_agent_thread(chat_id)
    except FileNotFoundError:
        thread = [chat_id]  # Fallback for very new agents

    # Hydrate events from all agents in the thread
    all_events = []
    for agent_id in thread:
        try:
            events = read_agent_events(agent_id)
            all_events.extend(events)
        except FileNotFoundError:
            # Agent file might not exist yet for very new agents
            pass

    # Check if the last agent in the thread is complete and how it ended
    is_complete = False
    end_state = None
    can_continue = False
    if thread:
        is_complete = get_agent_status(thread[-1]) == "completed"
        if is_complete:
            end_state = get_agent_end_state(thread[-1])
            can_continue = end_state == "finish"

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
        end_state=end_state,
        can_continue=can_continue,
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
    chat_data = _load_chat(chat_id)
    if not chat_data:
        resp = jsonify({"error": f"Chat not found: {chat_id}"})
        resp.status_code = 404
        return resp

    return jsonify(chat_data)


@chat_bp.route("/api/agent/<agent_id>/chat")
def find_chat_by_agent(agent_id: str) -> Any:
    """Find the chat that contains a given agent_id in its thread.

    Uses get_agent_thread() to derive the thread from agent files, then
    looks up the chat by the root agent ID (which equals chat_id).

    Args:
        agent_id: The agent ID to search for

    Returns:
        Chat metadata JSON or 404 if not found
    """
    from muse.cortex_client import get_agent_thread

    try:
        # Derive thread from agent - first element is the root/chat_id
        thread = get_agent_thread(agent_id)
        chat_id = thread[0]
    except FileNotFoundError:
        resp = jsonify({"error": f"Agent not found: {agent_id}"})
        resp.status_code = 404
        return resp

    chat_data = _load_chat(chat_id)
    if not chat_data:
        resp = jsonify({"error": f"Chat not found for agent: {agent_id}"})
        resp.status_code = 404
        return resp

    return jsonify(chat_data)


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


@chat_bp.route("/api/chat/<chat_id>/retry", methods=["POST"])
def retry_chat(chat_id: str) -> Any:
    """Retry the last failed message in a chat.

    Reads the last agent's prompt and spawns a new agent with the same prompt,
    continuing from the errored agent. Uses the backend specified in the request.

    Args:
        chat_id: The chat ID

    Returns:
        JSON with agent_id of the new retry attempt
    """
    from muse.cortex_client import (
        get_agent_end_state,
        get_agent_thread,
        read_agent_events,
    )

    payload = request.get_json(force=True) if request.data else {}
    backend = payload.get("backend")

    if not backend:
        resp = jsonify({"error": "backend is required"})
        resp.status_code = 400
        return resp

    # Validate chat exists
    chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
    if not (chats_dir / f"{chat_id}.json").exists():
        resp = jsonify({"error": f"Chat not found: {chat_id}"})
        resp.status_code = 404
        return resp

    # Get thread and verify last agent ended in error
    try:
        thread = get_agent_thread(chat_id)
    except FileNotFoundError:
        resp = jsonify({"error": "Chat thread not found"})
        resp.status_code = 404
        return resp

    if not thread:
        resp = jsonify({"error": "Chat thread is empty"})
        resp.status_code = 404
        return resp

    last_agent_id = thread[-1]
    end_state = get_agent_end_state(last_agent_id)

    if end_state != "error":
        resp = jsonify({"error": f"Cannot retry: last agent ended with '{end_state}'"})
        resp.status_code = 400
        return resp

    # Extract prompt from last agent's start event
    try:
        events = read_agent_events(last_agent_id)
    except FileNotFoundError:
        resp = jsonify({"error": "Could not read agent events"})
        resp.status_code = 500
        return resp

    prompt = None
    for event in events:
        if event.get("event") == "start":
            prompt = event.get("prompt")
            break

    if not prompt:
        resp = jsonify({"error": "Could not find original prompt to retry"})
        resp.status_code = 500
        return resp

    # Validate API key
    if not _get_backend_api_key(backend):
        key_name = _get_backend_key_name(backend)
        resp = jsonify({"error": f"{key_name} not set"})
        resp.status_code = 500
        return resp

    try:
        from convey.utils import spawn_agent

        # Get facet from chat metadata for context
        chat_data = load_json(chats_dir / f"{chat_id}.json")
        facet = chat_data.get("facet") if chat_data else None

        config: dict[str, Any] = {"continue_from": last_agent_id}
        if facet:
            config["facet"] = facet

        # Spawn retry agent
        agent_id = spawn_agent(
            prompt=prompt,
            persona="default",
            backend=backend,
            config=config,
        )

        # Update chat timestamp
        if chat_data:
            chat_data["updated_ts"] = int(time.time() * 1000)
            save_json(chats_dir / f"{chat_id}.json", chat_data)

        return jsonify(agent_id=agent_id)

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
                        chat_data["chat_id"] = link.stem
                        bookmarks.append(chat_data)

    # Sort by bookmarked timestamp, newest first
    bookmarks.sort(key=lambda c: c.get("bookmarked", 0), reverse=True)

    return jsonify({"bookmarks": bookmarks})
