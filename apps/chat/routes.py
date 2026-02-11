# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from apps.utils import get_app_storage_path
from convey.config import get_selected_facet
from convey.utils import error_response, load_json, save_json, success_response
from think.models import generate
from think.utils import now_ms


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


def _save_chat(chat_id: str, chat_data: dict) -> None:
    """Save chat metadata, stripping the injected chat_id."""
    chats_dir = get_app_storage_path("chat", "chats")
    data = {k: v for k, v in chat_data.items() if k != "chat_id"}
    save_json(chats_dir / f"{chat_id}.json", data)


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


def _check_provider_api_key(provider: str) -> str | None:
    """Check if provider API key is set and return error message if not.

    Args:
        provider: The provider name (openai, anthropic, google)

    Returns:
        Error message if API key is not set, None if valid
    """
    from think.providers import PROVIDER_METADATA

    key_name = PROVIDER_METADATA.get(provider, {}).get("env_key", "GOOGLE_API_KEY")
    if not os.getenv(key_name):
        return f"{key_name} not set"
    return None


def generate_chat_title(message: str) -> str:
    """Generate a short title for a chat message using configured provider."""
    from think.prompts import load_prompt

    prompt = load_prompt("title", base_dir=Path(__file__).parent)
    try:
        title = generate(
            contents=message,
            context="app.chat.title",
            system_instruction=prompt.text,
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
    provider = payload.get("provider")
    continue_chat = payload.get("continue_chat")  # chat_id to continue

    if not provider:
        return error_response("provider is required", 400)

    config: dict[str, Any] = {}
    is_continuation = False

    if continue_chat:
        chat_data = _load_chat(continue_chat)
        if chat_data:
            session_id = chat_data.get("session_id")
            chat_provider = chat_data.get("provider")

            if not session_id:
                return error_response("Chat has no session to continue", 400)

            if chat_provider and chat_provider != provider:
                return error_response(
                    f"Chat uses {chat_provider}, cannot switch to {provider}",
                    400,
                )

            config["session_id"] = session_id
            is_continuation = True

    api_key_error = _check_provider_api_key(provider)
    if api_key_error:
        return error_response(api_key_error, 500)

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

        # Pass chat_id so background service can find the chat from an agent.
        # For continuations, chat_id is known before spawn. For new chats,
        # chat_id == agent_id, and find_chat_by_agent's fallback handles it.
        if is_continuation:
            config["chat_id"] = continue_chat

        # Create agent request - events will be broadcast by shared watcher
        agent_id = spawn_agent(
            prompt=full_prompt,
            name="default",
            provider=provider,
            config=config,
        )
        if agent_id is None:
            return error_response("Failed to connect to agent service", 503)

        ts = now_ms()

        if is_continuation:
            # Continuation: append agent_id and update timestamp
            chat_data = _load_chat(continue_chat)
            if chat_data:
                agent_ids = chat_data.get("agent_ids", [])
                agent_ids.append(agent_id)
                chat_data["agent_ids"] = agent_ids
                chat_data["updated_ts"] = ts
                _save_chat(continue_chat, chat_data)
            chat_id = continue_chat
        else:
            # New chat: create metadata record
            chat_id = agent_id
            title = generate_chat_title(message)
            chat_record = {
                "ts": ts,
                "facet": facet,
                "provider": provider,
                "title": title,
                "agent_ids": [agent_id],
            }
            chats_dir = get_app_storage_path("chat", "chats")
            save_json(chats_dir / f"{chat_id}.json", chat_record)

        return jsonify(chat_id=chat_id, agent_id=agent_id)
    except Exception as e:
        return error_response(str(e), 500)


@chat_bp.route("/api/chats")
def list_chats() -> Any:
    """Return all saved chat metadata, sorted by timestamp descending."""
    chats, unread_count = _load_all_chats()

    # Sort by timestamp descending (most recent first)
    chats.sort(key=lambda c: c.get("ts", 0), reverse=True)

    return jsonify(chats=chats, unread_count=unread_count)


@chat_bp.route("/api/chat/<chat_id>/events")
def chat_events(chat_id: str) -> Any:
    """Return all events from a chat's agents.

    Reads agent_ids from chat metadata, then hydrates events from all agents.
    For active chats, client should subscribe to WebSocket for real-time updates.
    """
    from think.cortex_client import (
        get_agent_end_state,
        get_agent_log_status,
        read_agent_events,
    )

    chat = _load_chat(chat_id)
    if not chat:
        return error_response(f"Chat not found: {chat_id}", 404)

    agent_ids = chat.get("agent_ids", [])

    # Hydrate events from all agents
    all_events = []
    for agent_id in agent_ids:
        try:
            events = read_agent_events(agent_id)
            all_events.extend(events)
        except FileNotFoundError:
            pass

    # Check if the last agent is complete and how it ended
    is_complete = False
    end_state = None
    can_continue = False
    if agent_ids:
        last_agent_id = agent_ids[-1]
        is_complete = get_agent_log_status(last_agent_id) == "completed"
        if is_complete:
            end_state = get_agent_end_state(last_agent_id)
            # Can continue only if session_id is captured and ended successfully
            can_continue = end_state == "finish" and bool(chat.get("session_id"))

    return jsonify(
        events=all_events,
        chat=chat,
        agent_ids=agent_ids,
        is_complete=is_complete,
        end_state=end_state,
        can_continue=can_continue,
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
        return error_response(f"Chat not found: {chat_id}", 404)

    return jsonify(chat_data)


@chat_bp.route("/api/agent/<agent_id>/chat")
def find_chat_by_agent(agent_id: str) -> Any:
    """Find the chat that contains a given agent_id.

    Reads the agent's request event to find chat_id stored in config,
    then falls back to using agent_id as chat_id (for root agents).

    Args:
        agent_id: The agent ID to search for

    Returns:
        Chat metadata JSON or 404 if not found
    """
    from think.cortex_client import read_agent_events

    # Try to find chat_id from agent's request event
    chat_id = None
    try:
        events = read_agent_events(agent_id)
        for event in events:
            if event.get("event") == "request":
                chat_id = event.get("chat_id")
                break
    except FileNotFoundError:
        pass

    # Fall back to agent_id as chat_id (root agent case)
    if not chat_id:
        chat_id = agent_id

    chat_data = _load_chat(chat_id)
    if not chat_data:
        return error_response(f"Chat not found for agent: {agent_id}", 404)

    return jsonify(chat_data)


@chat_bp.route("/api/chat/<chat_id>/session", methods=["POST"])
def set_chat_session(chat_id: str) -> Any:
    """Set the CLI session ID for a chat (called once after first agent completes).

    Args:
        chat_id: The chat ID

    Returns:
        Success status or error
    """
    payload = request.get_json(force=True)
    session_id = payload.get("session_id")

    if not session_id:
        return error_response("session_id is required", 400)

    chat_data = _load_chat(chat_id)
    if not chat_data:
        return error_response(f"Chat not found: {chat_id}", 404)

    # Only set session_id once (it never changes)
    if not chat_data.get("session_id"):
        chat_data["session_id"] = session_id
        _save_chat(chat_id, chat_data)

    return success_response()


@chat_bp.route("/api/chat/<chat_id>/read", methods=["POST"])
def mark_chat_read(chat_id: str) -> Any:
    """Mark a chat as read by updating its metadata.

    Args:
        chat_id: The chat ID

    Returns:
        Success status or error
    """
    chat_data = _load_chat(chat_id)
    if not chat_data:
        return error_response(f"Chat not found: {chat_id}", 404)

    chat_data["unread"] = False
    _save_chat(chat_id, chat_data)
    return success_response()


@chat_bp.route("/api/chat/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id: str) -> Any:
    """Delete a chat by removing its metadata file.

    Args:
        chat_id: The chat ID

    Returns:
        Success status or error
    """
    try:
        chats_dir = get_app_storage_path("chat", "chats", ensure_exists=False)
        chat_file = chats_dir / f"{chat_id}.json"

        if not chat_file.exists():
            return error_response(f"Chat not found: {chat_id}", 404)

        # Delete the chat metadata file
        chat_file.unlink()

        return success_response()

    except Exception as e:
        return error_response(str(e), 500)


@chat_bp.route("/api/chat/<chat_id>/retry", methods=["POST"])
def retry_chat(chat_id: str) -> Any:
    """Retry the last failed message in a chat.

    Reads the last agent's prompt and spawns a new agent with the same prompt,
    resuming the CLI session. Provider is locked to the chat's provider.

    Args:
        chat_id: The chat ID

    Returns:
        JSON with agent_id of the new retry attempt
    """
    from think.cortex_client import (
        get_agent_end_state,
        read_agent_events,
    )

    chat_data = _load_chat(chat_id)
    if not chat_data:
        return error_response(f"Chat not found: {chat_id}", 404)

    agent_ids = chat_data.get("agent_ids", [])
    if not agent_ids:
        return error_response("Chat has no agents", 400)

    last_agent_id = agent_ids[-1]
    end_state = get_agent_end_state(last_agent_id)

    if end_state != "error":
        return error_response(
            f"Cannot retry: last agent ended with '{end_state}'",
            400,
        )

    # Extract prompt from last agent's start event
    try:
        events = read_agent_events(last_agent_id)
    except FileNotFoundError:
        return error_response("Could not read agent events", 500)

    prompt = None
    for event in events:
        if event.get("event") == "start":
            prompt = event.get("prompt")
            break

    if not prompt:
        return error_response("Could not find original prompt to retry", 500)

    # Use chat's locked provider
    provider = chat_data.get("provider")
    if not provider:
        return error_response("Chat has no provider set", 400)

    # Validate API key
    api_key_error = _check_provider_api_key(provider)
    if api_key_error:
        return error_response(api_key_error, 500)

    try:
        from convey.utils import spawn_agent

        facet = chat_data.get("facet")

        config: dict[str, Any] = {"chat_id": chat_id}
        # Resume CLI session if available
        session_id = chat_data.get("session_id")
        if session_id:
            config["session_id"] = session_id
        if facet:
            config["facet"] = facet

        # Spawn retry agent
        agent_id = spawn_agent(
            prompt=prompt,
            name="default",
            provider=provider,
            config=config,
        )
        if agent_id is None:
            return error_response("Failed to connect to agent service", 503)

        # Append agent to chat and update timestamp
        agent_ids.append(agent_id)
        chat_data["agent_ids"] = agent_ids
        chat_data["updated_ts"] = now_ms()
        _save_chat(chat_id, chat_data)

        return jsonify(agent_id=agent_id)

    except Exception as e:
        return error_response(str(e), 500)
