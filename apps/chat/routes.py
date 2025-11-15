from __future__ import annotations

import json
import os
from typing import Any

from flask import Blueprint, jsonify, render_template, request

chat_bp = Blueprint(
    "app:chat",
    __name__,
    url_prefix="/app/chat",
)


@chat_bp.route("/api/send", methods=["POST"])
def send_message() -> Any:
    from convey import state

    payload = request.get_json(force=True)
    message = payload.get("message", "")
    attachments = payload.get("attachments", [])
    backend = payload.get("backend", state.chat_backend)
    continue_agent_id = payload.get("continue")

    config: dict[str, Any] = {}
    if continue_agent_id:
        config["continue"] = continue_agent_id

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

        # Create agent request - events will be broadcast by shared watcher
        agent_id = spawn_agent(
            prompt=full_prompt,
            persona="default",
            backend=backend,
            config=config,
        )

        return jsonify(agent_id=agent_id)
    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


@chat_bp.route("/api/history")
def chat_history() -> Any:
    """Return empty history since we use one-shot pattern with no persistence."""

    return jsonify(history=[])


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
