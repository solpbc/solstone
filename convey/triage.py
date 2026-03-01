# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Triage endpoint for universal chat bar queries from non-chat apps."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from convey.utils import error_response

logger = logging.getLogger(__name__)

bp = Blueprint("triage", __name__, url_prefix="/api/triage")


@bp.route("", methods=["POST"])
def triage() -> Any:
    """Accept a message from the universal chat bar and return a response.

    Expects JSON: {message, app, path, facet}
    Returns JSON: {response}
    """
    payload = request.get_json(force=True)
    message = payload.get("message", "").strip()

    if not message:
        return error_response("message is required", 400)

    app_name = payload.get("app", "")
    path = payload.get("path", "")
    facet = payload.get("facet", "")

    # Build prompt with location context
    context_lines = []
    if app_name:
        context_lines.append(f"Current app: {app_name}")
    if path:
        context_lines.append(f"Current path: {path}")
    if facet:
        context_lines.append(f"Current facet: {facet}")

    if context_lines:
        full_prompt = "\n".join(context_lines) + "\n\n" + message
    else:
        full_prompt = message

    try:
        from convey.utils import spawn_agent
        from think.cortex_client import read_agent_events, wait_for_agents

        config: dict[str, Any] = {}
        if facet:
            config["facet"] = facet

        agent_id = spawn_agent(
            prompt=full_prompt,
            name="triage",
            provider=None,
            config=config,
        )
        if agent_id is None:
            return error_response("Failed to connect to agent service", 503)

        completed, timed_out = wait_for_agents([agent_id], timeout=60)

        if agent_id in timed_out:
            return error_response("Triage request timed out", 504)

        end_state = completed.get(agent_id)
        if end_state == "error":
            return error_response("Triage agent encountered an error", 500)

        # Extract result text from finish event
        try:
            events = read_agent_events(agent_id)
            for event in reversed(events):
                if event.get("event") == "finish":
                    return jsonify(response=event.get("result", ""))
        except FileNotFoundError:
            pass

        return error_response("No response from triage agent", 500)

    except Exception:
        logger.exception("Triage request failed")
        return error_response("Failed to process triage request", 500)
