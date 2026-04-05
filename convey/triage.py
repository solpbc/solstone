# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Triage endpoint for universal chat bar / conversation panel queries."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from convey.utils import error_response

logger = logging.getLogger(__name__)

bp = Blueprint("triage", __name__, url_prefix="/api/triage")


@bp.route("", methods=["POST"])
def triage() -> Any:
    """Accept a message from the conversation panel and spawn a triage agent.

    Expects JSON: {message, app, path, facet}
    Returns JSON: {agent_id}

    The agent runs asynchronously. The browser receives the result via
    WebSocket (cortex/finish event). For reload recovery, use GET /result/<agent_id>.

    All journals route to the unified talent.
    """
    payload = request.get_json(force=True)
    message = payload.get("message", "").strip()

    from think.awareness import ensure_sol_directory

    ensure_sol_directory()

    if not message:
        return error_response("message is required", 400)

    app_name = payload.get("app", "")
    path = payload.get("path", "")
    facet = payload.get("facet", "")
    agent_name = "unified"

    # Build prompt with location context
    context_lines = []
    if app_name:
        context_lines.append(f"Current app: {app_name}")
    if path:
        context_lines.append(f"Current path: {path}")
    if facet:
        context_lines.append(f"Current facet: {facet}")

    # Add import awareness context
    try:
        from think.awareness import get_imports

        imports = get_imports()
        if not imports.get("has_imported"):
            offer_declined = imports.get("offer_declined")
            last_nudge = imports.get("last_nudge")
            context_lines.append(
                f"Import state: no imports yet. "
                f"offer_declined={offer_declined}, last_nudge={last_nudge}. "
                "If contextually appropriate and no recent nudge, "
                "you may suggest importing once (then record with "
                "`sol call awareness imports --nudge`)."
            )
        else:
            count = imports.get("import_count", 0)
            sources = imports.get("sources_used", [])
            context_lines.append(
                f"Import state: {count} import(s) from {', '.join(sources)}. "
                "User has imported — no nudging needed. "
                "If they just returned from an import, offer another source."
            )
    except Exception:
        pass  # Don't let import context break triage

    # Add daily agent output context
    try:
        from datetime import datetime, timedelta
        from pathlib import Path

        from think.utils import get_journal

        journal = Path(get_journal())
        today = datetime.now().strftime("%Y%m%d")
        relevant_day = today
        agents_dir = journal / today / "agents"
        outputs = (
            sorted(p.stem for p in agents_dir.glob("*.md")) if agents_dir.is_dir() else []
        )
        if not outputs:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            agents_dir = journal / yesterday / "agents"
            outputs = (
                sorted(p.stem for p in agents_dir.glob("*.md"))
                if agents_dir.is_dir()
                else []
            )
            relevant_day = yesterday
        if outputs:
            names = ", ".join(outputs)
            context_lines.append(
                f"Daily analysis available: {names} (from {relevant_day}). "
                "The user can ask about any of these topics."
            )
    except Exception:
        pass  # Don't let context enrichment break triage

    # Add system health context when attention items exist
    try:
        from convey.apps import _resolve_attention
        from think.awareness import get_current

        attention = _resolve_attention(get_current())
        if attention:
            context_lines.extend(attention.context_lines)
    except Exception:
        pass  # Don't let health context break triage

    # Assemble the full prompt
    prompt_parts = []
    if context_lines:
        prompt_parts.append("\n".join(context_lines))
    prompt_parts.append(message)
    full_prompt = "\n\n".join(prompt_parts)

    try:
        from convey.utils import spawn_agent

        config: dict[str, Any] = {}
        if facet:
            config["facet"] = facet
        config["app"] = app_name
        config["path"] = path
        config["user_message"] = message

        agent_id = spawn_agent(
            prompt=full_prompt,
            name=agent_name,
            provider=None,
            config=config,
        )
        if agent_id is None:
            return error_response("Failed to connect to agent service", 503)

        return jsonify(agent_id=agent_id)

    except Exception:
        logger.exception("Triage request failed")
        return error_response("Failed to process triage request", 500)


@bp.route("/result/<agent_id>", methods=["GET"])
def triage_result(agent_id: str) -> Any:
    """Return the result of a completed triage agent.

    Returns {response, panel} if the agent has finished, 404 otherwise.
    Used for page-reload recovery when the WebSocket may have missed the finish event.
    """
    try:
        from think.cortex_client import read_agent_events

        events = read_agent_events(agent_id)
        for event in reversed(events):
            if event.get("event") == "finish":
                result = event.get("result", "")
                panel = (
                    len(result) >= 120
                    or "\n" in result
                    or len((result or "").split(". ")) > 2
                )
                return jsonify(response=result, panel=panel)
    except FileNotFoundError:
        pass
    except Exception:
        logger.debug("Failed to read triage result for %s", agent_id, exc_info=True)
    return jsonify(error="not found"), 404
