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

    from think.awareness import get_onboarding
    from think.facets import get_enabled_facets
    from think.utils import get_config

    onboarding = get_onboarding()
    onboarding_status = onboarding.get("status", "")
    _agent_cfg = get_config().get("agent", {})
    agent_display_name = _agent_cfg.get("name", "sol").capitalize()

    if onboarding_status in ("observing", "ready"):
        # Path A active — use triage with observation context
        agent_name = "triage"
    elif not get_enabled_facets() and onboarding_status not in (
        "complete",
        "skipped",
    ):
        # No facets and no onboarding state — new user, show welcome
        agent_name = "onboarding"
    else:
        agent_name = "triage"

    # Build prompt with location context
    context_lines = []
    if app_name:
        context_lines.append(f"Current app: {app_name}")
    if path:
        context_lines.append(f"Current path: {path}")
    if facet:
        context_lines.append(f"Current facet: {facet}")

    # Add observation context for Path A onboarding
    if onboarding_status == "observing":
        obs_count = onboarding.get("observation_count", 0)
        context_lines.append(
            f"Onboarding: Path A observation in progress ({obs_count} observations so far). "
            f"The user chose to let {agent_display_name} observe and learn. Capture is running. "
            "If they ask what you've noticed or how it's going, check the awareness log "
            "with `sol call awareness status onboarding` and summarize progress."
        )
    elif onboarding_status == "ready":
        context_lines.append(
            "Onboarding: Path A observation complete — recommendations are ready. "
            "Suggest the user review their recommendations. Use `sol call chat redirect` "
            "to open a chat with the recommendation agent if they want to proceed."
        )
    elif onboarding_status in ("complete", "skipped"):
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

        # Add daily agent output context for post-onboarding users
        try:
            from datetime import datetime, timedelta
            from pathlib import Path

            from think.utils import get_journal

            journal = Path(get_journal())
            today = datetime.now().strftime("%Y%m%d")
            relevant_day = today
            agents_dir = journal / today / "agents"
            outputs = (
                sorted(p.stem for p in agents_dir.glob("*.md"))
                if agents_dir.is_dir()
                else []
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
            name=agent_name,
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
