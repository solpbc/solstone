from __future__ import annotations

import json
import os
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from .. import state

bp = Blueprint("chat", __name__, template_folder="../templates")


@bp.route("/chat")
def chat_page() -> str:
    from think.utils import get_agents

    agents = get_agents()
    persona_titles = {aid: a["title"] for aid, a in agents.items()}
    return render_template("chat.html", active="chat", persona_titles=persona_titles)


@bp.route("/chat/api/send", methods=["POST"])
def send_message() -> Any:
    payload = request.get_json(force=True)
    message = payload.get("message", "")
    attachments = payload.get("attachments", [])
    backend = payload.get("backend", state.chat_backend)
    persona = payload.get("persona", "default")  # Get persona from request
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
        from pathlib import Path

        from muse.cortex_client import cortex_request

        # Prepare the full prompt with attachments
        if attachments:
            full_prompt = "\n".join([message] + attachments)
        else:
            full_prompt = message

        # Create agent request - events will be broadcast by shared watcher
        agent_file = cortex_request(
            prompt=full_prompt,
            persona=persona,
            backend=backend,
            config=config,
        )

        # Extract agent_id from the filename
        agent_id = Path(agent_file).stem.replace("_active", "")

        return jsonify(agent_id=agent_id)
    except Exception as e:
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


@bp.route("/chat/api/history")
def chat_history() -> Any:
    """Return empty history since we use one-shot pattern with no persistence."""

    return jsonify(history=[])


@bp.route("/chat/api/agent/<agent_id>")
def agent_events(agent_id: str) -> Any:
    """Return events from an agent run - either from Cortex if running, or from log file."""

    if not state.journal_root:
        return jsonify({"error": "Journal root not configured"}), 500

    # First, try to check if the agent is still running in Cortex
    from muse.cortex_client import cortex_agents

    try:
        # Get list of running agents from Cortex
        agent_list = cortex_agents(limit=100, offset=0)
        if agent_list and "agents" in agent_list:
            # Check if our agent is in the running list
            for agent in agent_list["agents"]:
                if agent.get("id") == agent_id and agent.get("status") == "running":
                    # Agent is still running - watcher will automatically pick it up

                    # First check if there's a log file with historical events
                    agents_dir = os.path.join(state.journal_root, "agents")
                    agent_file = os.path.join(agents_dir, f"{agent_id}.jsonl")

                    events = []
                    history = []

                    # Read any events that have already been written to disk
                    if os.path.isfile(agent_file):
                        try:
                            with open(agent_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    event = json.loads(line.strip())
                                    if not event:
                                        continue

                                    events.append(event)

                                    # Build chat history for display (raw data only)
                                    if event.get("event") == "start":
                                        history.append(
                                            {
                                                "role": "user",
                                                "text": event.get("prompt", ""),
                                            }
                                        )
                                    elif event.get("event") == "finish":
                                        history.append(
                                            {
                                                "role": "assistant",
                                                "text": event.get("result", ""),
                                            }
                                        )
                                    elif event.get("event") == "error":
                                        error_msg = event.get("error", "Unknown error")
                                        history.append(
                                            {
                                                "role": "assistant",
                                                "text": error_msg,
                                            }
                                        )
                        except Exception:
                            pass

                    # Return with source='cortex' to indicate it's live
                    # Include the agent_id so client can subscribe to updates
                    return jsonify(
                        events=events,
                        history=history,
                        source="cortex",
                        agent_id=agent_id,
                        is_running=True,
                    )
    except Exception:
        # If Cortex connection fails, fall through to file-based loading
        pass

    # Fall back to reading from the log file
    agents_dir = os.path.join(state.journal_root, "agents")
    agent_file = os.path.join(agents_dir, f"{agent_id}.jsonl")

    if not os.path.isfile(agent_file):
        return jsonify({"error": "Agent not found"}), 404

    events = []
    history = []

    try:
        with open(agent_file, "r", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line.strip())
                if not event:
                    continue

                events.append(event)

                # Build chat history for display (raw data only)
                if event.get("event") == "start":
                    history.append({"role": "user", "text": event.get("prompt", "")})
                elif event.get("event") == "finish":
                    history.append(
                        {"role": "assistant", "text": event.get("result", "")}
                    )
                elif event.get("event") == "error":
                    error_msg = event.get("error", "Unknown error")
                    history.append({"role": "assistant", "text": error_msg})

    except Exception as e:
        return jsonify({"error": f"Failed to read agent file: {str(e)}"}), 500

    return jsonify(events=events, history=history, source="file")


@bp.route("/chat/api/clear", methods=["POST"])
def clear_history() -> Any:
    """No-op since we use one-shot pattern with no persistent state."""

    return jsonify(ok=True)
