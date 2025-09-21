from __future__ import annotations

import json
import os
from typing import Any

import markdown  # type: ignore
from flask import Blueprint, jsonify, render_template, request

from .. import state
from ..cortex_utils import build_cortex_event_payload, run_agent_via_cortex
from ..push import push_server


def _push_event(event: dict) -> None:
    """Forward agent events to connected chat clients."""
    payload = build_cortex_event_payload(event, source="direct")
    push_server.push(payload)


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
        # Use the shared utility with event forwarding to push server
        result = run_agent_via_cortex(
            prompt=message,
            attachments=attachments,
            backend=backend,
            persona=persona,  # Pass the persona to the agent
            config=config,
            timeout=300,  # 5 minutes for chat
            on_event=_push_event,  # Forward events to push server
        )
        # Render markdown to HTML server-side
        html_result = markdown.markdown(result, extensions=["extra"])
        return jsonify(text=result, html=html_result)
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
    from think.cortex_client import cortex_agents

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

                                    # Add HTML rendering for finish and error events
                                    if event.get("event") == "finish":
                                        result_text = event.get("result", "")
                                        event["html"] = markdown.markdown(
                                            result_text, extensions=["extra"]
                                        )
                                    elif event.get("event") == "error":
                                        # Format error message
                                        error_msg = event.get("error", "Unknown error")
                                        trace = event.get("trace", "")
                                        error_text = (
                                            f"❌ **Error**: {error_msg}\n\n```\n{trace}\n```"
                                            if trace
                                            else f"❌ **Error**: {error_msg}"
                                        )
                                        event["html"] = markdown.markdown(
                                            error_text, extensions=["extra"]
                                        )
                                        event["result"] = error_text

                                    events.append(event)

                                    # Build chat history for display
                                    if event.get("event") == "start":
                                        history.append(
                                            {
                                                "role": "user",
                                                "text": event.get("prompt", ""),
                                            }
                                        )
                                    elif event.get("event") == "finish":
                                        result_text = event.get("result", "")
                                        html_result = markdown.markdown(
                                            result_text, extensions=["extra"]
                                        )
                                        history.append(
                                            {
                                                "role": "assistant",
                                                "text": result_text,
                                                "html": html_result,
                                            }
                                        )
                                    elif event.get("event") == "error":
                                        error_msg = event.get("error", "Unknown error")
                                        trace = event.get("trace", "")
                                        error_text = (
                                            f"❌ **Error**: {error_msg}\n\n```\n{trace}\n```"
                                            if trace
                                            else f"❌ **Error**: {error_msg}"
                                        )
                                        html_result = markdown.markdown(
                                            error_text, extensions=["extra"]
                                        )
                                        history.append(
                                            {
                                                "role": "assistant",
                                                "text": error_text,
                                                "html": html_result,
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

                # Add HTML rendering for finish and error events
                if event.get("event") == "finish":
                    result_text = event.get("result", "")
                    event["html"] = markdown.markdown(result_text, extensions=["extra"])
                elif event.get("event") == "error":
                    # Format error message
                    error_msg = event.get("error", "Unknown error")
                    trace = event.get("trace", "")
                    error_text = (
                        f"❌ **Error**: {error_msg}\n\n```\n{trace}\n```"
                        if trace
                        else f"❌ **Error**: {error_msg}"
                    )
                    event["html"] = markdown.markdown(error_text, extensions=["extra"])
                    event["result"] = error_text  # Add result field for consistency

                events.append(event)

                # Build chat history for display
                if event.get("event") == "start":
                    history.append({"role": "user", "text": event.get("prompt", "")})
                elif event.get("event") == "finish":
                    result_text = event.get("result", "")
                    # Convert markdown to HTML for proper display
                    html_result = markdown.markdown(result_text, extensions=["extra"])
                    history.append(
                        {"role": "assistant", "text": result_text, "html": html_result}
                    )
                elif event.get("event") == "error":
                    # Format error message for history
                    error_msg = event.get("error", "Unknown error")
                    trace = event.get("trace", "")
                    error_text = (
                        f"❌ **Error**: {error_msg}\n\n```\n{trace}\n```"
                        if trace
                        else f"❌ **Error**: {error_msg}"
                    )
                    html_result = markdown.markdown(error_text, extensions=["extra"])
                    history.append(
                        {"role": "assistant", "text": error_text, "html": html_result}
                    )

    except Exception as e:
        return jsonify({"error": f"Failed to read agent file: {str(e)}"}), 500

    return jsonify(events=events, history=history, source="file")


@bp.route("/chat/api/clear", methods=["POST"])
def clear_history() -> Any:
    """No-op since we use one-shot pattern with no persistent state."""

    return jsonify(ok=True)
