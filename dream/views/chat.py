from __future__ import annotations

import json
import os
from typing import Any, List

import markdown  # type: ignore
from flask import Blueprint, jsonify, render_template, request

from think.agents import run_agent

from .. import state
from ..push import push_server


def _push_event(event: dict) -> None:
    """Forward agent events to connected chat clients."""

    push_server.push({"view": "chat", **event})


bp = Blueprint("chat", __name__, template_folder="../templates")


async def ask_agent(prompt: str, attachments: List[str], backend: str) -> str:
    """Send ``prompt`` to the selected agent backend using one-shot pattern."""

    full_prompt = "\n".join([prompt] + attachments) if attachments else prompt
    return await run_agent(
        full_prompt,
        backend=backend,
        on_event=_push_event,
    )


@bp.route("/chat")
def chat_page() -> str:
    return render_template("chat.html", active="chat")


@bp.route("/chat/api/send", methods=["POST"])
async def send_message() -> Any:
    payload = request.get_json(force=True)
    message = payload.get("message", "")
    attachments = payload.get("attachments", [])
    backend = payload.get("backend", state.chat_backend)

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

    result = await ask_agent(message, attachments, backend)
    # Render markdown to HTML server-side
    html_result = markdown.markdown(result, extensions=["extra"])
    return jsonify(text=result, html=html_result)


@bp.route("/chat/api/history")
def chat_history() -> Any:
    """Return empty history since we use one-shot pattern with no persistence."""

    return jsonify(history=[])


@bp.route("/chat/api/agent/<agent_id>")
def agent_events(agent_id: str) -> Any:
    """Return events from a historical agent run."""
    
    if not state.journal_root:
        return jsonify({"error": "Journal root not configured"}), 500
    
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
                
                # Build chat history for display
                if event.get("event") == "start":
                    history.append({"role": "user", "text": event.get("prompt", "")})
                elif event.get("event") == "finish":
                    history.append({"role": "assistant", "text": event.get("result", "")})
                    
    except Exception as e:
        return jsonify({"error": f"Failed to read agent file: {str(e)}"}), 500
    
    return jsonify(events=events, history=history)


@bp.route("/chat/api/clear", methods=["POST"])
async def clear_history() -> Any:
    """No-op since we use one-shot pattern with no persistent state."""

    return jsonify(ok=True)
