from __future__ import annotations

import os
from typing import Any, List

from flask import Blueprint, jsonify, render_template, request

from think.google import AgentSession as GoogleAgent
from think.openai import AgentSession as OpenAIAgent

from .. import state

bp = Blueprint("chat", __name__, template_folder="../templates")


async def get_agent(backend: str):
    """Return the cached agent for ``backend`` creating one if needed."""

    if state.chat_agent is not None and state.chat_backend == backend:
        return state.chat_agent

    if state.chat_agent is not None:
        await state.chat_agent.__aexit__(None, None, None)

    if backend == "openai":
        state.chat_agent = OpenAIAgent()
    else:
        state.chat_agent = GoogleAgent()

    await state.chat_agent.__aenter__()
    state.chat_backend = backend
    return state.chat_agent


async def ask_agent(prompt: str, attachments: List[str], backend: str) -> str:
    """Send ``prompt`` to the selected agent backend."""

    agent = await get_agent(backend)
    full_prompt = "\n".join([prompt] + attachments) if attachments else prompt
    return await agent.run(full_prompt)


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
        if not os.getenv("OPENAI_API_KEY"):
            return "", 500
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            return "", 500

    result = await ask_agent(message, attachments, backend)
    return jsonify(text=result)


@bp.route("/chat/api/history")
def chat_history() -> Any:
    """Return the full cached chat history."""

    history = []
    if state.chat_agent is not None:
        for msg in state.chat_agent.history:
            history.append({"role": msg["role"], "text": msg["content"]})
    return jsonify(history=history)


@bp.route("/chat/api/clear", methods=["POST"])
async def clear_history() -> Any:
    """Clear the cached history."""

    agent = state.chat_agent
    if agent is not None:
        await agent.__aexit__(None, None, None)
    state.chat_agent = None
    return jsonify(ok=True)
