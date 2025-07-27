from __future__ import annotations

import asyncio
import os
from typing import Any, List

from flask import Blueprint, jsonify, render_template, request

from think.google import AgentSession

from .. import state

bp = Blueprint("chat", __name__, template_folder="../templates")


async def ask_gemini(prompt: str, attachments: List[str]) -> str:
    """Send ``prompt`` to Gemini using a shared :class:`AgentSession`."""

    if state.chat_agent is None:
        state.chat_agent = AgentSession()
        await state.chat_agent.__aenter__()

    full_prompt = "\n".join([prompt] + attachments) if attachments else prompt
    return await state.chat_agent.run(full_prompt)


@bp.route("/chat")
def chat_page() -> str:
    return render_template("chat.html", active="chat")


@bp.route("/chat/api/send", methods=["POST"])
def send_message() -> Any:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ("", 500)

    payload = request.get_json(force=True)
    message = payload.get("message", "")
    attachments = payload.get("attachments", [])

    result = ask_gemini(message, attachments)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
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
def clear_history() -> Any:
    """Clear the cached history."""

    agent = state.chat_agent
    if agent is not None:
        result = agent.__aexit__(None, None, None)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    state.chat_agent = None
    return jsonify(ok=True)
