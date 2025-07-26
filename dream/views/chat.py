from __future__ import annotations

import asyncio
import os
from typing import Any, List

from flask import Blueprint, jsonify, render_template, request

from think.genai import AgentSession

from .. import state

bp = Blueprint("chat", __name__, template_folder="../templates")


async def ask_gemini(prompt: str, attachments: List[str], api_key: str) -> str:
    """Send ``prompt`` along with prior chat history to Gemini."""

    async with AgentSession() as agent:
        for m in state.chat_history:
            role = "user" if m["role"] == "user" else "model"
            agent.add_history(role, m["text"])

        full_prompt = prompt
        if attachments:
            full_prompt = "\n".join([prompt] + attachments)

        text = await agent.run(full_prompt)

    state.chat_history.append({"role": "user", "text": prompt})
    state.chat_history.append({"role": "bot", "text": text})
    return text


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

    result = ask_gemini(message, attachments, api_key)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return jsonify(text=result)


@bp.route("/chat/api/history")
def chat_history() -> Any:
    """Return the full cached chat history."""

    return jsonify(history=state.chat_history)


@bp.route("/chat/api/clear", methods=["POST"])
def clear_history() -> Any:
    """Clear the cached history."""

    state.chat_history.clear()
    return jsonify(ok=True)
