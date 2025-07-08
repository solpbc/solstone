from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from flask import Blueprint, jsonify, render_template, request
from google import genai
from google.genai import types

from think.mcp_tools import get_sunstone_client
from think.models import GEMINI_FLASH

from .. import state

bp = Blueprint("chat", __name__, template_folder="../templates")


async def ask_gemini(prompt: str, attachments: List[str], api_key: str) -> str:
    """Send ``prompt`` along with prior chat history to Gemini."""

    client = genai.Client(api_key=api_key)
    mcp_client = get_sunstone_client()

    past: List[types.Content] = [
        types.Content(role=("user" if m["role"] == "user" else "model"), parts=[m["text"]])
        for m in state.chat_history
    ]

    past.append(types.Content(role="user", parts=[prompt]))
    for a in attachments:
        past.append(types.Content(role="user", parts=[a]))

    async with mcp_client:
        model = await client.aio.models.generate_content(
            model=GEMINI_FLASH,
            contents=past,
            config=types.GenerateContentConfig(
                tools=[mcp_client.session],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
            ),
        )

    state.chat_history.append({"role": "user", "text": prompt})
    state.chat_history.append({"role": "bot", "text": model.text})
    return model.text


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
