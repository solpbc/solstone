from __future__ import annotations

import asyncio
import os
from typing import Any, List

from flask import Blueprint, jsonify, render_template, request
from google import genai
from google.genai import types

from think.mcp_tools import sunstone_toolset
from think.models import GEMINI_FLASH

bp = Blueprint("chat", __name__, template_folder="../templates")

loop = asyncio.new_event_loop()
_toolset = None


async def _get_toolset():
    global _toolset
    if _toolset is None:
        _toolset = await sunstone_toolset()
    return _toolset


def ask_gemini(prompt: str, attachments: List[str], api_key: str) -> str:
    genai.configure(api_key=api_key)
    toolset = loop.run_until_complete(_get_toolset())
    model = genai.GenerativeModel(
        model_name=GEMINI_FLASH,
        tools=[toolset],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    )
    chat = model.start_chat(history=[])
    return chat.send_message([prompt] + attachments).text


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

    answer = loop.run_in_executor(None, ask_gemini, message, attachments, api_key).result()
    return jsonify(text=answer)
