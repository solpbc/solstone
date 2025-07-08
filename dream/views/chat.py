from __future__ import annotations

import asyncio
import os
from typing import Any, List

from flask import Blueprint, jsonify, render_template, request
from google import genai
from google.genai import types

from think.mcp_tools import get_sunstone_client
from think.models import GEMINI_FLASH

bp = Blueprint("chat", __name__, template_folder="../templates")


async def ask_gemini(prompt: str, attachments: List[str], api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    mcp_client = get_sunstone_client()

    async with mcp_client:
        model = await client.aio.models.generate_content(
            model=GEMINI_FLASH,
            contents=[prompt] + attachments,
            config=types.GenerateContentConfig(
                tools=[mcp_client.session],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
            ),
        )
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
