from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from flask import Blueprint, jsonify, render_template, request
from google import genai
from google.genai import types

from think.models import GEMINI_FLASH

from .. import state


def _create_mcp_client() -> Client:
    """Return a FastMCP client for the think tools."""

    server_url = os.getenv("SUNSTONE_MCP_URL")
    if server_url:
        return Client(server_url)

    server_path = Path(__file__).resolve().parents[2] / "think" / "mcp_server.py"
    transport = PythonStdioTransport(str(server_path), env=os.environ.copy())
    return Client(transport)


bp = Blueprint("chat", __name__, template_folder="../templates")


async def ask_gemini(prompt: str, attachments: List[str], api_key: str) -> str:
    """Send ``prompt`` along with prior chat history to Gemini."""

    client = genai.Client(api_key=api_key)
    mcp_client = _create_mcp_client()

    past: List[types.Content] = [
        types.Content(
            role=("user" if m["role"] == "user" else "model"),
            parts=[types.Part(text=m["text"])],
        )
        for m in state.chat_history
    ]

    past.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    for a in attachments:
        past.append(types.Content(role="user", parts=[types.Part(text=a)]))

    async with mcp_client:
        session = mcp_client.session

        original_call_tool = session.call_tool

        async def logged_call_tool(
            name: str, arguments: Dict[str, Any] | None = None, **kwargs
        ):
            print(f"Calling MCP tool {name} with args {arguments}")
            return await original_call_tool(name=name, arguments=arguments, **kwargs)

        session.call_tool = logged_call_tool  # type: ignore[assignment]

        model = await client.aio.models.generate_content(
            model=GEMINI_FLASH,
            contents=past,
            config=types.GenerateContentConfig(
                tools=[session],
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
