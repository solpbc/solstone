from __future__ import annotations

import os
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from google import genai
from google.genai import types

from think.models import GEMINI_FLASH

bp = Blueprint("chat", __name__, template_folder="../templates")


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
    contents = [message] + attachments

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_FLASH,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=1024),
    )
    return jsonify({"text": response.text})
