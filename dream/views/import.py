from __future__ import annotations

import json
import os
import tempfile
from typing import Any, List

from flask import Blueprint, jsonify, render_template, request
from google import genai
from google.genai import types
from werkzeug.utils import secure_filename

from think.models import GEMINI_FLASH

bp = Blueprint("import_view", __name__, template_folder="../templates")


@bp.route("/import")
def import_page() -> str:
    return render_template("import.html", active="import")


@bp.route("/import/api/process", methods=["POST"])
def import_process() -> Any:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ("", 500)

    parts: List[str] = []

    upload = request.files.get("file")
    if upload and upload.filename:
        tmp_dir = tempfile.mkdtemp(prefix="upload_")
        path = os.path.join(tmp_dir, secure_filename(upload.filename))
        upload.save(path)
        meta = f"File name: {upload.filename}; path: {path}; size: {os.path.getsize(path)}"
        parts.append(meta)

    text = request.form.get("text", "").strip()
    if text:
        parts.append(text)

    if not parts:
        return jsonify({"error": "No input provided"}), 400

    client = genai.Client(api_key=api_key)
    prompt = (
        "Return JSON object with fields day (YYYYMMDD), start (HHMMSS) and title "
        "derived from the provided inputs."
    )
    response = client.models.generate_content(
        model=GEMINI_FLASH,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=256,
            response_mime_type="application/json",
            system_instruction=prompt,
        ),
    )

    try:
        result = json.loads(response.text)
    except Exception:
        return jsonify({"error": "Failed to parse response"}), 500

    return jsonify(result)
