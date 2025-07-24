"""Media file metadata detection utilities."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_LITE


def _load_system_prompt() -> str:
    """Load the system prompt from created_detect.txt file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "created_detect.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _extract_metadata(path: str) -> str:
    """Return metadata for *path* using exiftool if available."""
    cmd = [
        "exiftool",
        "-all",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return proc.stdout
    except Exception as exc:  # pragma: no cover - exiftool optional
        return f"Error extracting metadata: {exc}"


def detect_creation_time(path: str, api_key: Optional[str] = None) -> Optional[dict]:
    """Return creation time information for *path* using Gemini."""

    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    metadata = _extract_metadata(path)

    lines = [
        f"# exiftool -all output for {path}",
        "",
        metadata,
    ]
    markdown = "\n".join(lines)

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_LITE,
        contents=[markdown],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=256 + 4096,
            thinking_config=types.ThinkingConfig(
                thinking_budget=4096,
            ),
            system_instruction=_load_system_prompt(),
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return None
