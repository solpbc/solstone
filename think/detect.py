"""Media file metadata detection utilities."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_LITE

DETECT_SYSTEM_PROMPT = (
    "do your best to determine the time or timestamp of when this file was created "
    "using all available information given, return your best guess as to the creation "
    'time in the json format \'{"day":"YYYYMMDD","time":"HHMMSS"}\''
)


def _extract_metadata(path: str) -> dict:
    """Return metadata for *path* using ffprobe if available."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)
    except Exception as exc:  # pragma: no cover - ffprobe optional
        return {"error": str(exc)}


def detect_media_timestamp(path: str, api_key: Optional[str] = None) -> Optional[dict]:
    """Return creation time information for *path* using Gemini."""

    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    ctime = datetime.fromtimestamp(os.path.getctime(path))
    metadata = _extract_metadata(path)

    lines = [
        f"# File information for {os.path.basename(path)}",
        "",
        f"File system creation timestamp: {ctime.isoformat()}",
        "",
        "## Metadata",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
    ]
    markdown = "\n".join(lines)

    client = genai.Client(api_key=api_key)
    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed")

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        response = client.models.generate_content(
            model=GEMINI_LITE,
            contents=[markdown],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                system_instruction=DETECT_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
    finally:
        done.set()
        t.join()

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return None
