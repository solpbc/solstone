"""Media file metadata detection utilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_LITE


def _load_system_prompt() -> str:
    """Load the system prompt from detect_created.txt file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "detect_created.txt")
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


def _debug_write_content(content: str, path: str) -> None:
    """Write content to a debug file in /tmp for diagnosis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_debug_{timestamp}_{os.path.basename(path)}.md"
    debug_path = os.path.join("/tmp", filename)

    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Debug: Content written to {debug_path}", file=sys.stderr)


def detect_created(
    path: str, api_key: Optional[str] = None, original_filename: Optional[str] = None
) -> Optional[dict]:
    """Return creation time information for *path* using Gemini.

    Parameters
    ----------
    path : str
        Path to the file to analyze
    api_key : Optional[str]
        Google API key for Gemini
    original_filename : Optional[str]
        Original filename if path is a temporary file
    """

    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    metadata = _extract_metadata(path)

    # Use original filename in header if provided, otherwise use the actual path
    display_path = original_filename if original_filename else path

    lines = [
        f"# exiftool -all output for {display_path}",
        "",
    ]

    # If we have an original filename and it's different from path, add a note
    if original_filename and original_filename != path:
        lines.extend(
            [
                f"Original filename: {original_filename}",
                f"(Analyzing temporary file: {path})",
                "",
            ]
        )

    lines.append(metadata)
    markdown = "\n".join(lines)

    # Debug: write content to temp file
    _debug_write_content(markdown, path)

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


def main():
    """Main CLI entry point for detect_created utility."""
    import argparse

    from .utils import setup_cli

    parser = argparse.ArgumentParser(
        description="Detect creation time information from media file metadata using Gemini"
    )
    parser.add_argument("file_path", help="Path to the media file to analyze")

    args = setup_cli(parser)

    result = detect_created(args.file_path)
    if result is not None:
        print(json.dumps(result, indent=2))
    else:
        print("Failed to detect creation time information", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
