"""Helpers for interacting with the Gemini API."""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from google.genai import types

USER_PROMPT = (
    "Process the provided audio clips and output your professional accurate "
    "transcription in the specified JSON format, each clip may contain one or more speakers."
)


def transcribe_segments(
    client,
    model: str,
    prompt_text: str,
    entities_text: str,
    segments: List[Dict[str, object]],
) -> dict:
    """Send audio segments to Gemini and return the parsed JSON result."""

    contents = [entities_text, USER_PROMPT]
    for seg in segments:
        contents.append(
            f"This clip starts at {seg['start']} and the source is '{seg['source']}':"
        )
        contents.append(
            types.Part.from_bytes(data=seg["bytes"], mime_type="audio/flac")
        )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192 * 2,
            response_mime_type="application/json",
            system_instruction=prompt_text,
        ),
    )
    result = json.loads(response.text)
    logging.info("Transcription result: %s", json.dumps(result, indent=2))
    return result
