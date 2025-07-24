"""Transcript segmentation utilities using Gemini."""

from __future__ import annotations

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_LITE


def _load_json_prompt() -> str:
    """Load the JSON system prompt."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "detect_transcript_json.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_segment_prompt() -> str:
    """Load the system prompt for segment detection."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "detect_transcript_segment.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def number_lines(text: str) -> tuple[str, List[str]]:
    """Return text with prefixed line numbers and the original lines."""
    lines = text.splitlines()
    numbered = "\n".join(f"{idx + 1}: {line}" for idx, line in enumerate(lines))
    return numbered, lines


def parse_line_numbers(json_text: str, num_lines: int) -> List[int]:
    """Validate and return line numbers parsed from ``json_text``."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - network errors
        raise ValueError("invalid JSON") from exc

    if not isinstance(data, list) or not data:
        raise ValueError("expected non-empty list")

    line_numbers: List[int] = []
    last = 0
    for item in data:
        if not isinstance(item, int):
            raise ValueError("line numbers must be integers")
        if item <= last or item < 1 or item > num_lines:
            raise ValueError("invalid line number")
        line_numbers.append(item)
        last = item
    return line_numbers


def segments_from_lines(lines: List[str], line_numbers: List[int]) -> List[str]:
    """Return transcript segments split at ``line_numbers``."""
    segments: List[str] = []
    start = 1
    for num in line_numbers:
        segments.append("\n".join(lines[start - 1 : num - 1]).strip())  # noqa: E203
        start = num
    segments.append("\n".join(lines[start - 1 :]).strip())  # noqa: E203
    return segments


def detect_transcript_segment(text: str, api_key: Optional[str] = None) -> List[str]:
    """Return transcript segments for ``text`` using Gemini."""
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    numbered, lines = number_lines(text)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_LITE,
        contents=[numbered],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(thinking_budget=512),
            system_instruction=_load_segment_prompt(),
            response_mime_type="application/json",
        ),
    )

    line_numbers = parse_line_numbers(response.text, len(lines))
    return segments_from_lines(lines, line_numbers)


def detect_transcript_json(
    text: str, api_key: Optional[str] = None
) -> Optional[list]:
    """Return transcript ``text`` converted to JSON using Gemini."""
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_LITE,
        contents=[text],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
            system_instruction=_load_json_prompt(),
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return None


__all__ = [
    "detect_transcript_segment",
    "detect_transcript_json",
    "number_lines",
    "parse_line_numbers",
    "segments_from_lines",
]
