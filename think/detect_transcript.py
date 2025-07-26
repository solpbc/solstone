"""Transcript segmentation utilities using Gemini."""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_FLASH, GEMINI_PRO


def _load_json_prompt() -> str:
    """Load the JSON system prompt."""
    prompt_path = os.path.join(os.path.dirname(__file__), "detect_transcript_json.txt")
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
        logging.error("Failed to parse JSON response")
        raise ValueError("invalid JSON") from exc

    if not isinstance(data, list) or not data:
        logging.error("JSON response is not a non-empty list")
        raise ValueError("expected non-empty list")

    line_numbers: List[int] = []
    last = 0
    for item in data:
        if not isinstance(item, int):
            logging.error(f"Invalid line number type: {type(item)}")
            raise ValueError("line numbers must be integers")
        if item <= last or item < 1 or item > num_lines:
            logging.error(
                f"Invalid line number: {item} (last: {last}, max: {num_lines})"
            )
            raise ValueError("invalid line number")
        line_numbers.append(item)
        last = item

    logging.info(
        f"Successfully parsed {len(line_numbers)} segment boundaries: {line_numbers}"
    )
    return line_numbers


def segments_from_lines(lines: List[str], line_numbers: List[int]) -> List[str]:
    """Return transcript segments split at ``line_numbers``."""
    segments: List[str] = []
    segment_start = 1

    for segment_boundary in line_numbers:
        if segment_boundary == 1:
            continue
        # Create segment from current start up to (but not including) the boundary
        segment_lines = lines[segment_start - 1 : segment_boundary - 1]  # noqa: E203
        segments.append("\n".join(segment_lines).strip())
        segment_start = segment_boundary

    # Add final segment from last boundary to end
    final_segment_lines = lines[segment_start - 1 :]  # noqa: E203
    segments.append("\n".join(final_segment_lines).strip())

    logging.info(f"Created {len(segments)} transcript segments")
    return segments


def detect_transcript_segment(text: str, api_key: Optional[str] = None) -> List[str]:
    """Return transcript segments for ``text`` using Gemini."""

    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY not found in environment")
            raise RuntimeError("GOOGLE_API_KEY not set")

    numbered, lines = number_lines(text)
    logging.info(
        f"Starting transcript segmentation with Gemini for: {numbered[:100]}..."
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_PRO,
        contents=[numbered],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096 + 8192,
            thinking_config=types.ThinkingConfig(thinking_budget=8192),
            system_instruction=_load_segment_prompt(),
            response_mime_type="application/json",
        ),
    )

    logging.info(f"Received response from Gemini: {response.text}")
    line_numbers = parse_line_numbers(response.text, len(lines))
    segments = segments_from_lines(lines, line_numbers)

    return segments


def detect_transcript_json(text: str, api_key: Optional[str] = None) -> Optional[list]:
    """Return transcript ``text`` converted to JSON using Gemini."""
    logging.info(
        f"Starting transcript JSON conversion with Gemini for text: {text[:100]}..."
    )

    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY not found in environment")
            raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_FLASH,
        contents=[text],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192 + 8192,
            thinking_config=types.ThinkingConfig(thinking_budget=8192),
            system_instruction=_load_json_prompt(),
            response_mime_type="application/json",
        ),
    )

    logging.info(f"Received response from Gemini: {response.text[:100]}")
    try:
        result = json.loads(response.text)
        logging.info("Successfully converted transcript to JSON")
        return result
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON response from Gemini")
        return None


__all__ = [
    "detect_transcript_segment",
    "detect_transcript_json",
    "number_lines",
    "parse_line_numbers",
    "segments_from_lines",
]
