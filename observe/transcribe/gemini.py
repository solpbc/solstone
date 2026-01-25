# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Gemini STT backend for speech-to-text transcription.

This module provides cloud-based speech-to-text transcription using Google's
Gemini API with speaker diarization (identifies who said what).

Enrichment (topics, setting, emotion, corrections) is handled separately by
the enrich step, same as other backends. This keeps the transcription focused
and avoids hallucinations from entity name hints in the prompt.

Environment:
- GOOGLE_API_KEY: API key (required)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from google.genai import types

from observe.utils import audio_to_flac_bytes
from think.models import generate
from think.utils import load_prompt

logger = logging.getLogger(__name__)

# Regex for parsing speaker strings like "Speaker 1", "Speaker 2"
SPEAKER_PATTERN = re.compile(r"(?:speaker\s*)?(\d+)", re.IGNORECASE)


def _parse_timestamp(ts: str) -> float | None:
    """Parse MM:SS or HH:MM:SS timestamp to seconds.

    Handles various formats:
    - "1:23" -> 83.0
    - "0:05" -> 5.0
    - "1:05:30" -> 3930.0
    - "5" -> 5.0 (just seconds)

    Args:
        ts: Timestamp string

    Returns:
        Seconds as float, or None if unparseable
    """
    if not ts or not isinstance(ts, str):
        return None

    ts = ts.strip()
    if not ts:
        return None

    try:
        parts = ts.split(":")
        if len(parts) == 1:
            # Just seconds
            seconds = float(parts[0])
        elif len(parts) == 2:
            # MM:SS
            minutes = int(parts[0])
            seconds = float(parts[1])
            seconds += minutes * 60
        elif len(parts) == 3:
            # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            seconds += hours * 3600 + minutes * 60
        else:
            return None

        # Clamp negative to 0
        return max(0.0, seconds)

    except (ValueError, TypeError):
        return None


def _parse_speaker(speaker: str | int | None) -> int | None:
    """Parse speaker identifier to 1-indexed integer.

    Handles:
    - "Speaker 1" -> 1
    - "speaker 2" -> 2
    - "1" -> 1
    - 1 -> 1

    Args:
        speaker: Speaker identifier (string or int)

    Returns:
        1-indexed speaker ID, or None if unparseable
    """
    if speaker is None:
        return None

    if isinstance(speaker, int):
        return speaker if speaker > 0 else None

    if isinstance(speaker, str):
        # Try regex match for "Speaker N" pattern
        match = SPEAKER_PATTERN.search(speaker)
        if match:
            val = int(match.group(1))
            return val if val > 0 else None

        # Try direct int conversion
        try:
            val = int(speaker)
            return val if val > 0 else None
        except ValueError:
            pass

    return None


def _normalize_segments(
    segments: list[dict],
    audio_duration: float,
) -> tuple[list[dict], int]:
    """Convert Gemini segments to standard statement format.

    Segments with empty text are dropped. IDs are assigned sequentially
    to the remaining segments.

    Args:
        segments: Raw segments from Gemini response
        audio_duration: Total audio duration in seconds

    Returns:
        Tuple of (statements list, count of segments with invalid timestamps)
    """
    statements = []
    invalid_count = 0
    statement_id = 1

    for seg in segments:
        # Get and strip text - skip if empty
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Parse timestamps
        start = _parse_timestamp(seg.get("start", ""))
        end = _parse_timestamp(seg.get("end", ""))

        # Track invalid timestamps but still include segment
        has_valid_timestamps = start is not None and end is not None
        if not has_valid_timestamps:
            invalid_count += 1

        # Clamp to audio duration if timestamps are valid
        if start is not None and start > audio_duration:
            start = audio_duration
        if end is not None and end > audio_duration:
            end = audio_duration

        # Build statement
        statement = {
            "id": statement_id,
            "start": start,
            "end": end,
            "text": text,
            "words": None,  # Not available from Gemini
        }
        statement_id += 1

        # Parse speaker
        speaker = _parse_speaker(seg.get("speaker"))
        if speaker is not None:
            statement["speaker"] = speaker

        statements.append(statement)

    return statements, invalid_count


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[dict]:
    """Transcribe audio using Gemini API.

    This is the standard backend interface. It converts audio to FLAC,
    sends to Gemini with a transcription prompt, and returns normalized
    statements with speaker diarization.

    Args:
        audio: Audio buffer (float32, mono)
        sample_rate: Sample rate in Hz (typically 16000)
        config: Backend configuration dict (currently unused)

    Returns:
        List of statements with id, start, end, text, speaker.
    """
    audio_duration = len(audio) / sample_rate

    logger.info(f"Transcribing audio with Gemini ({audio_duration:.1f}s)...")
    t0 = time.perf_counter()

    # Convert audio to FLAC bytes
    audio_bytes = audio_to_flac_bytes(audio, sample_rate)

    # Load prompt from gemini.md
    prompt_text = load_prompt("gemini", base_dir=Path(__file__).parent).text

    # Build contents: prompt text + audio
    contents: list = [
        prompt_text,
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/flac"),
    ]

    # Call Gemini via think.models.generate()
    response_text = generate(
        contents=contents,
        context="observe.transcribe.gemini",
        temperature=0.3,
        max_output_tokens=8192,
        json_output=True,
    )

    transcribe_time = time.perf_counter() - t0

    # Parse JSON response
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini returned invalid JSON: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        raise RuntimeError(f"Gemini returned invalid JSON: {e}") from e

    # Extract segments
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        logger.warning(f"Gemini 'segments' is not a list: {type(segments)}")
        segments = []

    # Normalize to standard statement format
    statements, invalid_count = _normalize_segments(segments, audio_duration)

    if invalid_count > 0:
        logger.warning(
            f"  {invalid_count} segment(s) had invalid timestamps "
            "(will be saved but not embedded)"
        )

    logger.info(
        f"  Gemini returned {len(statements)} segments in {transcribe_time:.2f}s"
    )

    return statements


def get_model_info(config: dict) -> dict:
    """Get model configuration info for metadata.

    Args:
        config: Backend configuration dict

    Returns:
        Dict with model info for JSONL metadata
    """
    # Model is resolved by think.models based on context
    # We report "gemini" as the model family
    return {
        "model": "gemini",
        "device": "cloud",
        "compute_type": "api",
    }
