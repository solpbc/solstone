# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Enrich audio transcripts with contextual information using Gemini Lite.

Takes Whisper transcript segments paired with audio clips and extracts:
- Per-segment corrected text (fixing transcription errors)
- Per-segment audio descriptions (tone, delivery, vocalizations)
- Overall topics discussed
- Setting classification (workplace, personal, etc.)
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from google.genai import types

from observe.utils import SAMPLE_RATE
from think.entities import load_recent_entity_names
from think.models import GEMINI_LITE, gemini_generate
from think.utils import load_prompt

logger = logging.getLogger(__name__)

# Number of recent entity names to include in enrichment prompt
ENTITY_NAMES_LIMIT = 40


def _load_entity_names() -> str | None:
    """Load recent entity names for enrichment prompt context.

    Returns:
        Comma-separated entity names string or None if no entities found
    """
    names = load_recent_entity_names(limit=ENTITY_NAMES_LIMIT)
    if not names:
        return None
    return ", ".join(names)


def _segment_to_flac_bytes(
    wav: np.ndarray, start: float, end: float, sample_rate: int
) -> bytes:
    """Extract a segment from audio and encode as FLAC bytes.

    Args:
        wav: Audio waveform (mono, float32)
        start: Start time in seconds
        end: End time in seconds
        sample_rate: Sample rate of the audio

    Returns:
        FLAC-encoded audio bytes
    """
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment_audio = wav[start_sample:end_sample]

    # Convert to int16 for FLAC encoding
    audio_int16 = (np.clip(segment_audio, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, audio_int16, sample_rate, format="FLAC")
    return buf.getvalue()


def enrich_transcript(
    audio_path: Path,
    segments: list[dict],
) -> dict | None:
    """Enrich transcript segments with audio context using Gemini Lite.

    Sends numbered segments with text and audio clips to Gemini to extract
    corrected text, per-segment descriptions, and overall topics/setting.

    Args:
        audio_path: Path to audio file (FLAC)
        segments: List of segment dicts with 'id', 'start', 'end', 'text'

    Returns:
        Dict with enrichment data or None on error:
        - segments: List of dicts with 'corrected' and 'description' keys
        - topics: Comma-separated topic string
        - setting: Setting classification string
    """
    if not segments:
        return None

    try:
        # Load audio
        logger.info("Loading audio for enrichment...")
        t0 = time.perf_counter()
        wav, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
        logger.info(f"  Audio loaded in {time.perf_counter() - t0:.2f}s")

        # Load entity names for context
        entity_names = _load_entity_names()

        # Build interleaved content: numbered text label + audio clip for each segment
        prompt_content = load_prompt(
            "enrich",
            base_dir=Path(__file__).parent,
            context={"entity_names": entity_names},
        )
        contents: list = [prompt_content.text]

        for i, seg in enumerate(segments, start=1):
            # Add numbered text label
            text = seg.get("text", "")
            contents.append(f"Segment {i}: {text}")

            # Add audio clip
            audio_bytes = _segment_to_flac_bytes(wav, seg["start"], seg["end"], sr)
            contents.append(
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/flac")
            )

        # Call Gemini Lite
        logger.info(f"Enriching {len(segments)} segments with Gemini Lite...")
        t0 = time.perf_counter()

        response_text = gemini_generate(
            contents=contents,
            model=GEMINI_LITE,
            temperature=0.3,
            max_output_tokens=8192,
            thinking_budget=4096,
            json_output=True,
            context="observe.enrich",
        )

        result = json.loads(response_text)
        logger.info(f"  Enrichment complete in {time.perf_counter() - t0:.2f}s")

        # Validate response structure
        if not isinstance(result, dict):
            logger.warning(f"Enrichment returned non-dict: {type(result)}")
            return None

        if "segments" not in result or "topics" not in result:
            logger.warning(f"Enrichment missing required fields: {result.keys()}")
            return None

        # Validate segments array
        enriched_segments = result["segments"]
        if not isinstance(enriched_segments, list):
            logger.warning("Enrichment 'segments' is not a list")
            return None

        if len(enriched_segments) != len(segments):
            logger.warning(
                f"Enrichment returned {len(enriched_segments)} segments "
                f"for {len(segments)} input segments"
            )
            # Still usable - we'll align what we can

        return result

    except Exception as e:
        logger.warning(f"Enrichment failed for {audio_path}: {e}")
        return None
