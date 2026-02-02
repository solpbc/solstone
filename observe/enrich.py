# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Enrich audio transcripts with contextual information using LLM analysis.

Takes transcript statements paired with audio clips and extracts:
- Per-statement corrected text (fixing transcription errors)
- Per-statement emotion (tone, delivery)
- Overall topics discussed
- Setting classification (workplace, personal, etc.)
- Audio quality warnings
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from google.genai import types

from observe.utils import audio_to_flac_bytes
from think.models import generate
from think.muse import load_prompt

logger = logging.getLogger(__name__)


def _statement_to_flac_bytes(
    wav: np.ndarray, start: float, end: float, sample_rate: int
) -> bytes:
    """Extract a statement's audio and encode as FLAC bytes.

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
    stmt_audio = wav[start_sample:end_sample]

    return audio_to_flac_bytes(stmt_audio, sample_rate)


def enrich_transcript(
    audio: np.ndarray,
    sample_rate: int,
    statements: list[dict],
    entity_names: list[str] | None = None,
) -> dict | None:
    """Enrich transcript statements with audio context using LLM analysis.

    Sends numbered statements with text and audio clips to extract corrected
    text, per-statement emotion, and overall topics/setting/warnings.

    Args:
        audio: Audio waveform (mono, float32)
        sample_rate: Sample rate in Hz
        statements: List of statement dicts with 'id', 'start', 'end', 'text'
        entity_names: Optional list of entity names for prompt context

    Returns:
        Dict with enrichment data or None on error:
        - statements: List of dicts with 'corrected' and 'emotion' keys
        - topics: Comma-separated topic string
        - setting: Setting classification string
        - warning: Audio quality issues (may be empty string)
    """
    if not statements:
        return None

    try:
        # Format entity names for prompt context
        entity_names_str = ", ".join(entity_names) if entity_names else None

        # Build interleaved content: numbered text label + audio clip for each statement
        prompt_content = load_prompt(
            "enrich",
            base_dir=Path(__file__).parent,
            context={"entity_names": entity_names_str},
        )
        contents: list = [prompt_content.text]

        for i, stmt in enumerate(statements, start=1):
            # Add numbered text label
            text = stmt.get("text", "")
            contents.append(f"Statement {i}: {text}")

            # Add audio clip
            audio_bytes = _statement_to_flac_bytes(
                audio, stmt["start"], stmt["end"], sample_rate
            )
            contents.append(
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/flac")
            )

        # Call LLM (tier from enrich.md frontmatter)
        logger.info(f"Enriching {len(statements)} statements...")
        t0 = time.perf_counter()

        response_text = generate(
            contents=contents,
            context="observe.enrich",
            temperature=0.3,
            max_output_tokens=8192,
            thinking_budget=4096,
            json_output=True,
        )

        result = json.loads(response_text)
        logger.info(f"  Enrichment complete in {time.perf_counter() - t0:.2f}s")

        # Validate response structure
        if not isinstance(result, dict):
            logger.warning(f"Enrichment returned non-dict: {type(result)}")
            return None

        if "statements" not in result or "topics" not in result:
            logger.warning(f"Enrichment missing required fields: {result.keys()}")
            return None

        # Validate statements array
        enriched_statements = result["statements"]
        if not isinstance(enriched_statements, list):
            logger.warning("Enrichment 'statements' is not a list")
            return None

        if len(enriched_statements) != len(statements):
            logger.warning(
                f"Enrichment returned {len(enriched_statements)} statements "
                f"for {len(statements)} input statements"
            )
            # Still usable - we'll align what we can

        return result

    except Exception as e:
        logger.warning(f"Enrichment failed: {e}")
        return None
