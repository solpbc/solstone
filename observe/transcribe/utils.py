# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities for STT backends.

This module provides common utilities used by multiple STT backends:
- Platform detection (is_apple_silicon)
- Segment building and re-segmentation by sentence boundaries
"""

from __future__ import annotations

import platform

# Sentence-ending punctuation marks
SENTENCE_ENDINGS = frozenset(".?!")


def is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def build_segment(segment_id: int, words: list[dict]) -> dict:
    """Build a segment dict from a list of words.

    Args:
        segment_id: Sequential segment ID
        words: List of word dicts with 'word', 'start', 'end', 'probability'

    Returns:
        Segment dict with id, start, end, text, words
    """
    # Join words - Whisper includes leading spaces in word text
    text = "".join(w.get("word", "") for w in words).strip()

    return {
        "id": segment_id,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
    }


def resegment_by_sentences(segments: list[dict]) -> list[dict]:
    """Re-segment transcript by sentence boundaries instead of acoustic pauses.

    Backends that return acoustic segments (VAD-driven) can use this to
    re-align to sentence boundaries (punctuation-driven). This function
    flattens all words and re-segments based on sentence-ending punctuation.

    Args:
        segments: List of segment dicts with 'words' containing word-level data

    Returns:
        New list of segments aligned to sentence boundaries
    """
    # Flatten all words from all segments
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        return segments

    # Build new segments based on sentence-ending punctuation
    new_segments = []
    current_words = []

    for word in all_words:
        current_words.append(word)

        # Check if word ends with sentence-ending punctuation
        word_text = word.get("word", "").strip()
        if word_text and word_text[-1] in SENTENCE_ENDINGS:
            # Complete this segment
            new_segments.append(build_segment(len(new_segments) + 1, current_words))
            current_words = []

    # Handle any remaining words (incomplete final sentence)
    if current_words:
        new_segments.append(build_segment(len(new_segments) + 1, current_words))

    return new_segments
