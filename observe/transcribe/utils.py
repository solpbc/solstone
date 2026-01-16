# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities for STT backends.

This module provides common utilities used by multiple STT backends:
- Platform detection (is_apple_silicon)
- Statement building from word-level data

Terminology:
- "statement" = individual transcript entry (sentence or speaker turn)
- "segment" = journal directory (HHMMSS_LEN/ time window) - NOT used here
"""

from __future__ import annotations

import platform

# Sentence-ending punctuation marks
SENTENCE_ENDINGS = frozenset(".?!")


def is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def build_statement(statement_id: int, words: list[dict]) -> dict:
    """Build a statement dict from a list of words.

    Args:
        statement_id: Sequential statement ID
        words: List of word dicts with 'word', 'start', 'end', 'probability'

    Returns:
        Statement dict with id, start, end, text, words
    """
    # Join words - Whisper includes leading spaces in word text
    text = "".join(w.get("word", "") for w in words).strip()

    return {
        "id": statement_id,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": text,
        "words": words,
    }


def build_statements_from_acoustic(acoustic_segments: list[dict]) -> list[dict]:
    """Build statements from acoustic segments by aligning to sentence boundaries.

    Backends that return acoustic segments (VAD-driven) can use this to
    re-align to sentence boundaries (punctuation-driven). This function
    flattens all words and builds statements based on sentence-ending punctuation.

    Args:
        acoustic_segments: List of acoustic segment dicts with 'words' containing
            word-level data

    Returns:
        List of statements aligned to sentence boundaries
    """
    # Flatten all words from all acoustic segments
    all_words = []
    for seg in acoustic_segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        return acoustic_segments

    # Build statements based on sentence-ending punctuation
    statements = []
    current_words = []

    for word in all_words:
        current_words.append(word)

        # Check if word ends with sentence-ending punctuation
        word_text = word.get("word", "").strip()
        if word_text and word_text[-1] in SENTENCE_ENDINGS:
            # Complete this statement
            statements.append(build_statement(len(statements) + 1, current_words))
            current_words = []

    # Handle any remaining words (incomplete final sentence)
    if current_words:
        statements.append(build_statement(len(statements) + 1, current_words))

    return statements
