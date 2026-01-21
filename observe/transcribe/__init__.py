# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speech-to-text backend registry and shared utilities.

This package provides a pluggable STT backend system with:
- Backend registry for dispatch to different STT providers
- Shared utilities for statement building from word-level data
- Normalized statement format for all backends

Terminology:
- "statement" = individual transcript entry (sentence or speaker turn)
- "segment" = journal directory (HHMMSS_LEN/ time window) - NOT used here

Available backends:
- whisper: Local faster-whisper (default, GPU/CPU)
- revai: Rev.ai cloud API (speaker diarization)
- gemini: Google Gemini API (integrated STT + enrichment)

Backend Interface:
    Each backend module must export a transcribe() function:

    def transcribe(
        audio: np.ndarray,      # float32, mono, sample_rate Hz
        sample_rate: int,       # typically 16000
        config: dict,           # backend-specific config
    ) -> list[dict]:
        '''Return statements with word-level timestamps (if available).'''

    Statement format:
    {
        "id": int,              # sequential, starting from 1
        "start": float,         # seconds
        "end": float,           # seconds
        "text": str,            # transcribed text
        "words": list[dict] | None,  # word-level data if available
        "speaker": int | None,  # speaker ID (revai/gemini, 1-indexed)
    }

    Word format (when available):
    {
        "word": str,
        "start": float,
        "end": float,
        "probability": float,
    }
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, str] = {
    "whisper": "observe.transcribe.whisper",
    "revai": "observe.transcribe.revai",
    "gemini": "observe.transcribe.gemini",
}

# ---------------------------------------------------------------------------
# Backend Metadata
# ---------------------------------------------------------------------------
# Display labels, descriptions, and settings schemas for each backend.
# Used by settings UI to dynamically build backend dropdowns and forms.
# ---------------------------------------------------------------------------

BACKEND_METADATA: dict[str, dict] = {
    "whisper": {
        "label": "Whisper - Local processing",
        "description": "Local speech recognition using faster-whisper",
        "env_key": None,
        "settings": ["device", "model", "compute_type"],
    },
    "revai": {
        "label": "Rev.ai - Cloud with speaker diarization",
        "description": "Cloud-based transcription with speaker identification",
        "env_key": "REVAI_ACCESS_TOKEN",
        "settings": ["model"],
    },
    "gemini": {
        "label": "Gemini - Cloud with integrated enrichment",
        "description": "Combines transcription and enrichment in one call",
        "env_key": "GOOGLE_API_KEY",
        "settings": [],
    },
}


def get_backend(name: str) -> ModuleType:
    """Get STT backend module by name.

    Args:
        name: Backend name (e.g., "whisper")

    Returns:
        Backend module with transcribe() function

    Raises:
        ValueError: If backend name is not registered
    """
    if name not in BACKEND_REGISTRY:
        valid = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown STT backend: {name!r}. Valid backends: {valid}")

    return import_module(BACKEND_REGISTRY[name])


def get_backend_list() -> list[dict]:
    """Get list of backends with metadata for UI display.

    Returns:
        List of backend info dicts, each containing:
        - name: Backend identifier (e.g., "whisper")
        - label: Display label
        - description: Short description
        - env_key: Environment variable for API key (None for local backends)
        - settings: List of configurable field names
    """
    return [
        {"name": name, **BACKEND_METADATA.get(name, {"label": name})}
        for name in BACKEND_REGISTRY
    ]


def transcribe(
    backend: str,
    audio: "np.ndarray",
    sample_rate: int,
    config: dict,
) -> list[dict]:
    """Dispatch transcription to the specified backend.

    Args:
        backend: Backend name (e.g., "whisper")
        audio: Audio buffer (float32, mono)
        sample_rate: Sample rate in Hz (typically 16000)
        config: Backend-specific configuration dict

    Returns:
        List of statement dicts with id, start, end, text, and optionally words
    """
    backend_mod = get_backend(backend)
    return backend_mod.transcribe(audio, sample_rate, config)


# ---------------------------------------------------------------------------
# Re-exports (utilities from utils.py, main entry point from main.py)
# ---------------------------------------------------------------------------

from observe.transcribe.main import (
    DEFAULT_MIN_SPEECH_SECONDS,
    MIN_STATEMENT_DURATION,
    main,
    process_audio,
)
from observe.transcribe.utils import (
    SENTENCE_ENDINGS,
    build_statement,
    build_statements_from_acoustic,
    is_apple_silicon,
)
from observe.transcribe.whisper import DEFAULT_COMPUTE, DEFAULT_DEVICE, DEFAULT_MODEL

__all__ = [
    # Registry
    "BACKEND_REGISTRY",
    "BACKEND_METADATA",
    "get_backend",
    "get_backend_list",
    "transcribe",
    # Utilities
    "SENTENCE_ENDINGS",
    "is_apple_silicon",
    "build_statement",
    "build_statements_from_acoustic",
    # Main entry point
    "main",
    "process_audio",
    # Constants (backwards compatibility)
    "DEFAULT_MODEL",
    "DEFAULT_DEVICE",
    "DEFAULT_COMPUTE",
    "DEFAULT_MIN_SPEECH_SECONDS",
    "MIN_STATEMENT_DURATION",
]
