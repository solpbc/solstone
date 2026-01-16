# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Speech-to-text backend registry and shared utilities.

This package provides a pluggable STT backend system with:
- Backend registry for dispatch to different STT providers
- Shared utilities for segment processing (resegmentation, building)
- Normalized segment format for all backends

Available backends:
- whisper: Local faster-whisper (default, GPU/CPU)
- revai: Rev.ai cloud API (speaker diarization)

Backend Interface:
    Each backend module must export a transcribe() function:

    def transcribe(
        audio: np.ndarray,      # float32, mono, sample_rate Hz
        sample_rate: int,       # typically 16000
        config: dict,           # backend-specific config
    ) -> list[dict]:
        '''Return segments with word-level timestamps (if available).'''

    Segment format:
    {
        "id": int,              # sequential, starting from 1
        "start": float,         # seconds
        "end": float,           # seconds
        "text": str,            # transcribed text
        "words": list[dict] | None,  # word-level data if available
        "speaker": int | None,  # speaker ID (revai only, 1-indexed)
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
        List of segment dicts with id, start, end, text, and optionally words
    """
    backend_mod = get_backend(backend)
    return backend_mod.transcribe(audio, sample_rate, config)


# ---------------------------------------------------------------------------
# Re-exports (utilities from utils.py, main entry point from main.py)
# ---------------------------------------------------------------------------

from observe.transcribe.main import (
    DEFAULT_MIN_SPEECH_SECONDS,
    MIN_SEGMENT_DURATION,
    main,
    process_audio,
)
from observe.transcribe.utils import (
    SENTENCE_ENDINGS,
    build_segment,
    is_apple_silicon,
    resegment_by_sentences,
)
from observe.transcribe.whisper import DEFAULT_COMPUTE, DEFAULT_DEVICE, DEFAULT_MODEL

__all__ = [
    # Registry
    "BACKEND_REGISTRY",
    "get_backend",
    "transcribe",
    # Utilities
    "SENTENCE_ENDINGS",
    "is_apple_silicon",
    "build_segment",
    "resegment_by_sentences",
    # Main entry point
    "main",
    "process_audio",
    # Constants (backwards compatibility)
    "DEFAULT_MODEL",
    "DEFAULT_DEVICE",
    "DEFAULT_COMPUTE",
    "DEFAULT_MIN_SPEECH_SECONDS",
    "MIN_SEGMENT_DURATION",
]
