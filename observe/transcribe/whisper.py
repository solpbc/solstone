# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Whisper STT backend using faster-whisper.

This module provides local speech-to-text transcription using faster-whisper,
a CTranslate2-based implementation of OpenAI's Whisper model.

Configuration keys (passed in config dict):
- model: Whisper model size (e.g., "medium.en", "small.en"). Default: "medium.en"
- device: Device for inference ("auto", "cpu", "cuda"). Default: "auto"
- compute_type: Precision ("default", "float32", "float16", "int8"). Default: "default"
- initial_prompt: Optional prompt for context/style. Default: None

Platform optimizations:
- CUDA GPU: Uses float16 for GPU-optimized inference
- Apple Silicon: Uses int8 for ~2x faster inference
- Other CPU: Uses int8 for best performance
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from observe.transcribe.utils import build_statements_from_acoustic

# Default configuration
DEFAULT_MODEL = "medium.en"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE = "default"

# Style prompt to establish punctuation pattern for Whisper
# Whisper is autoregressive and can get stuck in "no-punctuation mode" without this
STYLE_PROMPT = "Okay, let's get started. Here's what we've been working on."

# Module-level model cache
_whisper_model = None
_model_config: tuple | None = None  # Track config to detect changes


def _has_cuda() -> bool:
    """Check if CUDA is available via CTranslate2."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _get_optimal_compute_type(device: str) -> str:
    """Get optimal compute type for the current platform.

    When compute_type is "default", CTranslate2 auto-selects but makes suboptimal
    choices on some platforms. This function provides better defaults:

    - CUDA GPU: float16 for GPU-optimized inference
    - CPU (including Apple Silicon): int8 for ~2x faster inference and faster model load

    Args:
        device: The device being used ("cpu", "cuda", "auto")

    Returns:
        Optimal compute type string
    """
    # If CUDA is explicitly requested or auto-detected, float16 is optimal
    if device == "cuda" or (device == "auto" and _has_cuda()):
        return "float16"

    # For CPU (including Apple Silicon), int8 is fastest
    # This provides ~2x speedup and 76x faster model loading
    return "int8"


def _get_model(config: dict):
    """Get or create WhisperModel with caching.

    The model is cached at module level and reused across calls.
    If configuration changes, the model is reloaded.

    Args:
        config: Backend configuration dict

    Returns:
        WhisperModel instance
    """
    global _whisper_model, _model_config
    from faster_whisper import WhisperModel

    model_size = config.get("model", DEFAULT_MODEL)
    device = config.get("device", DEFAULT_DEVICE)
    compute_type = config.get("compute_type", DEFAULT_COMPUTE)

    # Resolve "default" compute_type to platform-optimal setting
    if compute_type == "default":
        compute_type = _get_optimal_compute_type(device)
        logging.info(f"Auto-selected compute_type={compute_type} for device={device}")

    # Check if we can reuse cached model
    cache_key = (model_size, device, compute_type)
    if _whisper_model is not None and _model_config == cache_key:
        return _whisper_model

    # Load new model
    logging.info(f"Loading faster-whisper model ({model_size})...")
    t0 = time.perf_counter()

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    actual_device = model.model.device
    actual_compute = model.model.compute_type
    logging.info(
        f"  Whisper loaded in {time.perf_counter() - t0:.2f}s "
        f"(device={actual_device}, compute={actual_compute})"
    )

    # Cache the model
    _whisper_model = model
    _model_config = cache_key

    return model


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[dict]:
    """Transcribe audio using faster-whisper.

    Args:
        audio: Audio buffer (float32, mono)
        sample_rate: Sample rate in Hz (typically 16000)
        config: Backend configuration dict with keys:
            - model: Whisper model size (default: "medium.en")
            - device: Device for inference (default: "auto")
            - compute_type: Precision (default: "default")
            - initial_prompt: Optional prompt for context

    Returns:
        List of statements (sentence-aligned) with word-level data.
        Each statement has: id, start, end, text, words
    """
    model = _get_model(config)

    # Write audio to temp file (faster-whisper requires a file path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            temp_path = Path(f.name)

        # Convert to int16 for FLAC encoding
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(temp_path, audio_int16, sample_rate, format="FLAC")

        logging.info(f"Transcribing audio ({len(audio) / sample_rate:.1f}s)...")
        t0 = time.perf_counter()

        # Build transcribe kwargs
        transcribe_kwargs = {
            "language": "en",
            "vad_filter": True,
            "beam_size": 5,
            "word_timestamps": True,
        }

        # Use provided prompt or fall back to default style prompt
        initial_prompt = config.get("initial_prompt", STYLE_PROMPT)
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        # Run transcription
        segments_gen, info = model.transcribe(str(temp_path), **transcribe_kwargs)

        # Consume generator and build acoustic segments
        acoustic_segments = []
        for seg in segments_gen:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                    )

            acoustic_segments.append(
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "words": words,
                }
            )

        transcribe_time = time.perf_counter() - t0

        # Get duration from last acoustic segment or 0
        duration = max((s["end"] for s in acoustic_segments), default=0)

        # Log transcription stats
        logging.info(
            f"  Transcribed {len(acoustic_segments)} acoustic segments, "
            f"{duration:.1f}s speech in {transcribe_time:.2f}s "
            f"(RTF: {transcribe_time / max(duration, 0.1):.3f}x)"
        )

        # Build statements aligned to sentence boundaries
        num_acoustic = len(acoustic_segments)
        statements = build_statements_from_acoustic(acoustic_segments)
        logging.info(
            f"  Built {len(statements)} statements from "
            f"{num_acoustic} acoustic segments"
        )

        return statements

    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()


# Export model metadata for use by orchestrator
def get_model_info(config: dict) -> dict:
    """Get model configuration info for metadata.

    Args:
        config: Backend configuration dict

    Returns:
        Dict with model, device, compute_type
    """
    model = _get_model(config)
    return {
        "model": config.get("model", DEFAULT_MODEL),
        "device": str(model.model.device),
        "compute_type": str(model.model.compute_type),
    }
