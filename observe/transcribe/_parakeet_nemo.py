# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Parakeet (NeMo) backend for Linux x86_64.

Uses nvidia/parakeet-tdt-0.6b-{v2,v3} via nemo_toolkit[asr] and torch.
First run downloads the model into ~/.cache/huggingface/hub/.

NeMo timestamp output does not expose per-word confidence, so this
backend emits probability=1.0 as a sentinel for each word and reports
per_word_confidence=False in get_model_info().
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from observe.transcribe.utils import build_statements_from_acoustic

logging.getLogger("nemo_logger").setLevel(logging.WARNING)

_DEFAULT_MODEL_VERSION = "v3"
_DEFAULT_TIMEOUT_SEC = 120.0
_DEFAULT_DEVICE = "auto"
_DEFAULT_PRECISION = "auto"
_MODEL_IDS = {
    "v2": "nvidia/parakeet-tdt-0.6b-v2",
    "v3": "nvidia/parakeet-tdt-0.6b-v3",
}
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _validate_config(config: dict) -> tuple[str, str, float, str]:
    """Validate backend configuration."""
    model_version = config.get("model_version", _DEFAULT_MODEL_VERSION)
    if model_version not in _MODEL_IDS:
        raise ValueError("model_version must be one of: v2, v3")

    device = config.get("device", _DEFAULT_DEVICE)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    precision = config.get("precision", _DEFAULT_PRECISION)
    if precision not in {"auto", "float16", "float32"}:
        raise ValueError("precision must be one of: auto, float16, float32")

    raw_timeout = config.get("timeout_sec", _DEFAULT_TIMEOUT_SEC)
    try:
        timeout_sec = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeout_sec must be > 0, got {raw_timeout!r}") from exc
    if timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be > 0, got {raw_timeout!r}")

    return model_version, device, timeout_sec, precision


def _resolve_device(device: str, torch_module: Any) -> str:
    """Resolve auto device selection."""
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if device == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            "Parakeet NeMo requested device=cuda but CUDA is unavailable"
        )
    return device


def _resolve_precision(precision: str, resolved_device: str) -> str:
    """Resolve auto precision selection."""
    if precision == "auto":
        return "float16" if resolved_device == "cuda" else "float32"
    if precision == "float16" and resolved_device != "cuda":
        raise ValueError("precision=float16 requires device=cuda")
    return precision


def _get_model(config: dict) -> tuple[Any, str, str]:
    """Load or reuse a cached NeMo model."""
    model_version, device, _timeout_sec, precision = _validate_config(config)

    import torch
    from nemo.collections.asr.models import ASRModel

    resolved_device = _resolve_device(device, torch)
    resolved_precision = _resolve_precision(precision, resolved_device)
    cache_key = (model_version, resolved_device, resolved_precision)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], resolved_device, resolved_precision

    try:
        model = ASRModel.from_pretrained(_MODEL_IDS[model_version])
    except Exception as exc:
        raise RuntimeError(f"Parakeet NeMo model download failed: {exc}") from exc

    model = model.to(resolved_device)
    model = model.half() if resolved_precision == "float16" else model.float()
    model.freeze()
    _MODEL_CACHE[cache_key] = model
    return model, resolved_device, resolved_precision


def _hypothesis_text(hypothesis: Any) -> str:
    """Extract transcript text from a NeMo hypothesis."""
    if hasattr(hypothesis, "text"):
        return str(hypothesis.text).strip()
    if isinstance(hypothesis, dict):
        return str(hypothesis.get("text", "")).strip()
    return ""


def _word_timestamps(hypothesis: Any) -> list[dict]:
    """Extract word timestamps from a NeMo hypothesis."""
    timestamp = getattr(hypothesis, "timestamp", None)
    if isinstance(timestamp, dict):
        words = timestamp.get("word", [])
        return words if isinstance(words, list) else []

    if isinstance(hypothesis, dict):
        timestamp = hypothesis.get("timestamp", {})
        if isinstance(timestamp, dict):
            words = timestamp.get("word", [])
            return words if isinstance(words, list) else []

    return []


def transcribe(audio: np.ndarray, sample_rate: int, config: dict) -> list[dict]:
    """Transcribe audio using NeMo Parakeet."""
    model_version, _device, _timeout_sec, _precision = _validate_config(config)
    model, resolved_device, _resolved_precision = _get_model(config)

    temp_path = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(temp_path, audio_int16, sample_rate, format="WAV", subtype="PCM_16")

        hypotheses = model.transcribe([str(temp_path)], timestamps=True)
        if not hypotheses:
            logging.debug("Parakeet NeMo returned no hypotheses")
            return []

        hypothesis = hypotheses[0]
        transcript_text = _hypothesis_text(hypothesis)
        raw_words = _word_timestamps(hypothesis)
        if not raw_words:
            logging.debug("Parakeet NeMo returned empty transcript timestamps")
            return []

        words = []
        for raw_word in raw_words:
            word_text = str(raw_word.get("word", "")).strip()
            if not word_text:
                continue
            word_text = (
                f" {word_text.lstrip()}"  # leading space required by build_statement()
            )
            words.append(
                {
                    "word": word_text,
                    "start": float(raw_word["start"]),
                    "end": float(raw_word["end"]),
                    "probability": 1.0,
                }
            )

        if not words:
            logging.debug("Parakeet NeMo word timestamps collapsed to empty output")
            return []

        acoustic_segments = [
            {
                "id": 1,
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": transcript_text,
                "words": words,
            }
        ]
        statements = build_statements_from_acoustic(acoustic_segments)
        for statement in statements:
            statement["speaker"] = None

        elapsed = time.perf_counter() - started
        audio_sec = words[-1]["end"] if words else len(audio) / sample_rate
        rtfx = audio_sec / max(elapsed, 0.001)
        logging.info(
            f"  Transcribed {len(statements)} statements, {audio_sec:.2f}s speech "
            f"in {elapsed:.2f}s (RTFx: {rtfx:.2f}) "
            f"[model={model_version} device={resolved_device}]"
        )
        return statements
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def get_model_info(config: dict) -> dict:
    """Return NeMo model metadata for transcript JSONL headers."""
    model_version, device, _timeout_sec, precision = _validate_config(config)

    import torch

    resolved_device = _resolve_device(device, torch)
    resolved_precision = _resolve_precision(precision, resolved_device)
    return {
        "model": _MODEL_IDS[model_version],
        "device": resolved_device,
        "compute_type": resolved_precision,
        "per_word_confidence": False,
    }
