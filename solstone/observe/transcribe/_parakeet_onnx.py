# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Linux x86_64 Parakeet backend via onnx-asr + onnxruntime.

macOS uses `_parakeet_coreml.py` separately.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from .utils import build_statements_from_acoustic

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_VERSION = "v3"
_DEFAULT_TIMEOUT_SEC = 300.0
_DEFAULT_DEVICE = "auto"
_DEFAULT_QUANTIZATION = "auto"
_MODEL_INFO_IDS = {
    "v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
}
_MODEL_LOAD_IDS = {
    "v2": "nemo-parakeet-tdt-0.6b-v2",
    "v3": "nemo-parakeet-tdt-0.6b-v3",
}
_ADAPTER_CACHE: dict[tuple[str, str, str], Any] = {}
_WARNED_INT8_CUDA: set[tuple[str, str]] = set()
_INT8_CUDA_WARNING = (
    "Parakeet ONNX int8 on CUDA is opt-in and may underperform fp32 on some GPUs"
)
_CUDA_REMEDIATION = (
    "device=cuda requires CUDAExecutionProvider; rerun: "
    "PARAKEET_ONNX_VARIANT=cuda make install"
)


def _validate_config(config: dict) -> tuple[str, str, float, str]:
    """Validate backend configuration."""
    model_version = config.get("model_version", _DEFAULT_MODEL_VERSION)
    if model_version not in _MODEL_INFO_IDS:
        raise ValueError("model_version must be one of: v2, v3")

    device = config.get("device", _DEFAULT_DEVICE)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    quantization = config.get("quantization", _DEFAULT_QUANTIZATION)
    if quantization not in {"auto", "fp32", "int8"}:
        raise ValueError("quantization must be one of: auto, fp32, int8")

    raw_timeout = config.get("timeout_sec", _DEFAULT_TIMEOUT_SEC)
    try:
        timeout_sec = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeout_sec must be > 0, got {raw_timeout!r}") from exc
    if timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be > 0, got {raw_timeout!r}")

    return model_version, device, timeout_sec, quantization


def _resolve_runtime(device: str, quantization: str) -> tuple[str, str, list[str]]:
    """Resolve runtime device, quantization, and provider list."""
    import onnxruntime

    available_providers = set(onnxruntime.get_available_providers())
    if device == "auto":
        resolved_device = (
            "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"
        )
    elif device == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(_CUDA_REMEDIATION)
        resolved_device = "cuda"
    else:
        resolved_device = "cpu"

    resolved_quantization = "fp32" if quantization == "auto" else quantization
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if resolved_device == "cuda"
        else ["CPUExecutionProvider"]
    )

    warning_key = (resolved_device, resolved_quantization)
    if warning_key == ("cuda", "int8") and warning_key not in _WARNED_INT8_CUDA:
        logger.warning(_INT8_CUDA_WARNING)
        _WARNED_INT8_CUDA.add(warning_key)

    return resolved_device, resolved_quantization, providers


def _get_adapter(model_version: str, device: str, quantization: str) -> Any:
    """Load or reuse a cached ONNX adapter."""
    import onnx_asr
    import onnxruntime

    resolved_device, resolved_quantization, providers = _resolve_runtime(
        device, quantization
    )
    cache_key = (model_version, resolved_device, resolved_quantization)
    if cache_key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[cache_key]

    if resolved_device == "cuda":
        try:
            onnxruntime.preload_dlls(cuda=True, cudnn=True)
        except Exception as exc:
            raise RuntimeError(_CUDA_REMEDIATION) from exc

    adapter = onnx_asr.load_model(
        _MODEL_LOAD_IDS[model_version],
        quantization=None if resolved_quantization == "fp32" else resolved_quantization,
        providers=providers,
    ).with_timestamps()
    _ADAPTER_CACHE[cache_key] = adapter
    return adapter


def _chunked_recognize(
    adapter: Any,
    audio: np.ndarray,
    sample_rate: int,
    chunk_sec: float = 25.0,
    stride_sec: float = 24.0,
) -> tuple[list[dict], float]:
    """Recognize audio in one pass or chunked sliding windows."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if chunk_sec <= 0 or stride_sec <= 0 or stride_sec > chunk_sec:
        raise ValueError(
            "chunk_sec and stride_sec must be > 0 with stride_sec <= chunk_sec"
        )
    if audio.size == 0:
        return [], 0.0

    audio_sec = len(audio) / sample_rate

    def recognize_words(chunk_audio: np.ndarray) -> list[dict]:
        result = adapter.recognize(chunk_audio, sample_rate=sample_rate)
        raw_tokens = None
        if isinstance(result, list):
            if (
                result
                and isinstance(result[0], dict)
                and ("token" in result[0] or "word" in result[0])
            ):
                raw_tokens = result
            else:
                result = result[0] if result else None
        if result is None:
            return []

        chunk_audio_sec = len(chunk_audio) / sample_rate
        words: list[dict] = []
        if raw_tokens is None:
            timestamps = getattr(result, "timestamps", None)
            tokens = getattr(result, "tokens", None)
            logprobs = getattr(result, "logprobs", None)
        else:
            timestamps = None
            tokens = None
            logprobs = None
        if timestamps is not None and tokens is not None:
            for index, token_text in enumerate(tokens):
                start = float(timestamps[index])
                end = (
                    float(timestamps[index + 1])
                    if index + 1 < len(timestamps)
                    else chunk_audio_sec
                )
                logprob = None
                if logprobs is not None and index < len(logprobs):
                    logprob = float(logprobs[index])
                words.append(
                    {
                        "word": f" {str(token_text).lstrip()}",
                        "start": start,
                        "end": end,
                        "probability": math.exp(logprob)
                        if logprob is not None
                        else 1.0,
                    }
                )
            return words

        if raw_tokens is None:
            if isinstance(result, dict):
                raw_tokens = result.get("token_timings") or result.get("tokens") or []
            else:
                raw_tokens = getattr(result, "token_timings", None)
                if raw_tokens is None:
                    raw_tokens = getattr(result, "tokens", [])

        for raw_token in raw_tokens:
            if isinstance(raw_token, dict):
                token_text = raw_token.get("token", raw_token.get("word", ""))
                start = raw_token.get("start")
                end = raw_token.get("end")
                logprob = raw_token.get("logprob")
            else:
                token_text = getattr(raw_token, "token", getattr(raw_token, "word", ""))
                start = getattr(raw_token, "start", None)
                end = getattr(raw_token, "end", None)
                logprob = getattr(raw_token, "logprob", None)

            token_text = str(token_text).strip()
            if not token_text or start is None or end is None:
                continue

            probability = math.exp(float(logprob)) if logprob is not None else 1.0
            words.append(
                {
                    "word": f" {token_text.lstrip()}",
                    "start": float(start),
                    "end": float(end),
                    "probability": probability,
                }
            )
        return words

    if audio_sec <= chunk_sec:
        return recognize_words(audio), audio_sec

    words = []
    offset_sec = 0.0
    while offset_sec < audio_sec:
        start_sample = int(offset_sec * sample_rate)
        end_sample = min(len(audio), int((offset_sec + chunk_sec) * sample_rate))
        chunk_audio = audio[start_sample:end_sample]
        is_final = end_sample >= len(audio)

        for word in recognize_words(chunk_audio):
            if not is_final and word["start"] >= stride_sec:
                continue
            adjusted = word.copy()
            adjusted["start"] += offset_sec
            adjusted["end"] += offset_sec
            words.append(adjusted)

        if is_final:
            break
        offset_sec += stride_sec

    return words, audio_sec


def transcribe(audio: np.ndarray, sample_rate: int, config: dict) -> list[dict]:
    """Transcribe audio using ONNX Parakeet."""
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim != 1:
        raise ValueError("audio must be a 1-D mono ndarray")

    model_version, device, _timeout_sec, quantization = _validate_config(config)
    resolved_device, resolved_quantization, _providers = _resolve_runtime(
        device, quantization
    )
    adapter = _get_adapter(model_version, device, quantization)

    started = time.perf_counter()
    words, audio_sec = _chunked_recognize(adapter, audio_array, sample_rate)
    if not words:
        return []

    acoustic_segments = [
        {
            "id": 1,
            "start": 0.0,
            "end": audio_sec,
            "text": "".join(word["word"] for word in words).strip(),
            "words": words,
        }
    ]
    statements = build_statements_from_acoustic(acoustic_segments)
    for statement in statements:
        statement["speaker"] = None

    elapsed = time.perf_counter() - started
    rtfx = audio_sec / max(elapsed, 0.001)
    logger.info(
        f"  Transcribed {len(statements)} statements, {audio_sec:.2f}s speech "
        f"in {elapsed:.2f}s (RTFx: {rtfx:.2f}) "
        f"[model={model_version} device={resolved_device} quant={resolved_quantization}]"
    )
    return statements


def get_model_info(config: dict) -> dict:
    """Return ONNX model metadata for transcript JSONL headers."""
    import onnxruntime

    model_version, device, _timeout_sec, quantization = _validate_config(config)
    resolved_device, resolved_quantization, providers = _resolve_runtime(
        device, quantization
    )
    return {
        "model": _MODEL_INFO_IDS[model_version],
        "device": resolved_device,
        "compute_type": resolved_quantization,
        "per_word_confidence": True,
        "onnxruntime_version": onnxruntime.__version__,
        "providers": providers,
    }
