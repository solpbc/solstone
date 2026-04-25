# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Parakeet backend dispatcher.

Routes to platform-specific implementations:
- Apple Silicon (darwin/arm64) -> _parakeet_coreml (FluidAudio helper)
- Linux x86_64 -> _parakeet_onnx (onnx-asr + onnxruntime)

Unsupported platforms raise RuntimeError with the detected platform
and the supported-platforms list.
Parakeet does not gate on internal silence; the upstream Silero VAD in
`observe/vad.py` (run by `_process_one`) is the gate and must remain in place.
"""

from __future__ import annotations

import platform
import sys

import numpy as np

from . import _parakeet_coreml, _parakeet_onnx


def _current_platform() -> tuple[str, str]:
    """Return normalized platform identifiers for dispatch."""
    os_name = "linux" if sys.platform.startswith("linux") else sys.platform
    return os_name, platform.machine().lower()


def _unsupported_platform_message(os_name: str, arch: str) -> str:
    """Build the unsupported-platform error message."""
    return (
        f"Unsupported Parakeet platform: {os_name}/{arch}. "
        "Supported platforms: darwin/arm64, linux/x86_64"
    )


def transcribe(audio: np.ndarray, sample_rate: int, config: dict) -> list[dict]:
    """Dispatch Parakeet transcription to the platform-specific backend."""
    os_name, arch = _current_platform()
    if os_name == "darwin" and arch == "arm64":
        return _parakeet_coreml.transcribe(audio, sample_rate, config)
    if os_name == "linux" and arch == "x86_64":
        return _parakeet_onnx.transcribe(audio, sample_rate, config)
    raise RuntimeError(_unsupported_platform_message(os_name, arch))


def get_model_info(config: dict) -> dict:
    """Dispatch Parakeet model-info lookup to the platform-specific backend."""
    os_name, arch = _current_platform()
    if os_name == "darwin" and arch == "arm64":
        return _parakeet_coreml.get_model_info(config)
    if os_name == "linux" and arch == "x86_64":
        return _parakeet_onnx.get_model_info(config)
    raise RuntimeError(_unsupported_platform_message(os_name, arch))
