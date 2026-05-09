# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared Parakeet owner-facing hint copy."""

PACKAGED_COREML_HINT = """solstone packaged installs ship the CoreML transcription helper on
Apple Silicon Macs running macOS 14 or newer. Your install does not
include it — likely because you're on an Intel Mac, on macOS 13 or
older, or pip selected the cross-platform fallback wheel.

Whisper, the Gemini cloud backend, and the macOS observer app continue
to work without the helper.

If you want CoreML-accelerated parakeet transcription, install solstone
from a source checkout: see https://github.com/solpbc/solstone/blob/main/CONTRIBUTING.md."""

PACKAGED_LINUX_PARAKEET_HINT = """solstone packaged installs on Linux don't include the parakeet-onnx
transcription stack — it ships as the `parakeet-onnx-cpu` (or
`parakeet-onnx-cuda` for NVIDIA GPUs) extra. Whisper, Gemini, OpenAI,
and Anthropic transcription continue to work without it.

To add local parakeet, reinstall solstone with the matching extra:
`uv tool install --reinstall 'solstone[parakeet-onnx-cpu]'` (or
`...[parakeet-onnx-cuda]`)."""
