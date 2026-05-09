# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared Parakeet owner-facing hint copy."""

PACKAGED_COREML_HINT = """solstone packaged installs on macOS don't include the CoreML transcription
helper — it's a Swift binary built from source. Whisper, the Gemini cloud
backend, and the macOS observer app continue to work without changes.

If you want CoreML-accelerated parakeet transcription, install solstone
from a source checkout: see https://github.com/solpbc/solstone/blob/main/CONTRIBUTING.md."""

PACKAGED_LINUX_PARAKEET_HINT = """solstone packaged installs on Linux don't include the parakeet-onnx
transcription stack — it ships as the `parakeet-onnx-cpu` (or
`parakeet-onnx-cuda` for NVIDIA GPUs) extra. Whisper, Gemini, OpenAI,
and Anthropic transcription continue to work without it.

To add local parakeet, reinstall solstone with the matching extra:
`uv tool install --reinstall 'solstone[parakeet-onnx-cpu]'` (or
`...[parakeet-onnx-cuda]`)."""
