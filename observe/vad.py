# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Voice Activity Detection (VAD) module using Silero VAD.

Provides a standalone VAD stage that can be run before transcription to:
- Filter out silent/low-speech audio files early (before loading heavy STT models)
- Provide speech duration metrics for logging and events

Uses Silero VAD via faster-whisper's bundled implementation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from observe.utils import SAMPLE_RATE


@dataclass
class VadResult:
    """Result of Voice Activity Detection analysis.

    Attributes:
        duration: Total audio duration in seconds
        speech_duration: Duration of detected speech in seconds
        has_speech: Whether speech duration meets minimum threshold
    """

    duration: float
    speech_duration: float
    has_speech: bool


def run_vad(
    audio_path: Path,
    min_speech_seconds: float,
) -> VadResult:
    """Run Voice Activity Detection on an audio file.

    Loads the audio and runs Silero VAD to identify speech segments.
    This can be used to filter silent files before loading heavier
    transcription models.

    Args:
        audio_path: Path to audio file (any format supported by ffmpeg/PyAV)
        min_speech_seconds: Minimum speech duration to set has_speech=True

    Returns:
        VadResult with duration info and has_speech flag
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    logging.info(f"Running VAD on {audio_path.name}...")
    t0 = time.perf_counter()

    # Load audio at 16kHz mono (Silero VAD requirement)
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    # Run Silero VAD with default options
    # threshold=0.5 for speech start, min_silence_duration_ms=2000
    vad_options = VadOptions()
    speech_chunks = get_speech_timestamps(audio, vad_options, sampling_rate=SAMPLE_RATE)

    # Calculate speech duration from chunks
    speech_samples = sum(chunk["end"] - chunk["start"] for chunk in speech_chunks)
    speech_duration = speech_samples / SAMPLE_RATE

    has_speech = speech_duration >= min_speech_seconds

    vad_time = time.perf_counter() - t0
    logging.info(
        f"  VAD complete in {vad_time:.2f}s: "
        f"{duration:.1f}s total, {speech_duration:.1f}s speech, "
        f"{len(speech_chunks)} chunks, has_speech={has_speech}"
    )

    return VadResult(
        duration=duration,
        speech_duration=speech_duration,
        has_speech=has_speech,
    )
