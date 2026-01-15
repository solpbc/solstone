# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.vad module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from observe.utils import SAMPLE_RATE
from observe.vad import VadResult, run_vad


class TestVadResult:
    """Test VadResult dataclass."""

    def test_vad_result_fields(self):
        """VadResult should have all expected fields."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
        )

        assert result.duration == 10.0
        assert result.speech_duration == 5.0
        assert result.has_speech is True

    def test_vad_result_no_speech(self):
        """VadResult with no speech should have has_speech=False."""
        result = VadResult(
            duration=5.0,
            speech_duration=0.0,
            has_speech=False,
        )

        assert result.duration == 5.0
        assert result.speech_duration == 0.0
        assert result.has_speech is False


class TestRunVad:
    """Test run_vad function."""

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_silent_audio_returns_no_speech(self, mock_decode, mock_get_timestamps):
        """Silent audio should return has_speech=False."""
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        mock_get_timestamps.return_value = []

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        assert result.duration == 5.0
        assert result.speech_duration == 0.0
        assert result.has_speech is False

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_speech_audio_returns_has_speech(self, mock_decode, mock_get_timestamps):
        """Audio with speech should return has_speech=True."""
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Mock: 2 seconds of speech (samples 16000-48000)
        mock_get_timestamps.return_value = [{"start": 16000, "end": 48000}]

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        assert result.duration == 5.0
        assert result.speech_duration == 2.0
        assert result.has_speech is True

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_speech_below_threshold(self, mock_decode, mock_get_timestamps):
        """Speech below threshold should return has_speech=False."""
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Mock: 0.5 seconds of speech (below 1.0s threshold)
        mock_get_timestamps.return_value = [{"start": 0, "end": 8000}]

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        assert result.duration == 5.0
        assert result.speech_duration == 0.5
        assert result.has_speech is False

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_custom_min_speech_threshold(self, mock_decode, mock_get_timestamps):
        """Custom min_speech_seconds threshold should be respected."""
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Mock: 0.5 seconds of speech
        mock_get_timestamps.return_value = [{"start": 0, "end": 8000}]

        # With 0.3s threshold, should have speech
        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=0.3)
        assert result.has_speech is True

        # With 1.0s threshold, should not have speech
        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)
        assert result.has_speech is False

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_multiple_speech_chunks(self, mock_decode, mock_get_timestamps):
        """Multiple speech chunks should be summed correctly."""
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Mock: Two 1-second speech segments
        mock_get_timestamps.return_value = [
            {"start": 16000, "end": 32000},  # 1 second
            {"start": 48000, "end": 64000},  # 1 second
        ]

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        assert result.duration == 5.0
        assert result.speech_duration == 2.0
        assert result.has_speech is True

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_calls_decode_audio_correctly(self, mock_decode, mock_get_timestamps):
        """run_vad should call decode_audio with correct parameters."""
        mock_decode.return_value = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
        mock_get_timestamps.return_value = [{"start": 0, "end": 32000}]

        run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        mock_decode.assert_called_once_with(
            "/fake/audio.flac", sampling_rate=SAMPLE_RATE
        )
