# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.vad module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from observe.utils import SAMPLE_RATE
from observe.vad import (
    GAP_BUFFER,
    AudioReduction,
    SpeechSegment,
    VadResult,
    compute_nonspeech_rms,
    get_nonspeech_segments,
    reduce_audio,
    restore_statement_timestamps,
    run_vad,
)


class TestVadResult:
    """Test VadResult dataclass."""

    def test_vad_result_fields(self):
        """VadResult should have all expected fields."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            speech_segments=[(1.0, 3.0), (5.0, 8.0)],
        )

        assert result.duration == 10.0
        assert result.speech_duration == 5.0
        assert result.has_speech is True
        assert result.speech_segments == [(1.0, 3.0), (5.0, 8.0)]

    def test_vad_result_no_speech(self):
        """VadResult with no speech should have has_speech=False."""
        result = VadResult(
            duration=5.0,
            speech_duration=0.0,
            has_speech=False,
            speech_segments=[],
        )

        assert result.duration == 5.0
        assert result.speech_duration == 0.0
        assert result.has_speech is False
        assert result.speech_segments == []

    def test_vad_result_default_speech_segments(self):
        """VadResult speech_segments should default to empty list."""
        result = VadResult(
            duration=5.0,
            speech_duration=0.0,
            has_speech=False,
        )

        assert result.speech_segments == []

    def test_vad_result_rms_fields(self):
        """VadResult should have RMS fields with defaults."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
        )

        # Default values
        assert result.noisy_rms is None
        assert result.noisy_s == 0.0

    def test_vad_result_with_rms(self):
        """VadResult should accept RMS values."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            noisy_rms=0.015,
            noisy_s=3.5,
        )

        assert result.noisy_rms == 0.015
        assert result.noisy_s == 3.5

    def test_is_noisy_above_threshold(self):
        """is_noisy() should return True when RMS exceeds threshold."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            noisy_rms=0.015,  # Above default 0.01 threshold
        )

        assert result.is_noisy() is True

    def test_is_noisy_below_threshold(self):
        """is_noisy() should return False when RMS is below threshold."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            noisy_rms=0.005,  # Below default 0.01 threshold
        )

        assert result.is_noisy() is False

    def test_is_noisy_none_rms(self):
        """is_noisy() should return False when RMS is None."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            noisy_rms=None,
        )

        assert result.is_noisy() is False

    def test_is_noisy_custom_threshold(self):
        """is_noisy() should respect custom threshold."""
        result = VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            noisy_rms=0.015,
        )

        # With default threshold (0.01), should be noisy
        assert result.is_noisy() is True

        # With higher threshold (0.02), should not be noisy
        assert result.is_noisy(threshold=0.02) is False


class TestGetNonspeechSegments:
    """Test get_nonspeech_segments function."""

    def test_leading_silence(self):
        """Should detect leading silence before first speech."""
        speech_segments = [(2.0, 4.0)]
        nonspeech = get_nonspeech_segments(speech_segments, 5.0)

        assert (0.0, 2.0) in nonspeech

    def test_trailing_silence(self):
        """Should detect trailing silence after last speech."""
        speech_segments = [(1.0, 3.0)]
        nonspeech = get_nonspeech_segments(speech_segments, 5.0)

        assert (3.0, 5.0) in nonspeech

    def test_gap_between_segments(self):
        """Should detect gaps between speech segments."""
        speech_segments = [(1.0, 2.0), (4.0, 5.0)]
        nonspeech = get_nonspeech_segments(speech_segments, 6.0)

        assert (2.0, 4.0) in nonspeech

    def test_all_regions(self):
        """Should detect leading, middle, and trailing silence."""
        speech_segments = [(1.0, 2.0), (4.0, 5.0)]
        nonspeech = get_nonspeech_segments(speech_segments, 7.0)

        assert nonspeech == [(0.0, 1.0), (2.0, 4.0), (5.0, 7.0)]

    def test_no_speech_segments(self):
        """Should return empty list when no speech segments."""
        nonspeech = get_nonspeech_segments([], 5.0)

        assert nonspeech == []

    def test_speech_fills_entire_audio(self):
        """Should return empty list when speech fills entire audio."""
        speech_segments = [(0.0, 5.0)]
        nonspeech = get_nonspeech_segments(speech_segments, 5.0)

        assert nonspeech == []

    def test_adjacent_segments(self):
        """Should not create zero-length gaps between adjacent segments."""
        speech_segments = [(1.0, 2.0), (2.0, 3.0)]
        nonspeech = get_nonspeech_segments(speech_segments, 4.0)

        # Should only have leading and trailing, no gap between adjacent segments
        assert nonspeech == [(0.0, 1.0), (3.0, 4.0)]


class TestComputeNonspeechRms:
    """Test compute_nonspeech_rms function."""

    def test_silent_audio_returns_zero_rms(self):
        """Silent audio should have RMS near zero."""
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        speech_segments = [(1.0, 2.0)]  # Speech in middle

        rms, duration = compute_nonspeech_rms(audio, speech_segments, SAMPLE_RATE)

        assert rms is not None
        assert rms < 0.001  # Effectively zero

    def test_noisy_audio_returns_high_rms(self):
        """Noisy audio should have measurable RMS."""
        # Create audio with noise (amplitude 0.1)
        audio = np.random.uniform(-0.1, 0.1, 5 * SAMPLE_RATE).astype(np.float32)
        # Put "speech" in middle (doesn't affect RMS calculation of non-speech)
        speech_segments = [(2.0, 3.0)]

        rms, duration = compute_nonspeech_rms(audio, speech_segments, SAMPLE_RATE)

        assert rms is not None
        assert rms > 0.01  # Noisy threshold

    def test_returns_duration_used(self):
        """Should return total duration of non-speech segments used."""
        audio = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
        # Speech from 2-4s and 6-8s, leaving gaps at 0-2, 4-6, 8-10
        speech_segments = [(2.0, 4.0), (6.0, 8.0)]

        rms, duration = compute_nonspeech_rms(audio, speech_segments, SAMPLE_RATE)

        # All three gaps are >= 0.5s (MIN_NONSPEECH_SEGMENT)
        # Total non-speech: 2 + 2 + 2 = 6 seconds
        assert duration == 6.0

    def test_filters_short_segments(self):
        """Should filter out non-speech segments shorter than min_segment."""
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Speech leaves only 0.3s gaps (below default 0.5s threshold)
        speech_segments = [(0.3, 1.0), (1.3, 2.0), (2.3, 5.0)]

        rms, duration = compute_nonspeech_rms(audio, speech_segments, SAMPLE_RATE)

        # No qualifying segments
        assert rms is None
        assert duration == 0.0

    def test_no_speech_segments_returns_none(self):
        """Should return None when no speech segments (can't compute non-speech)."""
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)

        rms, duration = compute_nonspeech_rms(audio, [], SAMPLE_RATE)

        assert rms is None
        assert duration == 0.0

    def test_custom_min_segment(self):
        """Should respect custom min_segment threshold."""
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Speech from 1-2s, leaving 1s gap at start
        speech_segments = [(1.0, 2.0)]

        # With default 0.5s threshold, should include leading gap
        rms, duration = compute_nonspeech_rms(
            audio, speech_segments, SAMPLE_RATE, min_segment=0.5
        )
        assert duration == 4.0  # 1s leading + 3s trailing

        # With 2.0s threshold, should only include trailing gap (3s)
        rms, duration = compute_nonspeech_rms(
            audio, speech_segments, SAMPLE_RATE, min_segment=2.0
        )
        assert duration == 3.0


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
        # Speech segments should be converted to seconds
        assert result.speech_segments == [(1.0, 3.0)]

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

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_returns_rms_for_silent_background(self, mock_decode, mock_get_timestamps):
        """run_vad should return low RMS for silent non-speech regions."""
        # Silent audio (zeros)
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        # Speech from 1-3s, leaving non-speech at 0-1s and 3-5s
        mock_get_timestamps.return_value = [{"start": 16000, "end": 48000}]

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        assert result.noisy_rms is not None
        assert result.noisy_rms < 0.001  # Effectively zero
        assert result.noisy_s == 3.0  # 1s leading + 2s trailing

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_returns_rms_for_noisy_background(self, mock_decode, mock_get_timestamps):
        """run_vad should return measurable RMS for noisy non-speech regions."""
        # Noisy audio
        np.random.seed(42)
        mock_decode.return_value = np.random.uniform(-0.1, 0.1, 5 * SAMPLE_RATE).astype(
            np.float32
        )
        # Speech from 1-3s
        mock_get_timestamps.return_value = [{"start": 16000, "end": 48000}]

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=1.0)

        assert result.noisy_rms is not None
        assert result.noisy_rms > 0.01  # Noisy threshold
        assert result.noisy_s == 3.0

    @patch("faster_whisper.vad.get_speech_timestamps")
    @patch("faster_whisper.audio.decode_audio")
    def test_returns_none_rms_when_no_qualifying_segments(
        self, mock_decode, mock_get_timestamps
    ):
        """run_vad should return None RMS when no qualifying non-speech segments."""
        mock_decode.return_value = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
        # Speech fills most of audio, leaving only 0.2s gaps (below 0.5s threshold)
        mock_get_timestamps.return_value = [
            {"start": 3200, "end": 12800},  # 0.2s to 0.8s
            {"start": 16000, "end": 28800},  # 1.0s to 1.8s
        ]

        result = run_vad(Path("/fake/audio.flac"), min_speech_seconds=0.5)

        assert result.noisy_rms is None
        assert result.noisy_s == 0.0


class TestSpeechSegment:
    """Test SpeechSegment dataclass."""

    def test_speech_segment_fields(self):
        """SpeechSegment should have all expected fields."""
        seg = SpeechSegment(
            original_start=5.0,
            original_end=10.0,
            reduced_start=2.0,
            reduced_end=7.0,
        )

        assert seg.original_start == 5.0
        assert seg.original_end == 10.0
        assert seg.reduced_start == 2.0
        assert seg.reduced_end == 7.0


class TestAudioReduction:
    """Test AudioReduction dataclass and timestamp restoration."""

    def test_empty_reduction(self):
        """Empty reduction should return timestamp unchanged."""
        reduction = AudioReduction()
        assert reduction.restore_timestamp(5.0) == 5.0

    def test_single_segment_restoration(self):
        """Single segment should restore timestamps within segment."""
        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=3.0,
                    original_end=8.0,
                    reduced_start=0.0,
                    reduced_end=5.0,
                )
            ],
            original_duration=10.0,
            reduced_duration=5.0,
        )

        # Reduced time 0.0 -> original 3.0
        assert reduction.restore_timestamp(0.0) == 3.0

        # Reduced time 2.5 -> original 5.5 (midpoint)
        assert reduction.restore_timestamp(2.5) == 5.5

        # Reduced time 5.0 -> original 8.0
        assert reduction.restore_timestamp(5.0) == 8.0

    def test_multiple_segments_restoration(self):
        """Multiple segments should restore timestamps correctly."""
        # Simulates: original 10s audio with speech at [1-3] and [7-9]
        # with 4s gap trimmed to 2s, so reduced audio is 6s
        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=1.0,
                    original_end=3.0,
                    reduced_start=1.0,
                    reduced_end=3.0,
                ),
                SpeechSegment(
                    original_start=7.0,
                    original_end=9.0,
                    reduced_start=5.0,  # 3.0 + 2.0 gap
                    reduced_end=7.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=8.0,
        )

        # First segment: reduced 1.0 -> original 1.0
        assert reduction.restore_timestamp(1.0) == 1.0

        # First segment: reduced 2.0 -> original 2.0
        assert reduction.restore_timestamp(2.0) == 2.0

        # Second segment: reduced 5.0 -> original 7.0
        assert reduction.restore_timestamp(5.0) == 7.0

        # Second segment: reduced 6.0 -> original 8.0
        assert reduction.restore_timestamp(6.0) == 8.0

    def test_timestamp_in_gap(self):
        """Timestamp in reduced gap should map proportionally to original gap."""
        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=0.0,
                    original_end=2.0,
                    reduced_start=0.0,
                    reduced_end=2.0,
                ),
                SpeechSegment(
                    original_start=8.0,
                    original_end=10.0,
                    reduced_start=4.0,  # 2.0 + 2.0 reduced gap
                    reduced_end=6.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=6.0,
        )

        # Gap in reduced: 2.0-4.0 (2s), original: 2.0-8.0 (6s)
        # Reduced 3.0 is midpoint of gap -> original 5.0 (midpoint of 2-8)
        result = reduction.restore_timestamp(3.0)
        assert abs(result - 5.0) < 0.1  # Allow small tolerance

    def test_timestamp_after_all_segments(self):
        """Timestamp after all segments should extrapolate from last segment."""
        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=0.0,
                    original_end=5.0,
                    reduced_start=0.0,
                    reduced_end=5.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=6.0,
        )

        # Reduced 6.0 is 1.0 after last segment end -> original 6.0
        assert reduction.restore_timestamp(6.0) == 6.0

    def test_timestamp_before_first_segment(self):
        """Timestamp before first segment should map to leading buffer region."""
        # Simulates: original audio with 5s silence then speech at [5-10]
        # Leading 5s gap reduced to 1s buffer, so speech starts at reduced 1.0
        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=5.0,
                    original_end=10.0,
                    reduced_start=1.0,  # 1s buffer before speech
                    reduced_end=6.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=6.0,
        )

        # Reduced 0.0 is 1.0 before first segment start (5.0) -> original 4.0
        assert reduction.restore_timestamp(0.0) == 4.0

        # Reduced 0.5 is 0.5 before first segment start (5.0) -> original 4.5
        assert reduction.restore_timestamp(0.5) == 4.5

        # Reduced 1.0 is exactly at first segment start -> original 5.0
        assert reduction.restore_timestamp(1.0) == 5.0


class TestRestoreSegmentTimestamps:
    """Test restore_statement_timestamps function."""

    def test_restores_segment_timestamps(self):
        """Should restore segment start and end timestamps."""
        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 2, "start": 4.0, "end": 6.0, "text": "World"},
        ]

        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=0.0,
                    original_end=2.0,
                    reduced_start=0.0,
                    reduced_end=2.0,
                ),
                SpeechSegment(
                    original_start=6.0,
                    original_end=8.0,
                    reduced_start=4.0,
                    reduced_end=6.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=8.0,
        )

        restored = restore_statement_timestamps(segments, reduction)

        assert restored[0]["start"] == 0.0
        assert restored[0]["end"] == 2.0
        assert restored[1]["start"] == 6.0
        assert restored[1]["end"] == 8.0

    def test_restores_word_timestamps(self):
        """Should restore word-level timestamps."""
        segments = [
            {
                "id": 1,
                "start": 4.0,
                "end": 6.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 4.0, "end": 5.0, "probability": 0.9},
                    {"word": "world", "start": 5.0, "end": 6.0, "probability": 0.9},
                ],
            },
        ]

        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=8.0,
                    original_end=10.0,
                    reduced_start=4.0,
                    reduced_end=6.0,
                ),
            ],
            original_duration=12.0,
            reduced_duration=8.0,
        )

        restored = restore_statement_timestamps(segments, reduction)

        assert restored[0]["start"] == 8.0
        assert restored[0]["end"] == 10.0
        assert restored[0]["words"][0]["start"] == 8.0
        assert restored[0]["words"][0]["end"] == 9.0
        assert restored[0]["words"][1]["start"] == 9.0
        assert restored[0]["words"][1]["end"] == 10.0

    def test_preserves_other_fields(self):
        """Should preserve non-timestamp fields."""
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "text": "Hello",
                "custom_field": "preserved",
            },
        ]

        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=0.0,
                    original_end=2.0,
                    reduced_start=0.0,
                    reduced_end=2.0,
                ),
            ],
            original_duration=5.0,
            reduced_duration=2.0,
        )

        restored = restore_statement_timestamps(segments, reduction)

        assert restored[0]["text"] == "Hello"
        assert restored[0]["custom_field"] == "preserved"
        assert restored[0]["id"] == 1

    def test_handles_empty_segments(self):
        """Should handle empty segment list."""
        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=0.0,
                    original_end=5.0,
                    reduced_start=0.0,
                    reduced_end=5.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=5.0,
        )

        restored = restore_statement_timestamps([], reduction)
        assert restored == []

    def test_handles_segments_without_words(self):
        """Should handle segments without words field."""
        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello"}]

        reduction = AudioReduction(
            segments=[
                SpeechSegment(
                    original_start=5.0,
                    original_end=7.0,
                    reduced_start=0.0,
                    reduced_end=2.0,
                ),
            ],
            original_duration=10.0,
            reduced_duration=2.0,
        )

        restored = restore_statement_timestamps(segments, reduction)

        assert restored[0]["start"] == 5.0
        assert restored[0]["end"] == 7.0
        assert "words" not in restored[0]


class TestReduceAudio:
    """Test reduce_audio function."""

    @patch("faster_whisper.audio.decode_audio")
    def test_no_gaps_to_reduce(self, mock_decode):
        """Should return None when no gaps > 2s exist."""
        # 5s audio with speech from 0.5-1.5s and 2.0-3.0s (gap = 0.5s < 2s)
        mock_decode.return_value = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)

        vad_result = VadResult(
            duration=5.0,
            speech_duration=2.0,
            has_speech=True,
            speech_segments=[(0.5, 1.5), (2.0, 3.0)],
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert reduced_audio is None
        assert reduction is None

    @patch("faster_whisper.audio.decode_audio")
    def test_no_speech_segments(self, mock_decode):
        """Should return None when no speech segments."""
        vad_result = VadResult(
            duration=5.0,
            speech_duration=0.0,
            has_speech=False,
            speech_segments=[],
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert reduced_audio is None
        assert reduction is None

    @patch("faster_whisper.audio.decode_audio")
    def test_leading_gap_reduction(self, mock_decode):
        """Should trim leading gap > 2s to GAP_BUFFER."""
        # 10s audio with speech starting at 5s (leading gap = 5s > 2s)
        mock_decode.return_value = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)

        vad_result = VadResult(
            duration=10.0,
            speech_duration=3.0,
            has_speech=True,
            speech_segments=[(5.0, 8.0)],  # Speech from 5-8s
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert reduced_audio is not None
        assert reduction is not None

        # Should have: GAP_BUFFER (1s) + speech (3s) + trailing (2s) = 6s
        # But trailing is <= 2s, so kept in full
        expected_duration = GAP_BUFFER + 3.0 + 2.0  # 6s
        actual_duration = len(reduced_audio) / SAMPLE_RATE
        assert abs(actual_duration - expected_duration) < 0.1

        # Check mapping: speech should start at GAP_BUFFER in reduced audio
        assert len(reduction.segments) == 1
        assert reduction.segments[0].original_start == 5.0
        assert reduction.segments[0].reduced_start == GAP_BUFFER

    @patch("faster_whisper.audio.decode_audio")
    def test_trailing_gap_reduction(self, mock_decode):
        """Should trim trailing gap > 2s to GAP_BUFFER."""
        # 10s audio with speech from 1-3s (trailing gap = 7s > 2s)
        mock_decode.return_value = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)

        vad_result = VadResult(
            duration=10.0,
            speech_duration=2.0,
            has_speech=True,
            speech_segments=[(1.0, 3.0)],  # Speech from 1-3s
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert reduced_audio is not None
        assert reduction is not None

        # Should have: leading (1s) + speech (2s) + GAP_BUFFER (1s) = 4s
        expected_duration = 1.0 + 2.0 + GAP_BUFFER  # 4s
        actual_duration = len(reduced_audio) / SAMPLE_RATE
        assert abs(actual_duration - expected_duration) < 0.1

    @patch("faster_whisper.audio.decode_audio")
    def test_middle_gap_reduction(self, mock_decode):
        """Should trim middle gap > 2s to 2*GAP_BUFFER."""
        # 10s audio with speech at 0-2s and 7-9s (gap = 5s > 2s)
        mock_decode.return_value = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)

        vad_result = VadResult(
            duration=10.0,
            speech_duration=4.0,
            has_speech=True,
            speech_segments=[(0.0, 2.0), (7.0, 9.0)],
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert reduced_audio is not None
        assert reduction is not None

        # Should have: speech1 (2s) + trimmed gap (2s) + speech2 (2s) + trailing (1s) = 7s
        expected_duration = 2.0 + 2 * GAP_BUFFER + 2.0 + 1.0  # 7s
        actual_duration = len(reduced_audio) / SAMPLE_RATE
        assert abs(actual_duration - expected_duration) < 0.1

        # Check mapping
        assert len(reduction.segments) == 2
        assert reduction.segments[0].original_start == 0.0
        assert reduction.segments[0].reduced_start == 0.0
        assert reduction.segments[1].original_start == 7.0
        # Second segment should start at: speech1_end + trimmed_gap = 2.0 + 2.0 = 4.0
        assert abs(reduction.segments[1].reduced_start - 4.0) < 0.1

    @patch("faster_whisper.audio.decode_audio")
    def test_multiple_gaps_reduction(self, mock_decode):
        """Should trim multiple gaps > 2s."""
        # 20s audio with speech at 5-7, 12-14, and 19-20 (two big gaps)
        mock_decode.return_value = np.zeros(20 * SAMPLE_RATE, dtype=np.float32)

        vad_result = VadResult(
            duration=20.0,
            speech_duration=5.0,
            has_speech=True,
            speech_segments=[(5.0, 7.0), (12.0, 14.0), (19.0, 20.0)],
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert reduced_audio is not None
        assert reduction is not None

        # Should have:
        # - Leading: GAP_BUFFER (1s)
        # - Speech1: 2s
        # - Gap1 (5s->2s): 2*GAP_BUFFER = 2s
        # - Speech2: 2s
        # - Gap2 (5s->2s): 2*GAP_BUFFER = 2s
        # - Speech3: 1s
        # Total: 1 + 2 + 2 + 2 + 2 + 1 = 10s
        expected_duration = (
            GAP_BUFFER + 2.0 + 2 * GAP_BUFFER + 2.0 + 2 * GAP_BUFFER + 1.0
        )
        actual_duration = len(reduced_audio) / SAMPLE_RATE
        assert abs(actual_duration - expected_duration) < 0.1

        # Check we have 3 speech segments in mapping
        assert len(reduction.segments) == 3

    @patch("faster_whisper.audio.decode_audio")
    def test_returns_numpy_array(self, mock_decode):
        """Should return numpy array, not file path."""
        mock_decode.return_value = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)

        vad_result = VadResult(
            duration=10.0,
            speech_duration=2.0,
            has_speech=True,
            speech_segments=[(5.0, 7.0)],  # Leading gap > 2s
        )

        reduced_audio, reduction = reduce_audio(Path("/fake/audio.flac"), vad_result)

        assert isinstance(reduced_audio, np.ndarray)
        assert reduced_audio.dtype == np.float32
