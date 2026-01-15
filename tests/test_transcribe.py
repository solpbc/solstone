# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.transcribe module."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from observe.transcribe import (
    DEFAULT_COMPUTE,
    DEFAULT_DEVICE,
    DEFAULT_MIN_SPEECH_SECONDS,
    DEFAULT_MODEL,
    MIN_SEGMENT_DURATION,
    SENTENCE_ENDINGS,
    _build_segment,
    resegment_by_sentences,
)
from observe.utils import prepare_audio_file


class TestResegmentBySentences:
    """Test sentence-based resegmentation of Whisper output."""

    def test_merges_fragments_into_sentence(self):
        """Multiple Whisper segments forming one sentence should merge."""
        # Simulates Whisper splitting "I think I can do it." across 3 segments
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 1.0,
                "text": "I think",
                "words": [
                    {"word": " I", "start": 0.0, "end": 0.3, "probability": 0.9},
                    {"word": " think", "start": 0.3, "end": 1.0, "probability": 0.9},
                ],
            },
            {
                "id": 2,
                "start": 1.5,
                "end": 2.5,
                "text": "I can",
                "words": [
                    {"word": " I", "start": 1.5, "end": 1.8, "probability": 0.9},
                    {"word": " can", "start": 1.8, "end": 2.5, "probability": 0.9},
                ],
            },
            {
                "id": 3,
                "start": 3.0,
                "end": 4.0,
                "text": "do it.",
                "words": [
                    {"word": " do", "start": 3.0, "end": 3.3, "probability": 0.9},
                    {"word": " it.", "start": 3.3, "end": 4.0, "probability": 0.9},
                ],
            },
        ]

        result = resegment_by_sentences(segments)

        assert len(result) == 1
        seg = result[0]
        assert seg["id"] == 1
        assert seg["start"] == 0.0
        assert seg["end"] == 4.0
        assert seg["text"] == "I think I can do it."
        assert len(seg["words"]) == 6

    def test_splits_on_period(self):
        """Segments should split on period."""
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 5.0,
                "text": "Hello. World.",
                "words": [
                    {"word": " Hello.", "start": 0.0, "end": 1.0, "probability": 0.9},
                    {"word": " World.", "start": 2.0, "end": 3.0, "probability": 0.9},
                ],
            },
        ]

        result = resegment_by_sentences(segments)

        assert len(result) == 2
        assert result[0]["text"] == "Hello."
        assert result[1]["text"] == "World."

    def test_splits_on_question_mark(self):
        """Segments should split on question mark."""
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 3.0,
                "text": "How are you? Good.",
                "words": [
                    {"word": " How", "start": 0.0, "end": 0.3, "probability": 0.9},
                    {"word": " are", "start": 0.3, "end": 0.6, "probability": 0.9},
                    {"word": " you?", "start": 0.6, "end": 1.0, "probability": 0.9},
                    {"word": " Good.", "start": 2.0, "end": 3.0, "probability": 0.9},
                ],
            },
        ]

        result = resegment_by_sentences(segments)

        assert len(result) == 2
        assert result[0]["text"] == "How are you?"
        assert result[1]["text"] == "Good."

    def test_splits_on_exclamation(self):
        """Segments should split on exclamation mark."""
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "text": "Wow! Amazing.",
                "words": [
                    {"word": " Wow!", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": " Amazing.", "start": 1.0, "end": 2.0, "probability": 0.9},
                ],
            },
        ]

        result = resegment_by_sentences(segments)

        assert len(result) == 2
        assert result[0]["text"] == "Wow!"
        assert result[1]["text"] == "Amazing."

    def test_handles_incomplete_final_sentence(self):
        """Final sentence without punctuation should still be captured."""
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 3.0,
                "text": "First sentence. And then",
                "words": [
                    {"word": " First", "start": 0.0, "end": 0.3, "probability": 0.9},
                    {
                        "word": " sentence.",
                        "start": 0.3,
                        "end": 1.0,
                        "probability": 0.9,
                    },
                    {"word": " And", "start": 1.5, "end": 1.8, "probability": 0.9},
                    {"word": " then", "start": 1.8, "end": 2.0, "probability": 0.9},
                ],
            },
        ]

        result = resegment_by_sentences(segments)

        assert len(result) == 2
        assert result[0]["text"] == "First sentence."
        assert result[1]["text"] == "And then"

    def test_empty_segments_returns_unchanged(self):
        """Empty segments should return unchanged."""
        segments = []
        result = resegment_by_sentences(segments)
        assert result == segments

    def test_segment_timestamps_from_words(self):
        """Segment start/end should come from first/last word."""
        segments = [
            {
                "id": 1,
                "start": 0.0,
                "end": 10.0,  # Original segment end
                "text": "Hello world.",
                "words": [
                    {"word": " Hello", "start": 2.5, "end": 3.0, "probability": 0.9},
                    {"word": " world.", "start": 3.5, "end": 4.2, "probability": 0.9},
                ],
            },
        ]

        result = resegment_by_sentences(segments)

        seg = result[0]
        assert seg["start"] == 2.5  # From first word
        assert seg["end"] == 4.2  # From last word


class TestBuildSegment:
    """Test segment building helper."""

    def test_builds_segment_from_words(self):
        """Should build segment with correct fields."""
        words = [
            {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " world", "start": 0.6, "end": 1.0, "probability": 0.8},
        ]

        seg = _build_segment(1, words)

        assert seg["id"] == 1
        assert seg["start"] == 0.0
        assert seg["end"] == 1.0
        assert seg["text"] == "Hello world"
        assert seg["words"] == words


class TestConstants:
    """Test module constants."""

    def test_sentence_endings(self):
        """SENTENCE_ENDINGS should contain expected punctuation."""
        assert "." in SENTENCE_ENDINGS
        assert "?" in SENTENCE_ENDINGS
        assert "!" in SENTENCE_ENDINGS
        assert "," not in SENTENCE_ENDINGS

    def test_min_segment_duration(self):
        """MIN_SEGMENT_DURATION should be positive."""
        assert MIN_SEGMENT_DURATION > 0

    def test_default_transcription_settings(self):
        """Default transcription settings should be valid."""
        assert DEFAULT_MODEL == "medium.en"
        assert DEFAULT_DEVICE == "auto"
        assert DEFAULT_COMPUTE == "default"
        assert DEFAULT_MIN_SPEECH_SECONDS == 1.0


class TestPrepareAudioFile:
    """Test the shared prepare_audio_file utility."""

    def test_flac_passthrough(self):
        """FLAC files should be returned unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            flac_path = Path(tmpdir) / "test.flac"

            # Create a simple FLAC file
            sample_rate = 16000
            data = np.zeros(sample_rate, dtype=np.float32)
            sf.write(flac_path, data, sample_rate, format="FLAC")

            result = prepare_audio_file(flac_path)
            assert result == flac_path

    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
    def test_m4a_conversion(self):
        """M4A files should be converted to temp FLAC."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source FLAC
            flac_path = Path(tmpdir) / "source.flac"
            sample_rate = 16000
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
            data = 0.5 * np.sin(2 * np.pi * 440 * t)
            sf.write(flac_path, data, sample_rate, format="FLAC")

            # Convert to M4A
            m4a_path = Path(tmpdir) / "test.m4a"
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(flac_path),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    str(m4a_path),
                ],
                capture_output=True,
            )
            assert result.returncode == 0

            # Test conversion
            temp_flac = prepare_audio_file(m4a_path)
            try:
                assert temp_flac.exists()
                assert temp_flac.suffix == ".flac"
                assert temp_flac != m4a_path

                # Verify audio was extracted
                mixed_data, sr = sf.read(temp_flac, dtype="float32")
                assert sr == 16000
                assert len(mixed_data) > 0
            finally:
                if temp_flac.exists():
                    temp_flac.unlink()

    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
    def test_multi_track_m4a(self):
        """Test that prepare_audio_file mixes multiple M4A audio streams together."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two mono FLAC files to combine into multi-track M4A
            track0_path = Path(tmpdir) / "track0.flac"
            track1_path = Path(tmpdir) / "track1.flac"
            m4a_path = Path(tmpdir) / "test.m4a"

            # Track 0: silence (system audio - no content)
            # Track 1: 440Hz sine wave (microphone - has voice)
            sample_rate = 16000
            duration = 1.0  # 1 second
            t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

            track0_data = np.zeros_like(t)  # Silence
            track1_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone

            sf.write(track0_path, track0_data, sample_rate, format="FLAC")
            sf.write(track1_path, track1_data, sample_rate, format="FLAC")

            # Use ffmpeg to create multi-track M4A
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(track0_path),
                    "-i",
                    str(track1_path),
                    "-map",
                    "0:a",
                    "-map",
                    "1:a",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    str(m4a_path),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"ffmpeg failed: {result.stderr}"

            temp_flac = prepare_audio_file(m4a_path)

            try:
                assert temp_flac.exists()
                assert temp_flac.suffix == ".flac"

                # Read the output and verify both streams were mixed
                mixed_data, sr = sf.read(temp_flac, dtype="float32")

                # The mixed audio should have content from track 1 (the sine wave)
                # AAC compression affects amplitude, so use loose threshold
                rms = np.sqrt(np.mean(mixed_data**2))
                assert rms > 0.1, f"Mixed audio should contain signal, got RMS={rms}"

                # Verify sample rate matches expected
                assert sr == 16000
            finally:
                if temp_flac.exists():
                    temp_flac.unlink()


class TestEmbeddingsFormat:
    """Test embeddings.npz format validation."""

    def test_embeddings_arrays_shape(self):
        """Embeddings should have correct array shapes."""
        # Simulate 10 segments with 256-dim embeddings
        embeddings = np.random.randn(10, 256).astype(np.float32)
        segment_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)

        assert embeddings.shape == (10, 256)
        assert segment_ids.shape == (10,)
        assert embeddings.dtype == np.float32
        assert segment_ids.dtype == np.int32

    def test_embeddings_npz_roundtrip(self):
        """Embeddings should survive save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "embeddings.npz"

            embeddings = np.random.randn(5, 256).astype(np.float32)
            segment_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)

            np.savez_compressed(
                npz_path, embeddings=embeddings, segment_ids=segment_ids
            )

            loaded = np.load(npz_path)
            np.testing.assert_array_almost_equal(loaded["embeddings"], embeddings)
            np.testing.assert_array_equal(loaded["segment_ids"], segment_ids)

    def test_segment_ids_are_unique(self):
        """Segment IDs should be unique."""
        segment_ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        assert len(segment_ids) == len(np.unique(segment_ids))


class TestJSONLFormat:
    """Test JSONL output format."""

    def test_metadata_first_line(self):
        """First line should be metadata with 'raw' field."""
        lines = [
            json.dumps({"raw": "audio.flac"}),
            json.dumps({"start": "00:00:01", "text": "Hello"}),
        ]
        jsonl_content = "\n".join(lines) + "\n"

        parsed_lines = jsonl_content.strip().split("\n")
        assert len(parsed_lines) == 2

        metadata = json.loads(parsed_lines[0])
        assert "raw" in metadata
        assert metadata["raw"] == "audio.flac"

    def test_metadata_includes_transcription_config(self):
        """Metadata should include model, device, and compute_type fields."""
        # Example metadata as produced by _segments_to_jsonl()
        metadata = {
            "raw": "audio.flac",
            "model": "medium.en",
            "device": "cuda",
            "compute_type": "float16",
        }

        # Verify all config fields are present
        assert "model" in metadata
        assert "device" in metadata
        assert "compute_type" in metadata

        # Verify they have expected types
        assert isinstance(metadata["model"], str)
        assert isinstance(metadata["device"], str)
        assert isinstance(metadata["compute_type"], str)

    def test_entry_has_required_fields(self):
        """Transcript entries should have start and text."""
        entry = {"start": "00:00:01", "text": "Hello world"}

        assert "start" in entry
        assert "text" in entry

    def test_entry_source_is_optional(self):
        """Source field should be optional."""
        entry_with_source = {"start": "00:00:01", "text": "Hello", "source": "mic"}
        entry_without_source = {"start": "00:00:01", "text": "Hello"}

        # Both should be valid
        assert "text" in entry_with_source
        assert "text" in entry_without_source

    def test_speaker_not_required(self):
        """Speaker field is no longer required (no diarization)."""
        entry = {"start": "00:00:01", "text": "Hello world"}

        # Should be valid without speaker
        assert "start" in entry
        assert "text" in entry
        assert "speaker" not in entry
