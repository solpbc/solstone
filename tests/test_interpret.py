# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.interpret module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


class TestTranscriptFormat:
    """Test transcript.json format validation."""

    def test_transcript_has_required_fields(self):
        """Transcript should have info, raw, and segments fields."""
        transcript = {
            "info": {
                "language": "en",
                "probability": 0.99,
                "duration": 10.0,
                "transcribe_time": 1.5,
                "model": "medium.en",
            },
            "raw": "audio.flac",
            "segments": [],
        }

        assert "info" in transcript
        assert "raw" in transcript
        assert "segments" in transcript
        assert transcript["info"]["model"] == "medium.en"

    def test_segment_has_required_fields(self):
        """Each segment should have id, start, end, text, words."""
        segment = {
            "id": 0,
            "start": 0.0,
            "end": 2.5,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.8, "probability": 0.95},
                {"word": "world", "start": 0.9, "end": 1.2, "probability": 0.92},
            ],
        }

        assert "id" in segment
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
        assert len(segment["words"]) == 2

    def test_word_has_required_fields(self):
        """Each word should have word, start, end, probability."""
        word = {"word": "Hello", "start": 0.0, "end": 0.8, "probability": 0.95}

        assert "word" in word
        assert "start" in word
        assert "end" in word
        assert "probability" in word
        assert isinstance(word["start"], float)
        assert isinstance(word["end"], float)


class TestEmbeddingsFormat:
    """Test embeddings.npz format validation."""

    def test_embeddings_arrays_shape(self):
        """Embeddings should have correct array shapes."""
        # Simulate 10 words with 256-dim embeddings
        embeddings = np.random.randn(10, 256).astype(np.float32)
        starts = np.array(
            [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=np.float32
        )

        assert embeddings.shape == (10, 256)
        assert starts.shape == (10,)
        assert embeddings.dtype == np.float32
        assert starts.dtype == np.float32

    def test_embeddings_npz_roundtrip(self):
        """Embeddings should survive save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "embeddings.npz"

            embeddings = np.random.randn(5, 256).astype(np.float32)
            starts = np.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)

            np.savez_compressed(npz_path, embeddings=embeddings, starts=starts)

            loaded = np.load(npz_path)
            np.testing.assert_array_almost_equal(loaded["embeddings"], embeddings)
            np.testing.assert_array_almost_equal(loaded["starts"], starts)

    def test_starts_are_unique(self):
        """Start times should be unique (used as word IDs)."""
        starts = np.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        assert len(starts) == len(np.unique(starts))


class TestPrepareAudioFile:
    """Test the shared prepare_audio_file utility."""

    def test_flac_passthrough(self):
        """FLAC files should be returned unchanged."""
        from observe.utils import prepare_audio_file

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

        from observe.utils import prepare_audio_file

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


class TestInterpreterOutputDir:
    """Test Interpreter output directory logic."""

    def test_output_dir_structure(self):
        """Output should go to <segment>/<stem>/ folder."""
        from observe.interpret import Interpreter

        # Create interpreter without loading models
        with patch("faster_whisper.WhisperModel"), patch("resemblyzer.VoiceEncoder"):
            interpreter = Interpreter.__new__(Interpreter)
            interpreter.whisper_model = MagicMock()
            interpreter.voice_encoder = MagicMock()
            interpreter.model_size = "medium.en"

            audio_path = Path("/journal/20250101/120000_300/audio.flac")
            output_dir = interpreter._get_output_dir(audio_path)

            assert output_dir == Path("/journal/20250101/120000_300/audio")
            assert output_dir.parent == audio_path.parent
            assert output_dir.name == audio_path.stem


class TestMinWordDuration:
    """Test word duration filtering."""

    def test_short_words_filtered(self):
        """Words shorter than MIN_WORD_DURATION should be skipped."""
        from observe.interpret import MIN_WORD_DURATION

        words = [
            {"word": "a", "start": 0.0, "end": 0.05},  # Too short
            {"word": "Hello", "start": 0.1, "end": 0.5},  # OK
            {"word": "I", "start": 0.6, "end": 0.65},  # Too short
            {"word": "world", "start": 0.7, "end": 1.2},  # OK
        ]

        filtered = [w for w in words if w["end"] - w["start"] >= MIN_WORD_DURATION]
        assert len(filtered) == 2
        assert filtered[0]["word"] == "Hello"
        assert filtered[1]["word"] == "world"


class TestResegmentBySentences:
    """Test sentence-based resegmentation of Whisper output."""

    def test_merges_fragments_into_sentence(self):
        """Multiple Whisper segments forming one sentence should merge."""
        from observe.interpret import resegment_by_sentences

        # Simulates Whisper splitting "I think I can do it." across 3 segments
        transcript = {
            "info": {"model": "medium.en"},
            "raw": "audio.flac",
            "segments": [
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
            ],
        }

        result = resegment_by_sentences(transcript)

        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert seg["id"] == 1
        assert seg["start"] == 0.0
        assert seg["end"] == 4.0
        assert seg["text"] == "I think I can do it."
        assert len(seg["words"]) == 6

    def test_splits_on_period(self):
        """Segments should split on period."""
        from observe.interpret import resegment_by_sentences

        transcript = {
            "info": {},
            "raw": "audio.flac",
            "segments": [
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
            ],
        }

        result = resegment_by_sentences(transcript)

        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Hello."
        assert result["segments"][1]["text"] == "World."

    def test_splits_on_question_mark(self):
        """Segments should split on question mark."""
        from observe.interpret import resegment_by_sentences

        transcript = {
            "info": {},
            "raw": "audio.flac",
            "segments": [
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
            ],
        }

        result = resegment_by_sentences(transcript)

        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "How are you?"
        assert result["segments"][1]["text"] == "Good."

    def test_splits_on_exclamation(self):
        """Segments should split on exclamation mark."""
        from observe.interpret import resegment_by_sentences

        transcript = {
            "info": {},
            "raw": "audio.flac",
            "segments": [
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
            ],
        }

        result = resegment_by_sentences(transcript)

        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "Wow!"
        assert result["segments"][1]["text"] == "Amazing."

    def test_handles_incomplete_final_sentence(self):
        """Final sentence without punctuation should still be captured."""
        from observe.interpret import resegment_by_sentences

        transcript = {
            "info": {},
            "raw": "audio.flac",
            "segments": [
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
            ],
        }

        result = resegment_by_sentences(transcript)

        assert len(result["segments"]) == 2
        assert result["segments"][0]["text"] == "First sentence."
        assert result["segments"][1]["text"] == "And then"

    def test_preserves_info_and_raw(self):
        """Info and raw fields should be preserved."""
        from observe.interpret import resegment_by_sentences

        transcript = {
            "info": {"model": "medium.en", "duration": 10.0},
            "raw": "audio.flac",
            "segments": [
                {
                    "id": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Test.",
                    "words": [
                        {"word": " Test.", "start": 0.0, "end": 1.0, "probability": 0.9}
                    ],
                }
            ],
        }

        result = resegment_by_sentences(transcript)

        assert result["info"] == {"model": "medium.en", "duration": 10.0}
        assert result["raw"] == "audio.flac"

    def test_empty_segments_returns_unchanged(self):
        """Empty segments should return transcript unchanged."""
        from observe.interpret import resegment_by_sentences

        transcript = {"info": {}, "raw": "audio.flac", "segments": []}

        result = resegment_by_sentences(transcript)

        assert result == transcript

    def test_segment_timestamps_from_words(self):
        """Segment start/end should come from first/last word."""
        from observe.interpret import resegment_by_sentences

        transcript = {
            "info": {},
            "raw": "audio.flac",
            "segments": [
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
            ],
        }

        result = resegment_by_sentences(transcript)

        seg = result["segments"][0]
        assert seg["start"] == 2.5  # From first word
        assert seg["end"] == 4.2  # From last word
