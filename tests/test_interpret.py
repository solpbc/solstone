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
