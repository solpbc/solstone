# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.enrich module."""

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from observe.enrich import _statement_to_flac_bytes


class TestStatementToFlacBytes:
    """Test audio statement extraction and encoding."""

    def test_extracts_statement_to_flac(self):
        """Should extract statement and encode as FLAC bytes."""
        # Create 2 seconds of 440Hz sine wave
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        wav = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Extract 0.5s to 1.5s
        flac_bytes = _statement_to_flac_bytes(wav, 0.5, 1.5, sample_rate)

        # Should return non-empty bytes
        assert isinstance(flac_bytes, bytes)
        assert len(flac_bytes) > 0

        # Should be valid FLAC that can be decoded
        buf = io.BytesIO(flac_bytes)
        data, sr = sf.read(buf)
        assert sr == sample_rate
        # 1 second at 16kHz = 16000 samples
        assert len(data) == 16000

    def test_handles_statement_at_end(self):
        """Should handle statement near end of audio."""
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)  # 1 second

        # Extract last 0.5s
        flac_bytes = _statement_to_flac_bytes(wav, 0.5, 1.0, sample_rate)

        buf = io.BytesIO(flac_bytes)
        data, _ = sf.read(buf)
        assert len(data) == sample_rate // 2

    def test_handles_empty_statement(self):
        """Should handle zero-length statement."""
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)

        # Extract 0-length statement
        flac_bytes = _statement_to_flac_bytes(wav, 0.5, 0.5, sample_rate)

        # Should still return valid (empty) FLAC
        assert isinstance(flac_bytes, bytes)


class TestEnrichTranscript:
    """Test the main enrichment function."""

    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_enrichment_data(self, mock_load, mock_generate):
        """Should return enrichment dict on success."""
        from observe.enrich import enrich_transcript

        # Mock audio loading
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)
        mock_load.return_value = (wav, sample_rate)

        # Mock Gemini response with statements array
        mock_response = json.dumps(
            {
                "statements": [
                    {"corrected": "Hello world.", "emotion": "calm tone"},
                    {"corrected": "This is a test.", "emotion": "excited voice"},
                ],
                "topics": "testing, software",
                "setting": "workplace",
            }
        )
        mock_generate.return_value = mock_response

        statements = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello world."},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "This is a test."},
        ]

        result = enrich_transcript(Path("/fake/audio.flac"), statements)

        assert result is not None
        assert "statements" in result
        assert "topics" in result
        assert "setting" in result
        assert len(result["statements"]) == 2
        assert result["statements"][0]["corrected"] == "Hello world."
        assert result["statements"][0]["emotion"] == "calm tone"
        assert result["topics"] == "testing, software"
        assert result["setting"] == "workplace"

    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_none_on_api_error(self, mock_load, mock_generate):
        """Should return None if Gemini call fails."""
        from observe.enrich import enrich_transcript

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_generate.side_effect = Exception("API error")

        statements = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]

        result = enrich_transcript(Path("/fake/audio.flac"), statements)

        assert result is None

    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_none_on_invalid_response(self, mock_load, mock_generate):
        """Should return None if response missing required fields."""
        from observe.enrich import enrich_transcript

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        # Missing 'statements' field
        mock_generate.return_value = json.dumps({"topics": "test"})

        statements = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]

        result = enrich_transcript(Path("/fake/audio.flac"), statements)

        assert result is None

    def test_returns_none_for_empty_statements(self):
        """Should return None for empty statement list."""
        from observe.enrich import enrich_transcript

        result = enrich_transcript(Path("/fake/audio.flac"), [])

        assert result is None

    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_builds_interleaved_content(self, mock_load, mock_generate):
        """Should send numbered text labels and audio clips interleaved."""
        from observe.enrich import enrich_transcript

        sample_rate = 16000
        wav = np.zeros(sample_rate * 10, dtype=np.float32)  # 10 seconds
        mock_load.return_value = (wav, sample_rate)

        mock_response = json.dumps(
            {
                "statements": [{"corrected": "Hello world.", "emotion": "neutral"}],
                "topics": "test",
                "setting": "other",
            }
        )
        mock_generate.return_value = mock_response

        statements = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello world."},
        ]

        # Pass entity names explicitly (caller's responsibility now)
        enrich_transcript(
            Path("/fake/audio.flac"), statements, entity_names=["Alice", "Bob"]
        )

        # Verify generate was called
        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs
        contents = call_kwargs.get("contents") or mock_generate.call_args.args[0]

        # Should have: prompt + (text label + audio clip) for each statement
        # = 1 + 2 * num_statements = 3 for 1 statement
        assert len(contents) == 3

        # First should be prompt text
        assert isinstance(contents[0], str)
        assert "corrected" in contents[0].lower()
        assert "Alice, Bob" in contents[0]  # Entity names should be in prompt

        # Second should be numbered text label
        assert "Statement 1:" in contents[1]
        assert "Hello world." in contents[1]

        # Third should be audio Part (check it's not a string)
        assert not isinstance(contents[2], str)
        # Check it has the Part's data attribute
        assert hasattr(contents[2], "inline_data") or hasattr(contents[2], "_pb")


class TestStatementsToJsonl:
    """Test JSONL output formatting with enrichment."""

    def test_statements_to_jsonl_without_enrichment(self):
        """_statements_to_jsonl should work without enrichment."""
        import datetime

        from observe.transcribe.main import _statements_to_jsonl

        statements = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        lines = _statements_to_jsonl(statements, "audio.flac", base_dt, model_info)

        assert len(lines) == 2
        metadata = json.loads(lines[0])
        assert metadata["raw"] == "audio.flac"
        assert "topics" not in metadata
        assert "setting" not in metadata

        entry = json.loads(lines[1])
        assert entry["start"] == "14:30:00"
        assert entry["text"] == "Hello."
        assert "emotion" not in entry
        assert "corrected" not in entry

    def test_statements_to_jsonl_with_enrichment(self):
        """_statements_to_jsonl should include enrichment data."""
        import datetime

        from observe.transcribe.main import _statements_to_jsonl

        statements = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "World."},
        ]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        # Enrichment with statements array (corrected + emotion)
        enrichment = {
            "statements": [
                {"corrected": "Hello!", "emotion": "friendly tone"},
                {"corrected": "World.", "emotion": "excited"},
            ],
            "topics": "greetings, testing",
            "setting": "personal",
        }

        lines = _statements_to_jsonl(
            statements, "audio.flac", base_dt, model_info, enrichment=enrichment
        )

        assert len(lines) == 3

        # Check metadata has topics and setting
        metadata = json.loads(lines[0])
        assert metadata["topics"] == "greetings, testing"
        assert metadata["setting"] == "personal"

        # Check entries have corrected text and emotions
        entry1 = json.loads(lines[1])
        assert entry1["emotion"] == "friendly tone"
        assert entry1["corrected"] == "Hello!"  # Different from original
        assert entry1["text"] == "Hello."  # Original preserved

        entry2 = json.loads(lines[2])
        assert entry2["emotion"] == "excited"
        assert "corrected" not in entry2  # Same as original, not included

    def test_statements_to_jsonl_corrected_same_as_original(self):
        """_statements_to_jsonl should not include corrected if same as original."""
        import datetime

        from observe.transcribe.main import _statements_to_jsonl

        statements = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        # Corrected text same as original
        enrichment = {
            "statements": [{"corrected": "Hello.", "emotion": "calm"}],
            "topics": "test",
            "setting": "other",
        }

        lines = _statements_to_jsonl(
            statements, "audio.flac", base_dt, model_info, enrichment=enrichment
        )

        entry = json.loads(lines[1])
        assert entry["text"] == "Hello."
        assert "corrected" not in entry  # Not included since same as original
        assert entry["emotion"] == "calm"

    def test_statements_to_jsonl_partial_enrichment(self):
        """_statements_to_jsonl should handle partial enrichment."""
        import datetime

        from observe.transcribe.main import _statements_to_jsonl

        statements = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "World."},
        ]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        # Enrichment only has one statement (fewer than input statements)
        enrichment = {
            "statements": [{"corrected": "Hello!", "emotion": "friendly tone"}],
            "topics": "test",
            "setting": "other",
        }

        lines = _statements_to_jsonl(
            statements, "audio.flac", base_dt, model_info, enrichment=enrichment
        )

        entry1 = json.loads(lines[1])
        assert "emotion" in entry1
        assert entry1["emotion"] == "friendly tone"
        assert entry1["corrected"] == "Hello!"

        entry2 = json.loads(lines[2])
        assert "emotion" not in entry2
        assert "corrected" not in entry2


class TestFormatAudioCorrectedText:
    """Test that format_audio prefers corrected text."""

    def test_prefers_corrected_over_text(self):
        """format_audio should display corrected text when available."""
        from observe.hear import format_audio

        entries = [
            {"raw": "audio.flac"},
            {
                "start": "10:00:00",
                "text": "Hello wrold.",
                "corrected": "Hello world.",
                "emotion": "calm",
            },
        ]

        chunks, meta = format_audio(entries)

        assert len(chunks) == 1
        # Should use corrected text, not original
        assert "Hello world." in chunks[0]["markdown"]
        assert "Hello wrold." not in chunks[0]["markdown"]
        # Description should still be appended
        assert "(calm)" in chunks[0]["markdown"]

    def test_falls_back_to_text_without_corrected(self):
        """format_audio should use text when corrected is not present."""
        from observe.hear import format_audio

        entries = [
            {"raw": "audio.flac"},
            {"start": "10:00:00", "text": "Hello world.", "emotion": "calm"},
        ]

        chunks, meta = format_audio(entries)

        assert len(chunks) == 1
        assert "Hello world." in chunks[0]["markdown"]
        assert "(calm)" in chunks[0]["markdown"]
