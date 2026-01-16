# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.enrich module."""

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from observe.enrich import _segment_to_flac_bytes


class TestSegmentToFlacBytes:
    """Test audio segment extraction and encoding."""

    def test_extracts_segment_to_flac(self):
        """Should extract segment and encode as FLAC bytes."""
        # Create 2 seconds of 440Hz sine wave
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        wav = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Extract 0.5s to 1.5s
        flac_bytes = _segment_to_flac_bytes(wav, 0.5, 1.5, sample_rate)

        # Should return non-empty bytes
        assert isinstance(flac_bytes, bytes)
        assert len(flac_bytes) > 0

        # Should be valid FLAC that can be decoded
        buf = io.BytesIO(flac_bytes)
        data, sr = sf.read(buf)
        assert sr == sample_rate
        # 1 second at 16kHz = 16000 samples
        assert len(data) == 16000

    def test_handles_segment_at_end(self):
        """Should handle segment near end of audio."""
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)  # 1 second

        # Extract last 0.5s
        flac_bytes = _segment_to_flac_bytes(wav, 0.5, 1.0, sample_rate)

        buf = io.BytesIO(flac_bytes)
        data, _ = sf.read(buf)
        assert len(data) == sample_rate // 2

    def test_handles_empty_segment(self):
        """Should handle zero-length segment."""
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)

        # Extract 0-length segment
        flac_bytes = _segment_to_flac_bytes(wav, 0.5, 0.5, sample_rate)

        # Should still return valid (empty) FLAC
        assert isinstance(flac_bytes, bytes)


class TestEnrichTranscript:
    """Test the main enrichment function."""

    @patch("observe.enrich._load_entity_names")
    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_enrichment_data(self, mock_load, mock_generate, mock_entities):
        """Should return enrichment dict on success."""
        from observe.enrich import enrich_transcript

        # Mock audio loading
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)
        mock_load.return_value = (wav, sample_rate)
        mock_entities.return_value = "Alice, Bob"

        # Mock Gemini response with segments array
        mock_response = json.dumps(
            {
                "segments": [
                    {"corrected": "Hello world.", "description": "calm tone"},
                    {"corrected": "This is a test.", "description": "excited voice"},
                ],
                "topics": "testing, software",
                "setting": "workplace",
            }
        )
        mock_generate.return_value = mock_response

        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello world."},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "This is a test."},
        ]

        result = enrich_transcript(Path("/fake/audio.flac"), segments)

        assert result is not None
        assert "segments" in result
        assert "topics" in result
        assert "setting" in result
        assert len(result["segments"]) == 2
        assert result["segments"][0]["corrected"] == "Hello world."
        assert result["segments"][0]["description"] == "calm tone"
        assert result["topics"] == "testing, software"
        assert result["setting"] == "workplace"

    @patch("observe.enrich._load_entity_names")
    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_none_on_api_error(self, mock_load, mock_generate, mock_entities):
        """Should return None if Gemini call fails."""
        from observe.enrich import enrich_transcript

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_generate.side_effect = Exception("API error")
        mock_entities.return_value = None

        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]

        result = enrich_transcript(Path("/fake/audio.flac"), segments)

        assert result is None

    @patch("observe.enrich._load_entity_names")
    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_none_on_invalid_response(
        self, mock_load, mock_generate, mock_entities
    ):
        """Should return None if response missing required fields."""
        from observe.enrich import enrich_transcript

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        # Missing 'segments' field
        mock_generate.return_value = json.dumps({"topics": "test"})
        mock_entities.return_value = None

        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]

        result = enrich_transcript(Path("/fake/audio.flac"), segments)

        assert result is None

    def test_returns_none_for_empty_segments(self):
        """Should return None for empty segment list."""
        from observe.enrich import enrich_transcript

        result = enrich_transcript(Path("/fake/audio.flac"), [])

        assert result is None

    @patch("observe.enrich._load_entity_names")
    @patch("observe.enrich.generate")
    @patch("observe.enrich.librosa.load")
    def test_builds_interleaved_content(self, mock_load, mock_generate, mock_entities):
        """Should send numbered text labels and audio clips interleaved."""
        from observe.enrich import enrich_transcript

        sample_rate = 16000
        wav = np.zeros(sample_rate * 10, dtype=np.float32)  # 10 seconds
        mock_load.return_value = (wav, sample_rate)
        mock_entities.return_value = "Alice, Bob"

        mock_response = json.dumps(
            {
                "segments": [{"corrected": "Hello world.", "description": "neutral"}],
                "topics": "test",
                "setting": "other",
            }
        )
        mock_generate.return_value = mock_response

        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello world."},
        ]

        enrich_transcript(Path("/fake/audio.flac"), segments)

        # Verify generate was called
        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs
        contents = call_kwargs.get("contents") or mock_generate.call_args.args[0]

        # Should have: prompt + (text label + audio clip) for each segment
        # = 1 + 2 * num_segments = 3 for 1 segment
        assert len(contents) == 3

        # First should be prompt text
        assert isinstance(contents[0], str)
        assert "corrected" in contents[0].lower()
        assert "Alice, Bob" in contents[0]  # Entity names should be in prompt

        # Second should be numbered text label
        assert "Segment 1:" in contents[1]
        assert "Hello world." in contents[1]

        # Third should be audio Part (check it's not a string)
        assert not isinstance(contents[2], str)
        # Check it has the Part's data attribute
        assert hasattr(contents[2], "inline_data") or hasattr(contents[2], "_pb")


class TestSegmentsToJsonl:
    """Test JSONL output formatting with enrichment."""

    def test_segments_to_jsonl_without_enrichment(self):
        """_segments_to_jsonl should work without enrichment."""
        import datetime

        from observe.transcribe.main import _segments_to_jsonl

        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        lines = _segments_to_jsonl(segments, "audio.flac", base_dt, model_info)

        assert len(lines) == 2
        metadata = json.loads(lines[0])
        assert metadata["raw"] == "audio.flac"
        assert "topics" not in metadata
        assert "setting" not in metadata

        entry = json.loads(lines[1])
        assert entry["start"] == "14:30:00"
        assert entry["text"] == "Hello."
        assert "description" not in entry
        assert "corrected" not in entry

    def test_segments_to_jsonl_with_enrichment(self):
        """_segments_to_jsonl should include enrichment data."""
        import datetime

        from observe.transcribe.main import _segments_to_jsonl

        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "World."},
        ]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        # Enrichment with segments array (corrected + description)
        enrichment = {
            "segments": [
                {"corrected": "Hello!", "description": "friendly tone"},
                {"corrected": "World.", "description": "excited"},
            ],
            "topics": "greetings, testing",
            "setting": "personal",
        }

        lines = _segments_to_jsonl(
            segments, "audio.flac", base_dt, model_info, enrichment=enrichment
        )

        assert len(lines) == 3

        # Check metadata has topics and setting
        metadata = json.loads(lines[0])
        assert metadata["topics"] == "greetings, testing"
        assert metadata["setting"] == "personal"

        # Check entries have corrected text and descriptions
        entry1 = json.loads(lines[1])
        assert entry1["description"] == "friendly tone"
        assert entry1["corrected"] == "Hello!"  # Different from original
        assert entry1["text"] == "Hello."  # Original preserved

        entry2 = json.loads(lines[2])
        assert entry2["description"] == "excited"
        assert "corrected" not in entry2  # Same as original, not included

    def test_segments_to_jsonl_corrected_same_as_original(self):
        """_segments_to_jsonl should not include corrected if same as original."""
        import datetime

        from observe.transcribe.main import _segments_to_jsonl

        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        # Corrected text same as original
        enrichment = {
            "segments": [{"corrected": "Hello.", "description": "calm"}],
            "topics": "test",
            "setting": "other",
        }

        lines = _segments_to_jsonl(
            segments, "audio.flac", base_dt, model_info, enrichment=enrichment
        )

        entry = json.loads(lines[1])
        assert entry["text"] == "Hello."
        assert "corrected" not in entry  # Not included since same as original
        assert entry["description"] == "calm"

    def test_segments_to_jsonl_partial_enrichment(self):
        """_segments_to_jsonl should handle partial enrichment."""
        import datetime

        from observe.transcribe.main import _segments_to_jsonl

        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "World."},
        ]
        base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}

        # Enrichment only has one segment (fewer than input segments)
        enrichment = {
            "segments": [{"corrected": "Hello!", "description": "friendly tone"}],
            "topics": "test",
            "setting": "other",
        }

        lines = _segments_to_jsonl(
            segments, "audio.flac", base_dt, model_info, enrichment=enrichment
        )

        entry1 = json.loads(lines[1])
        assert "description" in entry1
        assert entry1["description"] == "friendly tone"
        assert entry1["corrected"] == "Hello!"

        entry2 = json.loads(lines[2])
        assert "description" not in entry2
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
                "description": "calm",
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
            {"start": "10:00:00", "text": "Hello world.", "description": "calm"},
        ]

        chunks, meta = format_audio(entries)

        assert len(chunks) == 1
        assert "Hello world." in chunks[0]["markdown"]
        assert "(calm)" in chunks[0]["markdown"]
