# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe.enrich module."""

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from observe.enrich import _load_prompt, _segment_to_flac_bytes


class TestLoadPrompt:
    """Test prompt loading."""

    def test_loads_prompt_from_file(self):
        """Should load the enrich.txt prompt."""
        prompt = _load_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "topics" in prompt.lower()
        assert "setting" in prompt.lower()
        assert "description" in prompt.lower()


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

    @patch("observe.enrich.gemini_generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_enrichment_data(self, mock_load, mock_generate):
        """Should return enrichment dict on success."""
        from observe.enrich import enrich_transcript

        # Mock audio loading
        sample_rate = 16000
        wav = np.zeros(sample_rate, dtype=np.float32)
        mock_load.return_value = (wav, sample_rate)

        # Mock Gemini response - now uses descriptions array
        mock_response = json.dumps(
            {
                "descriptions": ["calm tone", "excited voice"],
                "topics": ["testing", "software"],
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
        assert "descriptions" in result
        assert "topics" in result
        assert "setting" in result
        assert len(result["descriptions"]) == 2
        assert result["descriptions"] == ["calm tone", "excited voice"]
        assert result["topics"] == ["testing", "software"]
        assert result["setting"] == "workplace"

    @patch("observe.enrich.gemini_generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_none_on_api_error(self, mock_load, mock_generate):
        """Should return None if Gemini call fails."""
        from observe.enrich import enrich_transcript

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_generate.side_effect = Exception("API error")

        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]

        result = enrich_transcript(Path("/fake/audio.flac"), segments)

        assert result is None

    @patch("observe.enrich.gemini_generate")
    @patch("observe.enrich.librosa.load")
    def test_returns_none_on_invalid_response(self, mock_load, mock_generate):
        """Should return None if response missing required fields."""
        from observe.enrich import enrich_transcript

        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        # Missing 'descriptions' field
        mock_generate.return_value = json.dumps({"topics": ["test"]})

        segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]

        result = enrich_transcript(Path("/fake/audio.flac"), segments)

        assert result is None

    def test_returns_none_for_empty_segments(self):
        """Should return None for empty segment list."""
        from observe.enrich import enrich_transcript

        result = enrich_transcript(Path("/fake/audio.flac"), [])

        assert result is None

    @patch("observe.enrich.gemini_generate")
    @patch("observe.enrich.librosa.load")
    def test_builds_interleaved_content(self, mock_load, mock_generate):
        """Should send numbered text labels and audio clips interleaved."""
        from observe.enrich import enrich_transcript

        sample_rate = 16000
        wav = np.zeros(sample_rate * 10, dtype=np.float32)  # 10 seconds
        mock_load.return_value = (wav, sample_rate)

        mock_response = json.dumps(
            {
                "descriptions": ["neutral"],
                "topics": ["test"],
                "setting": "other",
            }
        )
        mock_generate.return_value = mock_response

        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello world."},
        ]

        enrich_transcript(Path("/fake/audio.flac"), segments)

        # Verify gemini_generate was called
        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs
        contents = call_kwargs.get("contents") or mock_generate.call_args.args[0]

        # Should have: prompt + (text label + audio clip) for each segment
        # = 1 + 2 * num_segments = 3 for 1 segment
        assert len(contents) == 3

        # First should be prompt text
        assert isinstance(contents[0], str)
        assert "descriptions" in contents[0].lower()

        # Second should be numbered text label
        assert "Segment 1:" in contents[1]
        assert "Hello world." in contents[1]

        # Third should be audio Part (check it's not a string)
        assert not isinstance(contents[2], str)
        # Check it has the Part's data attribute
        assert hasattr(contents[2], "inline_data") or hasattr(contents[2], "_pb")


class TestTranscriberIntegration:
    """Test enrichment integration with Transcriber."""

    def test_segments_to_jsonl_without_enrichment(self):
        """_segments_to_jsonl should work without enrichment."""
        import datetime

        from observe.transcribe import Transcriber

        # Create a minimal mock transcriber
        with patch.object(Transcriber, "__init__", lambda self: None):
            t = Transcriber()
            t.model_size = "medium.en"
            t.device = "cpu"
            t.compute_type = "int8"

            segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."}]
            base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)

            lines = t._segments_to_jsonl(segments, "audio.flac", base_dt)

            assert len(lines) == 2
            metadata = json.loads(lines[0])
            assert metadata["raw"] == "audio.flac"
            assert "topics" not in metadata
            assert "setting" not in metadata

            entry = json.loads(lines[1])
            assert entry["start"] == "14:30:00"
            assert entry["text"] == "Hello."
            assert "description" not in entry

    def test_segments_to_jsonl_with_enrichment(self):
        """_segments_to_jsonl should include enrichment data."""
        import datetime

        from observe.transcribe import Transcriber

        with patch.object(Transcriber, "__init__", lambda self: None):
            t = Transcriber()
            t.model_size = "medium.en"
            t.device = "cpu"
            t.compute_type = "int8"

            segments = [
                {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."},
                {"id": 2, "start": 5.0, "end": 7.0, "text": "World."},
            ]
            base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)

            # Enrichment now uses descriptions array (positional)
            enrichment = {
                "descriptions": ["friendly tone", "excited"],
                "topics": ["greetings", "testing"],
                "setting": "personal",
            }

            lines = t._segments_to_jsonl(
                segments, "audio.flac", base_dt, enrichment=enrichment
            )

            assert len(lines) == 3

            # Check metadata has topics and setting
            metadata = json.loads(lines[0])
            assert metadata["topics"] == ["greetings", "testing"]
            assert metadata["setting"] == "personal"

            # Check entries have descriptions (matched by position)
            entry1 = json.loads(lines[1])
            assert entry1["description"] == "friendly tone"

            entry2 = json.loads(lines[2])
            assert entry2["description"] == "excited"

    def test_segments_to_jsonl_partial_enrichment(self):
        """_segments_to_jsonl should handle partial enrichment."""
        import datetime

        from observe.transcribe import Transcriber

        with patch.object(Transcriber, "__init__", lambda self: None):
            t = Transcriber()
            t.model_size = "medium.en"
            t.device = "cpu"
            t.compute_type = "int8"

            segments = [
                {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello."},
                {"id": 2, "start": 5.0, "end": 7.0, "text": "World."},
            ]
            base_dt = datetime.datetime(2026, 1, 10, 14, 30, 0)

            # Enrichment only has description for first segment (fewer than segments)
            enrichment = {
                "descriptions": ["friendly tone"],  # Only one description
                "topics": ["test"],
                "setting": "other",
            }

            lines = t._segments_to_jsonl(
                segments, "audio.flac", base_dt, enrichment=enrichment
            )

            entry1 = json.loads(lines[1])
            assert "description" in entry1
            assert entry1["description"] == "friendly tone"

            entry2 = json.loads(lines[2])
            assert "description" not in entry2
