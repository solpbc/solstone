# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the Rev.ai STT backend."""

import json

import pytest

from observe.transcribe.revai import convert_to_segments


class TestConvertToSegments:
    """Tests for convert_to_segments function."""

    def test_empty_json(self):
        """Empty JSON returns empty list."""
        assert convert_to_segments({}) == []
        assert convert_to_segments({"monologues": []}) == []

    def test_single_speaker_monologue(self):
        """Single speaker monologue produces one segment."""
        revai_json = {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {
                            "type": "text",
                            "value": "Hello",
                            "ts": 1.0,
                            "end_ts": 1.5,
                            "confidence": 0.95,
                        },
                        {"type": "punct", "value": " "},
                        {
                            "type": "text",
                            "value": "world",
                            "ts": 1.6,
                            "end_ts": 2.0,
                            "confidence": 0.98,
                        },
                        {"type": "punct", "value": "."},
                    ],
                }
            ]
        }

        segments = convert_to_segments(revai_json)

        assert len(segments) == 1
        seg = segments[0]
        assert seg["id"] == 1
        assert seg["start"] == 1.0
        assert seg["end"] == 2.0
        assert seg["text"] == "Hello world."
        assert seg["speaker"] == 1  # 0-indexed to 1-indexed
        assert seg["confidence"] == pytest.approx(0.965, rel=0.01)
        assert seg["words"] is not None
        assert len(seg["words"]) == 2

    def test_multiple_speakers(self):
        """Multiple speakers produce separate segments with correct speaker IDs."""
        revai_json = {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {
                            "type": "text",
                            "value": "Hi",
                            "ts": 0.0,
                            "end_ts": 0.5,
                            "confidence": 0.9,
                        },
                        {"type": "punct", "value": "."},
                    ],
                },
                {
                    "speaker": 1,
                    "elements": [
                        {
                            "type": "text",
                            "value": "Hello",
                            "ts": 1.0,
                            "end_ts": 1.5,
                            "confidence": 0.95,
                        },
                        {"type": "punct", "value": "."},
                    ],
                },
                {
                    "speaker": 0,
                    "elements": [
                        {
                            "type": "text",
                            "value": "Bye",
                            "ts": 2.0,
                            "end_ts": 2.5,
                            "confidence": 0.85,
                        },
                        {"type": "punct", "value": "."},
                    ],
                },
            ]
        }

        segments = convert_to_segments(revai_json)

        assert len(segments) == 3
        assert segments[0]["speaker"] == 1
        assert segments[0]["text"] == "Hi."
        assert segments[1]["speaker"] == 2
        assert segments[1]["text"] == "Hello."
        assert segments[2]["speaker"] == 1
        assert segments[2]["text"] == "Bye."

    def test_sequential_ids(self):
        """Segment IDs are sequential starting from 1."""
        revai_json = {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {"type": "text", "value": "A", "ts": 0.0, "end_ts": 0.5}
                    ],
                },
                {
                    "speaker": 1,
                    "elements": [
                        {"type": "text", "value": "B", "ts": 1.0, "end_ts": 1.5}
                    ],
                },
                {
                    "speaker": 2,
                    "elements": [
                        {"type": "text", "value": "C", "ts": 2.0, "end_ts": 2.5}
                    ],
                },
            ]
        }

        segments = convert_to_segments(revai_json)

        assert [s["id"] for s in segments] == [1, 2, 3]

    def test_word_data_preserved(self):
        """Word-level data is preserved in segments."""
        revai_json = {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {
                            "type": "text",
                            "value": "Test",
                            "ts": 0.0,
                            "end_ts": 0.5,
                            "confidence": 0.9,
                        },
                        {"type": "punct", "value": " "},
                        {
                            "type": "text",
                            "value": "data",
                            "ts": 0.6,
                            "end_ts": 1.0,
                            "confidence": 0.95,
                        },
                    ],
                }
            ]
        }

        segments = convert_to_segments(revai_json)

        assert len(segments) == 1
        words = segments[0]["words"]
        assert len(words) == 2
        assert words[0]["word"] == "Test"
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 0.5
        assert words[0]["probability"] == 0.9
        assert words[1]["word"] == "data"

    def test_missing_timestamps(self):
        """Elements without timestamps are handled gracefully."""
        revai_json = {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {"type": "text", "value": "No", "confidence": 0.9},
                        {"type": "text", "value": " timestamps"},
                    ],
                }
            ]
        }

        segments = convert_to_segments(revai_json)

        assert len(segments) == 1
        assert segments[0]["text"] == "No timestamps"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 0.0

    def test_empty_monologue_skipped(self):
        """Empty monologues are skipped."""
        revai_json = {
            "monologues": [
                {"speaker": 0, "elements": []},
                {
                    "speaker": 1,
                    "elements": [
                        {"type": "text", "value": "Hello", "ts": 0.0, "end_ts": 0.5}
                    ],
                },
            ]
        }

        segments = convert_to_segments(revai_json)

        assert len(segments) == 1
        assert segments[0]["speaker"] == 2

    def test_whitespace_only_skipped(self):
        """Monologues with only whitespace are skipped."""
        revai_json = {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {"type": "punct", "value": " "},
                        {"type": "punct", "value": "  "},
                    ],
                },
                {
                    "speaker": 1,
                    "elements": [
                        {"type": "text", "value": "Real", "ts": 0.0, "end_ts": 0.5}
                    ],
                },
            ]
        }

        segments = convert_to_segments(revai_json)

        # Whitespace-only monologue strips to empty and is skipped
        assert len(segments) == 1
        assert segments[0]["text"] == "Real"

    def test_fixture_data(self):
        """Test with actual fixture data from fixtures/revai.json."""
        from pathlib import Path

        fixture_path = Path(__file__).parent.parent / "fixtures" / "revai.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not found")

        with open(fixture_path) as f:
            revai_json = json.load(f)

        segments = convert_to_segments(revai_json)

        # Should have 2 monologues -> 2 segments
        assert len(segments) == 2

        # First segment from speaker 0 (becomes 1)
        assert segments[0]["speaker"] == 1
        assert "Okay" in segments[0]["text"]
        assert segments[0]["start"] == pytest.approx(0.395, rel=0.01)

        # Second segment from speaker 2 (becomes 3)
        assert segments[1]["speaker"] == 3
        assert "lunch" in segments[1]["text"]


class TestBackendRegistry:
    """Tests for backend registry integration."""

    def test_revai_registered(self):
        """Rev.ai backend is registered."""
        from observe.transcribe import BACKEND_REGISTRY

        assert "revai" in BACKEND_REGISTRY

    def test_get_backend(self):
        """Can get Rev.ai backend module."""
        from observe.transcribe import get_backend

        backend = get_backend("revai")
        assert hasattr(backend, "transcribe")
        assert hasattr(backend, "convert_to_segments")
        assert hasattr(backend, "get_model_info")


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_default_config(self):
        """Default config produces expected metadata."""
        from observe.transcribe.revai import get_model_info

        info = get_model_info({})

        assert info["model"] == "revai-fusion"
        assert info["device"] == "cloud"
        assert info["compute_type"] == "api"
        assert info["diarization"] == "premium"

    def test_custom_config(self):
        """Custom config is reflected in metadata."""
        from observe.transcribe.revai import get_model_info

        info = get_model_info({"model": "machine", "diarization_type": "standard"})

        assert info["model"] == "revai-machine"
        assert info["diarization"] == "standard"
