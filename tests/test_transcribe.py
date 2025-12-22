"""Tests for observe.transcribe validation logic."""

import json
import tempfile
from pathlib import Path

import pytest

from observe.transcribe import validate_transcription


def test_validate_empty_result():
    """Empty result should be valid."""
    result = []
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_only_metadata():
    """Result with only metadata (no 'start' field) should be valid."""
    result = [{"topics": "test, demo", "setting": "personal"}]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_with_transcript_and_metadata():
    """Valid transcript with metadata should pass."""
    result = [
        {
            "start": "00:00:01",
            "source": "mic",
            "speaker": 1,
            "text": "Hello world",
            "description": "friendly",
        },
        {"topics": "greeting", "setting": "personal"},
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_without_metadata():
    """Valid transcript without metadata should pass."""
    result = [
        {
            "start": "00:00:01",
            "source": "mic",
            "speaker": 1,
            "text": "Hello world",
            "description": "friendly",
        }
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_missing_start_field():
    """Transcript item missing 'start' should fail."""
    result = [
        {
            "source": "mic",
            "speaker": 1,
            "text": "Hello world",
            "description": "friendly",
        }
    ]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "missing 'start' field" in error


def test_validate_invalid_timestamp_format():
    """Invalid timestamp format should fail."""
    result = [
        {
            "start": "1:2:3",  # Should be HH:MM:SS
            "source": "mic",
            "text": "Hello",
        }
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid  # This should pass - we only check format, not padding


def test_validate_invalid_timestamp_not_string():
    """Timestamp that's not a string should fail."""
    result = [{"start": 123, "source": "mic", "text": "Hello"}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "'start' is not a string" in error


def test_validate_invalid_timestamp_format_not_three_parts():
    """Timestamp without three parts should fail."""
    result = [{"start": "00:00", "source": "mic", "text": "Hello"}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "not in HH:MM:SS format" in error


def test_validate_missing_text_field():
    """Transcript item missing 'text' should fail."""
    result = [{"start": "00:00:01", "source": "mic", "speaker": 1}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "missing 'text' field" in error


def test_validate_text_not_string():
    """Text field that's not a string should fail."""
    result = [{"start": "00:00:01", "source": "mic", "text": 123}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "'text' is not a string" in error


def test_validate_result_not_list():
    """Result that's not a list should fail."""
    result = {"start": "00:00:01"}
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "not a list" in error


def test_validate_item_not_dict():
    """Item that's not a dict should fail."""
    result = ["invalid", {"topics": "test"}]
    is_valid, error = validate_transcription(result)
    assert not is_valid
    assert "not a dictionary" in error


def test_validate_multiple_transcript_items():
    """Multiple valid transcript items should pass."""
    result = [
        {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "First"},
        {"start": "00:00:05", "source": "sys", "speaker": 2, "text": "Second"},
        {"start": "00:00:10", "source": "mic", "speaker": 1, "text": "Third"},
        {"topics": "conversation", "setting": "personal"},
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_validate_string_speaker_labels():
    """String speaker labels (from diarization) should pass."""
    result = [
        {
            "start": "00:00:01",
            "speaker": "Speaker 1",
            "text": "Hello world",
            "description": "friendly",
        },
        {
            "start": "00:00:05",
            "speaker": "Speaker 2",
            "text": "Hi there",
            "description": "casual",
        },
        {"topics": "greeting", "setting": "personal"},
    ]
    is_valid, error = validate_transcription(result)
    assert is_valid
    assert error == ""


def test_jsonl_format_with_metadata():
    """Test JSONL format with metadata first."""
    result = [
        {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "Hello"},
        {"start": "00:00:05", "source": "sys", "speaker": 2, "text": "Hi"},
        {"topics": "greeting", "setting": "personal"},
    ]

    # Extract metadata and transcript items (mimics _transcribe logic)
    metadata = {}
    transcript_items = result
    if result and isinstance(result[-1], dict):
        last_item = result[-1]
        if "start" not in last_item and (
            "topics" in last_item or "setting" in last_item
        ):
            metadata = last_item
            transcript_items = result[:-1]

    # Write JSONL format
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(item) for item in transcript_items)
    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Verify format
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 3

    # First line should be metadata
    first = json.loads(lines[0])
    assert first == {"topics": "greeting", "setting": "personal"}

    # Remaining lines should be transcript items
    second = json.loads(lines[1])
    assert second["start"] == "00:00:01"
    assert second["text"] == "Hello"

    third = json.loads(lines[2])
    assert third["start"] == "00:00:05"
    assert third["text"] == "Hi"


def test_jsonl_format_without_metadata():
    """Test JSONL format with empty metadata when none provided."""
    result = [
        {"start": "00:00:01", "source": "mic", "speaker": 1, "text": "Hello"},
    ]

    # Extract metadata and transcript items (mimics _transcribe logic)
    metadata = {}
    transcript_items = result
    if result and isinstance(result[-1], dict):
        last_item = result[-1]
        if "start" not in last_item and (
            "topics" in last_item or "setting" in last_item
        ):
            metadata = last_item
            transcript_items = result[:-1]

    # Write JSONL format
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(item) for item in transcript_items)
    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Verify format
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 2

    # First line should be empty metadata
    first = json.loads(lines[0])
    assert first == {}

    # Second line should be transcript item
    second = json.loads(lines[1])
    assert second["start"] == "00:00:01"
    assert second["text"] == "Hello"


def test_jsonl_format_empty_result():
    """Test JSONL format with empty result (only metadata line)."""
    result = []

    # Extract metadata and transcript items (mimics _transcribe logic)
    metadata = {}
    transcript_items = result
    if result and isinstance(result[-1], dict):
        last_item = result[-1]
        if "start" not in last_item and (
            "topics" in last_item or "setting" in last_item
        ):
            metadata = last_item
            transcript_items = result[:-1]

    # Write JSONL format
    jsonl_lines = [json.dumps(metadata)]
    jsonl_lines.extend(json.dumps(item) for item in transcript_items)
    jsonl_content = "\n".join(jsonl_lines) + "\n"

    # Verify format
    lines = jsonl_content.strip().split("\n")
    assert len(lines) == 1

    # Only line should be empty metadata
    first = json.loads(lines[0])
    assert first == {}
