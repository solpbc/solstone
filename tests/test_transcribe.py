"""Tests for observe.transcribe validation logic."""

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
