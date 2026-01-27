# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for RemoteClient with mocked HTTP calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_session():
    """Create a mock requests session."""
    with patch("observe.sync.requests.Session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


def test_remote_client_init():
    """Test RemoteClient initialization."""
    from observe.sync import RemoteClient

    client = RemoteClient("https://server:5000/app/remote/ingest/abc123")

    assert client.remote_url == "https://server:5000/app/remote/ingest/abc123"


def test_upload_segment_success(mock_session, tmp_path):
    """Test successful file upload."""
    from observe.sync import RemoteClient

    # Create test files
    file1 = tmp_path / "audio.flac"
    file1.write_bytes(b"audio data")
    file2 = tmp_path / "video.webm"
    file2.write_bytes(b"video data")

    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "files": ["audio.flac", "video.webm"],
        "bytes": 20,
    }
    mock_session.post.return_value = mock_response

    client = RemoteClient("https://server/ingest/key")
    result = client.upload_segment("20250103", "120000_300", [file1, file2])

    assert result is True
    mock_session.post.assert_called_once()

    # Check the call arguments
    call_args = mock_session.post.call_args
    assert call_args[0][0] == "https://server/ingest/key"
    # Verify required fields (host/platform are also sent but vary by machine)
    data = call_args[1]["data"]
    assert data["day"] == "20250103"
    assert data["segment"] == "120000_300"
    assert "host" in data
    assert "platform" in data


def test_upload_segment_retry_on_failure(mock_session, tmp_path):
    """Test that upload retries on failure."""
    from observe.sync import RemoteClient

    # Create test file
    file1 = tmp_path / "audio.flac"
    file1.write_bytes(b"audio data")

    # Mock failure then success
    mock_failure = MagicMock()
    mock_failure.status_code = 500
    mock_failure.text = "Server error"

    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.json.return_value = {"files": ["audio.flac"], "bytes": 10}

    mock_session.post.side_effect = [mock_failure, mock_success]

    # Patch sleep to avoid delays
    with patch("observe.sync.time.sleep"):
        client = RemoteClient("https://server/ingest/key")
        result = client.upload_segment("20250103", "120000_300", [file1])

    assert result is True
    assert mock_session.post.call_count == 2


def test_upload_segment_all_retries_fail(mock_session, tmp_path):
    """Test that upload returns False after all retries fail."""
    from observe.sync import RETRY_BACKOFF, RemoteClient

    # Create test file
    file1 = tmp_path / "audio.flac"
    file1.write_bytes(b"audio data")

    # Mock all failures
    mock_failure = MagicMock()
    mock_failure.status_code = 500
    mock_failure.text = "Server error"
    mock_session.post.return_value = mock_failure

    # Patch sleep to avoid delays
    with patch("observe.sync.time.sleep"):
        client = RemoteClient("https://server/ingest/key")
        result = client.upload_segment("20250103", "120000_300", [file1])

    assert result is False
    assert mock_session.post.call_count == len(RETRY_BACKOFF)


def test_upload_segment_skips_missing_files(mock_session, tmp_path):
    """Test that upload skips missing files."""
    from observe.sync import RemoteClient

    # Create one existing file
    file1 = tmp_path / "exists.flac"
    file1.write_bytes(b"data")

    # Reference a missing file
    file2 = tmp_path / "missing.flac"

    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"files": ["exists.flac"], "bytes": 4}
    mock_session.post.return_value = mock_response

    client = RemoteClient("https://server/ingest/key")
    result = client.upload_segment("20250103", "120000_300", [file1, file2])

    assert result is True


def test_upload_segment_fails_if_all_missing(mock_session, tmp_path):
    """Test that upload fails if all files are missing."""
    from observe.sync import RemoteClient

    # Reference missing files
    file1 = tmp_path / "missing1.flac"
    file2 = tmp_path / "missing2.flac"

    client = RemoteClient("https://server/ingest/key")
    result = client.upload_segment("20250103", "120000_300", [file1, file2])

    assert result is False
    mock_session.post.assert_not_called()


def test_upload_segment_empty_list(mock_session):
    """Test that upload fails with empty file list."""
    from observe.sync import RemoteClient

    client = RemoteClient("https://server/ingest/key")
    result = client.upload_segment("20250103", "120000_300", [])

    assert result is False
    mock_session.post.assert_not_called()
