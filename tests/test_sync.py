# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observe/sync.py - sync service for remote uploads."""

from unittest.mock import MagicMock, patch

import pytest
import requests


@pytest.fixture
def sync_journal(tmp_path):
    """Create a temporary journal structure for sync tests.

    Returns a dict with 'path' and 'day' keys.
    """
    from datetime import datetime

    journal = tmp_path / "journal"
    journal.mkdir()

    # Use today's date so get_pending_segments finds it within days_back
    day = datetime.now().strftime("%Y%m%d")
    day_dir = journal / day
    day_dir.mkdir()

    # Create segment with files under default stream
    segment = "120000_300"
    stream_dir = day_dir / "default"
    stream_dir.mkdir()
    segment_dir = stream_dir / segment
    segment_dir.mkdir()

    audio_file = segment_dir / "audio.flac"
    audio_file.write_bytes(b"audio data for testing")

    video_file = segment_dir / "screen.webm"
    video_file.write_bytes(b"video data for testing")

    # Create health directory
    health_dir = day_dir / "health"
    health_dir.mkdir()

    return {"path": journal, "day": day}


def test_compute_file_sha256(sync_journal):
    """Test SHA256 computation."""
    from observe.utils import compute_file_sha256

    journal = sync_journal["path"]
    day = sync_journal["day"]
    test_file = journal / day / "default" / "120000_300" / "audio.flac"
    sha = compute_file_sha256(test_file)

    # Just verify it's a valid SHA256 hex string
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_get_sync_state_path(sync_journal, monkeypatch):
    """Test sync state path generation."""
    from observe.sync import get_sync_state_path

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    path = get_sync_state_path(day)
    assert path == journal / day / "health" / "sync.jsonl"


def test_append_and_load_sync_state(sync_journal, monkeypatch):
    """Test appending and loading sync state records."""
    from observe.sync import append_sync_record, load_sync_state

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # Initially empty
    records = load_sync_state(day)
    assert records == []

    # Append a pending record
    record1 = {
        "ts": 1234567890000,
        "segment": "120000_300",
        "status": "pending",
        "files": [{"name": "audio.flac", "sha256": "abc123"}],
    }
    append_sync_record(day, record1)

    # Append a confirmed record
    record2 = {
        "ts": 1234567891000,
        "segment": "120000_300",
        "status": "confirmed",
    }
    append_sync_record(day, record2)

    # Load and verify
    records = load_sync_state(day)
    assert len(records) == 2
    assert records[0]["status"] == "pending"
    assert records[1]["status"] == "confirmed"


def test_get_pending_segments(sync_journal, monkeypatch):
    """Test scanning for pending segments."""
    from observe.sync import append_sync_record, get_pending_segments

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # Add pending segment
    append_sync_record(
        day,
        {
            "ts": 1234567890000,
            "segment": "120000_300",
            "status": "pending",
            "files": [{"name": "audio.flac", "sha256": "abc123"}],
        },
    )

    # Add another pending segment
    segment2_dir = journal / day / "default" / "130000_300"
    segment2_dir.mkdir(parents=True)
    append_sync_record(
        day,
        {
            "ts": 1234567890001,
            "segment": "130000_300",
            "status": "pending",
            "files": [{"name": "audio.flac", "sha256": "def456"}],
        },
    )

    # Add a confirmed segment (should not be returned)
    append_sync_record(
        day,
        {
            "ts": 1234567890002,
            "segment": "140000_300",
            "status": "pending",
            "files": [{"name": "audio.flac", "sha256": "ghi789"}],
        },
    )
    append_sync_record(
        day,
        {
            "ts": 1234567890003,
            "segment": "140000_300",
            "status": "confirmed",
        },
    )

    # Get pending
    pending = get_pending_segments(days_back=7)

    assert len(pending) == 2
    segments = {p.segment for p in pending}
    assert "120000_300" in segments
    assert "130000_300" in segments
    assert "140000_300" not in segments  # Already confirmed


def test_get_pending_segments_empty(sync_journal, monkeypatch):
    """Test scanning when no pending segments exist."""
    from observe.sync import get_pending_segments

    journal = sync_journal["path"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    pending = get_pending_segments(days_back=7)
    assert pending == []


class TestSyncService:
    """Tests for SyncService class."""

    @pytest.fixture
    def mock_remote_client(self):
        """Create a mock RemoteClient."""
        with patch("observe.sync.RemoteClient") as mock:
            client = MagicMock()
            client.session = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def mock_callosum(self):
        """Create a mock CallosumConnection."""
        with patch("observe.sync.CallosumConnection") as mock:
            conn = MagicMock()
            mock.return_value = conn
            yield conn

    def test_sync_service_init(self, sync_journal, monkeypatch):
        """Test SyncService initialization."""
        from observe.sync import SyncService

        journal = sync_journal["path"]
        monkeypatch.setenv("JOURNAL_PATH", str(journal))

        service = SyncService(
            remote_url="https://server/ingest/key",
            days_back=7,
        )

        assert service.remote_url == "https://server/ingest/key"
        assert service.days_back == 7

    def test_check_confirmation_success(
        self, sync_journal, monkeypatch, mock_remote_client, mock_callosum
    ):
        """Test successful sha256 confirmation check."""
        from observe.sync import SyncService

        journal = sync_journal["path"]
        day = sync_journal["day"]
        monkeypatch.setenv("JOURNAL_PATH", str(journal))

        service = SyncService("https://server/ingest/key")
        service._client = mock_remote_client

        # Mock segments endpoint response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key": "120000_300",
                "files": [
                    {"name": "audio.flac", "sha256": "abc123", "size": 100},
                    {"name": "screen.webm", "sha256": "def456", "size": 200},
                ],
            }
        ]
        mock_remote_client.session.get.return_value = mock_response

        # Check with matching sha256s
        result = service._check_confirmation(
            day,
            "120000_300",
            {"audio.flac": "abc123", "screen.webm": "def456"},
        )

        assert result is True
        mock_remote_client.session.get.assert_called_once()

    def test_check_confirmation_mismatch(
        self, sync_journal, monkeypatch, mock_remote_client, mock_callosum
    ):
        """Test sha256 mismatch returns False."""
        from observe.sync import SyncService

        journal = sync_journal["path"]
        day = sync_journal["day"]
        monkeypatch.setenv("JOURNAL_PATH", str(journal))

        service = SyncService("https://server/ingest/key")
        service._client = mock_remote_client

        # Mock response with wrong sha256
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "key": "120000_300",
                "files": [
                    {"name": "audio.flac", "sha256": "wrong_hash", "size": 100},
                ],
            }
        ]
        mock_remote_client.session.get.return_value = mock_response

        result = service._check_confirmation(
            day,
            "120000_300",
            {"audio.flac": "abc123"},
        )

        assert result is False

    def test_check_confirmation_segment_not_found(
        self, sync_journal, monkeypatch, mock_remote_client, mock_callosum
    ):
        """Test segment not in response returns False."""
        from observe.sync import SyncService

        journal = sync_journal["path"]
        day = sync_journal["day"]
        monkeypatch.setenv("JOURNAL_PATH", str(journal))

        service = SyncService("https://server/ingest/key")
        service._client = mock_remote_client

        # Mock empty response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_remote_client.session.get.return_value = mock_response

        result = service._check_confirmation(
            day,
            "120000_300",
            {"audio.flac": "abc123"},
        )

        assert result is False

    def test_cleanup_segment(self, sync_journal, monkeypatch):
        """Test segment cleanup deletes files."""
        from observe.sync import SyncService

        journal = sync_journal["path"]
        day = sync_journal["day"]
        monkeypatch.setenv("JOURNAL_PATH", str(journal))

        service = SyncService("https://server/ingest/key")

        segment_dir = journal / day / "default" / "120000_300"
        audio_file = segment_dir / "audio.flac"
        video_file = segment_dir / "screen.webm"

        # Verify files exist
        assert audio_file.exists()
        assert video_file.exists()

        # Cleanup
        service._cleanup_segment(segment_dir, [audio_file, video_file])

        # Files should be deleted
        assert not audio_file.exists()
        assert not video_file.exists()
        # Directory should be removed if empty
        assert not segment_dir.exists()

    def test_handle_observing_message(
        self, sync_journal, monkeypatch, mock_remote_client, mock_callosum
    ):
        """Test handling observe.observing message."""
        from observe.sync import SyncService, load_sync_state

        journal = sync_journal["path"]
        day = sync_journal["day"]
        monkeypatch.setenv("JOURNAL_PATH", str(journal))

        service = SyncService("https://server/ingest/key")
        service._callosum = mock_callosum

        # Simulate observing message with metadata
        message = {
            "tract": "observe",
            "event": "observing",
            "day": day,
            "segment": "120000_300",
            "files": ["audio.flac", "screen.webm"],
            "host": "testhost",
            "platform": "linux",
            "stream": "default",
            "meta": {"facet": "work"},
        }

        service._handle_message(message)

        # Check pending record was written with metadata
        records = load_sync_state(day)
        assert len(records) == 1
        assert records[0]["status"] == "pending"
        assert records[0]["segment"] == "120000_300"
        assert len(records[0]["files"]) == 2
        # Verify metadata was extracted and merged
        assert records[0]["meta"]["host"] == "testhost"
        assert records[0]["meta"]["platform"] == "linux"
        assert records[0]["meta"]["facet"] == "work"

        # Check segment was queued with metadata
        assert service._queue.qsize() == 1
        seg_info = service._queue.get_nowait()
        assert seg_info.meta["host"] == "testhost"
        assert seg_info.meta["facet"] == "work"


def test_sync_service_startup_with_pending(sync_journal, monkeypatch):
    """Test that startup loads pending segments into the queue with metadata."""
    from observe.sync import SyncService, append_sync_record

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # Add pending segment with metadata
    append_sync_record(
        day,
        {
            "ts": 1234567890000,
            "segment": "120000_300",
            "status": "pending",
            "files": [{"name": "audio.flac", "sha256": "abc123"}],
            "meta": {"host": "remote-host", "platform": "darwin"},
        },
    )

    with patch("observe.sync.RemoteClient"), patch("observe.sync.CallosumConnection"):
        service = SyncService("https://server/ingest/key")
        # Replace worker with no-op so thread exits immediately
        service._sync_worker = lambda: None
        service.start()

        # Pending segment should have been queued with metadata
        assert service._queue.qsize() == 1
        seg_info = service._queue.get_nowait()
        assert seg_info.segment == "120000_300"
        assert seg_info.day == day
        assert seg_info.meta["host"] == "remote-host"
        assert seg_info.meta["platform"] == "darwin"

        service.stop()


def test_process_segment_skips_upload_if_already_confirmed(sync_journal, monkeypatch):
    """Test that segment already on server is skipped without upload."""
    from observe.sync import SegmentInfo, SyncService

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # Create SegmentInfo
    seg_info = SegmentInfo(
        day=day,
        segment="120000_300",
        files=[
            {"name": "audio.flac", "sha256": "abc123"},
            {"name": "screen.webm", "sha256": "def456"},
        ],
        meta={"stream": "default"},
    )

    with patch("observe.sync.CallosumConnection") as mock_callosum_class:
        mock_callosum = MagicMock()
        mock_callosum_class.return_value = mock_callosum

        with patch("observe.sync.RemoteClient") as mock_client_class:
            mock_client = MagicMock()
            mock_session = MagicMock()

            # Simulate server already has the segment with matching SHA256
            server_response = MagicMock()
            server_response.status_code = 200
            server_response.json.return_value = [
                {
                    "key": "120000_300",
                    "files": [
                        {"name": "audio.flac", "sha256": "abc123"},
                        {"name": "screen.webm", "sha256": "def456"},
                    ],
                }
            ]
            mock_session.get.return_value = server_response
            mock_client.session = mock_session
            mock_client.upload_segment = MagicMock(return_value=True)
            mock_client_class.return_value = mock_client

            service = SyncService("https://server/ingest/key")

            # Call _process_segment directly (internal method)
            service._process_segment(seg_info)

            # Upload should NOT have been called (already confirmed)
            mock_client.upload_segment.assert_not_called()


def test_process_segment_uploads_if_not_on_server(sync_journal, monkeypatch):
    """Test that segment not on server is uploaded."""
    from observe.sync import SegmentInfo, SyncService

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    seg_info = SegmentInfo(
        day=day,
        segment="120000_300",
        files=[
            {"name": "audio.flac", "sha256": "abc123"},
        ],
        meta={"stream": "default"},
    )

    with patch("observe.sync.CallosumConnection") as mock_callosum_class:
        mock_callosum = MagicMock()
        mock_callosum_class.return_value = mock_callosum

        with patch("observe.sync.RemoteClient") as mock_client_class:
            mock_client = MagicMock()
            mock_session = MagicMock()

            # First call: server doesn't have segment (pre-check)
            # Second call: server has segment (post-upload confirm)
            responses = [
                MagicMock(status_code=200, json=MagicMock(return_value=[])),
                MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value=[
                            {
                                "key": "120000_300",
                                "files": [{"name": "audio.flac", "sha256": "abc123"}],
                            }
                        ]
                    ),
                ),
            ]
            mock_session.get.side_effect = responses
            mock_client.session = mock_session
            mock_client.upload_segment = MagicMock(return_value=True)
            mock_client_class.return_value = mock_client

            service = SyncService("https://server/ingest/key")
            service._process_segment(seg_info)

            # Upload SHOULD have been called
            mock_client.upload_segment.assert_called_once()


def test_process_segment_passes_metadata_to_upload(sync_journal, monkeypatch):
    """Test that metadata is passed through to upload_segment call."""
    from observe.sync import SegmentInfo, SyncService

    journal = sync_journal["path"]
    day = sync_journal["day"]
    monkeypatch.setenv("JOURNAL_PATH", str(journal))

    # Create SegmentInfo with metadata
    seg_info = SegmentInfo(
        day=day,
        segment="120000_300",
        files=[
            {"name": "audio.flac", "sha256": "abc123"},
        ],
        meta={
            "host": "laptop",
            "platform": "linux",
            "facet": "meetings",
            "stream": "default",
        },
    )

    with patch("observe.sync.CallosumConnection") as mock_callosum_class:
        mock_callosum = MagicMock()
        mock_callosum_class.return_value = mock_callosum

        with patch("observe.sync.RemoteClient") as mock_client_class:
            mock_client = MagicMock()
            mock_session = MagicMock()

            # First call: server doesn't have segment (pre-check)
            # Second call: server has segment (post-upload confirm)
            responses = [
                MagicMock(status_code=200, json=MagicMock(return_value=[])),
                MagicMock(
                    status_code=200,
                    json=MagicMock(
                        return_value=[
                            {
                                "key": "120000_300",
                                "files": [{"name": "audio.flac", "sha256": "abc123"}],
                            }
                        ]
                    ),
                ),
            ]
            mock_session.get.side_effect = responses
            mock_client.session = mock_session
            mock_client.upload_segment = MagicMock(return_value=True)
            mock_client_class.return_value = mock_client

            service = SyncService("https://server/ingest/key")
            service._process_segment(seg_info)

            # Verify upload was called with metadata
            mock_client.upload_segment.assert_called_once()
            call_kwargs = mock_client.upload_segment.call_args.kwargs
            assert call_kwargs["meta"] == {
                "host": "laptop",
                "platform": "linux",
                "facet": "meetings",
                "stream": "default",
            }


class TestCheckRemoteHealth:
    """Tests for check_remote_health() function."""

    def test_health_check_success(self):
        """Test successful health check returns True with connection info."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            success, message = check_remote_health(
                "http://server.local:8000/app/remote/ingest/abc12345xyz"
            )

            assert success is True
            assert "server.local:8000" in message
            assert "abc12345" in message  # Key prefix

    def test_health_check_invalid_key(self):
        """Test 401 response returns False with appropriate message."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_get.return_value = mock_response

            success, message = check_remote_health(
                "http://server.local:8000/app/remote/ingest/badkey"
            )

            assert success is False
            assert "401" in message or "Invalid key" in message

    def test_health_check_revoked_key(self):
        """Test 403 response returns False with error details."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.text = '{"error": "Remote revoked"}'
            mock_response.json.return_value = {"error": "Remote revoked"}
            mock_get.return_value = mock_response

            success, message = check_remote_health(
                "http://server.local:8000/app/remote/ingest/revokedkey"
            )

            assert success is False
            assert "403" in message or "revoked" in message.lower()

    def test_health_check_403_non_json_body(self):
        """Test 403 with non-JSON body doesn't crash."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.text = "Forbidden"
            mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
                "", "", 0
            )
            mock_get.return_value = mock_response

            success, message = check_remote_health(
                "http://server.local:8000/app/remote/ingest/badkey"
            )

            assert success is False
            assert "403" in message

    def test_health_check_connection_refused(self):
        """Test connection refused returns False with clear message."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )

            success, message = check_remote_health(
                "http://server.local:8000/app/remote/ingest/key123"
            )

            assert success is False
            assert "refused" in message.lower() or "connection" in message.lower()

    def test_health_check_timeout(self):
        """Test timeout returns False with timeout message."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("timed out")

            success, message = check_remote_health(
                "http://server.local:8000/app/remote/ingest/key123",
                timeout=5.0,
            )

            assert success is False
            assert "timeout" in message.lower()

    def test_health_check_host_not_found(self):
        """Test DNS failure returns False with host not found message."""
        from observe.sync import check_remote_health

        with patch("observe.sync.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Name or service not known"
            )

            success, message = check_remote_health(
                "http://nonexistent.invalid/app/remote/ingest/key123"
            )

            assert success is False
            assert "not found" in message.lower() or "connection" in message.lower()
