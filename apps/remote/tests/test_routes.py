# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for remote app routes."""

from __future__ import annotations

import io
import json


def test_api_list_empty(remote_env):
    """Test listing remotes when none exist."""
    env = remote_env()

    resp = env.client.get("/app/remote/api/list")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_create_remote(remote_env):
    """Test creating a new remote."""
    env = remote_env()

    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test-laptop"},
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()

    assert "key" in data
    assert len(data["key"]) > 32  # 256 bits = 43 base64 chars
    assert data["key_prefix"] == data["key"][:8]
    assert data["name"] == "test-laptop"
    assert "/app/remote/ingest/" in data["ingest_url"]


def test_api_create_requires_name(remote_env):
    """Test that creating a remote requires a name."""
    env = remote_env()

    # Missing name
    resp = env.client.post(
        "/app/remote/api/create",
        json={},
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Name is required" in resp.get_json()["error"]

    # Empty name
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "   "},
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_api_list_shows_created_remote(remote_env):
    """Test that created remotes appear in the list."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "my-remote"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    key_prefix = resp.get_json()["key_prefix"]

    # List should show it
    resp = env.client.get("/app/remote/api/list")
    assert resp.status_code == 200
    remotes = resp.get_json()

    assert len(remotes) == 1
    assert remotes[0]["key_prefix"] == key_prefix
    assert remotes[0]["name"] == "my-remote"
    assert remotes[0]["enabled"] is True
    assert remotes[0]["stats"]["segments_received"] == 0


def test_api_delete_remote(remote_env):
    """Test revoking a remote (soft-delete)."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "to-revoke"},
        content_type="application/json",
    )
    key_prefix = resp.get_json()["key_prefix"]

    # Revoke it
    resp = env.client.delete(f"/app/remote/api/{key_prefix}")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

    # List should still show it, but marked as revoked
    resp = env.client.get("/app/remote/api/list")
    remotes = resp.get_json()
    assert len(remotes) == 1
    assert remotes[0]["key_prefix"] == key_prefix
    assert remotes[0]["revoked"] is True
    assert remotes[0]["revoked_at"] is not None


def test_api_delete_nonexistent(remote_env):
    """Test deleting a nonexistent remote returns 404."""
    env = remote_env()

    resp = env.client.delete("/app/remote/api/nonexistent")
    assert resp.status_code == 404


def test_ingest_invalid_key(remote_env):
    """Test that ingest rejects invalid keys."""
    env = remote_env()

    resp = env.client.post(
        "/app/remote/ingest/invalid-key-12345",
        data={"day": "20250103", "segment": "120000_300"},
    )
    assert resp.status_code == 401
    assert "Invalid key" in resp.get_json()["error"]


def test_ingest_missing_segment(remote_env):
    """Test that ingest requires segment."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without segment
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "20250103"},
    )
    assert resp.status_code == 400
    assert "Missing segment" in resp.get_json()["error"]


def test_ingest_missing_day(remote_env):
    """Test that ingest requires day."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without day
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"segment": "120000_300"},
    )
    assert resp.status_code == 400
    assert "Missing day" in resp.get_json()["error"]


def test_ingest_invalid_segment_format(remote_env):
    """Test that ingest validates segment format."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Invalid segment format
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "20250103", "segment": "invalid"},
    )
    assert resp.status_code == 400
    assert "Invalid segment format" in resp.get_json()["error"]


def test_ingest_invalid_day_format(remote_env):
    """Test that ingest validates day format."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Invalid day format
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "2025-01-03", "segment": "120000_300"},
    )
    assert resp.status_code == 400
    assert "Invalid day format" in resp.get_json()["error"]


def test_ingest_no_files(remote_env):
    """Test that ingest requires files."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without files
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={"day": "20250103", "segment": "120000_300"},
    )
    assert resp.status_code == 400
    assert "No files uploaded" in resp.get_json()["error"]


def test_ingest_success(remote_env):
    """Test successful file ingest."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test-remote"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "test_audio.flac"),
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["files"] == ["test_audio.flac"]
    assert data["bytes"] == len(test_data)

    # Verify file was written (in segment directory)
    expected_file = env.journal / "20250103" / "120000_300" / "test_audio.flac"
    assert expected_file.exists()
    assert expected_file.read_bytes() == test_data


def test_ingest_updates_stats(remote_env):
    """Test that ingest updates remote stats."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "stats-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Check stats updated
    resp = env.client.get("/app/remote/api/list")
    remotes = resp.get_json()
    assert len(remotes) == 1
    assert remotes[0]["stats"]["segments_received"] == 1
    assert remotes[0]["stats"]["bytes_received"] == len(test_data)
    assert remotes[0]["last_segment"] == "120000_300"
    assert remotes[0]["last_seen"] is not None


def test_ingest_event_relay(remote_env):
    """Test event relay endpoint."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "event-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Send an event
    resp = env.client.post(
        f"/app/remote/ingest/{key}/event",
        json={"tract": "observe", "event": "status", "mode": "screencast"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_ingest_event_missing_tract(remote_env):
    """Test that event relay requires tract."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Missing tract
    resp = env.client.post(
        f"/app/remote/ingest/{key}/event",
        json={"event": "status"},
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Missing tract or event" in resp.get_json()["error"]


def test_ingest_revoked_key(remote_env):
    """Test that ingest rejects revoked keys."""
    env = remote_env()

    # Create and revoke a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "revoked-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    resp = env.client.delete(f"/app/remote/api/{key_prefix}")
    assert resp.status_code == 200

    # Try to upload - should fail
    test_data = b"test content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 403
    assert "Remote revoked" in resp.get_json()["error"]


def test_ingest_event_revoked_key(remote_env):
    """Test that event relay rejects revoked keys."""
    env = remote_env()

    # Create and revoke a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "revoked-event-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    resp = env.client.delete(f"/app/remote/api/{key_prefix}")
    assert resp.status_code == 200

    # Try to send event - should fail
    resp = env.client.post(
        f"/app/remote/ingest/{key}/event",
        json={"tract": "observe", "event": "status"},
        content_type="application/json",
    )
    assert resp.status_code == 403
    assert "Remote revoked" in resp.get_json()["error"]


def test_api_get_key(remote_env):
    """Test retrieving full key for a remote."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "key-test"},
        content_type="application/json",
    )
    create_data = resp.get_json()
    key = create_data["key"]
    key_prefix = create_data["key_prefix"]

    # Get the key
    resp = env.client.get(f"/app/remote/api/{key_prefix}/key")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["key"] == key
    assert data["name"] == "key-test"
    assert data["ingest_url"] == f"/app/remote/ingest/{key}"


def test_api_get_key_nonexistent(remote_env):
    """Test getting key for nonexistent remote returns 404."""
    env = remote_env()

    resp = env.client.get("/app/remote/api/nonexistent/key")
    assert resp.status_code == 404


# === Segment collision helper tests ===


def test_find_available_segment_no_conflict(remote_env):
    """Test find_available_segment returns original when no conflict."""
    from observe.utils import find_available_segment

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    result = find_available_segment(day_dir, "120000_300")
    assert result == "120000_300"


def test_find_available_segment_with_conflict(remote_env):
    """Test find_available_segment finds alternative when conflict exists."""
    from observe.utils import find_available_segment

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create conflicting segment directory
    (day_dir / "120000_300").mkdir()

    result = find_available_segment(day_dir, "120000_300")

    # Should find a different segment
    assert result is not None
    assert result != "120000_300"
    # Should be a valid segment format
    assert "_" in result
    time_part, dur_part = result.split("_")
    assert len(time_part) == 6
    assert dur_part.isdigit()


def test_find_available_segment_with_limited_attempts(remote_env):
    """Test find_available_segment respects max_attempts limit."""
    from observe.utils import find_available_segment

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create conflicting segment directory
    (day_dir / "120000_300").mkdir()

    # With max_attempts=0, should return None immediately (no attempts allowed)
    result = find_available_segment(day_dir, "120000_300", max_attempts=0)
    assert result is None


def test_save_to_failed_creates_directory(remote_env):
    """Test _save_to_failed creates failed directory structure."""
    from apps.remote.routes import _save_to_failed

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create mock file_data tuples: (submitted_filename, simple_filename, content, sha256)
    file_data = [
        ("120000_300_audio.flac", "audio.flac", b"audio data", "sha256_audio"),
        ("120000_300_screen.webm", "screen.webm", b"video data", "sha256_video"),
    ]

    failed_dir = _save_to_failed(day_dir, file_data, "120000_300")

    # Verify structure includes segment key
    assert failed_dir.exists()
    assert "remote/failed/120000_300/" in str(failed_dir)
    assert (failed_dir / "120000_300_audio.flac").exists()
    assert (failed_dir / "120000_300_screen.webm").exists()
    # Verify actual content was written
    assert (failed_dir / "120000_300_audio.flac").read_bytes() == b"audio data"
    assert (failed_dir / "120000_300_screen.webm").read_bytes() == b"video data"


# === Integration tests for collision handling ===


def test_ingest_collision_adjusts_segment(remote_env):
    """Test that ingest adjusts segment key on collision."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "collision-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Create a conflicting segment directory
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300").mkdir()
    (day_dir / "120000_300" / "audio.flac").write_bytes(b"existing")

    # Upload with same segment key
    test_data = b"new audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "collision"  # New status indicates adjustment

    # The segment key should have been adjusted, file is stripped of prefix
    saved_file = data["files"][0]
    assert saved_file == "audio.flac"

    # Verify both segments exist
    assert (day_dir / "120000_300" / "audio.flac").exists()  # Original
    # New one is in adjusted segment directory (not 120000_300)
    adjusted_segments = [
        d for d in day_dir.iterdir() if d.is_dir() and d.name != "120000_300"
    ]
    assert len(adjusted_segments) == 1
    assert (adjusted_segments[0] / "audio.flac").exists()


def test_ingest_no_collision_preserves_segment(remote_env):
    """Test that ingest preserves segment key when no collision."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "no-collision-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload without any conflicting segment directory
    test_data = b"audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["files"] == ["audio.flac"]  # Segment prefix stripped

    # Verify file saved in segment directory
    expected_file = env.journal / "20250103" / "120000_300" / "audio.flac"
    assert expected_file.exists()


def test_ingest_stats_use_adjusted_segment(remote_env):
    """Test that remote stats record the adjusted segment key."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "stats-adjust-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Create a conflicting segment directory
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300").mkdir()

    # Upload with same segment key
    test_data = b"new audio"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )

    assert resp.status_code == 200

    # Check stats - last_segment should be the adjusted one
    resp = env.client.get("/app/remote/api/list")
    remotes = resp.get_json()
    assert len(remotes) == 1
    last_segment = remotes[0]["last_segment"]
    assert last_segment is not None
    # It should be adjusted (not the original conflicting one)
    assert last_segment != "120000_300"
    # The adjusted segment directory should exist
    assert (day_dir / last_segment).exists()


# === Sync history tests ===


def test_ingest_creates_sync_history(remote_env):
    """Test that ingest creates sync history record."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "history-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    # Upload a file
    test_data = b"test audio content for history"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Check history file exists
    hist_path = (
        env.journal
        / "apps"
        / "remote"
        / "remotes"
        / key_prefix
        / "hist"
        / "20250103.jsonl"
    )
    assert hist_path.exists()

    # Load and verify history
    with open(hist_path) as f:
        record = json.loads(f.readline())

    assert record["segment"] == "120000_300"
    assert "segment_original" not in record  # No collision
    assert len(record["files"]) == 1

    file_rec = record["files"][0]
    assert file_rec["submitted"] == "120000_300_audio.flac"
    assert file_rec["written"] == "audio.flac"  # Segment prefix stripped
    assert file_rec["size"] == len(test_data)
    assert len(file_rec["sha256"]) == 64  # SHA256 hex length
    assert file_rec["inode"] > 0


def test_ingest_history_with_collision(remote_env):
    """Test that sync history records collision adjustment."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "collision-history-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    # Create conflicting segment directory
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300").mkdir()

    # Upload with same segment key
    test_data = b"new audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Load history
    hist_path = (
        env.journal
        / "apps"
        / "remote"
        / "remotes"
        / key_prefix
        / "hist"
        / "20250103.jsonl"
    )
    with open(hist_path) as f:
        record = json.loads(f.readline())

    # Should record original segment
    assert record["segment_original"] == "120000_300"
    assert record["segment"] != "120000_300"

    # File names should reflect stripping of segment prefix
    file_rec = record["files"][0]
    assert file_rec["submitted"] == "120000_300_audio.flac"
    assert file_rec["written"] == "audio.flac"  # Segment prefix stripped


def test_segments_endpoint_empty(remote_env):
    """Test segments endpoint returns empty for no uploads."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-empty-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Query segments - should be empty
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_segments_endpoint_invalid_key(remote_env):
    """Test segments endpoint rejects invalid key."""
    env = remote_env()

    resp = env.client.get("/app/remote/ingest/invalid-key/segments/20250103")
    assert resp.status_code == 401


def test_segments_endpoint_invalid_day(remote_env):
    """Test segments endpoint validates day format."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-day-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    resp = env.client.get(f"/app/remote/ingest/{key}/segments/2025-01-03")
    assert resp.status_code == 400
    assert "Invalid day format" in resp.get_json()["error"]


def test_segments_endpoint_lists_uploads(remote_env):
    """Test segments endpoint lists uploaded segments."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-list-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Query segments
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    assert resp.status_code == 200
    data = resp.get_json()

    assert len(data) == 1
    segment = data[0]
    assert segment["key"] == "120000_300"
    assert segment["observed"] is False  # Not yet processed
    assert "original_key" not in segment  # No collision
    assert len(segment["files"]) == 1

    file_info = segment["files"][0]
    assert file_info["name"] == "audio.flac"  # Segment prefix stripped
    assert file_info["size"] == len(test_data)
    assert len(file_info["sha256"]) == 64
    assert file_info["status"] == "present"
    assert (
        file_info["submitted_name"] == "120000_300_audio.flac"
    )  # Original name preserved


def test_segments_endpoint_shows_collision(remote_env):
    """Test segments endpoint shows collision info."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-collision-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Create conflicting segment directory
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300").mkdir()

    # Upload with collision
    test_data = b"new audio"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Query segments
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    data = resp.get_json()

    assert len(data) == 1
    segment = data[0]
    assert segment["key"] != "120000_300"
    assert segment["original_key"] == "120000_300"

    file_info = segment["files"][0]
    assert file_info["submitted_name"] == "120000_300_audio.flac"
    assert file_info["name"] == "audio.flac"  # Segment prefix stripped
    assert file_info["status"] == "present"


def test_segments_endpoint_missing_file(remote_env):
    """Test segments endpoint reports missing files."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-missing-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test audio"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Delete the file (now in segment directory with stripped name)
    (env.journal / "20250103" / "120000_300" / "audio.flac").unlink()

    # Query segments
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    data = resp.get_json()

    assert len(data) == 1
    file_info = data[0]["files"][0]
    assert file_info["status"] == "missing"


def test_segments_endpoint_relocated_file(remote_env):
    """Test segments endpoint detects relocated files by inode."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-relocate-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test audio for relocation"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Move the file to a different name (simulating some file reorganization)
    day_dir = env.journal / "20250103"
    segment_dir = day_dir / "120000_300"
    original_path = segment_dir / "audio.flac"
    new_path = segment_dir / "renamed_audio.flac"
    original_path.rename(new_path)

    # Query segments - should detect relocation by inode
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    data = resp.get_json()

    assert len(data) == 1
    file_info = data[0]["files"][0]
    assert file_info["status"] == "relocated"
    assert file_info["current_path"] == "120000_300/renamed_audio.flac"


def test_find_by_inode(remote_env):
    """Test _find_by_inode helper."""
    from apps.remote.routes import _find_by_inode

    env = remote_env()
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)

    # Create a file and get its inode
    test_file = day_dir / "test.txt"
    test_file.write_bytes(b"hello")
    inode = test_file.stat().st_ino

    # Should find it at original location
    found = _find_by_inode(day_dir, inode)
    assert found == test_file

    # Move to subdirectory
    subdir = day_dir / "subdir"
    subdir.mkdir()
    new_path = subdir / "renamed.txt"
    test_file.rename(new_path)

    # Should still find by inode
    found = _find_by_inode(day_dir, inode)
    assert found == new_path

    # Non-existent inode returns None
    found = _find_by_inode(day_dir, 999999999)
    assert found is None


def test_segments_endpoint_revoked_key(remote_env):
    """Test segments endpoint rejects revoked key."""
    env = remote_env()

    # Create and revoke a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-revoked-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    env.client.delete(f"/app/remote/api/{key_prefix}")

    # Query segments - should be rejected
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    assert resp.status_code == 403
    assert "Remote revoked" in resp.get_json()["error"]


def test_segments_endpoint_deduplicates_by_sha256(remote_env):
    """Test that duplicate file uploads are rejected (not duplicated on disk).

    With duplicate detection enabled, re-uploading the same content returns
    status='duplicate' and the segment is not written again.
    """
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "segments-dedup-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Upload a file
    test_data = b"test audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"

    # Upload the same file again (same content = same sha256)
    # With duplicate detection, this should be rejected
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "duplicate"

    # Query segments - should have only one segment (duplicate was rejected)
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    data = resp.get_json()

    # Should have 1 segment (duplicate rejected, not 2 segments)
    assert len(data) == 1
    assert data[0]["key"] == "120000_300"
    assert len(data[0]["files"]) == 1
    assert data[0]["files"][0]["status"] == "present"


def test_segments_endpoint_shows_observed_status(remote_env):
    """Test that segments endpoint includes observed status."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "observed-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    # Upload a file
    test_data = b"test audio content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "120000_300_audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Query segments - should show observed: false
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["observed"] is False

    # Manually add an observed record to simulate event handler
    hist_dir = env.journal / "apps" / "remote" / "remotes" / key_prefix / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / "20250103.jsonl"
    with open(hist_path, "a") as f:
        f.write('{"ts": 1704312345000, "type": "observed", "segment": "120000_300"}\n')

    # Query again - should now show observed: true
    resp = env.client.get(f"/app/remote/ingest/{key}/segments/20250103")
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]["observed"] is True


def test_api_list_includes_segments_observed_stat(remote_env):
    """Test that api_list includes segments_observed stat."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "stats-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key_prefix = data["key_prefix"]

    # Initially no segments_observed
    resp = env.client.get("/app/remote/api/list")
    data = resp.get_json()
    assert len(data) == 1
    assert "segments_observed" not in data[0]["stats"]

    # Manually add segments_observed stat
    remote_path = env.journal / "apps" / "remote" / "remotes" / f"{key_prefix}.json"
    with open(remote_path) as f:
        remote_data = json.load(f)
    remote_data["stats"]["segments_observed"] = 5
    with open(remote_path, "w") as f:
        json.dump(remote_data, f)

    # Should now show in list
    resp = env.client.get("/app/remote/api/list")
    data = resp.get_json()
    assert data[0]["stats"]["segments_observed"] == 5


# === Duplicate detection tests ===


def test_ingest_duplicate_segment_returns_duplicate_status(remote_env):
    """Test that re-submitting identical files returns duplicate status."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "duplicate-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # First upload
    test_data = b"test audio content for duplicate test"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    first_segment = data["segment"]

    # Second upload with identical content
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "duplicate"
    assert data["existing_segment"] == first_segment
    assert "message" in data


def test_ingest_duplicate_does_not_emit_event(remote_env, monkeypatch):
    """Test that duplicate submission does not emit observe.observing event."""
    from unittest.mock import MagicMock

    env = remote_env()

    # Mock emit
    import apps.remote.routes as routes_module

    emit_mock = MagicMock()
    monkeypatch.setattr(routes_module, "emit", emit_mock)

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "no-event-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    test_data = b"test audio for event test"

    # First upload - should emit
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200
    assert emit_mock.call_count == 1

    # Second upload - should NOT emit
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "duplicate"
    assert emit_mock.call_count == 1  # No new emit


def test_ingest_duplicate_increments_duplicates_rejected_stat(remote_env):
    """Test that duplicate submission increments duplicates_rejected stat."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "dup-stat-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    test_data = b"test audio for stat test"

    # First upload
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Check stats - no duplicates_rejected yet
    resp = env.client.get("/app/remote/api/list")
    stats = resp.get_json()[0]["stats"]
    assert stats.get("duplicates_rejected", 0) == 0

    # Submit duplicate
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "duplicate"

    # Check stats - should have 1 duplicate rejected
    resp = env.client.get("/app/remote/api/list")
    stats = resp.get_json()[0]["stats"]
    assert stats["duplicates_rejected"] == 1


def test_ingest_partial_duplicate_creates_new_segment(remote_env):
    """Test that partial duplicate (some files match) creates new segment."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "partial-dup-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    audio_data = b"test audio content"
    screen_data = b"test screen content"
    new_screen_data = b"different screen content"

    # First upload with audio and screen
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
        },
        content_type="multipart/form-data",
    )
    # Add files manually for multipart
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": [
                (io.BytesIO(audio_data), "audio.flac"),
                (io.BytesIO(screen_data), "screen.mp4"),
            ],
        },
    )
    assert resp.status_code == 200
    first_data = resp.get_json()
    assert first_data["status"] == "ok"
    first_segment = first_data["segment"]

    # Second upload with same audio but different screen
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": [
                (io.BytesIO(audio_data), "audio.flac"),
                (io.BytesIO(new_screen_data), "screen.mp4"),
            ],
        },
    )
    assert resp.status_code == 200
    second_data = resp.get_json()
    # Should be collision (new segment) not duplicate
    assert second_data["status"] in ("ok", "collision")
    # Should be a different segment (collision resolution)
    assert second_data["segment"] != first_segment


def test_ingest_partial_match_logged_in_history(remote_env):
    """Test that partial SHA256 matches are logged in history record."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "partial-log-test"},
        content_type="application/json",
    )
    data = resp.get_json()
    key = data["key"]
    key_prefix = data["key_prefix"]

    audio_data = b"test audio for partial log"

    # First upload
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(audio_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200

    # Second upload with same audio but new additional file
    new_data = b"brand new file"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": [
                (io.BytesIO(audio_data), "audio.flac"),
                (io.BytesIO(new_data), "new_file.txt"),
            ],
        },
    )
    assert resp.status_code == 200

    # Load history and check for partial_match_sha256s in latest record
    hist_path = (
        env.journal
        / "apps"
        / "remote"
        / "remotes"
        / key_prefix
        / "hist"
        / "20250103.jsonl"
    )
    with open(hist_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Should have 2 upload records
    upload_records = [r for r in records if "type" not in r]
    assert len(upload_records) == 2

    # The second record should have partial_match_sha256s
    assert "partial_match_sha256s" in upload_records[1]
    assert len(upload_records[1]["partial_match_sha256s"]) == 1


def test_ingest_returns_collision_status_when_adjusted(remote_env):
    """Test that collision resolution returns status='collision'."""
    env = remote_env()

    # Create a remote
    resp = env.client.post(
        "/app/remote/api/create",
        json={"name": "collision-status-test"},
        content_type="application/json",
    )
    key = resp.get_json()["key"]

    # Create existing segment directory
    day_dir = env.journal / "20250103"
    day_dir.mkdir(parents=True)
    (day_dir / "120000_300").mkdir()
    (day_dir / "120000_300" / "existing.txt").write_bytes(b"existing content")

    # Upload - will need collision resolution
    test_data = b"new content"
    resp = env.client.post(
        f"/app/remote/ingest/{key}",
        data={
            "day": "20250103",
            "segment": "120000_300",
            "files": (io.BytesIO(test_data), "audio.flac"),
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "collision"
    assert data["segment"] != "120000_300"  # Adjusted
