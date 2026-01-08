# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for remote app utilities."""

from __future__ import annotations

import json

import pytest

from apps.remote.utils import (
    append_history_record,
    find_remote_by_name,
    find_segment_by_sha256,
    get_hist_dir,
    get_remotes_dir,
    increment_stat,
    list_remotes,
    load_history,
    load_remote,
    save_remote,
)


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Create a temporary journal environment for storage tests."""
    from convey import state

    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setattr(state, "journal_root", str(journal))

    # Create remotes directory
    remotes_dir = journal / "apps" / "remote" / "remotes"
    remotes_dir.mkdir(parents=True)

    class Env:
        def __init__(self):
            self.journal = journal
            self.remotes_dir = remotes_dir

    return Env()


class TestRemoteStorage:
    """Tests for remote metadata storage."""

    def test_get_remotes_dir_creates_directory(self, storage_env):
        """get_remotes_dir creates and returns remotes directory."""
        result = get_remotes_dir()
        assert result.exists()
        assert result == storage_env.remotes_dir

    def test_save_and_load_remote(self, storage_env):
        """save_remote and load_remote work together."""
        remote = {
            "key": "testkey123456789",
            "name": "test-remote",
            "stats": {"segments_received": 0},
        }

        assert save_remote(remote) is True

        loaded = load_remote("testkey123456789")
        assert loaded is not None
        assert loaded["name"] == "test-remote"

    def test_load_remote_wrong_key(self, storage_env):
        """load_remote returns None for wrong key."""
        remote = {
            "key": "testkey123456789",
            "name": "test-remote",
            "stats": {},
        }
        save_remote(remote)

        # Same prefix but different key
        result = load_remote("testkey1xxxxxxxx")
        assert result is None

    def test_load_remote_not_found(self, storage_env):
        """load_remote returns None when remote doesn't exist."""
        result = load_remote("nonexistent12345")
        assert result is None

    def test_list_remotes_empty(self, storage_env):
        """list_remotes returns empty list when no remotes."""
        result = list_remotes()
        assert result == []

    def test_list_remotes_returns_all(self, storage_env):
        """list_remotes returns all registered remotes."""
        for i in range(3):
            save_remote(
                {
                    "key": f"remote{i}0123456789",
                    "name": f"remote-{i}",
                    "created_at": 1000 + i,
                    "stats": {},
                }
            )

        result = list_remotes()
        assert len(result) == 3
        # Sorted by created_at descending
        assert result[0]["name"] == "remote-2"
        assert result[1]["name"] == "remote-1"
        assert result[2]["name"] == "remote-0"

    def test_find_remote_by_name(self, storage_env):
        """find_remote_by_name finds existing remote."""
        save_remote(
            {
                "key": "findme123456789",
                "name": "find-me",
                "stats": {},
            }
        )

        result = find_remote_by_name("find-me")
        assert result is not None
        assert result["key"] == "findme123456789"

    def test_find_remote_by_name_not_found(self, storage_env):
        """find_remote_by_name returns None for unknown name."""
        result = find_remote_by_name("unknown")
        assert result is None


class TestHistoryStorage:
    """Tests for sync history storage."""

    def test_get_hist_dir_creates_directory(self, storage_env):
        """get_hist_dir creates history directory."""
        result = get_hist_dir("testkey1")
        assert result.exists()
        assert result == storage_env.remotes_dir / "testkey1" / "hist"

    def test_get_hist_dir_no_create(self, storage_env):
        """get_hist_dir with ensure_exists=False doesn't create."""
        result = get_hist_dir("nonexistent", ensure_exists=False)
        assert not result.exists()

    def test_append_history_record(self, storage_env):
        """append_history_record creates and appends to JSONL file."""
        append_history_record(
            "testkey1", "20250103", {"type": "upload", "segment": "120000_300"}
        )
        append_history_record(
            "testkey1", "20250103", {"type": "observed", "segment": "120000_300"}
        )

        hist_path = storage_env.remotes_dir / "testkey1" / "hist" / "20250103.jsonl"
        assert hist_path.exists()

        with open(hist_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "upload"
        assert json.loads(lines[1])["type"] == "observed"

    def test_load_history_empty(self, storage_env):
        """load_history returns empty list when no history."""
        result = load_history("testkey1", "20250103")
        assert result == []

    def test_load_history(self, storage_env):
        """load_history returns all records."""
        append_history_record("testkey1", "20250103", {"segment": "a"})
        append_history_record("testkey1", "20250103", {"segment": "b"})

        result = load_history("testkey1", "20250103")
        assert len(result) == 2
        assert result[0]["segment"] == "a"
        assert result[1]["segment"] == "b"


class TestIncrementStat:
    """Tests for stat increment."""

    def test_increment_stat_new_counter(self, storage_env):
        """increment_stat creates new counter."""
        save_remote(
            {
                "key": "testkey123456789",
                "name": "test",
                "stats": {},
            }
        )

        increment_stat("testkey1", "segments_observed")

        loaded = load_remote("testkey123456789")
        assert loaded["stats"]["segments_observed"] == 1

    def test_increment_stat_existing_counter(self, storage_env):
        """increment_stat increments existing counter."""
        save_remote(
            {
                "key": "testkey123456789",
                "name": "test",
                "stats": {"segments_observed": 5},
            }
        )

        increment_stat("testkey1", "segments_observed")

        loaded = load_remote("testkey123456789")
        assert loaded["stats"]["segments_observed"] == 6

    def test_increment_stat_missing_remote(self, storage_env):
        """increment_stat handles missing remote gracefully."""
        # Should not raise
        increment_stat("nonexistent", "segments_observed")


class TestFindSegmentBySha256:
    """Tests for find_segment_by_sha256."""

    def test_no_history_returns_no_match(self, storage_env):
        """Returns (None, empty set) when no history exists."""
        segment, matched = find_segment_by_sha256(
            "testkey1", "20250103", {"sha256_abc"}
        )
        assert segment is None
        assert matched == set()

    def test_full_match_returns_segment(self, storage_env):
        """Returns segment key when all SHA256s match."""
        # Create history with segment upload
        append_history_record(
            "testkey1",
            "20250103",
            {
                "segment": "120000_300",
                "files": [
                    {"sha256": "sha256_aaa", "written": "audio.flac"},
                    {"sha256": "sha256_bbb", "written": "screen.mp4"},
                ],
            },
        )

        segment, matched = find_segment_by_sha256(
            "testkey1", "20250103", {"sha256_aaa", "sha256_bbb"}
        )
        assert segment == "120000_300"
        assert matched == {"sha256_aaa", "sha256_bbb"}

    def test_partial_match_returns_matched_set(self, storage_env):
        """Returns (None, matched set) when only some SHA256s match."""
        append_history_record(
            "testkey1",
            "20250103",
            {
                "segment": "120000_300",
                "files": [
                    {"sha256": "sha256_aaa", "written": "audio.flac"},
                ],
            },
        )

        # Request includes one matching and one new
        segment, matched = find_segment_by_sha256(
            "testkey1", "20250103", {"sha256_aaa", "sha256_new"}
        )
        assert segment is None
        assert matched == {"sha256_aaa"}

    def test_no_match_returns_empty_set(self, storage_env):
        """Returns (None, empty set) when no SHA256s match."""
        append_history_record(
            "testkey1",
            "20250103",
            {
                "segment": "120000_300",
                "files": [
                    {"sha256": "sha256_aaa", "written": "audio.flac"},
                ],
            },
        )

        segment, matched = find_segment_by_sha256(
            "testkey1", "20250103", {"sha256_xxx", "sha256_yyy"}
        )
        assert segment is None
        assert matched == set()

    def test_skips_observed_records(self, storage_env):
        """Ignores records with type field (e.g., 'observed')."""
        # Upload record
        append_history_record(
            "testkey1",
            "20250103",
            {
                "segment": "120000_300",
                "files": [
                    {"sha256": "sha256_aaa", "written": "audio.flac"},
                ],
            },
        )
        # Observed record
        append_history_record(
            "testkey1",
            "20250103",
            {
                "type": "observed",
                "segment": "120000_300",
            },
        )

        segment, matched = find_segment_by_sha256(
            "testkey1", "20250103", {"sha256_aaa"}
        )
        assert segment == "120000_300"
        assert matched == {"sha256_aaa"}

    def test_subset_match_returns_segment(self, storage_env):
        """Returns segment when incoming is subset of existing files."""
        # Segment has 3 files
        append_history_record(
            "testkey1",
            "20250103",
            {
                "segment": "120000_300",
                "files": [
                    {"sha256": "sha256_aaa", "written": "audio.flac"},
                    {"sha256": "sha256_bbb", "written": "screen.mp4"},
                    {"sha256": "sha256_ccc", "written": "transcript.json"},
                ],
            },
        )

        # Request only 2 of the 3 files (subset)
        segment, matched = find_segment_by_sha256(
            "testkey1", "20250103", {"sha256_aaa", "sha256_bbb"}
        )
        assert segment == "120000_300"
        assert matched == {"sha256_aaa", "sha256_bbb"}
