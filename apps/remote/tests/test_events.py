# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for remote app event handlers."""

from __future__ import annotations

import json

import pytest

from apps.events import EventContext
from apps.remote.events import handle_observed


@pytest.fixture
def remote_journal(tmp_path, monkeypatch):
    """Create a temporary journal with a remote registered."""
    from convey import state

    journal = tmp_path / "journal"
    journal.mkdir()

    # Set JOURNAL_PATH env var and convey state
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setattr(state, "journal_root", str(journal))

    # Create remotes directory
    remotes_dir = journal / "apps" / "remote" / "remotes"
    remotes_dir.mkdir(parents=True)

    # Create a test remote
    remote_data = {
        "key": "testkey123456789abcdef",
        "name": "test-remote",
        "created_at": 1704312000000,
        "last_seen": None,
        "last_segment": None,
        "enabled": True,
        "stats": {
            "segments_received": 5,
            "bytes_received": 1024,
        },
    }
    remote_path = remotes_dir / "testkey1.json"
    with open(remote_path, "w") as f:
        json.dump(remote_data, f)

    class Env:
        def __init__(self):
            self.journal = journal
            self.remotes_dir = remotes_dir
            self.remote_path = remote_path

    return Env()


class TestHandleObserved:
    """Tests for handle_observed event handler."""

    def test_records_observed_for_remote(self, remote_journal):
        """Handler records observed status for remote segment."""
        ctx = EventContext(
            msg={
                "tract": "observe",
                "event": "observed",
                "remote": "test-remote",
                "segment": "120000_300",
                "day": "20250103",
            },
            app="remote",
            tract="observe",
            event="observed",
            journal_root=str(remote_journal.journal),
        )

        handle_observed(ctx)

        # Check history was written
        hist_path = remote_journal.remotes_dir / "testkey1" / "hist" / "20250103.jsonl"
        assert hist_path.exists()

        with open(hist_path) as f:
            record = json.loads(f.readline())

        assert record["type"] == "observed"
        assert record["segment"] == "120000_300"
        assert "ts" in record

        # Check stat was incremented
        with open(remote_journal.remote_path) as f:
            data = json.load(f)
        assert data["stats"]["segments_observed"] == 1

    def test_multiple_observed_events(self, remote_journal):
        """Handler appends multiple observed records."""
        for segment in ["120000_300", "130000_300", "140000_300"]:
            ctx = EventContext(
                msg={
                    "tract": "observe",
                    "event": "observed",
                    "remote": "test-remote",
                    "segment": segment,
                    "day": "20250103",
                },
                app="remote",
                tract="observe",
                event="observed",
                journal_root=str(remote_journal.journal),
            )
            handle_observed(ctx)

        # Check all records written
        hist_path = remote_journal.remotes_dir / "testkey1" / "hist" / "20250103.jsonl"
        with open(hist_path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0])["segment"] == "120000_300"
        assert json.loads(lines[1])["segment"] == "130000_300"
        assert json.loads(lines[2])["segment"] == "140000_300"

        # Check stat incremented 3 times
        with open(remote_journal.remote_path) as f:
            data = json.load(f)
        assert data["stats"]["segments_observed"] == 3

    def test_ignores_non_remote_events(self, remote_journal):
        """Handler ignores events without remote field."""
        ctx = EventContext(
            msg={
                "tract": "observe",
                "event": "observed",
                "segment": "120000_300",
                "day": "20250103",
            },
            app="remote",
            tract="observe",
            event="observed",
            journal_root=str(remote_journal.journal),
        )

        handle_observed(ctx)

        # No history should be created
        hist_dir = remote_journal.remotes_dir / "testkey1" / "hist"
        assert not hist_dir.exists()

    def test_ignores_unknown_remote(self, remote_journal):
        """Handler ignores events for unknown remotes."""
        ctx = EventContext(
            msg={
                "tract": "observe",
                "event": "observed",
                "remote": "unknown-remote",
                "segment": "120000_300",
                "day": "20250103",
            },
            app="remote",
            tract="observe",
            event="observed",
            journal_root=str(remote_journal.journal),
        )

        handle_observed(ctx)

        # No history should be created for unknown remote
        hist_dir = remote_journal.remotes_dir / "testkey1" / "hist"
        assert not hist_dir.exists()

    def test_handles_missing_segment(self, remote_journal):
        """Handler handles events missing segment field."""
        ctx = EventContext(
            msg={
                "tract": "observe",
                "event": "observed",
                "remote": "test-remote",
                "day": "20250103",
            },
            app="remote",
            tract="observe",
            event="observed",
            journal_root=str(remote_journal.journal),
        )

        # Should not raise
        handle_observed(ctx)

        # No history should be created
        hist_dir = remote_journal.remotes_dir / "testkey1" / "hist"
        assert not hist_dir.exists()

    def test_handles_missing_day(self, remote_journal):
        """Handler handles events missing day field."""
        ctx = EventContext(
            msg={
                "tract": "observe",
                "event": "observed",
                "remote": "test-remote",
                "segment": "120000_300",
            },
            app="remote",
            tract="observe",
            event="observed",
            journal_root=str(remote_journal.journal),
        )

        # Should not raise
        handle_observed(ctx)

        # No history should be created
        hist_dir = remote_journal.remotes_dir / "testkey1" / "hist"
        assert not hist_dir.exists()
