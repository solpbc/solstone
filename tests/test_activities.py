# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the activities module and activities agent hooks."""

import json
import os
import tempfile
from pathlib import Path

# Set up test environment before importing the module
os.environ["JOURNAL_PATH"] = "tests/fixtures/journal"


def test_get_default_activities():
    """Test that default activities are returned correctly."""
    from think.activities import get_default_activities

    defaults = get_default_activities()

    # Should return a list
    assert isinstance(defaults, list)
    assert len(defaults) > 0

    # Each activity should have required fields
    for activity in defaults:
        assert "id" in activity
        assert "name" in activity
        assert "description" in activity

    # Check some known activities exist
    ids = [a["id"] for a in defaults]
    assert "meeting" in ids
    assert "coding" in ids
    assert "browsing" in ids


def test_get_default_activities_returns_copy():
    """Test that get_default_activities returns a copy, not the original."""
    from think.activities import get_default_activities

    defaults1 = get_default_activities()
    defaults2 = get_default_activities()

    # Should be equal but not the same object
    assert defaults1 == defaults2
    assert defaults1 is not defaults2

    # Modifying one should not affect the other
    defaults1[0]["id"] = "modified"
    assert defaults2[0]["id"] != "modified"


def test_generate_activity_id():
    """Test activity ID generation from names."""
    from think.activities import generate_activity_id

    assert generate_activity_id("My Activity") == "my_activity"
    assert generate_activity_id("Research & Development") == "research_development"
    assert generate_activity_id("  Spaces  ") == "spaces"
    assert generate_activity_id("123-Numbers!") == "123_numbers"
    assert generate_activity_id("") == "activity"


def test_facet_activities_empty():
    """Test loading activities from a facet with no activities file."""
    from think.activities import get_facet_activities

    # The test journal may not have activities set up
    activities = get_facet_activities("personal")

    # Should return empty list if no file
    assert isinstance(activities, list)


def test_facet_activities_roundtrip():
    """Test saving and loading activities."""
    from think.activities import (
        _get_activities_path,
        get_facet_activities,
        save_facet_activities,
    )

    # Create a temp journal
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override JOURNAL_PATH
        original_path = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = tmpdir

        # Create facet directory
        facet_path = Path(tmpdir) / "facets" / "test_facet"
        facet_path.mkdir(parents=True)

        try:
            # Save some activities
            activities = [
                {"id": "meeting", "priority": "high"},
                {"id": "coding", "description": "Custom coding description"},
                {
                    "id": "custom_activity",
                    "name": "Custom",
                    "description": "A custom activity",
                    "custom": True,
                },
            ]
            save_facet_activities("test_facet", activities)

            # Verify file was created
            path = _get_activities_path("test_facet")
            assert path.exists()

            # Load and verify
            loaded = get_facet_activities("test_facet")
            assert len(loaded) == 3

            # Check meeting (predefined with priority override)
            meeting = next(a for a in loaded if a["id"] == "meeting")
            assert meeting["priority"] == "high"
            assert meeting["custom"] is False
            assert "name" in meeting  # Should have default name

            # Check coding (predefined with description override)
            coding = next(a for a in loaded if a["id"] == "coding")
            assert coding["description"] == "Custom coding description"

            # Check custom activity
            custom = next(a for a in loaded if a["id"] == "custom_activity")
            assert custom["custom"] is True
            assert custom["name"] == "Custom"

        finally:
            if original_path:
                os.environ["JOURNAL_PATH"] = original_path


def test_add_activity_to_facet():
    """Test adding an activity to a facet."""
    from think.activities import (
        add_activity_to_facet,
        get_facet_activities,
        remove_activity_from_facet,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = tmpdir

        facet_path = Path(tmpdir) / "facets" / "test_facet"
        facet_path.mkdir(parents=True)

        try:
            # Add a predefined activity
            result = add_activity_to_facet("test_facet", "meeting", priority="high")
            assert result["id"] == "meeting"

            # Verify it was added
            activities = get_facet_activities("test_facet")
            assert len(activities) == 1

            # Adding same activity again should not duplicate
            add_activity_to_facet("test_facet", "meeting")
            activities = get_facet_activities("test_facet")
            assert len(activities) == 1

            # Remove it
            removed = remove_activity_from_facet("test_facet", "meeting")
            assert removed is True
            activities = get_facet_activities("test_facet")
            assert len(activities) == 0

            # Removing non-existent should return False
            removed = remove_activity_from_facet("test_facet", "meeting")
            assert removed is False

        finally:
            if original_path:
                os.environ["JOURNAL_PATH"] = original_path


def test_update_activity_in_facet():
    """Test updating an activity in a facet."""
    from think.activities import (
        add_activity_to_facet,
        get_activity_by_id,
        update_activity_in_facet,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = os.environ.get("JOURNAL_PATH")
        os.environ["JOURNAL_PATH"] = tmpdir

        facet_path = Path(tmpdir) / "facets" / "test_facet"
        facet_path.mkdir(parents=True)

        try:
            # Add an activity
            add_activity_to_facet("test_facet", "meeting")

            # Update it
            updated = update_activity_in_facet(
                "test_facet", "meeting", priority="low", description="Updated desc"
            )
            assert updated is not None
            assert updated["priority"] == "low"
            assert updated["description"] == "Updated desc"

            # Verify via lookup
            activity = get_activity_by_id("test_facet", "meeting")
            assert activity["priority"] == "low"

            # Update non-existent should return None
            result = update_activity_in_facet(
                "test_facet", "nonexistent", priority="high"
            )
            assert result is None

        finally:
            if original_path:
                os.environ["JOURNAL_PATH"] = original_path


# ---------------------------------------------------------------------------
# Activity Records (think/activities.py)
# ---------------------------------------------------------------------------


class TestLevelAvg:
    """Tests for level_avg computation."""

    def test_all_high(self):
        from think.activities import level_avg

        assert level_avg(["high", "high", "high"]) == 1.0

    def test_all_medium(self):
        from think.activities import level_avg

        assert level_avg(["medium", "medium"]) == 0.5

    def test_all_low(self):
        from think.activities import level_avg

        assert level_avg(["low", "low"]) == 0.25

    def test_mixed(self):
        from think.activities import level_avg

        assert level_avg(["high", "medium"]) == 0.75

    def test_empty_defaults_to_medium(self):
        from think.activities import level_avg

        assert level_avg([]) == 0.5

    def test_unknown_defaults_to_medium(self):
        from think.activities import level_avg

        assert level_avg(["unknown", "high"]) == 0.75


class TestActivityRecordIO:
    """Tests for append/load/update of activity records."""

    def test_append_and_load(self, monkeypatch):
        from think.activities import append_activity_record, load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            record = {
                "id": "coding_100000_300",
                "activity": "coding",
                "segments": ["100000_300", "100500_300"],
                "level_avg": 0.75,
                "description": "Test coding session",
                "active_entities": ["VS Code"],
                "created_at": 1234567890000,
            }

            assert append_activity_record("work", "20260209", record) is True
            records = load_activity_records("work", "20260209")
            assert len(records) == 1
            assert records[0]["id"] == "coding_100000_300"
            assert records[0]["segments"] == ["100000_300", "100500_300"]

    def test_append_idempotent(self, monkeypatch):
        from think.activities import append_activity_record, load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            record = {
                "id": "coding_100000_300",
                "activity": "coding",
                "segments": ["100000_300"],
                "created_at": 1234567890000,
            }

            assert append_activity_record("work", "20260209", record) is True
            assert append_activity_record("work", "20260209", record) is False

            records = load_activity_records("work", "20260209")
            assert len(records) == 1

    def test_load_nonexistent_returns_empty(self, monkeypatch):
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)
            assert load_activity_records("work", "20260209") == []

    def test_update_description(self, monkeypatch):
        from think.activities import (
            append_activity_record,
            load_activity_records,
            update_record_description,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            record = {
                "id": "coding_100000_300",
                "activity": "coding",
                "description": "Original description",
                "segments": ["100000_300"],
                "created_at": 1234567890000,
            }

            append_activity_record("work", "20260209", record)
            result = update_record_description(
                "work", "20260209", "coding_100000_300", "Updated description"
            )
            assert result is True

            records = load_activity_records("work", "20260209")
            assert records[0]["description"] == "Updated description"

    def test_update_nonexistent_returns_false(self, monkeypatch):
        from think.activities import update_record_description

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)
            assert (
                update_record_description("work", "20260209", "nonexistent", "desc")
                is False
            )

    def test_update_preserves_other_records(self, monkeypatch):
        from think.activities import (
            append_activity_record,
            load_activity_records,
            update_record_description,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            r1 = {
                "id": "coding_100000_300",
                "activity": "coding",
                "description": "First",
                "segments": ["100000_300"],
                "created_at": 1,
            }
            r2 = {
                "id": "meeting_110000_300",
                "activity": "meeting",
                "description": "Second",
                "segments": ["110000_300"],
                "created_at": 2,
            }

            append_activity_record("work", "20260209", r1)
            append_activity_record("work", "20260209", r2)

            update_record_description(
                "work", "20260209", "coding_100000_300", "Updated first"
            )

            records = load_activity_records("work", "20260209")
            assert len(records) == 2
            assert records[0]["description"] == "Updated first"
            assert records[1]["description"] == "Second"


# ---------------------------------------------------------------------------
# Activities Agent Hooks (muse/activities.py)
# ---------------------------------------------------------------------------


def _setup_segment(tmpdir, day, segment, facet, state):
    """Helper to create an activity_state.json file in a segment."""
    agents_dir = Path(tmpdir) / day / "default" / segment / "agents" / facet
    agents_dir.mkdir(parents=True, exist_ok=True)
    state_file = agents_dir / "activity_state.json"
    state_file.write_text(json.dumps(state))


class TestMakeActivityId:
    def test_basic(self):
        from think.activities import make_activity_id

        assert make_activity_id("coding", "095809_303") == "coding_095809_303"

    def test_with_custom_type(self):
        from think.activities import make_activity_id

        assert (
            make_activity_id("video_editing", "120000_300")
            == "video_editing_120000_300"
        )


class TestListFacetsWithActivityState:
    def test_finds_facets(self, monkeypatch):
        from muse.activities import _list_facets_with_activity_state

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(tmpdir, "20260209", "100000_300", "personal", [])
            _setup_segment(tmpdir, "20260209", "100000_300", "work", [])

            facets = _list_facets_with_activity_state("20260209", "100000_300", stream="default")
            assert facets == ["personal", "work"]

    def test_returns_empty_for_nonexistent(self, monkeypatch):
        from muse.activities import _list_facets_with_activity_state

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)
            assert _list_facets_with_activity_state("20260209", "100000_300", stream="default") == []


class TestDetectEndedActivities:
    def test_explicit_ended(self):
        from muse.activities import _detect_ended_activities

        prev = [
            {"activity": "coding", "state": "active", "since": "100000_300"},
        ]
        curr = [
            {"activity": "coding", "state": "ended", "since": "100000_300"},
        ]
        ended = _detect_ended_activities(prev, curr, timed_out=False)
        assert len(ended) == 1
        assert ended[0]["activity"] == "coding"

    def test_implicit_ended(self):
        from muse.activities import _detect_ended_activities

        prev = [
            {"activity": "coding", "state": "active", "since": "100000_300"},
            {"activity": "meeting", "state": "active", "since": "100000_300"},
        ]
        curr = [
            {"activity": "meeting", "state": "active", "since": "100000_300"},
        ]
        ended = _detect_ended_activities(prev, curr, timed_out=False)
        assert len(ended) == 1
        assert ended[0]["activity"] == "coding"

    def test_timeout_ends_all(self):
        from muse.activities import _detect_ended_activities

        prev = [
            {"activity": "coding", "state": "active", "since": "100000_300"},
            {"activity": "meeting", "state": "active", "since": "100000_300"},
        ]
        ended = _detect_ended_activities(prev, [], timed_out=True)
        assert len(ended) == 2

    def test_continuing_not_ended(self):
        from muse.activities import _detect_ended_activities

        prev = [
            {"activity": "coding", "state": "active", "since": "100000_300"},
        ]
        curr = [
            {"activity": "coding", "state": "active", "since": "100000_300"},
        ]
        ended = _detect_ended_activities(prev, curr, timed_out=False)
        assert len(ended) == 0

    def test_ignores_previously_ended(self):
        from muse.activities import _detect_ended_activities

        prev = [
            {"activity": "coding", "state": "ended", "since": "090000_300"},
            {"activity": "meeting", "state": "active", "since": "100000_300"},
        ]
        curr = []
        ended = _detect_ended_activities(prev, curr, timed_out=False)
        assert len(ended) == 1
        assert ended[0]["activity"] == "meeting"

    def test_new_activity_same_type(self):
        """A new activity of same type with different since is not the same."""
        from muse.activities import _detect_ended_activities

        prev = [
            {"activity": "coding", "state": "active", "since": "100000_300"},
        ]
        curr = [
            {"activity": "coding", "state": "active", "since": "110000_300"},
        ]
        ended = _detect_ended_activities(prev, curr, timed_out=False)
        assert len(ended) == 1
        assert ended[0]["since"] == "100000_300"


class TestWalkActivitySegments:
    def test_walks_segments(self, monkeypatch):
        from muse.activities import _walk_activity_segments

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Starting work",
                        "level": "high",
                        "active_entities": ["VS Code"],
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100500_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Continuing work",
                        "level": "medium",
                        "active_entities": ["VS Code", "Claude Code"],
                    }
                ],
            )

            result = _walk_activity_segments(
                "20260209", "work", "coding", "100000_300", "100500_300"
            )

            assert result["segments"] == ["100000_300", "100500_300"]
            assert len(result["descriptions"]) == 2
            assert result["levels"] == ["high", "medium"]
            assert result["active_entities"] == ["VS Code", "Claude Code"]

    def test_deduplicates_entities(self, monkeypatch):
        from muse.activities import _walk_activity_segments

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "level": "high",
                        "active_entities": ["VS Code", "Git"],
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100500_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "level": "high",
                        "active_entities": ["VS Code", "Claude Code"],
                    }
                ],
            )

            result = _walk_activity_segments(
                "20260209", "work", "coding", "100000_300", "100500_300"
            )

            assert result["active_entities"] == ["VS Code", "Git", "Claude Code"]

    def test_empty_when_no_match(self, monkeypatch):
        from muse.activities import _walk_activity_segments

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)
            (Path(tmpdir) / "20260209").mkdir()

            result = _walk_activity_segments(
                "20260209", "work", "coding", "100000_300", "100500_300"
            )
            assert result["segments"] == []


class TestPreProcess:
    """Tests for the activities pre_process hook."""

    def test_skips_when_no_previous_segment(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            day_dir = Path(tmpdir) / "20260209"
            day_dir.mkdir()
            (day_dir / "default" / "100000_300").mkdir(parents=True)

            result = pre_process({"day": "20260209", "segment": "100000_300", "stream": "default"})
            assert result is not None
            assert "skip_reason" in result

    def test_skips_when_no_ended_activities(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "level": "high",
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100500_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "level": "high",
                    }
                ],
            )

            result = pre_process({"day": "20260209", "segment": "100500_300", "stream": "default"})
            assert result is not None
            assert result.get("skip_reason") == "no_ended_activities"

    def test_detects_ended_and_writes_record(self, monkeypatch):
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Starting work",
                        "level": "high",
                        "active_entities": ["VS Code"],
                    }
                ],
            )
            _setup_segment(tmpdir, "20260209", "100500_300", "work", [])

            result = pre_process({"day": "20260209", "segment": "100500_300", "stream": "default"})

            assert "skip_reason" not in result
            assert "transcript" in result
            assert "coding_100000_300" in result["transcript"]

            records = load_activity_records("work", "20260209")
            assert len(records) == 1
            assert records[0]["id"] == "coding_100000_300"
            assert records[0]["segments"] == ["100000_300"]

    def test_idempotent_on_rerun(self, monkeypatch):
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Coding",
                        "level": "high",
                    }
                ],
            )
            _setup_segment(tmpdir, "20260209", "100500_300", "work", [])

            context = {"day": "20260209", "segment": "100500_300", "stream": "default"}
            pre_process(context)
            pre_process(context)

            records = load_activity_records("work", "20260209")
            assert len(records) == 1

    def test_multi_facet_detection(self, monkeypatch):
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Work coding",
                        "level": "high",
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "personal",
                [
                    {
                        "activity": "meeting",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Team standup",
                        "level": "medium",
                    }
                ],
            )

            _setup_segment(tmpdir, "20260209", "100500_300", "work", [])
            _setup_segment(tmpdir, "20260209", "100500_300", "personal", [])

            result = pre_process({"day": "20260209", "segment": "100500_300", "stream": "default"})

            assert "skip_reason" not in result
            assert "#work" in result["transcript"]
            assert "#personal" in result["transcript"]

            work_records = load_activity_records("work", "20260209")
            personal_records = load_activity_records("personal", "20260209")
            assert len(work_records) == 1
            assert len(personal_records) == 1

    def test_multi_segment_span(self, monkeypatch):
        """Activity spanning multiple segments should collect all segments."""
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Starting",
                        "level": "high",
                        "active_entities": ["VS Code"],
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100500_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Continuing",
                        "level": "medium",
                        "active_entities": ["VS Code", "Git"],
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "101000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Finishing",
                        "level": "high",
                        "active_entities": ["Claude Code"],
                    }
                ],
            )
            # Coding ends
            _setup_segment(tmpdir, "20260209", "101500_300", "work", [])

            pre_process({"day": "20260209", "segment": "101500_300", "stream": "default"})

            records = load_activity_records("work", "20260209")
            assert len(records) == 1
            r = records[0]
            assert r["segments"] == ["100000_300", "100500_300", "101000_300"]
            assert r["active_entities"] == ["VS Code", "Git", "Claude Code"]
            assert r["level_avg"] == 0.83  # (1.0 + 0.5 + 1.0) / 3


class TestPostProcess:
    """Tests for the activities post_process hook."""

    def test_updates_descriptions(self, monkeypatch):
        from muse.activities import post_process
        from think.activities import append_activity_record, load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            record = {
                "id": "coding_100000_300",
                "activity": "coding",
                "description": "Preliminary description",
                "segments": ["100000_300"],
                "created_at": 1,
            }
            append_activity_record("work", "20260209", record)

            llm_result = json.dumps(
                {
                    "work": [
                        {
                            "id": "coding_100000_300",
                            "description": "Synthesized full description of coding session",
                        }
                    ]
                }
            )

            post_process(llm_result, {"day": "20260209"})

            records = load_activity_records("work", "20260209")
            assert (
                records[0]["description"]
                == "Synthesized full description of coding session"
            )

    def test_handles_invalid_json(self):
        from muse.activities import post_process

        result = post_process("not json", {"day": "20260209"})
        assert result is None

    def test_handles_non_object(self):
        from muse.activities import post_process

        result = post_process("[]", {"day": "20260209"})
        assert result is None

    def test_returns_none(self, monkeypatch):
        from muse.activities import post_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            result = post_process("{}", {"day": "20260209"})
            assert result is None


class TestEstimateDurationMinutes:
    def test_single_segment(self):
        from think.activities import estimate_duration_minutes

        assert estimate_duration_minutes(["100000_300"]) == 5

    def test_multiple_segments(self):
        from think.activities import estimate_duration_minutes

        assert estimate_duration_minutes(["100000_300", "100500_300"]) == 10

    def test_empty_returns_1(self):
        from think.activities import estimate_duration_minutes

        assert estimate_duration_minutes([]) == 1


class TestPreProcessMeta:
    """Tests for pre-hook stashing record data in meta."""

    def test_meta_contains_activity_records(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "level": "high",
                        "description": "Writing code",
                        "active_entities": ["VS Code"],
                    }
                ],
            )
            _setup_segment(tmpdir, "20260209", "100500_300", "work", [])

            result = pre_process(
                {"day": "20260209", "segment": "100500_300", "stream": "default", "meta": {}}
            )

            assert "meta" in result
            records = result["meta"]["activity_records"]
            assert "work" in records
            assert "coding_100000_300" in records["work"]
            rec = records["work"]["coding_100000_300"]
            assert rec["activity"] == "coding"
            assert rec["segments"] == ["100000_300"]
            assert rec["level_avg"] == 1.0
            assert rec["active_entities"] == ["VS Code"]
            assert rec["description"] == "Writing code"

    def test_meta_multiple_facets(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "level": "high",
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "personal",
                [
                    {
                        "activity": "browsing",
                        "state": "active",
                        "since": "100000_300",
                        "level": "low",
                    }
                ],
            )
            _setup_segment(tmpdir, "20260209", "100500_300", "work", [])
            _setup_segment(tmpdir, "20260209", "100500_300", "personal", [])

            result = pre_process({"day": "20260209", "segment": "100500_300", "stream": "default"})

            records = result["meta"]["activity_records"]
            assert "work" in records
            assert "personal" in records


class TestPostProcessEvents:
    """Tests for post-hook callosum event emission."""

    def test_emits_events_with_llm_description(self, monkeypatch):
        from unittest.mock import patch

        from muse.activities import post_process
        from think.activities import append_activity_record

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            record = {
                "id": "coding_100000_300",
                "activity": "coding",
                "description": "Preliminary",
                "segments": ["100000_300"],
                "created_at": 1,
            }
            append_activity_record("work", "20260209", record)

            llm_result = json.dumps(
                {
                    "work": [
                        {
                            "id": "coding_100000_300",
                            "description": "Full synthesized description",
                        }
                    ]
                }
            )

            meta = {
                "activity_records": {
                    "work": {
                        "coding_100000_300": {
                            "activity": "coding",
                            "segments": ["100000_300"],
                            "description": "Preliminary",
                            "level_avg": 1.0,
                            "active_entities": ["VS Code"],
                        }
                    }
                }
            }

            with patch("muse.activities.callosum_send") as mock_send:
                mock_send.return_value = True
                post_process(
                    llm_result,
                    {"day": "20260209", "segment": "100500_300", "meta": meta},
                )

                mock_send.assert_called_once_with(
                    "activity",
                    "recorded",
                    facet="work",
                    day="20260209",
                    segment="100500_300",
                    id="coding_100000_300",
                    activity="coding",
                    segments=["100000_300"],
                    level_avg=1.0,
                    description="Full synthesized description",
                    active_entities=["VS Code"],
                )

    def test_falls_back_to_prehook_description_with_warning(self, monkeypatch, caplog):
        from unittest.mock import patch

        from muse.activities import post_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            meta = {
                "activity_records": {
                    "work": {
                        "coding_100000_300": {
                            "activity": "coding",
                            "segments": ["100000_300"],
                            "description": "Pre-hook fallback desc",
                            "level_avg": 0.5,
                            "active_entities": [],
                        }
                    }
                }
            }

            # LLM returns empty — no descriptions to update
            with patch("muse.activities.callosum_send") as mock_send:
                mock_send.return_value = True
                import logging

                with caplog.at_level(logging.WARNING, logger="muse.activities"):
                    post_process(
                        "{}",
                        {"day": "20260209", "segment": "100500_300", "meta": meta},
                    )

                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["description"] == "Pre-hook fallback desc"
                assert "No LLM description" in caplog.text

    def test_no_events_without_meta(self, monkeypatch):
        from unittest.mock import patch

        from muse.activities import post_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            with patch("muse.activities.callosum_send") as mock_send:
                post_process("{}", {"day": "20260209", "segment": "100500_300"})
                mock_send.assert_not_called()

    def test_event_emission_failure_does_not_raise(self, monkeypatch):
        from unittest.mock import patch

        from muse.activities import post_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            meta = {
                "activity_records": {
                    "work": {
                        "coding_100000_300": {
                            "activity": "coding",
                            "segments": ["100000_300"],
                            "description": "desc",
                            "level_avg": 0.5,
                            "active_entities": [],
                        }
                    }
                }
            }

            with patch("muse.activities.callosum_send") as mock_send:
                mock_send.side_effect = OSError("socket error")
                # Should not raise
                result = post_process(
                    "{}",
                    {"day": "20260209", "segment": "100500_300", "meta": meta},
                )
                assert result is None


class TestHandleActivityRecorded:
    """Tests for supervisor's _handle_activity_recorded handler."""

    def test_queues_dream_task(self):
        from unittest.mock import MagicMock, patch

        from think.supervisor import _handle_activity_recorded

        mock_queue = MagicMock()
        with patch("think.supervisor._task_queue", mock_queue):
            _handle_activity_recorded(
                {
                    "tract": "activity",
                    "event": "recorded",
                    "id": "coding_100000_300",
                    "facet": "work",
                    "day": "20260209",
                }
            )

            mock_queue.submit.assert_called_once_with(
                [
                    "sol",
                    "dream",
                    "--activity",
                    "coding_100000_300",
                    "--facet",
                    "work",
                    "--day",
                    "20260209",
                ]
            )

    def test_ignores_wrong_tract(self):
        from unittest.mock import MagicMock, patch

        from think.supervisor import _handle_activity_recorded

        mock_queue = MagicMock()
        with patch("think.supervisor._task_queue", mock_queue):
            _handle_activity_recorded(
                {
                    "tract": "dream",
                    "event": "recorded",
                    "id": "x",
                    "facet": "w",
                    "day": "d",
                }
            )
            mock_queue.submit.assert_not_called()

    def test_ignores_wrong_event(self):
        from unittest.mock import MagicMock, patch

        from think.supervisor import _handle_activity_recorded

        mock_queue = MagicMock()
        with patch("think.supervisor._task_queue", mock_queue):
            _handle_activity_recorded(
                {
                    "tract": "activity",
                    "event": "other",
                    "id": "x",
                    "facet": "w",
                    "day": "d",
                }
            )
            mock_queue.submit.assert_not_called()

    def test_warns_on_missing_fields(self, caplog):
        from unittest.mock import MagicMock, patch

        from think.supervisor import _handle_activity_recorded

        mock_queue = MagicMock()
        import logging

        with patch("think.supervisor._task_queue", mock_queue):
            with caplog.at_level(logging.WARNING):
                _handle_activity_recorded(
                    {"tract": "activity", "event": "recorded", "id": "x"}
                )
            mock_queue.submit.assert_not_called()

    def test_warns_when_no_task_queue(self, caplog):
        import logging
        from unittest.mock import patch

        from think.supervisor import _handle_activity_recorded

        with patch("think.supervisor._task_queue", None):
            with caplog.at_level(logging.WARNING):
                _handle_activity_recorded(
                    {
                        "tract": "activity",
                        "event": "recorded",
                        "id": "coding_100000_300",
                        "facet": "work",
                        "day": "20260209",
                    }
                )
            assert "No task queue" in caplog.text


# ---------------------------------------------------------------------------
# Flush Mode Tests
# ---------------------------------------------------------------------------


class TestPreProcessFlush:
    """Tests for the activities pre_process hook in flush mode."""

    def test_flush_ends_all_active_activities(self, monkeypatch):
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            # Set up a segment with an active activity (no following segment)
            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Working on feature",
                        "level": "high",
                        "active_entities": ["VS Code"],
                    }
                ],
            )

            result = pre_process(
                {"day": "20260209", "segment": "100000_300", "stream": "default", "flush": True}
            )

            # Should detect the active activity as ended
            assert "skip_reason" not in result
            assert "transcript" in result
            assert "coding_100000_300" in result["transcript"]

            # Record should be written
            records = load_activity_records("work", "20260209")
            assert len(records) == 1
            assert records[0]["id"] == "coding_100000_300"
            assert records[0]["segments"] == ["100000_300"]

    def test_flush_skips_when_no_active_activities(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            # Set up a segment with only ended activities
            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "ended",
                        "since": "090000_300",
                    }
                ],
            )

            result = pre_process(
                {"day": "20260209", "segment": "100000_300", "stream": "default", "flush": True}
            )
            assert result["skip_reason"] == "no_active_activities"

    def test_flush_skips_when_no_activity_state(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            # Create segment dir but no activity_state files
            seg_dir = Path(tmpdir) / "20260209" / "default" / "100000_300"
            seg_dir.mkdir(parents=True)

            result = pre_process(
                {"day": "20260209", "segment": "100000_300", "stream": "default", "flush": True}
            )
            assert result["skip_reason"] == "no_activity_state"

    def test_flush_handles_multiple_facets(self, monkeypatch):
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Work coding",
                        "level": "high",
                    }
                ],
            )
            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "personal",
                [
                    {
                        "activity": "browsing",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Personal browsing",
                        "level": "low",
                    }
                ],
            )

            result = pre_process(
                {"day": "20260209", "segment": "100000_300", "stream": "default", "flush": True}
            )

            assert "skip_reason" not in result
            assert "transcript" in result

            work_records = load_activity_records("work", "20260209")
            personal_records = load_activity_records("personal", "20260209")
            assert len(work_records) == 1
            assert len(personal_records) == 1

    def test_flush_is_idempotent(self, monkeypatch):
        from muse.activities import pre_process
        from think.activities import load_activity_records

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Coding",
                        "level": "high",
                    }
                ],
            )

            context = {"day": "20260209", "segment": "100000_300", "stream": "default", "flush": True}
            pre_process(context)
            pre_process(context)

            records = load_activity_records("work", "20260209")
            assert len(records) == 1

    def test_flush_stashes_meta_for_post_hook(self, monkeypatch):
        from muse.activities import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("JOURNAL_PATH", tmpdir)

            _setup_segment(
                tmpdir,
                "20260209",
                "100000_300",
                "work",
                [
                    {
                        "activity": "coding",
                        "state": "active",
                        "since": "100000_300",
                        "description": "Coding",
                        "level": "high",
                    }
                ],
            )

            result = pre_process(
                {"day": "20260209", "segment": "100000_300", "stream": "default", "flush": True}
            )

            assert "meta" in result
            records = result["meta"]["activity_records"]
            assert "work" in records
            assert "coding_100000_300" in records["work"]


class TestCheckSegmentFlush:
    """Tests for supervisor's _check_segment_flush."""

    def test_queues_flush_after_timeout(self):
        import time as time_mod
        from unittest.mock import MagicMock, patch

        from think.supervisor import _check_segment_flush, _flush_state

        # Set up state as if a segment arrived over an hour ago
        _flush_state["last_segment_ts"] = time_mod.time() - 4000
        _flush_state["day"] = "20260209"
        _flush_state["segment"] = "100000_300"
        _flush_state["flushed"] = False

        mock_queue = MagicMock()
        with (
            patch("think.supervisor._task_queue", mock_queue),
            patch("think.supervisor._is_remote_mode", False),
        ):
            _check_segment_flush()

        mock_queue.submit.assert_called_once_with(
            [
                "sol",
                "dream",
                "-v",
                "--day",
                "20260209",
                "--segment",
                "100000_300",
                "--flush",
            ]
        )
        assert _flush_state["flushed"] is True

    def test_does_not_flush_before_timeout(self):
        import time as time_mod
        from unittest.mock import MagicMock, patch

        from think.supervisor import _check_segment_flush, _flush_state

        _flush_state["last_segment_ts"] = time_mod.time() - 100  # Only 100s ago
        _flush_state["day"] = "20260209"
        _flush_state["segment"] = "100000_300"
        _flush_state["flushed"] = False

        mock_queue = MagicMock()
        with (
            patch("think.supervisor._task_queue", mock_queue),
            patch("think.supervisor._is_remote_mode", False),
        ):
            _check_segment_flush()

        mock_queue.submit.assert_not_called()
        assert _flush_state["flushed"] is False

    def test_does_not_flush_twice(self):
        import time as time_mod
        from unittest.mock import MagicMock, patch

        from think.supervisor import _check_segment_flush, _flush_state

        _flush_state["last_segment_ts"] = time_mod.time() - 4000
        _flush_state["day"] = "20260209"
        _flush_state["segment"] = "100000_300"
        _flush_state["flushed"] = True  # Already flushed

        mock_queue = MagicMock()
        with (
            patch("think.supervisor._task_queue", mock_queue),
            patch("think.supervisor._is_remote_mode", False),
        ):
            _check_segment_flush()

        mock_queue.submit.assert_not_called()

    def test_skips_in_remote_mode(self):
        import time as time_mod
        from unittest.mock import MagicMock, patch

        from think.supervisor import _check_segment_flush, _flush_state

        _flush_state["last_segment_ts"] = time_mod.time() - 4000
        _flush_state["day"] = "20260209"
        _flush_state["segment"] = "100000_300"
        _flush_state["flushed"] = False

        mock_queue = MagicMock()
        with (
            patch("think.supervisor._task_queue", mock_queue),
            patch("think.supervisor._is_remote_mode", True),
        ):
            _check_segment_flush()

        mock_queue.submit.assert_not_called()

    def test_force_flushes_before_timeout(self):
        import time as time_mod
        from unittest.mock import MagicMock, patch

        from think.supervisor import _check_segment_flush, _flush_state

        # Only 100s ago — would NOT flush normally
        _flush_state["last_segment_ts"] = time_mod.time() - 100
        _flush_state["day"] = "20260209"
        _flush_state["segment"] = "100000_300"
        _flush_state["flushed"] = False

        mock_queue = MagicMock()
        with (
            patch("think.supervisor._task_queue", mock_queue),
            patch("think.supervisor._is_remote_mode", False),
        ):
            _check_segment_flush(force=True)

        mock_queue.submit.assert_called_once()
        assert _flush_state["flushed"] is True

    def test_segment_observed_resets_flush_state(self):
        from think.supervisor import _flush_state, _handle_segment_observed

        _flush_state["flushed"] = True
        _flush_state["last_segment_ts"] = 0

        # We need to mock the thread start to avoid side effects
        from unittest.mock import patch

        with patch("think.supervisor.threading"):
            _handle_segment_observed(
                {
                    "tract": "observe",
                    "event": "observed",
                    "day": "20260209",
                    "segment": "110000_300",
                }
            )

        assert _flush_state["flushed"] is False
        assert _flush_state["day"] == "20260209"
        assert _flush_state["segment"] == "110000_300"
        assert _flush_state["last_segment_ts"] > 0
