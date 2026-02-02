# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the activity_state pre-hook module."""

import json
import os
import tempfile
from pathlib import Path

# Set up test environment before importing the module
os.environ["JOURNAL_PATH"] = "fixtures/journal"


class TestExtractFacetFromOutputPath:
    """Tests for _extract_facet_from_output_path."""

    def test_extracts_facet_from_valid_path(self):
        from muse.activity_state import _extract_facet_from_output_path

        path = "/journal/20260130/143000_300/activity_state_work.json"
        assert _extract_facet_from_output_path(path) == "work"

    def test_extracts_facet_with_hyphen(self):
        from muse.activity_state import _extract_facet_from_output_path

        path = "/journal/20260130/143000_300/activity_state_my-project.json"
        assert _extract_facet_from_output_path(path) == "my-project"

    def test_returns_none_for_empty_path(self):
        from muse.activity_state import _extract_facet_from_output_path

        assert _extract_facet_from_output_path("") is None
        assert _extract_facet_from_output_path(None) is None

    def test_returns_none_for_non_matching_path(self):
        from muse.activity_state import _extract_facet_from_output_path

        # Different generator name
        assert _extract_facet_from_output_path("/path/to/facets.json") is None
        # No facet suffix
        assert _extract_facet_from_output_path("/path/to/activity_state.json") is None


class TestFindPreviousSegment:
    """Tests for find_previous_segment."""

    def test_finds_previous_segment(self):
        from muse.activity_state import find_previous_segment

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                # Create day directory with segments
                day_dir = Path(tmpdir) / "20260130"
                day_dir.mkdir()
                (day_dir / "100000_300").mkdir()
                (day_dir / "110000_300").mkdir()
                (day_dir / "120000_300").mkdir()

                # Test finding previous
                assert find_previous_segment("20260130", "120000_300") == "110000_300"
                assert find_previous_segment("20260130", "110000_300") == "100000_300"
                assert find_previous_segment("20260130", "100000_300") is None

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path

    def test_returns_none_for_nonexistent_day(self):
        from muse.activity_state import find_previous_segment

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                assert find_previous_segment("20260130", "100000_300") is None
            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path

    def test_handles_segments_with_suffix(self):
        from muse.activity_state import find_previous_segment

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                day_dir = Path(tmpdir) / "20260130"
                day_dir.mkdir()
                (day_dir / "100000_300_audio").mkdir()
                (day_dir / "110000_300").mkdir()

                # Should still find previous
                assert (
                    find_previous_segment("20260130", "110000_300")
                    == "100000_300_audio"
                )

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path


class TestCheckTimeout:
    """Tests for check_timeout."""

    def test_no_timeout_within_threshold(self):
        from muse.activity_state import check_timeout

        # 5 minute gap (300 seconds)
        assert check_timeout("100500_300", "100000_300", timeout_seconds=3600) is False

    def test_timeout_exceeds_threshold(self):
        from muse.activity_state import check_timeout

        # 2 hour gap
        assert check_timeout("120000_300", "100000_300", timeout_seconds=3600) is True

    def test_uses_segment_end_time(self):
        from muse.activity_state import check_timeout

        # Previous segment: 10:00:00 - 10:05:00 (300 seconds)
        # Current segment: 10:10:00
        # Gap should be 5 minutes (10:10:00 - 10:05:00)
        assert check_timeout("101000_300", "100000_300", timeout_seconds=600) is False


class TestLoadPreviousState:
    """Tests for load_previous_state."""

    def test_loads_valid_state(self):
        from muse.activity_state import load_previous_state

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                # Create state file
                segment_dir = Path(tmpdir) / "20260130" / "100000_300"
                segment_dir.mkdir(parents=True)

                state = {
                    "active": [
                        {
                            "activity": "meeting",
                            "since": "100000_300",
                            "description": "Standup",
                            "level": "high",
                        }
                    ],
                    "ended": [],
                }
                (segment_dir / "activity_state_work.json").write_text(json.dumps(state))

                loaded, segment = load_previous_state("20260130", "100000_300", "work")
                assert segment == "100000_300"
                assert loaded["active"][0]["activity"] == "meeting"

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path

    def test_returns_none_for_missing_file(self):
        from muse.activity_state import load_previous_state

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                segment_dir = Path(tmpdir) / "20260130" / "100000_300"
                segment_dir.mkdir(parents=True)

                loaded, segment = load_previous_state("20260130", "100000_300", "work")
                assert loaded is None
                assert segment is None

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path


class TestFormatActivitiesContext:
    """Tests for format_activities_context."""

    def test_formats_activities_list(self):
        from muse.activity_state import format_activities_context

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                # Create facet with activities
                facet_dir = Path(tmpdir) / "facets" / "work" / "activities"
                facet_dir.mkdir(parents=True)

                activities = [
                    {"id": "meeting"},
                    {"id": "coding", "priority": "high"},
                ]
                (facet_dir / "activities.jsonl").write_text(
                    "\n".join(json.dumps(a) for a in activities)
                )

                result = format_activities_context("work")
                assert "## Facet Activities" in result
                assert "meeting" in result
                assert "coding" in result
                assert "[high priority]" in result

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path

    def test_handles_empty_activities(self):
        from muse.activity_state import format_activities_context

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                # Create facet without activities
                facet_dir = Path(tmpdir) / "facets" / "work"
                facet_dir.mkdir(parents=True)

                result = format_activities_context("work")
                assert "No activities configured" in result

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path


class TestFormatPreviousState:
    """Tests for format_previous_state."""

    def test_formats_active_activities(self):
        from muse.activity_state import format_previous_state

        state = {
            "active": [
                {
                    "activity": "meeting",
                    "since": "100000_300",
                    "description": "Team standup",
                    "level": "high",
                }
            ],
            "ended": [],
        }

        result = format_previous_state(
            state, "100000_300", "100500_300", timed_out=False
        )
        assert "Previous State" in result
        assert "meeting" in result
        assert "since 100000_300" in result
        assert "Team standup" in result

    def test_formats_ended_activities(self):
        from muse.activity_state import format_previous_state

        state = {
            "active": [],
            "ended": [
                {
                    "activity": "email",
                    "since": "093000_300",
                    "description": "Replied to boss",
                }
            ],
        }

        result = format_previous_state(
            state, "100000_300", "100500_300", timed_out=False
        )
        assert "Recently ended" in result
        assert "email" in result

    def test_handles_timeout(self):
        from muse.activity_state import format_previous_state

        state = {"active": [{"activity": "meeting"}], "ended": []}
        result = format_previous_state(
            state, "100000_300", "120000_300", timed_out=True
        )
        assert "Starting fresh" in result
        assert "meeting" not in result

    def test_handles_no_previous_state(self):
        from muse.activity_state import format_previous_state

        result = format_previous_state(None, None, "100000_300", timed_out=False)
        assert "No previous segment state" in result


class TestPreProcess:
    """Tests for the pre_process hook function."""

    def test_builds_enriched_context(self):
        from muse.activity_state import pre_process

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = os.environ.get("JOURNAL_PATH")
            os.environ["JOURNAL_PATH"] = tmpdir

            try:
                # Create day and segments
                day_dir = Path(tmpdir) / "20260130"
                day_dir.mkdir()
                (day_dir / "100000_300").mkdir()
                segment_dir = day_dir / "110000_300"
                segment_dir.mkdir()

                # Create facet with activities
                facet_dir = Path(tmpdir) / "facets" / "work" / "activities"
                facet_dir.mkdir(parents=True)
                (facet_dir / "activities.jsonl").write_text(
                    '{"id": "meeting"}\n{"id": "coding"}'
                )

                # Create previous state
                prev_state = {
                    "active": [
                        {
                            "activity": "meeting",
                            "since": "100000_300",
                            "description": "Standup",
                            "level": "high",
                        }
                    ],
                    "ended": [],
                }
                (day_dir / "100000_300" / "activity_state_work.json").write_text(
                    json.dumps(prev_state)
                )

                context = {
                    "day": "20260130",
                    "segment": "110000_300",
                    "output_path": "/journal/20260130/110000_300/activity_state_work.json",
                    "transcript": "User is typing code...",
                    "meta": {},
                }

                result = pre_process(context)
                assert result is not None
                assert "transcript" in result

                transcript = result["transcript"]
                assert "## Facet Activities" in transcript
                assert "meeting" in transcript
                assert "## Previous State" in transcript
                assert "Standup" in transcript
                assert "## Current Segment Content" in transcript
                assert "User is typing code" in transcript

            finally:
                if original_path:
                    os.environ["JOURNAL_PATH"] = original_path

    def test_returns_none_without_day(self):
        from muse.activity_state import pre_process

        context = {
            "segment": "100000_300",
            "output_path": "/path/to/activity_state_work.json",
        }
        assert pre_process(context) is None

    def test_returns_none_without_segment(self):
        from muse.activity_state import pre_process

        context = {
            "day": "20260130",
            "output_path": "/path/to/activity_state_work.json",
        }
        assert pre_process(context) is None

    def test_returns_none_without_facet_in_path(self):
        from muse.activity_state import pre_process

        context = {
            "day": "20260130",
            "segment": "100000_300",
            "output_path": "/path/to/something_else.json",
        }
        assert pre_process(context) is None
