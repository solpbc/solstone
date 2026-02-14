# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for calendar app routes — activities API and output serving."""

import os

import pytest

from apps.calendar.routes import calendar_bp


@pytest.fixture
def fixture_journal():
    """Set JOURNAL_PATH to tests/fixtures/journal for testing."""
    old = os.environ.get("JOURNAL_PATH")
    os.environ["JOURNAL_PATH"] = "tests/fixtures/journal"
    yield
    if old is None:
        os.environ.pop("JOURNAL_PATH", None)
    else:
        os.environ["JOURNAL_PATH"] = old


@pytest.fixture
def calendar_client(fixture_journal):
    """Create a Flask test client with calendar blueprint."""
    from flask import Flask

    from convey import state

    app = Flask(__name__)
    app.register_blueprint(calendar_bp)
    state.journal_root = "tests/fixtures/journal"
    return app.test_client()


class TestCalendarDayActivities:
    """Tests for GET /api/day/<day>/activities."""

    def test_returns_enriched_records(self, calendar_client):
        """Activities endpoint returns records with metadata and timing."""
        resp = calendar_client.get(
            "/app/calendar/api/day/20260214/activities?facet=full-featured"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) >= 2

        # Find the coding activity
        coding = next(a for a in data if a["activity"] == "coding")
        assert coding["id"] == "coding_093000_300"
        assert coding["facet"] == "full-featured"
        assert coding["description"] != ""
        assert coding["level_avg"] == 0.88
        assert coding["duration_minutes"] > 0
        assert "startTime" in coding
        assert "endTime" in coding
        assert "segments" in coding
        assert len(coding["segments"]) == 4

    def test_includes_activity_metadata(self, calendar_client):
        """Activities include name and icon from activity definitions."""
        resp = calendar_client.get(
            "/app/calendar/api/day/20260214/activities?facet=full-featured"
        )
        data = resp.get_json()
        coding = next(a for a in data if a["activity"] == "coding")
        # coding is a default activity with name and icon
        assert coding["name"] != ""
        assert coding["icon"] != ""

    def test_lists_output_files(self, calendar_client):
        """Activities with output files include them in the outputs array."""
        resp = calendar_client.get(
            "/app/calendar/api/day/20260214/activities?facet=full-featured"
        )
        data = resp.get_json()
        coding = next(a for a in data if a["activity"] == "coding")
        assert len(coding["outputs"]) >= 1
        output = coding["outputs"][0]
        assert output["filename"] == "session_review.md"
        assert "facets/full-featured/activities/" in output["path"]

    def test_no_outputs_for_activity_without_files(self, calendar_client):
        """Activities without output files have empty outputs array."""
        resp = calendar_client.get(
            "/app/calendar/api/day/20260214/activities?facet=full-featured"
        )
        data = resp.get_json()
        meeting = next(a for a in data if a["activity"] == "meeting")
        assert meeting["outputs"] == []

    def test_empty_day_returns_empty_list(self, calendar_client):
        """Day with no activity records returns empty array."""
        resp = calendar_client.get(
            "/app/calendar/api/day/20260101/activities?facet=full-featured"
        )
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_invalid_day_returns_400(self, calendar_client):
        """Invalid day format returns 400."""
        resp = calendar_client.get("/app/calendar/api/day/badday/activities")
        assert resp.status_code == 400

    def test_sorted_by_start_time(self, calendar_client):
        """Activities are sorted by start time."""
        resp = calendar_client.get(
            "/app/calendar/api/day/20260214/activities?facet=full-featured"
        )
        data = resp.get_json()
        times = [a.get("startTime", "z") for a in data]
        assert times == sorted(times)


class TestCalendarActivityOutput:
    """Tests for GET /api/activity_output/<path:filename>."""

    def test_serves_activity_output(self, calendar_client):
        """Serves markdown output file content."""
        resp = calendar_client.get(
            "/app/calendar/api/activity_output/"
            "facets/full-featured/activities/20260214/"
            "coding_093000_300/session_review.md"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "# Coding Session Review" in data["content"]
        assert data["format"] == "md"
        assert data["filename"] == "session_review.md"

    def test_rejects_non_facets_path(self, calendar_client):
        """Paths not starting with facets/ are rejected."""
        resp = calendar_client.get(
            "/app/calendar/api/activity_output/20260214/agents/flow.md"
        )
        assert resp.status_code == 400

    def test_missing_file_returns_404(self, calendar_client):
        """Non-existent file returns 404."""
        resp = calendar_client.get(
            "/app/calendar/api/activity_output/"
            "facets/full-featured/activities/20260214/"
            "coding_093000_300/nonexistent.md"
        )
        assert resp.status_code == 404
