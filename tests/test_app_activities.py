# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for activities app routes — activities API and output serving."""

import json
import os

import pytest

from apps.activities.routes import activities_bp


@pytest.fixture
def fixture_journal():
    """Set SOLSTONE_JOURNAL to tests/fixtures/journal for testing."""
    old = os.environ.get("SOLSTONE_JOURNAL")
    os.environ["SOLSTONE_JOURNAL"] = "tests/fixtures/journal"
    yield
    if old is None:
        os.environ.pop("SOLSTONE_JOURNAL", None)
    else:
        os.environ["SOLSTONE_JOURNAL"] = old


@pytest.fixture
def activities_client(fixture_journal):
    """Create a Flask test client with the activities blueprint."""
    from flask import Flask

    from convey import state

    app = Flask(__name__)
    app.register_blueprint(activities_bp)
    state.journal_root = "tests/fixtures/journal"
    return app.test_client()


class TestActivitiesDayRoutes:
    def test_returns_enriched_records(self, activities_client):
        resp = activities_client.get(
            "/app/activities/api/day/20260214/activities?facet=full-featured"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) >= 2

        coding = next(a for a in data if a["activity"] == "coding")
        assert coding["id"] == "coding_093000_300"
        assert coding["facet"] == "full-featured"
        assert coding["description"] != ""
        assert coding["level_avg"] == 0.88
        assert coding["duration_minutes"] > 0
        assert "startTime" in coding
        assert "endTime" in coding
        assert len(coding["segments"]) == 4

    def test_includes_activity_metadata(self, activities_client):
        resp = activities_client.get(
            "/app/activities/api/day/20260214/activities?facet=full-featured"
        )
        data = resp.get_json()
        coding = next(a for a in data if a["activity"] == "coding")
        assert coding["name"] != ""
        assert coding["icon"] != ""

    def test_lists_output_files(self, activities_client):
        resp = activities_client.get(
            "/app/activities/api/day/20260214/activities?facet=full-featured"
        )
        data = resp.get_json()
        coding = next(a for a in data if a["activity"] == "coding")
        assert len(coding["outputs"]) >= 1
        output = coding["outputs"][0]
        assert output["filename"] == "session_review.md"
        assert "facets/full-featured/activities/" in output["path"]

    def test_invalid_day_returns_400(self, activities_client):
        resp = activities_client.get("/app/activities/api/day/badday/activities")
        assert resp.status_code == 400


class TestActivitiesOutputRoutes:
    def test_serves_activity_output(self, activities_client):
        resp = activities_client.get(
            "/app/activities/api/activity_output/"
            "facets/full-featured/activities/20260214/"
            "coding_093000_300/session_review.md"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "# Coding Session Review" in data["content"]
        assert data["format"] == "md"
        assert data["filename"] == "session_review.md"

    def test_rejects_non_facets_path(self, activities_client):
        resp = activities_client.get(
            "/app/activities/api/activity_output/20260214/talents/flow.md"
        )
        assert resp.status_code == 400


class TestActivitiesStatsRoutes:
    def test_returns_month_activity_counts(self, tmp_path, monkeypatch):
        from flask import Flask

        from convey import state

        journal = tmp_path / "journal"
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

        for facet in ("work", "personal"):
            facet_dir = journal / "facets" / facet
            facet_dir.mkdir(parents=True)
            (facet_dir / "facet.json").write_text(
                json.dumps({"title": facet.title()}), encoding="utf-8"
            )
            (facet_dir / "activities").mkdir()

        (journal / "facets" / "work" / "activities" / "20260418.jsonl").write_text(
            json.dumps({"id": "coding_1", "activity": "coding", "segments": []})
            + "\n"
            + json.dumps({"id": "coding_2", "activity": "coding", "segments": []})
            + "\n",
            encoding="utf-8",
        )
        (journal / "facets" / "personal" / "activities" / "20260418.jsonl").write_text(
            json.dumps({"id": "walk_1", "activity": "walking", "segments": []}) + "\n",
            encoding="utf-8",
        )
        (journal / "facets" / "work" / "activities" / "20260419.jsonl").write_text(
            json.dumps({"id": "coding_3", "activity": "coding", "segments": []}) + "\n",
            encoding="utf-8",
        )

        app = Flask(__name__)
        app.register_blueprint(activities_bp)
        monkeypatch.setattr(state, "journal_root", str(journal), raising=False)
        client = app.test_client()

        resp = client.get("/app/activities/api/stats/202604")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {
            "20260418": {"personal": 1, "work": 2},
            "20260419": {"work": 1},
        }
        assert "20260420" not in data
