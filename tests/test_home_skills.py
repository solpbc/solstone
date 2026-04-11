# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for home pulse skill surfacing."""

import json
from datetime import datetime, timedelta

import pytest

from apps.home.routes import (
    _collect_skills,
    _load_skills_state,
    _save_skills_state,
    home_bp,
)


@pytest.fixture
def home_client():
    """Create a Flask test client with home routes registered."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(home_bp)
    return app.test_client()


def _write_skill_fixtures(tmp_path, facet_name, patterns, skill_files):
    """Write patterns.jsonl and skill .md files for a facet.

    Parameters
    ----------
    patterns : list[dict]
        Each dict is one line in patterns.jsonl.
    skill_files : dict[str, str]
        Mapping of {slug: markdown_content} for skill files.
    """
    skills_dir = tmp_path / "facets" / facet_name / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Write facet.json so get_enabled_facets() finds this facet
    facet_dir = tmp_path / "facets" / facet_name
    (facet_dir / "facet.json").write_text(
        json.dumps(
            {
                "title": facet_name.title(),
                "description": "",
                "color": "#000",
                "emoji": "📁",
            }
        ),
        encoding="utf-8",
    )

    # Write patterns.jsonl
    lines = [json.dumps(p) for p in patterns]
    (skills_dir / "patterns.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Write skill markdown files
    for slug, content in skill_files.items():
        (skills_dir / f"{slug}.md").write_text(content, encoding="utf-8")


def test_collect_skills_no_facets(monkeypatch, tmp_path):
    """No facets directory yields empty skills list."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    assert _collect_skills() == []


def test_collect_skills_no_skills_dir(monkeypatch, tmp_path):
    """Facet exists but has no skills directory."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    facet_dir = tmp_path / "facets" / "work"
    facet_dir.mkdir(parents=True)
    (facet_dir / "facet.json").write_text(
        json.dumps(
            {"title": "Work", "description": "", "color": "#000", "emoji": "💼"}
        ),
        encoding="utf-8",
    )

    assert _collect_skills() == []


def test_collect_skills_with_mature_skill(monkeypatch, tmp_path):
    """Mature skill (skill_generated: true) is collected with correct fields."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    skill_md = """---
name: Morning Standup
activity_type: meeting
facet: work
observations: 5
first_seen: "2026-03-01T09:00:00"
last_seen: "2026-04-10T09:15:00"
typical_duration: 15m
typical_time: "9:00 AM"
key_entities:
  - Engineering Team
---

## when this happens

Daily morning standup with the engineering team.

## what the owner does

Gives status updates and listens to blockers.
"""

    _write_skill_fixtures(
        tmp_path,
        "work",
        [{"id": "morning-standup", "skill_generated": True, "observation_count": 5}],
        {"morning-standup": skill_md},
    )

    skills = _collect_skills()

    assert len(skills) == 1
    assert skills[0]["id"] == "morning-standup"
    assert skills[0]["name"] == "Morning Standup"
    assert skills[0]["facet"] == "work"
    assert skills[0]["observations"] == 5
    assert "meeting" in skills[0]["summary"]
    assert "9:00 AM" in skills[0]["summary"]
    assert "when this happens" in skills[0]["content"]
    assert skills[0]["seen"] is False


def test_collect_skills_immature_excluded(monkeypatch, tmp_path):
    """Patterns with skill_generated: false are excluded."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    skill_md = """---
name: Random Chat
activity_type: conversation
observations: 1
---

Some content.
"""

    _write_skill_fixtures(
        tmp_path,
        "work",
        [{"id": "random-chat", "skill_generated": False, "observation_count": 1}],
        {"random-chat": skill_md},
    )

    assert _collect_skills() == []


def test_collect_skills_seen_flag(monkeypatch, tmp_path):
    """Skills modified before skills_last_seen are marked seen."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    skill_md = """---
name: Daily Review
activity_type: review
observations: 3
last_seen: "2026-04-09T17:00:00"
---

Review content.
"""

    _write_skill_fixtures(
        tmp_path,
        "work",
        [{"id": "daily-review", "skill_generated": True, "observation_count": 3}],
        {"daily-review": skill_md},
    )

    # Mark as seen AFTER the file was written (file mtime < skills_last_seen)
    _save_skills_state(
        {"skills_last_seen": (datetime.now() + timedelta(minutes=5)).isoformat()}
    )

    skills = _collect_skills()

    assert len(skills) == 1
    assert skills[0]["seen"] is True


def test_api_skills_seen(monkeypatch, tmp_path, home_client):
    """Seen endpoint persists the skills seen timestamp."""
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    resp = home_client.post("/app/home/api/skills/seen")

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}
    state = _load_skills_state()
    assert "skills_last_seen" in state


def test_api_pulse_includes_skills(monkeypatch, home_client):
    """Pulse API includes skills data from the context builder."""
    monkeypatch.setattr(
        "apps.home.routes.get_current", lambda: {"capture": {"status": "ok"}}
    )
    monkeypatch.setattr("apps.home.routes.get_cached_state", lambda: {})
    monkeypatch.setattr("apps.home.routes._resolve_attention", lambda awareness: None)
    monkeypatch.setattr("apps.home.routes._load_stats", lambda today: {})
    monkeypatch.setattr("apps.home.routes._load_flow_md", lambda today: (None, None))
    monkeypatch.setattr("apps.home.routes._load_pulse_md", lambda: (None, None, []))
    monkeypatch.setattr("apps.home.routes._collect_events", lambda today: [])
    monkeypatch.setattr("apps.home.routes._collect_activities", lambda today: [])
    monkeypatch.setattr("apps.home.routes._collect_todos", lambda today: [])
    monkeypatch.setattr("apps.home.routes._collect_entities_today", lambda today: [])
    monkeypatch.setattr("apps.home.routes._collect_routines", lambda: [])
    monkeypatch.setattr(
        "apps.home.routes._collect_skills",
        lambda: [
            {
                "id": "morning-standup",
                "name": "Morning Standup",
                "facet": "work",
                "summary": "meeting · 9:00 AM",
                "observations": 5,
                "last_seen": "2026-04-10T09:15:00",
                "content": "# Standup\n\nDaily standup.",
                "seen": False,
            }
        ],
    )

    resp = home_client.get("/app/home/api/pulse")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "skills" in data
    assert data["skills"][0]["name"] == "Morning Standup"
    assert "skills_summary" in data
    assert "skills_content" in data
    assert "morning-standup" in data["skills_content"]
