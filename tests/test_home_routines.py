# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for home pulse routine surfacing."""

import json
from datetime import datetime, timedelta

import pytest

from solstone.apps.home.routes import (
    _collect_routines,
    _load_routines_state,
    _save_routines_state,
    home_bp,
)


@pytest.fixture
def home_client():
    """Create a Flask test client with home routes registered."""
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(home_bp)
    return app.test_client()


def _write_routines_config(tmp_path, config):
    routines_dir = tmp_path / "routines"
    routines_dir.mkdir(exist_ok=True)
    (routines_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_collect_routines_empty_config(monkeypatch, tmp_path):
    """Missing routines config yields no pulse routines."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    assert _collect_routines() == []


def test_collect_routines_with_recent_output(monkeypatch, tmp_path):
    """Recent enabled routine output is returned with an extracted summary."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    routine_id = "morning-briefing"
    _write_routines_config(
        tmp_path,
        {
            routine_id: {
                "id": routine_id,
                "name": "Morning Briefing",
                "cadence": "0 8 * * *",
                "enabled": True,
                "last_run": (datetime.now() - timedelta(hours=2)).isoformat(),
            }
        },
    )
    output_dir = tmp_path / "routines" / routine_id
    output_dir.mkdir()
    (output_dir / "20260327.md").write_text(
        "---\nupdated: 2026-03-27T08:00:00\n---\n# Heading\n\nYour day looks clear with one meeting at 2pm.\n",
        encoding="utf-8",
    )

    routines = _collect_routines()

    assert len(routines) == 1
    assert routines[0]["id"] == routine_id
    assert routines[0]["name"] == "Morning Briefing"
    assert routines[0]["summary"] == "Your day looks clear with one meeting at 2pm."
    assert routines[0]["seen"] is False


def test_collect_routines_multi_output_picks_newest(monkeypatch, tmp_path):
    """When multiple outputs exist for a routine, the newest by mtime is used."""
    import time

    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    routine_id = "multi-run"
    _write_routines_config(
        tmp_path,
        {
            routine_id: {
                "id": routine_id,
                "name": "Multi Run",
                "cadence": "0 8 * * *",
                "enabled": True,
                "last_run": (datetime.now() - timedelta(hours=1)).isoformat(),
            }
        },
    )
    output_dir = tmp_path / "routines" / routine_id
    output_dir.mkdir()
    # Older file (plain date name)
    older = output_dir / "20260327.md"
    older.write_text("Old output from first run.", encoding="utf-8")
    # Ensure mtime difference
    time.sleep(0.05)
    # Newer file (collision name with timestamp)
    newer = output_dir / "20260327-120000.md"
    newer.write_text("Updated output from second run.", encoding="utf-8")

    routines = _collect_routines()

    assert len(routines) == 1
    assert routines[0]["summary"] == "Updated output from second run."


def test_collect_routines_stale_excluded(monkeypatch, tmp_path):
    """Stale routine runs are excluded from pulse."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    _write_routines_config(
        tmp_path,
        {
            "morning-briefing": {
                "id": "morning-briefing",
                "name": "Morning Briefing",
                "cadence": "0 8 * * *",
                "enabled": True,
                "last_run": (datetime.now() - timedelta(days=2)).isoformat(),
            }
        },
    )

    assert _collect_routines() == []


def test_collect_routines_disabled_excluded(monkeypatch, tmp_path):
    """Disabled routines are excluded even with recent runs."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    _write_routines_config(
        tmp_path,
        {
            "morning-briefing": {
                "id": "morning-briefing",
                "name": "Morning Briefing",
                "cadence": "0 8 * * *",
                "enabled": False,
                "last_run": datetime.now().isoformat(),
            }
        },
    )

    assert _collect_routines() == []


def test_collect_routines_seen_flag(monkeypatch, tmp_path):
    """Routine runs before the last-seen marker are marked seen."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    last_run = datetime.now() - timedelta(hours=2)
    _write_routines_config(
        tmp_path,
        {
            "morning-briefing": {
                "id": "morning-briefing",
                "name": "Morning Briefing",
                "cadence": "0 8 * * *",
                "enabled": True,
                "last_run": last_run.isoformat(),
            }
        },
    )
    _save_routines_state(
        {"routines_last_seen": (last_run + timedelta(minutes=30)).isoformat()}
    )

    routines = _collect_routines()

    assert len(routines) == 1
    assert routines[0]["seen"] is True


def test_api_routines_seen(monkeypatch, tmp_path, home_client):
    """Seen endpoint persists the routines seen timestamp."""
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))

    resp = home_client.post("/app/home/api/routines/seen")

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}
    state = _load_routines_state()
    assert "routines_last_seen" in state


def test_api_pulse_includes_routines(monkeypatch, home_client):
    """Pulse API includes the routines payload from the context builder."""
    monkeypatch.setattr(
        "solstone.apps.home.routes.get_capture_health",
        lambda: {"status": "active", "observers": []},
    )
    monkeypatch.setattr("solstone.apps.home.routes.get_cached_state", lambda: {})
    monkeypatch.setattr(
        "solstone.apps.home.routes._resolve_attention", lambda awareness: None
    )
    monkeypatch.setattr("solstone.apps.home.routes._load_stats", lambda today: {})
    monkeypatch.setattr(
        "solstone.apps.home.routes._load_flow_md", lambda today: (None, None)
    )
    monkeypatch.setattr(
        "solstone.apps.home.routes._load_pulse_md", lambda: (None, None, [])
    )
    monkeypatch.setattr(
        "solstone.apps.home.routes._collect_anticipated_activities", lambda today: []
    )
    monkeypatch.setattr(
        "solstone.apps.home.routes._collect_activities", lambda today: []
    )
    monkeypatch.setattr("solstone.apps.home.routes._collect_todos", lambda today: [])
    monkeypatch.setattr(
        "solstone.apps.home.routes._collect_entities_today", lambda today: []
    )
    monkeypatch.setattr(
        "solstone.apps.home.routes._collect_routines",
        lambda: [
            {
                "id": "morning-briefing",
                "name": "Morning Briefing",
                "last_run": datetime.now().isoformat(),
                "run_time_display": "just now",
                "summary": "Clear day ahead",
                "seen": False,
            }
        ],
    )

    resp = home_client.get("/app/home/api/pulse")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "routines" in data
    assert data["routines"][0]["name"] == "Morning Briefing"
