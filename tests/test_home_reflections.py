# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from datetime import datetime

from solstone.convey import create_app


def _make_client(journal_path: str):
    app = create_app(journal_path)
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return client


def _minimal_pulse_context(latest_weekly_reflection):
    return {
        "today": "20260310",
        "now": datetime(2026, 3, 10, 12, 0, 0),
        "capture_status": "active",
        "last_observe_relative": None,
        "attention": None,
        "pipeline_status": None,
        "segment_count": 0,
        "duration_minutes": 0,
        "facet_data": {},
        "narrative_content": None,
        "narrative_updated_at": None,
        "narrative_source": None,
        "narrative_header": "today's flow",
        "pulse_needs": [],
        "flow_content": None,
        "flow_updated_at": None,
        "anticipated_activities": [],
        "activities": [],
        "todos": [],
        "entities": [],
        "routines": [],
        "skills": [],
        "skills_summary": "",
        "skills_content": {},
        "briefing_sections": {},
        "briefing_meta": None,
        "briefing_phase": "eod",
        "briefing_exists": False,
        "briefing_summary": None,
        "briefing_needs_deduped": [],
        "briefing_needs_shared_count": 0,
        "briefing_needs_badge": None,
        "latest_weekly_reflection": latest_weekly_reflection,
        "yesterday_processing": {
            "has_story": False,
            "framing": "",
            "summary": "",
            "details": [],
            "label": "",
        },
        "show_welcome": False,
        "narrative_summary": "",
        "routines_summary": "",
        "today_summary": "",
        "needs_summary": "",
        "network_summary": "",
    }


def test_home_shows_latest_weekly_reflection_link(monkeypatch, journal_copy):
    client = _make_client(str(journal_copy))
    monkeypatch.setattr(
        "solstone.apps.home.routes._build_pulse_context",
        lambda: _minimal_pulse_context(
            {
                "day": "20260308",
                "label": "Sunday March 8th",
                "url": "/app/reflections/20260308",
            }
        ),
    )

    response = client.get("/app/home/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "weekly reflection" in html
    assert 'href="/app/reflections/20260308"' in html


def test_home_omits_weekly_reflection_card_when_missing(monkeypatch, journal_copy):
    client = _make_client(str(journal_copy))
    monkeypatch.setattr(
        "solstone.apps.home.routes._build_pulse_context",
        lambda: _minimal_pulse_context(None),
    )

    response = client.get("/app/home/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'href="/app/reflections/' not in html
