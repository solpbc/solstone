# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from convey import create_app


@pytest.fixture
def voice_app(journal_copy):
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True
    return app


@pytest.fixture
def voice_client(voice_app):
    return voice_app.test_client()


def test_session_rejects_non_object_json(voice_client):
    response = voice_client.post("/api/voice/session", json=["bad"])
    assert response.status_code == 400
    assert response.get_json() == {"error": "request body must be a JSON object"}


def test_session_requires_openai_key(voice_client, monkeypatch):
    monkeypatch.setattr("convey.voice.get_openai_api_key", lambda: None)
    response = voice_client.post("/api/voice/session")
    assert response.status_code == 503
    assert response.get_json() == {
        "error": "voice unavailable — openai key not configured"
    }


def test_session_returns_brain_not_ready(voice_client, monkeypatch):
    monkeypatch.setattr("convey.voice.get_openai_api_key", lambda: "sk-test")
    monkeypatch.setattr(
        "convey.voice.brain.wait_until_ready", lambda app, timeout: False
    )
    response = voice_client.post("/api/voice/session")
    assert response.status_code == 503
    assert response.get_json() == {"error": "voice unavailable — brain not ready"}


def test_connect_requires_call_id(voice_client, monkeypatch):
    monkeypatch.setattr("convey.voice.get_openai_api_key", lambda: "sk-test")
    response = voice_client.post("/api/voice/connect", json={})
    assert response.status_code == 400
    assert response.get_json() == {"error": "call_id is required"}


def test_nav_hints_unknown_call_id_returns_empty(voice_client):
    response = voice_client.get("/api/voice/nav-hints?call_id=missing")
    assert response.status_code == 200
    assert response.get_json() == {"hints": [], "consumed": True}


def test_status_reports_all_fields(voice_client, voice_app, monkeypatch):
    pending: Future[None] = Future()
    done: Future[None] = Future()
    done.set_result(None)
    voice_app.voice_tasks.update({pending, done})
    voice_app.voice_brain_instruction = "Ready voice"
    voice_app.voice_brain_refreshed_at = 1.0
    monkeypatch.setattr("convey.voice.get_openai_api_key", lambda: "sk-test")
    monkeypatch.setattr("convey.voice.brain.brain_age_seconds", lambda app: 12)

    response = voice_client.get("/api/voice/status")

    assert response.status_code == 200
    assert response.get_json() == {
        "brain_ready": True,
        "brain_age_seconds": 12,
        "openai_configured": True,
        "active_sessions": 1,
    }


def test_refresh_brain_returns_202_while_running(voice_client, monkeypatch):
    class PendingFuture:
        def result(self, timeout=None):
            raise FutureTimeoutError()

    monkeypatch.setattr(
        "convey.voice.brain.schedule_refresh",
        lambda app, force: PendingFuture(),
    )

    response = voice_client.post("/api/voice/refresh-brain")

    assert response.status_code == 202
    assert response.get_json() == {"status": "refreshing"}


def test_refresh_brain_returns_preview(voice_client, monkeypatch, voice_app):
    future: Future[tuple[str, str]] = Future()
    future.set_result(("session-1", "Voice preview"))
    voice_app.voice_brain_instruction = "Voice preview"
    monkeypatch.setattr(
        "convey.voice.brain.schedule_refresh", lambda app, force: future
    )
    monkeypatch.setattr("convey.voice.brain.brain_age_seconds", lambda app: 0)

    response = voice_client.post("/api/voice/refresh-brain")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "refreshed",
        "instruction_preview": "Voice preview",
        "brain_ready": True,
        "brain_age_seconds": 0,
    }
