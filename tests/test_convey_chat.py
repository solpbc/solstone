# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from datetime import datetime

import pytest
from flask import Flask

from convey.chat import chat_bp
from convey.chat_stream import append_chat_event


def _setup_journal(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    return journal


def _reset_chat_state(chat_module) -> None:
    chat_module.stop_all_chat_runtime()
    with chat_module._state_lock:
        chat_module._current_chat_use_id = None
        chat_module._current_chat_state = None
        chat_module._queued_trigger = None
        chat_module._active_execs.clear()
        chat_module._recovery_day = None
        chat_module._last_use_id = 0


def _ms(year: int, month: int, day: int, hour: int, minute: int, second: int) -> int:
    return int(datetime(year, month, day, hour, minute, second).timestamp() * 1000)


@pytest.fixture
def chat_client(tmp_path, monkeypatch):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(chat_bp)
    return app.test_client()


def test_post_chat_appends_owner_message_and_returns_reserved_use_id(
    chat_client, monkeypatch
):
    starts: list[dict] = []
    monkeypatch.setattr("think.identity.ensure_identity_directory", lambda: None)
    monkeypatch.setattr(
        "convey.chat._spawn_chat_generate", lambda action: starts.append(action) or True
    )

    response = chat_client.post(
        "/api/chat",
        json={
            "message": "hello there",
            "app": "sol",
            "path": "/app/sol",
            "facet": "work",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["queued"] is False
    assert payload["use_id"].isdigit()
    assert starts and starts[-1]["logical_use_id"] == payload["use_id"]


def test_session_endpoint_reduces_from_chat_stream(chat_client):
    day = "20260420"
    append_chat_event(
        "sol_message",
        ts=_ms(2026, 4, 20, 12, 0, 0),
        use_id="1713626000000",
        text="hello",
        notes="ready",
        requested_exec=False,
        requested_task=None,
    )
    append_chat_event(
        "talent_spawned",
        ts=_ms(2026, 4, 20, 12, 1, 0),
        use_id="1713626000001",
        name="exec",
        task="research",
        started_at=1713626000001,
    )

    response = chat_client.get("/api/chat/session")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["latest_sol_message"]["text"] == "hello"
    assert payload["active_talents"][0]["task"] == "research"
    assert chat_client.get(f"/api/chat/stream/{day}").status_code == 200


def test_stream_endpoint_ordered_with_limit(chat_client):
    start = _ms(2026, 4, 20, 12, 0, 0)
    for index in range(4):
        append_chat_event(
            "owner_message",
            ts=start + (index * 300_000),
            text=f"m{index}",
            app="sol",
            path="/app/sol",
            facet="work",
        )

    response = chat_client.get("/api/chat/stream/20260420?limit=2")
    assert response.status_code == 200
    payload = response.get_json()
    assert [event["text"] for event in payload["events"]] == ["m2", "m3"]


def test_result_endpoint_reads_stream_not_talent_log(chat_client, tmp_path):
    use_id = str(_ms(2026, 4, 20, 12, 0, 0))
    append_chat_event(
        "sol_message",
        use_id=use_id,
        text="stream reply",
        notes="done",
        requested_exec=False,
        requested_task=None,
    )

    talents_dir = tmp_path / "journal" / "talents" / "chat"
    talents_dir.mkdir(parents=True, exist_ok=True)
    (talents_dir / f"{use_id}.jsonl").write_text(
        '{"event":"finish","result":"log reply"}\n'
    )

    response = chat_client.get(f"/api/chat/result/{use_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "finished"
    assert payload["summary"] == "stream reply"
