# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
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
        chat_module._active_talents.clear()
        chat_module._last_use_id = 0


def _ms(year: int, month: int, day: int, hour: int, minute: int, second: int) -> int:
    return int(datetime(year, month, day, hour, minute, second).timestamp() * 1000)


def _write_talent_log(
    journal, talent_name: str, filename: str, events: list[dict]
) -> None:
    talent_dir = journal / "talents" / talent_name
    talent_dir.mkdir(parents=True, exist_ok=True)
    log_path = talent_dir / filename
    log_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


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


def test_session_endpoint_reduces_from_chat_stream(chat_client, monkeypatch):
    day = "20260420"
    monkeypatch.setattr("convey.chat._today_day", lambda: day)
    started_at = _ms(2026, 4, 20, 12, 1, 0)
    finished_at = _ms(2026, 4, 20, 12, 2, 0)
    append_chat_event(
        "sol_message",
        ts=_ms(2026, 4, 20, 12, 0, 0),
        use_id="1713626000000",
        text="hello",
        notes="ready",
        requested_target=None,
        requested_task=None,
    )
    append_chat_event(
        "talent_spawned",
        ts=started_at,
        use_id="1713626000001",
        name="exec",
        task="research",
        started_at=started_at,
    )
    append_chat_event(
        "talent_finished",
        ts=finished_at,
        use_id="1713626000001",
        name="exec",
        summary="done",
    )

    response = chat_client.get("/api/chat/session")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["latest_sol_message"]["text"] == "hello"
    assert payload["active_talents"] == []
    assert payload["completed_talents"] == [
        {
            "finished_at": finished_at,
            "name": "exec",
            "summary": "done",
            "task": "research",
            "use_id": "1713626000001",
        }
    ]
    assert chat_client.get(f"/api/chat/stream/{day}").status_code == 200


def test_chat_session_retries_unresolved_trigger_when_idle(chat_client, monkeypatch):
    day = "20260420"
    monkeypatch.setattr("convey.chat._today_day", lambda: day)
    append_chat_event(
        "owner_message",
        ts=_ms(2026, 4, 20, 12, 0, 0),
        text="retry me",
        app="sol",
        path="/app/sol",
        facet="work",
    )

    starts: list[dict] = []
    monkeypatch.setattr(
        "convey.chat._spawn_chat_generate", lambda action: starts.append(action) or True
    )

    response = chat_client.get("/api/chat/session")

    assert response.status_code == 200
    assert len(starts) == 1
    assert starts[0]["trigger"]["type"] == "owner_message"


def test_chat_session_retries_again_when_spawn_fails_and_trigger_remains_unresolved(
    chat_client, monkeypatch
):
    day = "20260420"
    monkeypatch.setattr("convey.chat._today_day", lambda: day)
    append_chat_event(
        "owner_message",
        ts=_ms(2026, 4, 20, 12, 0, 0),
        text="retry me again",
        app="sol",
        path="/app/sol",
        facet="work",
    )

    starts: list[dict] = []

    def fake_spawn(action):
        starts.append(action)
        return len(starts) > 1

    monkeypatch.setattr("convey.chat._spawn_chat_generate", fake_spawn)
    monkeypatch.setattr("convey.chat._emit_error", lambda *_args, **_kwargs: None)

    first = chat_client.get("/api/chat/session")
    second = chat_client.get("/api/chat/session")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(starts) == 2
    assert starts[0]["trigger"]["type"] == "owner_message"
    assert starts[1]["trigger"]["type"] == "owner_message"


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
        ts=_ms(2026, 4, 20, 12, 0, 0),
        use_id=use_id,
        text="stream reply",
        notes="done",
        requested_target=None,
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


def test_talent_log_endpoint_returns_completed_run(chat_client, tmp_path):
    use_id = "1700000000001"
    _write_talent_log(
        tmp_path / "journal",
        "default",
        f"{use_id}.jsonl",
        [
            {
                "event": "request",
                "ts": 1700000000001,
                "use_id": use_id,
                "prompt": "Search for meetings about project updates",
                "name": "default",
                "provider": "openai",
            },
            {
                "event": "start",
                "ts": 1700000000100,
                "use_id": use_id,
                "model": "gpt-4o",
                "provider": "openai",
            },
            {
                "event": "thinking",
                "ts": 1700000000300,
                "use_id": use_id,
                "content": "reasoning",
                "raw": {"provider": "openai"},
            },
            {
                "event": "finish",
                "ts": 1700000000600,
                "use_id": use_id,
                "result": "done",
            },
        ],
    )

    response = chat_client.get(f"/api/chat/talent-log/{use_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["use_id"] == use_id
    assert payload["status"] == "completed"
    assert payload["task"] == "Search for meetings about project updates"
    assert payload["started_at"] == 1700000000100
    assert payload["finished_at"] == 1700000000600
    assert len(payload["events"]) == 3
    assert payload["events"][1]["event"] == "thinking"
    assert "raw" not in payload["events"][1]


def test_talent_log_endpoint_returns_running_active_run(chat_client, tmp_path):
    use_id = "1700000000002"
    _write_talent_log(
        tmp_path / "journal",
        "default",
        f"{use_id}_active.jsonl",
        [
            {
                "event": "request",
                "ts": 1700000000002,
                "use_id": use_id,
                "task": "Analyze conversation flow",
            },
            {
                "event": "start",
                "ts": 1700000000102,
                "use_id": use_id,
                "model": "gpt-4o-mini",
            },
            {
                "event": "thinking",
                "ts": 1700000000202,
                "use_id": use_id,
                "content": "still working",
            },
        ],
    )

    response = chat_client.get(f"/api/chat/talent-log/{use_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "running"
    assert payload["task"] == "Analyze conversation flow"
    assert payload["finished_at"] is None
    assert payload["events"][-1]["event"] == "thinking"


def test_talent_log_endpoint_prefers_active_log(chat_client, tmp_path):
    use_id = "1700000000003"
    journal = tmp_path / "journal"
    _write_talent_log(
        journal,
        "default",
        f"{use_id}_active.jsonl",
        [
            {
                "event": "request",
                "ts": 1700000000003,
                "use_id": use_id,
                "prompt": "active prompt",
            },
            {
                "event": "thinking",
                "ts": 1700000000103,
                "use_id": use_id,
                "content": "active content",
            },
        ],
    )
    _write_talent_log(
        journal,
        "flow",
        f"{use_id}.jsonl",
        [
            {
                "event": "request",
                "ts": 1700000000003,
                "use_id": use_id,
                "prompt": "completed prompt",
            },
            {
                "event": "finish",
                "ts": 1700000000203,
                "use_id": use_id,
                "result": "completed result",
            },
        ],
    )

    response = chat_client.get(f"/api/chat/talent-log/{use_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "running"
    assert payload["task"] == "active prompt"
    assert payload["events"][0]["content"] == "active content"


def test_talent_log_endpoint_returns_errored_run(chat_client, tmp_path):
    use_id = "1700000000004"
    _write_talent_log(
        tmp_path / "journal",
        "flow",
        f"{use_id}.jsonl",
        [
            {
                "event": "request",
                "ts": 1700000000004,
                "use_id": use_id,
                "prompt": "Analyze flow",
            },
            {
                "event": "error",
                "ts": 1700000000204,
                "use_id": use_id,
                "error": "Rate limit exceeded",
            },
        ],
    )

    response = chat_client.get(f"/api/chat/talent-log/{use_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "errored"
    assert payload["finished_at"] == 1700000000204
    assert payload["events"][-1]["event"] == "error"


def test_talent_log_endpoint_returns_missing(chat_client):
    use_id = "1700000000999"

    response = chat_client.get(f"/api/chat/talent-log/{use_id}")

    assert response.status_code == 404
    assert response.get_json() == {"error": f"Talent log not found for use_id {use_id}"}


def test_talent_log_endpoint_task_falls_back_to_prompt(chat_client, tmp_path):
    use_id = "1700000000005"
    _write_talent_log(
        tmp_path / "journal",
        "default",
        f"{use_id}.jsonl",
        [
            {
                "event": "request",
                "ts": 1700000000005,
                "use_id": use_id,
                "prompt": "Fallback prompt",
            },
            {
                "event": "finish",
                "ts": 1700000000305,
                "use_id": use_id,
                "result": "done",
            },
        ],
    )

    response = chat_client.get(f"/api/chat/talent-log/{use_id}")

    assert response.status_code == 200
    assert response.get_json()["task"] == "Fallback prompt"
