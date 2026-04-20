# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from flask import Flask

from convey.chat_stream import append_chat_event, read_chat_events


def _reset_chat_state(chat_module) -> None:
    chat_module.stop_all_chat_runtime()
    with chat_module._state_lock:
        chat_module._current_chat_use_id = None
        chat_module._current_chat_state = None
        chat_module._queued_trigger = None
        chat_module._active_execs.clear()
        chat_module._recovery_day = None
        chat_module._last_use_id = 0


def _setup_journal(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    return journal


def test_chat_result_with_two_active_execs_retriggers_with_max_active_reason(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    append_chat_event(
        "talent_spawned",
        use_id="1713620000001",
        name="exec",
        task="first task",
        started_at=1713620000001,
    )
    append_chat_event(
        "talent_spawned",
        use_id="1713620000002",
        name="exec",
        task="second task",
        started_at=1713620000002,
    )

    actions: list[dict] = []
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: actions.append(action)
    )
    monkeypatch.setattr("convey.chat._emit_finish", lambda *args, **kwargs: None)
    monkeypatch.setattr("convey.chat._emit_error", lambda *args, **kwargs: None)

    with chat._state_lock:
        chat._current_chat_use_id = "1713620000100"
        chat._current_chat_state = {
            "raw_use_id": "1713620000101",
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }

    chat._on_cortex_finish(
        {
            "use_id": "1713620000101",
            "result": (
                '{"message":"I am looking into that.","notes":"need exec",'
                '"talent_request":{"task":"research it","context":{"k":"v"}}}'
            ),
        }
    )

    assert actions
    assert actions[-1]["kind"] == "chat"
    assert actions[-1]["trigger"] == {
        "type": "synthetic-max-active",
        "reason": "max active — waiting for one to finish",
    }

    sol_messages = [
        e for e in read_chat_events(chat._today_day()) if e["kind"] == "sol_message"
    ]
    assert sol_messages[-1]["requested_exec"] is True
    assert sol_messages[-1]["requested_task"] == "research it"


def test_exec_retrigger_loop_stops_after_three_without_owner_reset(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    append_chat_event(
        "owner_message",
        text="dig deeper",
        app="sol",
        path="/app/sol",
        facet="work",
    )
    for index in range(3):
        append_chat_event(
            "talent_finished",
            use_id=f"171362100000{index}",
            name="exec",
            summary=f"summary {index}",
        )
        if index < 2:
            append_chat_event(
                "sol_message",
                use_id="1713621999999",
                text=f"follow up {index}",
                notes="retrying",
                requested_exec=True,
                requested_task=f"task {index}",
            )

    emitted_errors: list[tuple[str, str]] = []
    actions: list[dict | None] = []
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: actions.append(action)
    )
    monkeypatch.setattr("convey.chat._emit_finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "convey.chat._emit_error",
        lambda use_id, reason: emitted_errors.append((use_id, reason)),
    )

    with chat._state_lock:
        chat._current_chat_use_id = "1713621999999"
        chat._current_chat_state = {
            "raw_use_id": "1713622000000",
            "trigger": {"type": "talent_finished", "summary": "summary 2"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }

    chat._on_cortex_finish(
        {
            "use_id": "1713622000000",
            "result": (
                '{"message":"Still digging.","notes":"loop",'
                '"talent_request":{"task":"one more pass","context":{}}}'
            ),
        }
    )

    assert emitted_errors == [("1713621999999", "chat had trouble — try again")]
    assert actions == [None]
    errors = [
        e for e in read_chat_events(chat._today_day()) if e["kind"] == "chat_error"
    ]
    assert errors[-1]["reason"] == "chat had trouble — try again"


def test_cortex_finish_and_error_append_exec_terminal_events_by_use_id(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    actions: list[dict] = []
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: actions.append(action)
    )
    monkeypatch.setattr("convey.chat._emit_finish", lambda *args, **kwargs: None)
    monkeypatch.setattr("convey.chat._emit_error", lambda *args, **kwargs: None)

    with chat._state_lock:
        chat._current_chat_use_id = "1713623000000"
        chat._current_chat_state = {
            "raw_use_id": None,
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }
        chat._active_execs["1713623000001"] = {
            "chat_use_id": "1713623000000",
            "task": "summarize",
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
        }

    chat._on_cortex_finish({"use_id": "1713623000001", "result": "done"})
    finished_events = [
        e for e in read_chat_events(chat._today_day()) if e["kind"] == "talent_finished"
    ]
    assert finished_events[-1]["use_id"] == "1713623000001"
    assert actions[-1]["trigger"]["type"] == "talent_finished"

    _reset_chat_state(chat)
    actions.clear()
    with chat._state_lock:
        chat._current_chat_use_id = "1713624000000"
        chat._current_chat_state = {
            "raw_use_id": None,
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }
        chat._active_execs["1713624000001"] = {
            "chat_use_id": "1713624000000",
            "task": "summarize",
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
        }

    chat._on_cortex_error({"use_id": "1713624000001", "error": "boom"})
    errored_events = [
        e for e in read_chat_events(chat._today_day()) if e["kind"] == "talent_errored"
    ]
    assert errored_events[-1]["use_id"] == "1713624000001"
    assert actions[-1]["trigger"]["type"] == "talent_errored"
    assert actions[-1]["trigger"]["reason"] == "boom"


def test_start_chat_runtime_recovers_exactly_one_unresponded_trigger(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    append_chat_event(
        "owner_message",
        text="recover me",
        app="sol",
        path="/app/sol",
        facet="work",
    )

    starts: list[dict] = []
    monkeypatch.setattr(
        "convey.chat.CallosumConnection.start", lambda self, callback=None: None
    )
    monkeypatch.setattr("convey.chat.CallosumConnection.stop", lambda self: None)
    monkeypatch.setattr(
        "convey.chat._spawn_chat_generate", lambda action: starts.append(action) or True
    )

    app = Flask(__name__)
    chat.start_chat_runtime(app)
    chat.start_chat_runtime(app)

    assert len(starts) == 1


def test_chat_generate_schema_violation_retries_once_then_chat_errors(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    actions: list[dict | None] = []
    emitted_errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: actions.append(action)
    )
    monkeypatch.setattr("convey.chat._emit_finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "convey.chat._emit_error",
        lambda use_id, reason: emitted_errors.append((use_id, reason)),
    )

    with chat._state_lock:
        chat._current_chat_use_id = "1713625000000"
        chat._current_chat_state = {
            "raw_use_id": "1713625000001",
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }

    chat._on_cortex_finish({"use_id": "1713625000001", "result": "not json"})

    assert actions and actions[-1]["kind"] == "chat"
    assert actions[-1]["logical_use_id"] == "1713625000000"
    assert emitted_errors == []

    with chat._state_lock:
        retry_use_id = chat._current_chat_state["raw_use_id"]

    chat._on_cortex_finish({"use_id": retry_use_id, "result": "still not json"})

    assert emitted_errors == [("1713625000000", "chat had trouble — try again")]
    errors = [
        e for e in read_chat_events(chat._today_day()) if e["kind"] == "chat_error"
    ]
    assert errors[-1]["use_id"] == "1713625000000"
