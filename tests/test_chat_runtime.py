# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import pytest
from flask import Flask

from convey.chat_stream import append_chat_event, read_chat_events


def _reset_chat_state(chat_module) -> None:
    chat_module.stop_all_chat_runtime()
    with chat_module._state_lock:
        chat_module._current_chat_use_id = None
        chat_module._current_chat_state = None
        chat_module._queued_trigger = None
        chat_module._active_talents.clear()
        for timer in chat_module._watchdog_timers.values():
            timer.cancel()
        chat_module._watchdog_timers.clear()
        chat_module._last_use_id = 0


def _setup_journal(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    return journal


def _install_fake_timers(monkeypatch):
    timers: list[FakeTimer] = []

    class FakeTimer:
        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = tuple(args or ())
            self.kwargs = dict(kwargs or {})
            self.started = False
            self.cancelled = False
            self.daemon = False
            timers.append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

        def fire(self) -> None:
            if self.cancelled:
                return
            self.function(*self.args, **self.kwargs)

    monkeypatch.setattr("convey.chat.threading.Timer", FakeTimer)
    return timers


def test_chat_result_with_two_active_talents_retriggers_with_max_active_reason(
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
    assert sol_messages[-1]["requested_target"] == "exec"
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
                requested_target="exec",
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
        chat._active_talents["1713623000001"] = {
            "chat_use_id": "1713623000000",
            "target": "exec",
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
        chat._active_talents["1713624000001"] = {
            "chat_use_id": "1713624000000",
            "target": "exec",
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


def test_exec_dispatch_appends_sol_message_and_spawns_talent_real_path(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)
    timers = _install_fake_timers(monkeypatch)

    spawn_calls: list[dict[str, object]] = []
    monkeypatch.setattr("convey.chat._emit_cortex_event", lambda *args, **kwargs: None)

    def fake_spawn_agent(prompt, name, provider=None, config=None, use_id=None):
        spawn_calls.append(
            {
                "prompt": prompt,
                "name": name,
                "provider": provider,
                "config": config,
                "use_id": use_id,
            }
        )
        return use_id

    monkeypatch.setattr("convey.utils.spawn_agent", fake_spawn_agent)

    with chat._state_lock:
        start_info = chat._activate_current_locked(
            "1713625500000",
            {"type": "owner_message", "message": "help"},
            {"app": "sol", "path": "/app/sol", "facet": "work"},
        )

    raw_use_id = start_info["raw_use_id"]
    chat._on_cortex_finish(
        {
            "use_id": raw_use_id,
            "result": (
                '{"message":"I am looking into that.","notes":"need exec",'
                '"talent_request":{"target":"exec","task":"research it",'
                '"context":{"k":"v"}}}'
            ),
        }
    )

    events = read_chat_events(chat._today_day())
    sol_messages = [event for event in events if event["kind"] == "sol_message"]
    spawned_events = [event for event in events if event["kind"] == "talent_spawned"]

    assert sol_messages[-1]["text"] == "I am looking into that."
    assert sol_messages[-1]["requested_target"] == "exec"
    assert sol_messages[-1]["requested_task"] == "research it"
    assert spawned_events[-1]["name"] == "exec"
    assert spawned_events[-1]["task"] == "research it"
    assert spawn_calls == [
        {
            "prompt": "Task: research it\n\nContext hints:\n{'k': 'v'}\n\n"
            "Location: app=sol path=/app/sol facet=work\n\n"
            "Recent chat:\n**Sol**: I am looking into that.",
            "name": "exec",
            "provider": None,
            "config": {
                "app": "sol",
                "path": "/app/sol",
                "facet": "work",
                "chat_parent_use_id": "1713625500000",
            },
            "use_id": spawned_events[-1]["use_id"],
        }
    ]
    assert len(timers) == 2
    assert timers[0].cancelled is True
    with chat._state_lock:
        assert spawned_events[-1]["use_id"] in chat._active_talents


def test_chat_watchdog_times_out_current_chat_generate(tmp_path, monkeypatch):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)
    timers = _install_fake_timers(monkeypatch)

    emitted_errors: list[tuple[str, str]] = []
    run_actions: list[dict | None] = []
    monkeypatch.setattr(
        "convey.chat._emit_error",
        lambda use_id, reason: emitted_errors.append((use_id, reason)),
    )
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: run_actions.append(action)
    )

    with chat._state_lock:
        start_info = chat._activate_current_locked(
            "1713628000000",
            {"type": "owner_message", "message": "help"},
            {"app": "sol", "path": "/app/sol", "facet": "work"},
        )

    raw_use_id = start_info["raw_use_id"]
    assert raw_use_id in chat._watchdog_timers

    timers[-1].fire()

    errors = [
        event
        for event in read_chat_events(chat._today_day())
        if event["kind"] == "chat_error"
    ]
    assert emitted_errors == [("1713628000000", "chat took too long — try again")]
    assert run_actions == [None]
    assert errors[-1]["use_id"] == "1713628000000"
    assert errors[-1]["reason"] == "chat took too long — try again"
    with chat._state_lock:
        assert chat._current_chat_use_id is None
        assert chat._current_chat_state is None
        assert raw_use_id not in chat._watchdog_timers


def test_chat_watchdog_times_out_active_talent_and_clears_blocked_chat(
    tmp_path, monkeypatch
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)
    timers = _install_fake_timers(monkeypatch)

    emitted_errors: list[tuple[str, str]] = []
    monkeypatch.setattr("convey.chat._emit_cortex_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "convey.chat._emit_error",
        lambda use_id, reason: emitted_errors.append((use_id, reason)),
    )
    monkeypatch.setattr(
        "convey.utils.spawn_agent", lambda *args, **kwargs: kwargs["use_id"]
    )

    with chat._state_lock:
        chat._current_chat_use_id = "1713629000000"
        chat._current_chat_state = {
            "raw_use_id": None,
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }
        chat._active_talents["1713629000001"] = {
            "chat_use_id": "1713629000000",
            "target": "exec",
            "task": "summarize",
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
        }

    chat._run_next_action(
        {
            "kind": "talent",
            "logical_use_id": "1713629000000",
            "target": "exec",
            "use_id": "1713629000001",
            "task": "summarize",
            "context": {},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
        }
    )

    assert "1713629000001" in chat._watchdog_timers
    timers[-1].fire()

    errors = [
        event
        for event in read_chat_events(chat._today_day())
        if event["kind"] == "chat_error"
    ]
    assert emitted_errors == [("1713629000000", "chat took too long — try again")]
    assert errors[-1]["use_id"] == "1713629000000"
    assert errors[-1]["reason"] == "chat took too long — try again"
    with chat._state_lock:
        assert "1713629000001" not in chat._active_talents
        assert chat._current_chat_use_id is None
        assert chat._current_chat_state is None
        assert "1713629000001" not in chat._watchdog_timers


def test_cortex_finish_logs_warning_for_unrouteable_use_id(
    tmp_path, monkeypatch, caplog
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    with caplog.at_level("WARNING"):
        chat._on_cortex_finish({"use_id": "1713630000000", "result": "done"})

    assert (
        "unrouteable cortex event use_id=1713630000000 event=finish "
        "reason=no matching active chat-generate or talent"
    ) in caplog.text


def test_cortex_error_logs_warning_for_unrouteable_use_id(
    tmp_path, monkeypatch, caplog
):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    with caplog.at_level("WARNING"):
        chat._on_cortex_error({"use_id": "1713631000000", "error": "boom"})

    assert (
        "unrouteable cortex event use_id=1713631000000 event=error "
        "reason=no matching active chat-generate or talent"
    ) in caplog.text


def test_parse_chat_result_accepts_reflection_target():
    import convey.chat as chat

    parsed = chat._parse_chat_result(
        {
            "message": "Let me think about that.",
            "notes": "dispatch reflection",
            "talent_request": {
                "target": "reflection",
                "task": "Reflect on the last week",
                "context": {"facet": "work"},
            },
        }
    )

    assert parsed["talent_request"] == {
        "target": "reflection",
        "task": "Reflect on the last week",
        "context": {"facet": "work"},
    }


def test_parse_chat_result_rejects_unknown_target():
    import convey.chat as chat

    with pytest.raises(ValueError, match="unknown talent target: foo"):
        chat._parse_chat_result(
            {
                "message": "Let me think about that.",
                "notes": "dispatch reflection",
                "talent_request": {"target": "foo", "task": "Reflect on the week"},
            }
        )


def test_parse_chat_result_defaults_legacy_target_to_exec():
    import convey.chat as chat

    parsed = chat._parse_chat_result(
        {
            "message": "I am looking into that.",
            "notes": "need exec",
            "talent_request": {"task": "Research it", "context": {"k": "v"}},
        }
    )

    assert parsed["talent_request"] == {
        "target": "exec",
        "task": "Research it",
        "context": {"k": "v"},
    }


def test_reflection_dispatch_spawns_reflection_talent(tmp_path, monkeypatch):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    actions: list[dict | None] = []
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: actions.append(action)
    )
    monkeypatch.setattr("convey.chat._emit_finish", lambda *args, **kwargs: None)
    monkeypatch.setattr("convey.chat._emit_error", lambda *args, **kwargs: None)

    with chat._state_lock:
        chat._current_chat_use_id = "1713626000000"
        chat._current_chat_state = {
            "raw_use_id": "1713626000001",
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }

    chat._on_cortex_finish(
        {
            "use_id": "1713626000001",
            "result": (
                '{"message":"I want to sit with that.","notes":"need reflection",'
                '"talent_request":{"target":"reflection","task":"Reflect on the week",'
                '"context":{"facet":"work"}}}'
            ),
        }
    )

    assert actions[-1]["kind"] == "talent"
    assert actions[-1]["target"] == "reflection"

    events = read_chat_events(chat._today_day())
    sol_message = next(event for event in events if event["kind"] == "sol_message")
    spawned = next(event for event in events if event["kind"] == "talent_spawned")
    assert sol_message["requested_target"] == "reflection"
    assert spawned["name"] == "reflection"


def test_reflection_finish_retriggers_chat_like_exec(tmp_path, monkeypatch):
    import convey.chat as chat

    _setup_journal(tmp_path, monkeypatch)
    _reset_chat_state(chat)

    actions: list[dict | None] = []
    monkeypatch.setattr(
        "convey.chat._run_next_action", lambda action: actions.append(action)
    )
    monkeypatch.setattr("convey.chat._emit_finish", lambda *args, **kwargs: None)
    monkeypatch.setattr("convey.chat._emit_error", lambda *args, **kwargs: None)

    with chat._state_lock:
        chat._current_chat_use_id = "1713627000000"
        chat._current_chat_state = {
            "raw_use_id": None,
            "trigger": {"type": "owner_message", "message": "help"},
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
            "retry_count": 0,
        }
        chat._active_talents["1713627000001"] = {
            "chat_use_id": "1713627000000",
            "target": "reflection",
            "task": "Reflect on the week",
            "location": {"app": "sol", "path": "/app/sol", "facet": "work"},
        }

    chat._on_cortex_finish({"use_id": "1713627000001", "result": "A reflective note"})

    finished_events = [
        e for e in read_chat_events(chat._today_day()) if e["kind"] == "talent_finished"
    ]
    assert finished_events[-1]["name"] == "reflection"
    assert actions[-1]["trigger"]["type"] == "talent_finished"
    assert actions[-1]["trigger"]["name"] == "reflection"
