# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import pytest
from flask import Flask

from solstone.think.push import runtime
from solstone.think.push.runtime import (
    get_runtime_state,
    start_push_runtime,
    stop_all_push_runtime,
    stop_push_runtime,
)


@pytest.fixture(autouse=True)
def reset_runtime():
    stop_all_push_runtime()
    yield
    stop_all_push_runtime()


def test_start_push_runtime_attaches_state(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.start",
        lambda self, callback=None: calls.append("start"),
    )
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.stop",
        lambda self: calls.append("stop"),
    )
    app = Flask(__name__)

    start_push_runtime(app)
    try:
        runtime_state = get_runtime_state()
        assert app.push_runtime_started is True
        assert runtime_state is not None
        assert runtime_state.loop is not None
        assert runtime_state.thread is not None
        assert calls == ["start"]
    finally:
        stop_push_runtime(app)


def test_start_push_runtime_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.start",
        lambda self, callback=None: None,
    )
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.stop", lambda self: None
    )
    app = Flask(__name__)

    start_push_runtime(app)
    runtime_state = get_runtime_state()
    first_loop = runtime_state.loop if runtime_state else None
    first_thread = runtime_state.thread if runtime_state else None
    try:
        start_push_runtime(app)
        runtime_state = get_runtime_state()
        assert runtime_state is not None
        assert runtime_state.loop is first_loop
        assert runtime_state.thread is first_thread
        assert runtime_state.apps.count(app) == 1
    finally:
        stop_push_runtime(app)


def test_stop_push_runtime_cleans_last_app(monkeypatch):
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.start",
        lambda self, callback=None: None,
    )
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.stop", lambda self: None
    )
    app = Flask(__name__)

    start_push_runtime(app)
    stop_push_runtime(app)

    assert app.push_runtime_started is False
    assert get_runtime_state() is None


def test_stop_all_push_runtime_clears_runtime(monkeypatch):
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.start",
        lambda self, callback=None: None,
    )
    monkeypatch.setattr(
        "solstone.think.push.runtime.CallosumConnection.stop", lambda self: None
    )
    app = Flask(__name__)

    start_push_runtime(app)
    stop_all_push_runtime()

    assert app.push_runtime_started is False
    assert get_runtime_state() is None


def test_on_callosum_message_calls_both_handlers(monkeypatch):
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(
        runtime.triggers,
        "handle_briefing_finish",
        lambda message: calls.append(("briefing", message)),
    )
    monkeypatch.setattr(
        runtime.triggers,
        "handle_weekly_reflection_finish",
        lambda message: calls.append(("weekly_reflection", message)),
    )
    message = {"tract": "cortex", "event": "finish", "name": "weekly_reflection"}

    runtime._on_callosum_message(message)

    assert calls == [("briefing", message), ("weekly_reflection", message)]
