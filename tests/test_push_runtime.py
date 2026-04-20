# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import pytest
from flask import Flask

from think.push.runtime import (
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
        "think.push.runtime.CallosumConnection.start",
        lambda self, callback=None: calls.append("start"),
    )
    monkeypatch.setattr(
        "think.push.runtime.CallosumConnection.stop",
        lambda self: calls.append("stop"),
    )
    app = Flask(__name__)

    start_push_runtime(app)
    try:
        runtime = get_runtime_state()
        assert app.push_runtime_started is True
        assert runtime is not None
        assert runtime.loop is not None
        assert runtime.thread is not None
        assert calls == ["start"]
    finally:
        stop_push_runtime(app)


def test_start_push_runtime_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        "think.push.runtime.CallosumConnection.start", lambda self, callback=None: None
    )
    monkeypatch.setattr("think.push.runtime.CallosumConnection.stop", lambda self: None)
    app = Flask(__name__)

    start_push_runtime(app)
    runtime = get_runtime_state()
    first_loop = runtime.loop if runtime else None
    first_thread = runtime.thread if runtime else None
    try:
        start_push_runtime(app)
        runtime = get_runtime_state()
        assert runtime is not None
        assert runtime.loop is first_loop
        assert runtime.thread is first_thread
        assert runtime.apps.count(app) == 1
    finally:
        stop_push_runtime(app)


def test_stop_push_runtime_cleans_last_app(monkeypatch):
    monkeypatch.setattr(
        "think.push.runtime.CallosumConnection.start", lambda self, callback=None: None
    )
    monkeypatch.setattr("think.push.runtime.CallosumConnection.stop", lambda self: None)
    app = Flask(__name__)

    start_push_runtime(app)
    stop_push_runtime(app)

    assert app.push_runtime_started is False
    assert get_runtime_state() is None


def test_stop_all_push_runtime_clears_runtime(monkeypatch):
    monkeypatch.setattr(
        "think.push.runtime.CallosumConnection.start", lambda self, callback=None: None
    )
    monkeypatch.setattr("think.push.runtime.CallosumConnection.stop", lambda self: None)
    app = Flask(__name__)

    start_push_runtime(app)
    stop_all_push_runtime()

    assert app.push_runtime_started is False
    assert get_runtime_state() is None
