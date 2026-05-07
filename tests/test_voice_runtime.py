# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from concurrent.futures import Future

from flask import Flask

from solstone.think.voice import brain
from solstone.think.voice.runtime import (
    get_runtime_state,
    start_voice_runtime,
    stop_all_voice_runtime,
    stop_voice_runtime,
)


def test_start_voice_runtime_attaches_state(monkeypatch, tmp_path):
    app = Flask(__name__)
    monkeypatch.setattr(
        "solstone.think.voice.runtime.get_journal",
        lambda: str((tmp_path / "journal").resolve()),
    )

    start_voice_runtime(app)
    try:
        runtime = get_runtime_state()
        assert app.voice_runtime_started is True
        assert app.voice_tasks == set()
        assert app.voice_journal_root == str((tmp_path / "journal").resolve())
        assert app.voice_brain_instruction == ""
        assert runtime.loop is not None
        assert runtime.thread is not None
    finally:
        stop_voice_runtime(app)


def test_start_voice_runtime_is_idempotent(monkeypatch, tmp_path):
    app = Flask(__name__)
    monkeypatch.setattr(
        "solstone.think.voice.runtime.get_journal",
        lambda: str((tmp_path / "journal").resolve()),
    )

    start_voice_runtime(app)
    runtime = get_runtime_state()
    first_loop = runtime.loop
    first_thread = runtime.thread
    try:
        start_voice_runtime(app)
        assert runtime.loop is first_loop
        assert runtime.thread is first_thread
        assert runtime.apps.count(app) == 1
    finally:
        stop_voice_runtime(app)


def test_stop_voice_runtime_cancels_registered_futures(monkeypatch, tmp_path):
    app = Flask(__name__)
    monkeypatch.setattr(
        "solstone.think.voice.runtime.get_journal",
        lambda: str((tmp_path / "journal").resolve()),
    )
    start_voice_runtime(app)
    pending: Future[None] = Future()
    app.voice_tasks.add(pending)

    stop_voice_runtime(app)

    assert pending.cancelled()
    assert app.voice_runtime_started is False


def test_stop_voice_runtime_cancels_pending_brain_futures(monkeypatch, tmp_path):
    app = Flask(__name__)
    monkeypatch.setattr(
        "solstone.think.voice.runtime.get_journal",
        lambda: str((tmp_path / "journal").resolve()),
    )
    start_voice_runtime(app)
    pending: Future[tuple[str, str]] = Future()
    brain.clear_brain_state()
    brain._BRAIN_STATE.start_future = pending

    stop_voice_runtime(app)

    assert pending.cancelled()


def test_stop_all_voice_runtime_cleans_registered_apps(monkeypatch, tmp_path):
    app = Flask(__name__)
    monkeypatch.setattr(
        "solstone.think.voice.runtime.get_journal",
        lambda: str((tmp_path / "journal").resolve()),
    )
    start_voice_runtime(app)

    stop_all_voice_runtime()

    runtime = get_runtime_state()
    assert runtime.loop is None
    assert runtime.thread is None
