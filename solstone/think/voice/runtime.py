# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Background runtime for voice tasks."""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from solstone.think.utils import get_journal

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    started_event: threading.Event = field(default_factory=threading.Event)
    apps: list[Any] = field(default_factory=list)
    atexit_registered: bool = False


_RUNTIME_STATE = RuntimeState()


def get_runtime_state() -> RuntimeState:
    return _RUNTIME_STATE


def _run_loop(loop: asyncio.AbstractEventLoop, started_event: threading.Event) -> None:
    asyncio.set_event_loop(loop)
    started_event.set()
    loop.run_forever()


def _attach_app(app: Any) -> None:
    app.voice_tasks = set()
    app.voice_journal_root = get_journal()
    app.voice_brain_session = None
    app.voice_brain_instruction = ""
    app.voice_brain_refreshed_at = None
    app.voice_runtime_started = True


def start_voice_runtime(app: Any) -> None:
    _attach_app(app)
    with _RUNTIME_STATE.lock:
        if app not in _RUNTIME_STATE.apps:
            _RUNTIME_STATE.apps.append(app)
        if _RUNTIME_STATE.loop is None or _RUNTIME_STATE.thread is None:
            loop = asyncio.new_event_loop()
            started_event = threading.Event()
            thread = threading.Thread(
                target=_run_loop,
                args=(loop, started_event),
                name="voice-runtime",
                daemon=True,
            )
            _RUNTIME_STATE.loop = loop
            _RUNTIME_STATE.thread = thread
            _RUNTIME_STATE.started_event = started_event
            thread.start()
        if not _RUNTIME_STATE.atexit_registered:
            atexit.register(stop_all_voice_runtime)
            _RUNTIME_STATE.atexit_registered = True
        started_event = _RUNTIME_STATE.started_event
    started_event.wait(timeout=1.0)


def _cancel_app_futures(app: Any) -> None:
    tasks = getattr(app, "voice_tasks", None)
    if not isinstance(tasks, set):
        return
    for future in list(tasks):
        if isinstance(future, Future) and not future.done():
            future.cancel()
    tasks.clear()


def stop_voice_runtime(app: Any) -> None:
    with _RUNTIME_STATE.lock:
        _cancel_app_futures(app)
        if app in _RUNTIME_STATE.apps:
            _RUNTIME_STATE.apps.remove(app)
        should_stop = not _RUNTIME_STATE.apps
        loop = _RUNTIME_STATE.loop if should_stop else None
        thread = _RUNTIME_STATE.thread if should_stop else None
        if should_stop:
            _RUNTIME_STATE.loop = None
            _RUNTIME_STATE.thread = None
            _RUNTIME_STATE.started_event = threading.Event()
    app.voice_runtime_started = False
    if loop is not None:
        from solstone.think.voice import brain

        brain.clear_brain_state()
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=1.0)
        loop.close()


def stop_all_voice_runtime() -> None:
    with _RUNTIME_STATE.lock:
        apps = list(_RUNTIME_STATE.apps)
    for app in apps:
        try:
            stop_voice_runtime(app)
        except Exception:
            logger.exception("voice runtime shutdown failed")


__all__ = [
    "RuntimeState",
    "get_runtime_state",
    "start_voice_runtime",
    "stop_all_voice_runtime",
    "stop_voice_runtime",
]
