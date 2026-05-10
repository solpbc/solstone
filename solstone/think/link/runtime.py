# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Background runtime for link route helpers in the Convey process."""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .interface_watcher import (
    InterfaceWatcher,
    set_interface_watcher,
)

logger = logging.getLogger("link.runtime")


@dataclass
class RuntimeState:
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    started_event: threading.Event = field(default_factory=threading.Event)
    apps: list[Any] = field(default_factory=list)
    watcher: InterfaceWatcher = field(default_factory=InterfaceWatcher)


_RUNTIME_LOCK = threading.Lock()
_runtime: RuntimeState | None = None
_atexit_registered = False


def get_runtime_state() -> RuntimeState | None:
    return _runtime


def _thread_main(runtime: RuntimeState) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runtime.loop = loop
    set_interface_watcher(runtime.watcher)
    loop.run_until_complete(runtime.watcher.start())
    runtime.started_event.set()
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(runtime.watcher.stop())
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        set_interface_watcher(None)
        loop.close()


def start_link_runtime(app: Any) -> None:
    """Start the interface watcher in the Convey process."""
    global _runtime, _atexit_registered

    with _RUNTIME_LOCK:
        if _runtime is None:
            runtime = RuntimeState()
            thread = threading.Thread(
                target=_thread_main,
                args=(runtime,),
                name="link-runtime",
                daemon=True,
            )
            runtime.thread = thread
            _runtime = runtime
            thread.start()
        runtime = _runtime
        if app not in runtime.apps:
            runtime.apps.append(app)
        app.link_runtime_started = True
        if not _atexit_registered:
            atexit.register(stop_all_link_runtime)
            _atexit_registered = True
        started_event = runtime.started_event
    started_event.wait(timeout=1.0)


def stop_link_runtime(app: Any) -> None:
    runtime = _runtime
    app.link_runtime_started = False
    if runtime is None:
        return
    with _RUNTIME_LOCK:
        if app in runtime.apps:
            runtime.apps.remove(app)
        remaining = list(runtime.apps)
    if not remaining:
        stop_all_link_runtime()


def stop_all_link_runtime() -> None:
    global _runtime

    with _RUNTIME_LOCK:
        runtime = _runtime
        _runtime = None
    if runtime is None:
        return
    for app in list(runtime.apps):
        try:
            app.link_runtime_started = False
        except Exception:
            logger.exception("link runtime app cleanup failed")
    if runtime.loop is not None:
        runtime.loop.call_soon_threadsafe(runtime.loop.stop)
    if runtime.thread is not None:
        runtime.thread.join(timeout=1.0)
    set_interface_watcher(None)


__all__ = [
    "RuntimeState",
    "get_runtime_state",
    "start_link_runtime",
    "stop_all_link_runtime",
    "stop_link_runtime",
]
