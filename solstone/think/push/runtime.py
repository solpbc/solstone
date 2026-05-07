# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Background runtime for push tasks."""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from solstone.think.callosum import CallosumConnection
from solstone.think.push import triggers

logger = logging.getLogger("solstone.push.runtime")


@dataclass
class RuntimeState:
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    started_event: threading.Event = field(default_factory=threading.Event)
    apps: list[Any] = field(default_factory=list)
    callosum: CallosumConnection | None = None
    periodic_task: asyncio.Task[Any] | None = None


_RUNTIME_LOCK = threading.Lock()
_runtime: RuntimeState | None = None
_atexit_registered = False


def get_runtime_state() -> RuntimeState | None:
    return _runtime


def _on_callosum_message(message: dict[str, Any]) -> None:
    try:
        triggers.handle_briefing_finish(message)
        triggers.handle_weekly_reflection_finish(message)
    except Exception:
        logger.exception("push callosum handler failed")


async def _periodic_loop() -> None:
    while True:
        await asyncio.sleep(60)
        try:
            triggers.check_pre_meeting_prep(datetime.now())
        except Exception:
            logger.exception("push periodic check failed")


def _thread_main(runtime: RuntimeState) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runtime.loop = loop
    runtime.callosum = CallosumConnection()
    runtime.callosum.start(callback=_on_callosum_message)
    runtime.periodic_task = loop.create_task(_periodic_loop())
    runtime.started_event.set()
    try:
        loop.run_forever()
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def start_push_runtime(app: Any) -> None:
    global _runtime, _atexit_registered

    with _RUNTIME_LOCK:
        if _runtime is None:
            runtime = RuntimeState()
            thread = threading.Thread(
                target=_thread_main,
                args=(runtime,),
                name="push-runtime",
                daemon=True,
            )
            runtime.thread = thread
            _runtime = runtime
            thread.start()
        runtime = _runtime
        if app not in runtime.apps:
            runtime.apps.append(app)
        app.push_runtime_started = True
        if not _atexit_registered:
            atexit.register(stop_all_push_runtime)
            _atexit_registered = True
        started_event = runtime.started_event
    started_event.wait(timeout=1.0)


def stop_push_runtime(app: Any) -> None:
    runtime = _runtime
    app.push_runtime_started = False
    if runtime is None:
        return
    with _RUNTIME_LOCK:
        if app in runtime.apps:
            runtime.apps.remove(app)
        remaining = list(runtime.apps)
    if not remaining:
        stop_all_push_runtime()


def stop_all_push_runtime() -> None:
    global _runtime

    with _RUNTIME_LOCK:
        runtime = _runtime
        _runtime = None
    if runtime is None:
        return
    for app in list(runtime.apps):
        try:
            app.push_runtime_started = False
        except Exception:
            logger.exception("push runtime app cleanup failed")
    if runtime.callosum is not None:
        runtime.callosum.stop()
    if runtime.loop is not None:
        if runtime.periodic_task is not None:
            runtime.loop.call_soon_threadsafe(runtime.periodic_task.cancel)
        runtime.loop.call_soon_threadsafe(runtime.loop.stop)
    if runtime.thread is not None:
        runtime.thread.join(timeout=1.0)


__all__ = [
    "RuntimeState",
    "get_runtime_state",
    "start_push_runtime",
    "stop_all_push_runtime",
    "stop_push_runtime",
]
