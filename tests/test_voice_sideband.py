# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future

from solstone.think.voice import sideband


class _FakeEvent:
    def __init__(
        self, event_type: str, *, name: str = "", arguments: str = "", call_id: str = ""
    ):
        self.type = event_type
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _FakeConversationItem:
    def __init__(self) -> None:
        self.items: list[dict] = []

    async def create(self, *, item):
        self.items.append(item)


class _FakeResponse:
    def __init__(self) -> None:
        self.count = 0

    async def create(self):
        self.count += 1


class _FakeConversation:
    def __init__(self) -> None:
        self.item = _FakeConversationItem()


class _FakeConn:
    def __init__(self, events):
        self._events = iter(events)
        self.conversation = _FakeConversation()
        self.response = _FakeResponse()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


def test_sideband_loop_dispatches_function_calls(monkeypatch):
    conn = _FakeConn(
        [
            _FakeEvent("session.created"),
            _FakeEvent(
                "response.function_call_arguments.done",
                name="journal.get_day",
                arguments='{"day":"2026-03-04"}',
                call_id="call-1",
            ),
        ]
    )
    seen: list[tuple[str, str, str]] = []

    async def fake_dispatch(name, arguments, call_id, app):
        seen.append((name, arguments, call_id))
        return json.dumps({"day": "2026-03-04"})

    monkeypatch.setattr(sideband, "dispatch_tool_call", fake_dispatch)

    asyncio.run(sideband._sideband_loop(conn, "call-1", object()))

    assert seen == [("journal.get_day", '{"day":"2026-03-04"}', "call-1")]
    assert conn.conversation.item.items == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"day": "2026-03-04"}',
        }
    ]
    assert conn.response.count == 1


def test_sideband_loop_ignores_non_tool_events(monkeypatch):
    conn = _FakeConn([_FakeEvent("session.created")])
    called = False

    async def fake_dispatch(name, arguments, call_id, app):
        nonlocal called
        called = True
        return "{}"

    monkeypatch.setattr(sideband, "dispatch_tool_call", fake_dispatch)

    asyncio.run(sideband._sideband_loop(conn, "call-1", object()))

    assert called is False


def test_register_voice_task_tracks_and_prunes():
    class DummyApp:
        def __init__(self):
            self.voice_tasks = set()

    app = DummyApp()
    future: Future[None] = Future()

    sideband.register_voice_task(app, future)
    assert future in app.voice_tasks

    future.set_result(None)
    assert future not in app.voice_tasks
