# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import pytest
import requests

from solstone.apps.observer.routes import OBSERVER_CALLOSUM_SSE_ROUTE
from solstone.observe import observer_client


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        lines: Iterable[str] = (),
        *,
        text: str = "",
        block: bool = False,
    ):
        self.status_code = status_code
        self.text = text
        self.closed = False
        self.started = False
        self._lines = list(lines)
        self._block = block

    def iter_lines(self, decode_unicode: bool = False):
        self.started = True
        for line in self._lines:
            if self.closed:
                return
            yield line
        while self._block and not self.closed:
            time.sleep(0.01)

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, results: list[FakeResponse | Exception]):
        self.results = results
        self.get_calls: list[tuple[tuple, dict]] = []
        self.closed = False

    def get(self, *args, **kwargs):
        self.get_calls.append((args, kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    session: FakeSession,
    *,
    key: str = "test-key",
):
    monkeypatch.setattr(observer_client.requests, "Session", lambda: session)
    monkeypatch.setattr(
        observer_client,
        "get_config",
        lambda: {
            "observe": {
                "observer": {
                    "url": "https://convey.test/",
                    "key": key,
                    "auto_register": False,
                }
            }
        },
    )
    monkeypatch.setattr(observer_client, "read_service_port", lambda _name: None)
    return observer_client.ObserverClient("main-stream")


def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


def test_subscribe_callosum_starts_once_and_sends_bearer_header(monkeypatch):
    response = FakeResponse(block=True)
    session = FakeSession([response])
    client = _build_client(monkeypatch, session, key="test key/123")

    client.subscribe_callosum(lambda _payload: None)
    _wait_until(lambda: bool(session.get_calls) and response.started)

    with pytest.raises(RuntimeError, match="already active"):
        client.subscribe_callosum(lambda _payload: None)

    url = session.get_calls[0][0][0]
    kwargs = session.get_calls[0][1]
    expected_path = OBSERVER_CALLOSUM_SSE_ROUTE.replace("<key>", "test%20key%2F123")
    assert url == f"https://convey.test{expected_path}"
    assert kwargs["headers"] == {"Authorization": "Bearer test key/123"}
    assert kwargs["stream"] is True
    assert kwargs["timeout"] == (observer_client.EVENT_TIMEOUT, None)

    client.stop()


def test_subscribe_callosum_parses_sse_frames(monkeypatch):
    session = FakeSession(
        [
            FakeResponse(
                lines=[
                    ": heartbeat",
                    'data: {"tract": "test",',
                    'data: "event": "ping", "ts": 1}',
                    "",
                    ": ignored",
                    'data: {"tract": "chat", "event": "append", "ts": 2}',
                    "",
                ]
            )
        ]
    )
    client = _build_client(monkeypatch, session)
    received: list[dict] = []

    def callback(payload: dict) -> None:
        received.append(payload)
        if len(received) == 2:
            client._callosum_stop.set()

    client.subscribe_callosum(callback)
    client._callosum_thread.join(timeout=1.0)

    assert received == [
        {"tract": "test", "event": "ping", "ts": 1},
        {"tract": "chat", "event": "append", "ts": 2},
    ]
    client.stop()


def test_subscribe_callosum_reconnects_after_transport_error(monkeypatch):
    waits: list[float] = []
    session = FakeSession(
        [
            requests.ConnectionError("no route"),
            FakeResponse(lines=['data: {"tract": "test", "event": "ok", "ts": 1}', ""]),
        ]
    )
    client = _build_client(monkeypatch, session)
    original_wait = client._callosum_stop.wait
    received: list[dict] = []

    def wait(delay: float) -> bool:
        waits.append(delay)
        return original_wait(0)

    monkeypatch.setattr(client._callosum_stop, "wait", wait)

    def callback(payload: dict) -> None:
        received.append(payload)
        client._callosum_stop.set()

    client.subscribe_callosum(callback)
    client._callosum_thread.join(timeout=1.0)

    assert waits == [1]
    assert len(session.get_calls) == 2
    assert received == [{"tract": "test", "event": "ok", "ts": 1}]
    client.stop()


def test_subscribe_callosum_revocation_exits_without_reconnect(monkeypatch):
    response = FakeResponse(status_code=401, text="Unauthorized")
    session = FakeSession([response])
    client = _build_client(monkeypatch, session)

    client.subscribe_callosum(lambda _payload: None)
    client._callosum_thread.join(timeout=1.0)

    assert client._revoked is True
    assert client._callosum_thread is not None
    assert not client._callosum_thread.is_alive()
    assert len(session.get_calls) == 1
    assert response.closed is True
    client.stop()


def test_stop_closes_callosum_response_before_session(monkeypatch):
    response = FakeResponse(block=True)
    session = FakeSession([response])
    client = _build_client(monkeypatch, session)

    client.subscribe_callosum(lambda _payload: None)
    _wait_until(lambda: response.started)

    start = time.monotonic()
    client.stop()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert response.closed is True
    assert session.closed is True
    assert client._callosum_thread is not None
    assert not client._callosum_thread.is_alive()
