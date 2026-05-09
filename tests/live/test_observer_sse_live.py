# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests

import solstone.apps.observer.routes as observer_routes
import solstone.convey.bridge as convey_bridge
import solstone.convey.chat as chat_module
from solstone.observe import observer_client
from solstone.think.callosum import CallosumServer
from tests.link.live_helpers import running_convey_server

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("SOLSTONE_LIVE_SANDBOX"),
        reason="SOLSTONE_LIVE_SANDBOX not set",
    ),
]


def test_observer_sse_live_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))
    monkeypatch.setattr(observer_routes, "_SSE_HEARTBEAT_SECONDS", 0.1)

    with (
        _running_callosum_server(tmp_journal) as callosum,
        running_convey_server(tmp_journal) as base_url,
    ):
        convey_bridge.start_bridge()
        key, key_prefix = _create_observer(base_url)
        client, received, got_ping = _subscribed_client(monkeypatch, base_url, key)
        try:
            _wait_until(lambda: _observer_live(base_url, key_prefix) is True)
            _wait_until(lambda: callosum.client_count() >= 2)

            assert convey_bridge.emit("test", "ping")
            assert got_ping.wait(timeout=5)
            assert received[-1]["tract"] == "test"
            assert received[-1]["event"] == "ping"

            client.stop()
            _wait_until(lambda: _observer_live(base_url, key_prefix) is False)
        finally:
            client.stop()
            convey_bridge.stop_bridge()
            chat_module.stop_all_chat_runtime()


def _subscribed_client(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    key: str,
) -> tuple[observer_client.ObserverClient, list[dict], threading.Event]:
    monkeypatch.setattr(
        observer_client,
        "get_config",
        lambda: {
            "observe": {
                "observer": {
                    "url": base_url,
                    "key": key,
                    "auto_register": False,
                }
            }
        },
    )
    client = observer_client.ObserverClient("live-stream")
    received: list[dict] = []
    got_ping = threading.Event()

    def callback(payload: dict) -> None:
        received.append(payload)
        if payload.get("tract") == "test" and payload.get("event") == "ping":
            got_ping.set()

    client.subscribe_callosum(callback)
    return client, received, got_ping


def _create_observer(base_url: str) -> tuple[str, str]:
    response = requests.post(
        f"{base_url}/app/observer/api/create",
        json={"name": "live-sse"},
        timeout=5,
    )
    response.raise_for_status()
    key = response.json()["key"]
    return key, key[:8]


def _observer_live(base_url: str, key_prefix: str) -> bool | None:
    response = requests.get(f"{base_url}/app/observer/api/list", timeout=5)
    response.raise_for_status()
    for observer in response.json()["observers"]:
        if observer["key_prefix"] == key_prefix:
            return observer["live"]
    return None


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for condition")


@contextlib.contextmanager
def _running_callosum_server(journal_path: Path) -> Iterator[CallosumServer]:
    server = CallosumServer()
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    socket_path = journal_path / "health" / "callosum.sock"
    _wait_until(socket_path.exists)
    try:
        yield server
    finally:
        server.stop()
        thread.join(timeout=2)
