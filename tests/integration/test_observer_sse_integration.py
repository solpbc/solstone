# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests

import solstone.apps.observer.routes as observer_routes
import solstone.convey.bridge as convey_bridge
import solstone.convey.chat as chat_module
import solstone.convey.chat_stream as chat_stream
from solstone.apps.observer.utils import load_observer, save_observer
from solstone.observe import observer_client
from solstone.think.callosum import CallosumServer
from tests.link.live_helpers import running_convey_server

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def cleanup_runtime_state() -> Iterator[None]:
    _clear_sse_subscribers()
    yield
    convey_bridge.stop_bridge()
    chat_module.stop_all_chat_runtime()
    _clear_sse_subscribers()


def test_observer_sse_receives_multiple_tracts_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))
    monkeypatch.setattr(observer_routes, "_SSE_HEARTBEAT_SECONDS", 0.1)

    with running_convey_server(tmp_journal) as base_url:
        key, key_prefix = _create_observer(base_url, "events")
        client, received, got_all = _subscribed_client(
            monkeypatch,
            base_url,
            key,
            expected_count=5,
        )
        try:
            _wait_until(lambda: convey_bridge.subscription_count(key_prefix) == 1)

            messages = [
                {"tract": "test", "event": "one", "ts": 1, "seq": 1},
                {"tract": "chat", "event": "owner_message", "ts": 2, "seq": 2},
                {"tract": "test", "event": "two", "ts": 3, "seq": 3},
                {"tract": "cortex", "event": "finished", "ts": 4, "seq": 4},
                {"tract": "chat", "event": "sol_message", "ts": 5, "seq": 5},
            ]
            for message in messages:
                convey_bridge._broadcast_callosum_event(message)

            assert got_all.wait(timeout=5)
            assert received == messages
        finally:
            client.stop()


def test_live_bit_tracks_active_subscription_and_stop_within_five_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))
    monkeypatch.setattr(observer_routes, "_SSE_HEARTBEAT_SECONDS", 0.1)

    with running_convey_server(tmp_journal) as base_url:
        key, key_prefix = _create_observer(base_url, "live-bit")
        client, _received, _got_one = _subscribed_client(
            monkeypatch,
            base_url,
            key,
            expected_count=1,
        )
        try:
            _wait_until(lambda: _observer_live(base_url, key_prefix) is True)
            start = time.monotonic()
            client.stop()
            elapsed = time.monotonic() - start

            assert elapsed < 5.0
            _wait_until(lambda: _observer_live(base_url, key_prefix) is False)
        finally:
            client.stop()


def test_revoke_tears_down_subscription_and_live_bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))
    monkeypatch.setattr(observer_routes, "_SSE_HEARTBEAT_SECONDS", 0.1)

    with running_convey_server(tmp_journal) as base_url:
        key, key_prefix = _create_observer(base_url, "revoked")
        client, _received, _got_one = _subscribed_client(
            monkeypatch,
            base_url,
            key,
            expected_count=1,
        )
        try:
            _wait_until(lambda: _observer_live(base_url, key_prefix) is True)
            response = requests.delete(
                f"{base_url}/app/observer/api/{key_prefix}",
                timeout=5,
            )
            response.raise_for_status()

            _wait_until(lambda: not client._callosum_thread.is_alive(), timeout=5)
            _wait_until(lambda: _observer_live(base_url, key_prefix) is False)
            assert client._revoked is True
        finally:
            client.stop()


def test_disabled_observer_tears_down_subscription_and_live_bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))
    monkeypatch.setattr(observer_routes, "_SSE_HEARTBEAT_SECONDS", 0.1)

    with running_convey_server(tmp_journal) as base_url:
        key, key_prefix = _create_observer(base_url, "disabled")
        client, _received, _got_one = _subscribed_client(
            monkeypatch,
            base_url,
            key,
            expected_count=1,
        )
        try:
            _wait_until(lambda: _observer_live(base_url, key_prefix) is True)
            observer = load_observer(key)
            assert observer is not None
            observer["enabled"] = False
            assert save_observer(observer)

            _wait_until(lambda: not client._callosum_thread.is_alive(), timeout=5)
            _wait_until(lambda: _observer_live(base_url, key_prefix) is False)
            assert client._revoked is True
        finally:
            client.stop()


def test_chat_append_writes_disk_before_sse_callback(
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
        key, key_prefix = _create_observer(base_url, "chat")
        client, received, got_one = _subscribed_client(
            monkeypatch,
            base_url,
            key,
            expected_count=1,
        )
        try:
            _wait_until(lambda: convey_bridge.subscription_count(key_prefix) == 1)
            _wait_until(lambda: callosum.client_count() >= 2)

            chat_stream.append_chat_event(
                "owner_message",
                text="hello",
                app=None,
                path=None,
                facet=None,
            )
            chat_file = _only_chat_file(tmp_journal)
            assert chat_file.stat().st_size > 0
            assert (
                json.loads(chat_file.read_text(encoding="utf-8").splitlines()[0])[
                    "kind"
                ]
                == "owner_message"
            )

            assert got_one.wait(timeout=5)
            assert received[0]["tract"] == "chat"
            assert received[0]["event"] == "owner_message"
        finally:
            client.stop()


def test_normal_sse_paths_do_not_write_chat_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))
    monkeypatch.setattr(observer_routes, "_SSE_HEARTBEAT_SECONDS", 0.1)

    with running_convey_server(tmp_journal) as base_url:
        before = _chat_snapshot(tmp_journal)
        key, key_prefix = _create_observer(base_url, "no-chat-write")
        client, _received, _got_one = _subscribed_client(
            monkeypatch,
            base_url,
            key,
            expected_count=1,
        )
        slow = convey_bridge.register_sse_subscriber("slowkey1")
        try:
            _wait_until(lambda: _observer_live(base_url, key_prefix) is True)
            convey_bridge._broadcast_callosum_event(
                {"tract": "test", "event": "ping", "ts": 1}
            )
            for idx in range(convey_bridge._SSE_QUEUE_MAXSIZE + 1):
                convey_bridge._broadcast_callosum_event(
                    {"tract": "overflow", "event": "ping", "ts": idx}
                )
            response = requests.delete(
                f"{base_url}/app/observer/api/{key_prefix}",
                timeout=5,
            )
            response.raise_for_status()
            _wait_until(lambda: slow.dropped.is_set())
            _wait_until(lambda: not client._callosum_thread.is_alive(), timeout=5)
        finally:
            client.stop()
            convey_bridge.unregister_sse_subscriber(slow)

        assert _chat_snapshot(tmp_journal) == before


def test_slow_sse_subscriber_does_not_delay_chat_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_journal = tmp_path / "journal"
    tmp_journal.mkdir()
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_journal))

    def broadcast_via_bridge(stored_event: dict) -> None:
        convey_bridge._broadcast_callosum_event(
            {
                "tract": "chat",
                "event": stored_event["kind"],
                **stored_event,
            }
        )

    monkeypatch.setattr(chat_stream, "_broadcast_chat_event", broadcast_via_bridge)

    start = time.monotonic()
    chat_stream.append_chat_event(
        "owner_message",
        text="baseline",
        app=None,
        path=None,
        facet=None,
    )
    baseline = time.monotonic() - start

    slow = convey_bridge.register_sse_subscriber("slowkey1")
    try:
        start = time.monotonic()
        chat_stream.append_chat_event(
            "owner_message",
            text="with slow subscriber",
            app=None,
            path=None,
            facet=None,
        )
        with_slow = time.monotonic() - start
    finally:
        convey_bridge.unregister_sse_subscriber(slow)

    assert with_slow <= max(baseline * 2, 0.05)


def _subscribed_client(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    key: str,
    *,
    expected_count: int,
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
    client = observer_client.ObserverClient("main-stream")
    received: list[dict] = []
    got_expected = threading.Event()

    def callback(payload: dict) -> None:
        received.append(payload)
        if len(received) >= expected_count:
            got_expected.set()

    client.subscribe_callosum(callback)
    return client, received, got_expected


def _create_observer(base_url: str, name: str) -> tuple[str, str]:
    response = requests.post(
        f"{base_url}/app/observer/api/create",
        json={"name": name},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["key"], payload["key"][:8]


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


def _clear_sse_subscribers() -> None:
    with convey_bridge._SSE_LOCK:
        convey_bridge._SSE_SUBSCRIBERS_BY_KEY.clear()


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


def _only_chat_file(journal_path: Path) -> Path:
    files = list((journal_path / "chronicle").glob("*/chat/*/chat.jsonl"))
    assert len(files) == 1
    return files[0]


def _chat_snapshot(journal_path: Path) -> list[tuple[str, int, int]]:
    chronicle = journal_path / "chronicle"
    if not chronicle.exists():
        return []
    rows: list[tuple[str, int, int]] = []
    for path in chronicle.rglob("*"):
        if "chat" not in path.relative_to(chronicle).parts:
            continue
        stat = path.stat()
        rows.append((str(path.relative_to(chronicle)), stat.st_mtime_ns, stat.st_size))
    return sorted(rows)
