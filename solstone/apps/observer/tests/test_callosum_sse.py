# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for observer Callosum SSE route and bridge fan-out."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import pytest

import solstone.apps.observer.routes as routes_module
import solstone.convey.bridge as convey_bridge
from solstone.apps.observer.routes import OBSERVER_CALLOSUM_SSE_ROUTE
from solstone.apps.observer.utils import load_observer, save_observer


@pytest.fixture(autouse=True)
def clear_sse_subscribers() -> Iterator[None]:
    with convey_bridge._SSE_LOCK:
        convey_bridge._SSE_SUBSCRIBERS_BY_KEY.clear()
    yield
    with convey_bridge._SSE_LOCK:
        convey_bridge._SSE_SUBSCRIBERS_BY_KEY.clear()


def _create_observer(env, name: str = "sse-test") -> tuple[str, str]:
    resp = env.client.post(
        "/app/observer/api/create",
        json={"name": name},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    return data["key"], data["key_prefix"]


def _route_for(key: str) -> str:
    return OBSERVER_CALLOSUM_SSE_ROUTE.replace("<key>", key)


def _next_chunk(response) -> str:
    chunk = next(iter(response.response))
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8")
    return str(chunk)


def _parse_sse_data(chunk: str) -> dict:
    for line in chunk.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"No data line found in chunk: {chunk!r}")


def _next_data(response) -> dict:
    for chunk in response.response:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        if "data: " in text:
            return _parse_sse_data(text)
    raise AssertionError("SSE stream ended before a data frame was received")


def test_callosum_sse_missing_key_returns_401(observer_env):
    env = observer_env()
    with env.app.test_request_context(_route_for("unused")):
        response, status = routes_module.callosum_sse("")
    assert status == 401
    assert response.get_json()["error"] == "Authorization required"


def test_callosum_sse_unknown_key_returns_401(observer_env):
    env = observer_env()
    resp = env.client.get(_route_for("unknown-key"), buffered=False)
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid key"


def test_callosum_sse_revoked_key_returns_403(observer_env):
    env = observer_env()
    key, key_prefix = _create_observer(env, "revoked-sse")
    revoke = env.client.delete(f"/app/observer/api/{key_prefix}")
    assert revoke.status_code == 200

    resp = env.client.get(_route_for(key), buffered=False)
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "Observer revoked"


def test_callosum_sse_disabled_key_returns_403(observer_env):
    env = observer_env()
    key, _ = _create_observer(env, "disabled-sse")
    observer = load_observer(key)
    assert observer is not None
    observer["enabled"] = False
    assert save_observer(observer)

    resp = env.client.get(_route_for(key), buffered=False)
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "Observer disabled"


def test_callosum_sse_bearer_header_overrides_path_key(observer_env):
    env = observer_env()
    valid_key, _ = _create_observer(env, "valid-sse")
    bogus_key = "bogus-key"

    resp = env.client.get(
        _route_for(bogus_key),
        headers={"Authorization": f"Bearer {valid_key}"},
        buffered=False,
    )
    try:
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/event-stream")
    finally:
        resp.close()

    resp = env.client.get(
        _route_for(valid_key),
        headers={"Authorization": "Bearer invalid-key"},
        buffered=False,
    )
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Invalid key"


def test_callosum_sse_success_content_type(observer_env):
    env = observer_env()
    key, _ = _create_observer(env)

    resp = env.client.get(_route_for(key), buffered=False)
    try:
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/event-stream")
    finally:
        resp.close()


def test_callosum_sse_round_trip_payload(observer_env):
    env = observer_env()
    key, key_prefix = _create_observer(env)
    resp = env.client.get(_route_for(key), buffered=False)
    try:
        assert resp.status_code == 200
        assert convey_bridge.subscription_count(key_prefix) == 1
        message = {"tract": "test", "event": "ping", "ts": 0, "extra": "value"}
        convey_bridge._broadcast_callosum_event(message)

        parsed = _next_data(resp)
        assert parsed == message
    finally:
        resp.close()


def test_callosum_sse_heartbeat(observer_env, monkeypatch):
    env = observer_env()
    key, _ = _create_observer(env)
    monkeypatch.setattr(routes_module, "_SSE_HEARTBEAT_SECONDS", 0.01)

    resp = env.client.get(_route_for(key), buffered=False)
    try:
        assert resp.status_code == 200
        assert _next_chunk(resp) == ": heartbeat\n\n"
    finally:
        resp.close()


def test_sse_registry_lifecycle():
    handle = convey_bridge.register_sse_subscriber("aaaaaaaa")
    assert convey_bridge.subscription_count("aaaaaaaa") == 1

    convey_bridge.unregister_sse_subscriber(handle)
    assert convey_bridge.subscription_count("aaaaaaaa") == 0

    convey_bridge.unregister_sse_subscriber(handle)
    assert convey_bridge.subscription_count("aaaaaaaa") == 0


def test_slow_sse_subscriber_is_dropped_without_blocking_healthy_subscriber():
    slow = convey_bridge.register_sse_subscriber("aaaaaaaa")
    healthy = convey_bridge.register_sse_subscriber("bbbbbbbb")
    received: list[dict] = []

    start = time.perf_counter()
    for i in range(convey_bridge._SSE_QUEUE_MAXSIZE + 1):
        convey_bridge._broadcast_callosum_event(
            {"tract": "test", "event": "ping", "ts": i}
        )
        received.append(json.loads(healthy.queue.get_nowait()))
    elapsed = time.perf_counter() - start

    assert slow.dropped.is_set()
    assert slow.drop_reason == "overflow"
    assert "aaaaaaaa" not in convey_bridge._SSE_SUBSCRIBERS_BY_KEY
    assert len(received) == convey_bridge._SSE_QUEUE_MAXSIZE + 1
    assert [message["ts"] for message in received] == list(
        range(convey_bridge._SSE_QUEUE_MAXSIZE + 1)
    )
    assert elapsed < 0.5

    convey_bridge.unregister_sse_subscriber(healthy)
