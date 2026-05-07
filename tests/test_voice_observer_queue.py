# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from solstone.think.voice.observer_queue import ObserverActionQueue


def test_observer_queue_returns_empty_for_unknown_call_id():
    queue = ObserverActionQueue()
    assert queue.drain("call-1", now=100.0) == []


def test_observer_queue_drain_clears_queue():
    queue = ObserverActionQueue()
    queue.push("call-1", {"type": "start_observer", "mode": "meeting"}, now=100.0)

    assert queue.drain("call-1", now=100.0) == [
        {"type": "start_observer", "mode": "meeting"}
    ]
    assert queue.drain("call-1", now=100.0) == []


def test_observer_queue_drops_expired_actions():
    queue = ObserverActionQueue(ttl_seconds=10)
    queue.push("call-1", {"type": "start_observer", "mode": "meeting"}, now=100.0)
    queue.push(
        "call-1",
        {"type": "start_observer", "mode": "voice_memo"},
        now=111.0,
    )

    assert queue.drain("call-1", now=111.0) == [
        {"type": "start_observer", "mode": "voice_memo"}
    ]


def test_observer_queue_enforces_fifo_capacity():
    queue = ObserverActionQueue(capacity=3)
    for idx in range(5):
        queue.push(
            "call-1",
            {"type": "start_observer", "mode": f"mode-{idx}"},
            now=float(idx),
        )

    assert queue.drain("call-1", now=10.0) == [
        {"type": "start_observer", "mode": "mode-2"},
        {"type": "start_observer", "mode": "mode-3"},
        {"type": "start_observer", "mode": "mode-4"},
    ]


def test_observer_queue_rejects_blank_call_id():
    queue = ObserverActionQueue()
    queue.push(" ", {"type": "start_observer", "mode": "meeting"})

    assert queue.drain("call-1", now=100.0) == []


def test_observer_queue_rejects_blank_payload():
    queue = ObserverActionQueue()
    queue.push("call-1", {})

    assert queue.drain("call-1", now=100.0) == []


def test_observer_queue_is_thread_safe_for_push_then_drain():
    queue = ObserverActionQueue(capacity=16)

    def push_action(index: int) -> None:
        queue.push(
            "call-1",
            {"type": "start_observer", "mode": f"mode-{index}"},
            now=float(index),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(push_action, range(8)))

    drained = queue.drain("call-1", now=8.0)
    assert len(drained) == 8
    assert {item["mode"] for item in drained} == {f"mode-{idx}" for idx in range(8)}
