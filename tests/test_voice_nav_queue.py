# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from think.voice.nav_queue import NavHintQueue


def test_nav_queue_returns_empty_for_unknown_call_id():
    queue = NavHintQueue()
    assert queue.drain("call-1", now=100.0) == []


def test_nav_queue_drain_clears_queue():
    queue = NavHintQueue()
    queue.push("call-1", "today", now=100.0)

    assert queue.drain("call-1", now=100.0) == ["today"]
    assert queue.drain("call-1", now=100.0) == []


def test_nav_queue_drops_expired_hints():
    queue = NavHintQueue(ttl_seconds=10)
    queue.push("call-1", "today", now=100.0)
    queue.push("call-1", "entity/sarah", now=111.0)

    assert queue.drain("call-1", now=111.0) == ["entity/sarah"]


def test_nav_queue_enforces_fifo_capacity():
    queue = NavHintQueue(capacity=3)
    for idx in range(5):
        queue.push("call-1", f"hint-{idx}", now=float(idx))

    assert queue.drain("call-1", now=10.0) == ["hint-2", "hint-3", "hint-4"]


def test_nav_queue_ignores_blank_values():
    queue = NavHintQueue()
    queue.push(" ", "today")
    queue.push("call-1", " ")

    assert queue.drain("call-1", now=100.0) == []


def test_nav_queue_is_thread_safe_for_push_then_drain():
    queue = NavHintQueue(capacity=16)

    def push_hint(index: int) -> None:
        queue.push("call-1", f"hint-{index}", now=float(index))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(push_hint, range(8)))

    drained = queue.drain("call-1", now=8.0)
    assert len(drained) == 8
    assert set(drained) == {f"hint-{idx}" for idx in range(8)}
