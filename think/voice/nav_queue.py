# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""In-memory nav hints for voice turns."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import DefaultDict, Deque

logger = logging.getLogger(__name__)

NAV_HINT_TTL_SECONDS = 60
NAV_HINT_CAPACITY = 8


@dataclass(frozen=True)
class QueuedHint:
    value: str
    created_at: float


class NavHintQueue:
    """Thread-safe FIFO queue for voice nav hints."""

    def __init__(
        self,
        *,
        ttl_seconds: int = NAV_HINT_TTL_SECONDS,
        capacity: int = NAV_HINT_CAPACITY,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._lock = threading.Lock()
        self._queues: DefaultDict[str, Deque[QueuedHint]] = defaultdict(deque)

    def push(self, call_id: str, hint: str, *, now: float | None = None) -> None:
        cleaned_call_id = call_id.strip()
        cleaned_hint = hint.strip()
        if not cleaned_call_id or not cleaned_hint:
            return
        current = time.time() if now is None else now
        with self._lock:
            queue = self._queues[cleaned_call_id]
            self._drop_expired(queue, current)
            queue.append(QueuedHint(cleaned_hint, current))
            while len(queue) > self.capacity:
                dropped = queue.popleft()
                logger.debug("voice nav hint dropped for capacity: %s", dropped.value)

    def drain(self, call_id: str, *, now: float | None = None) -> list[str]:
        cleaned_call_id = call_id.strip()
        if not cleaned_call_id:
            return []
        current = time.time() if now is None else now
        with self._lock:
            queue = self._queues.get(cleaned_call_id)
            if not queue:
                return []
            self._drop_expired(queue, current)
            hints = [entry.value for entry in queue]
            if cleaned_call_id in self._queues:
                del self._queues[cleaned_call_id]
            return hints

    def clear(self) -> None:
        with self._lock:
            self._queues.clear()

    def _drop_expired(self, queue: Deque[QueuedHint], now: float) -> None:
        while queue and now - queue[0].created_at > self.ttl_seconds:
            dropped = queue.popleft()
            logger.debug("voice nav hint expired: %s", dropped.value)


_NAV_QUEUE = NavHintQueue()


def get_nav_queue() -> NavHintQueue:
    return _NAV_QUEUE


__all__ = [
    "NAV_HINT_CAPACITY",
    "NAV_HINT_TTL_SECONDS",
    "NavHintQueue",
    "QueuedHint",
    "get_nav_queue",
]
