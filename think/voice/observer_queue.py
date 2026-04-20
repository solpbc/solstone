# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""In-memory observer actions for voice turns."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, DefaultDict, Deque

logger = logging.getLogger(__name__)

OBSERVER_ACTION_TTL_SECONDS = 60
OBSERVER_ACTION_CAPACITY = 8


@dataclass(frozen=True)
class QueuedAction:
    payload: dict[str, Any]
    created_at: float


class ObserverActionQueue:
    """Thread-safe FIFO queue for voice observer actions."""

    def __init__(
        self,
        *,
        ttl_seconds: int = OBSERVER_ACTION_TTL_SECONDS,
        capacity: int = OBSERVER_ACTION_CAPACITY,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._lock = threading.Lock()
        self._queues: DefaultDict[str, Deque[QueuedAction]] = defaultdict(deque)

    def push(
        self,
        call_id: str,
        action: dict[str, Any],
        *,
        now: float | None = None,
    ) -> None:
        cleaned_call_id = call_id.strip()
        if not cleaned_call_id:
            logger.warning("voice observer action rejected blank call_id")
            return
        if not action:
            logger.warning("voice observer action rejected empty payload")
            return
        current = time.time() if now is None else now
        with self._lock:
            queue = self._queues[cleaned_call_id]
            self._drop_expired(queue, current)
            queue.append(QueuedAction(copy.deepcopy(action), current))
            while len(queue) > self.capacity:
                queue.popleft()
                logger.warning("voice observer action dropped for capacity")

    def drain(self, call_id: str, *, now: float | None = None) -> list[dict[str, Any]]:
        cleaned_call_id = call_id.strip()
        if not cleaned_call_id:
            return []
        current = time.time() if now is None else now
        with self._lock:
            queue = self._queues.get(cleaned_call_id)
            if not queue:
                return []
            self._drop_expired(queue, current)
            actions = [entry.payload for entry in queue]
            if cleaned_call_id in self._queues:
                del self._queues[cleaned_call_id]
            return actions

    def clear(self) -> None:
        with self._lock:
            self._queues.clear()

    def _drop_expired(self, queue: Deque[QueuedAction], now: float) -> None:
        while queue and now - queue[0].created_at > self.ttl_seconds:
            queue.popleft()


_OBSERVER_QUEUE = ObserverActionQueue()


def get_observer_queue() -> ObserverActionQueue:
    return _OBSERVER_QUEUE


__all__ = [
    "OBSERVER_ACTION_CAPACITY",
    "OBSERVER_ACTION_TTL_SECONDS",
    "ObserverActionQueue",
    "QueuedAction",
    "get_observer_queue",
]
