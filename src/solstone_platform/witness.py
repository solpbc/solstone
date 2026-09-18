# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test-only semantic and filesystem witness accounting."""

import threading
from typing import Optional


class WitnessRegistry:
    def __init__(self) -> None:
        self.enabled = False
        self.counts: dict[str, int] = {}
        self.live_source_accesses: list[str] = []

    def reset(self) -> None:
        self.counts.clear()
        self.live_source_accesses.clear()

    def record(self, event: str, path: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self.counts[event] = self.counts.get(event, 0) + 1
        if path:
            self.live_source_accesses.append(f"{event}:{path}")

    def total_semantic_count(self) -> int:
        return sum(v for k, v in self.counts.items() if not k.startswith("raw_path_"))


_tls = threading.local()


def get_witness() -> WitnessRegistry:
    if not hasattr(_tls, "witness"):
        _tls.witness = WitnessRegistry()
    return _tls.witness


def enable_witness() -> WitnessRegistry:
    w = get_witness()
    w.enabled = True
    w.reset()
    return w


def disable_witness() -> None:
    w = get_witness()
    w.enabled = False
    w.reset()
