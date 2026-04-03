# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""File-based rate limiter for cross-process API budget management.

Uses fcntl.flock() for cross-process synchronization following the pattern
in think/entities/saving.py. Budget file lives at journal/health/rate_budget.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from think.utils import get_config, get_journal

_rate_budget = None


class RateBudgetExhausted(Exception):
    """Raised when rate budget cannot be acquired within timeout."""


class RateBudget:
    """File-based fixed-window request budget."""

    def __init__(self, budget_path: Path, rpm: int = 1500, window_s: float = 60.0):
        self.budget_path = budget_path
        self.rpm = rpm
        self.window_s = window_s

    def _default_state(self, now: float) -> dict:
        return {
            "remaining": self.rpm,
            "window_start": now,
            "window_duration_s": self.window_s,
            "budget_per_window": self.rpm,
        }

    def try_acquire(self) -> bool:
        import fcntl

        now = time.time()
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.budget_path.with_name(f"{self.budget_path.name}.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if self.budget_path.exists():
                    try:
                        state = json.loads(self.budget_path.read_text())
                    except (json.JSONDecodeError, OSError):
                        state = self._default_state(now)
                else:
                    state = self._default_state(now)

                window_start = state.get("window_start", now)
                window_duration_s = state.get("window_duration_s", self.window_s)
                budget_per_window = state.get("budget_per_window", self.rpm)
                remaining = state.get("remaining", budget_per_window)

                if now - window_start >= window_duration_s:
                    remaining = self.rpm
                    state = {
                        "remaining": remaining,
                        "window_start": now,
                        "window_duration_s": self.window_s,
                        "budget_per_window": self.rpm,
                    }

                if remaining > 0:
                    state["remaining"] = remaining - 1
                    tmp = self.budget_path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(state, indent=2))
                    tmp.replace(self.budget_path)
                    return True

                tmp = self.budget_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(state, indent=2))
                tmp.replace(self.budget_path)
                return False
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def acquire(self, timeout_s: float = 30.0) -> None:
        deadline = time.time() + timeout_s
        delay = 0.1
        while time.time() < deadline:
            if self.try_acquire():
                return
            time.sleep(delay)
            delay = min(delay * 2, 1.0)
        raise RateBudgetExhausted(
            f"Rate budget not available within {timeout_s:.1f}s"
        )

    async def aacquire(self, timeout_s: float = 30.0) -> None:
        deadline = time.time() + timeout_s
        delay = 0.1
        while time.time() < deadline:
            if self.try_acquire():
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)
        raise RateBudgetExhausted(
            f"Rate budget not available within {timeout_s:.1f}s"
        )


def get_rate_budget() -> RateBudget:
    """Get or create the global rate budget instance.

    Reads budget RPM from:
    1. SOL_RATE_BUDGET_RPM env var
    2. journal config providers.rate_budget_rpm
    3. Default: 1500
    """
    global _rate_budget
    if _rate_budget is None:
        rpm_str = os.getenv("SOL_RATE_BUDGET_RPM")
        if rpm_str is not None:
            rpm = int(rpm_str)
        else:
            rpm = get_config().get("providers", {}).get("rate_budget_rpm", 1500)
        budget_path = Path(get_journal()) / "health" / "rate_budget.json"
        _rate_budget = RateBudget(budget_path, rpm=rpm)
    return _rate_budget
