# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import json
import time

import pytest

from think.rate_limiter import RateBudget, RateBudgetExhausted


def test_try_acquire_success(tmp_path):
    budget = RateBudget(tmp_path / "rate_budget.json", rpm=3)
    assert budget.try_acquire() is True
    state = json.loads((tmp_path / "rate_budget.json").read_text())
    assert state["remaining"] == 2


def test_try_acquire_exhausted(tmp_path):
    budget_path = tmp_path / "rate_budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "remaining": 0,
                "window_start": time.time(),
                "window_duration_s": 60.0,
                "budget_per_window": 3,
            }
        )
    )
    budget = RateBudget(budget_path, rpm=3)
    assert budget.try_acquire() is False


def test_window_replenishment(tmp_path):
    budget_path = tmp_path / "rate_budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "remaining": 0,
                "window_start": time.time() - 61,
                "window_duration_s": 60.0,
                "budget_per_window": 3,
            }
        )
    )
    budget = RateBudget(budget_path, rpm=3)
    assert budget.try_acquire() is True
    state = json.loads(budget_path.read_text())
    assert state["remaining"] == 2


def test_acquire_blocks_then_succeeds(tmp_path, monkeypatch):
    budget = RateBudget(tmp_path / "rate_budget.json", rpm=1)
    calls = {"count": 0}

    def fake_try_acquire():
        calls["count"] += 1
        return calls["count"] >= 2

    monkeypatch.setattr(budget, "try_acquire", fake_try_acquire)
    budget.acquire(timeout_s=1)
    assert calls["count"] == 2


def test_acquire_timeout_raises(tmp_path):
    budget = RateBudget(tmp_path / "rate_budget.json", rpm=1)
    assert budget.try_acquire() is True
    with pytest.raises(RateBudgetExhausted):
        budget.acquire(timeout_s=0.2)


def test_aacquire_success(tmp_path):
    async def run():
        budget = RateBudget(tmp_path / "rate_budget.json", rpm=1)
        await budget.aacquire(timeout_s=1)
        state = json.loads((tmp_path / "rate_budget.json").read_text())
        assert state["remaining"] == 0

    asyncio.run(run())


def test_atomic_write_no_tmp_file_persists(tmp_path):
    budget_path = tmp_path / "rate_budget.json"
    budget = RateBudget(budget_path, rpm=3)
    budget.try_acquire()
    assert not budget_path.with_suffix(".json.tmp").exists()
    state = json.loads(budget_path.read_text())
    assert "remaining" in state


def test_concurrent_acquire(tmp_path):
    async def run():
        budget = RateBudget(tmp_path / "rate_budget.json", rpm=10)

        async def worker():
            successes = 0
            for _ in range(5):
                if budget.try_acquire():
                    successes += 1
                await asyncio.sleep(0)
            return successes

        results = await asyncio.gather(worker(), worker(), worker())
        assert sum(results) == 10

    asyncio.run(run())
