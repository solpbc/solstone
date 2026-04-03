# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from think.providers.shared import CircuitBreaker, CircuitOpenError


class FakeClientError(Exception):
    def __init__(self, code, response_json=None):
        self.code = code
        self.response_json = response_json or {}
        super().__init__(f"{code}")


def test_starts_closed():
    cb = CircuitBreaker("google")
    assert cb.state == cb.CLOSED


def test_stays_closed_below_threshold():
    cb = CircuitBreaker("google", failure_threshold=5)
    for _ in range(4):
        cb.record_failure(FakeClientError(429))
    assert cb.state == cb.CLOSED


def test_opens_at_threshold():
    cb = CircuitBreaker("google", failure_threshold=5)
    for _ in range(5):
        cb.record_failure(FakeClientError(429))
    assert cb.state == cb.OPEN


def test_open_rejects_requests():
    cb = CircuitBreaker("google", failure_threshold=1)
    cb.record_failure(FakeClientError(429))
    with pytest.raises(CircuitOpenError):
        cb.check()


def test_circuit_open_error_attributes():
    err = CircuitOpenError("google", 12.5)
    assert err.provider == "google"
    assert err.cooldown_remaining == 12.5


def test_half_open_after_cooldown():
    cb = CircuitBreaker("google", failure_threshold=1, cooldown_s=5)
    cb.record_failure(FakeClientError(429))
    cb._opened_at = time.time() - cb._current_cooldown - 1
    assert cb.state == cb.HALF_OPEN


def test_half_open_success_closes():
    cb = CircuitBreaker("google", failure_threshold=1, cooldown_s=5)
    cb.record_failure(FakeClientError(429))
    cb._opened_at = time.time() - cb._current_cooldown - 1
    assert cb.state == cb.HALF_OPEN
    cb.record_success()
    assert cb.state == cb.CLOSED
    assert cb._failure_count == 0


def test_half_open_failure_reopens_with_doubled_cooldown():
    cb = CircuitBreaker("google", failure_threshold=1, cooldown_s=5)
    cb.record_failure(FakeClientError(429))
    cb._opened_at = time.time() - cb._current_cooldown - 1
    assert cb.state == cb.HALF_OPEN
    cb.record_failure(FakeClientError(429))
    assert cb.state == cb.OPEN
    assert cb._current_cooldown == 10


def test_cooldown_cap():
    cb = CircuitBreaker("google", failure_threshold=1, cooldown_s=400, max_cooldown_s=600)
    cb.record_failure(FakeClientError(429))
    cb._opened_at = time.time() - cb._current_cooldown - 1
    assert cb.state == cb.HALF_OPEN
    cb.record_failure(FakeClientError(429))
    assert cb._current_cooldown == 600


def test_success_resets_failure_count():
    cb = CircuitBreaker("google", failure_threshold=5)
    cb.record_failure(FakeClientError(429))
    cb.record_failure(FakeClientError(429))
    cb.record_success()
    assert cb._failure_count == 0
    assert cb.state == cb.CLOSED


def test_circuit_opens_on_consecutive_429s(monkeypatch):
    from think.providers import google as google_provider

    # Pre-populate with a clean breaker (no health_path) to avoid fixture writes
    cb = CircuitBreaker("google")
    monkeypatch.setattr(google_provider, "_circuit_breakers", {"google": cb})
    monkeypatch.setattr(
        google_provider, "_is_quota_error", lambda error: getattr(error, "code", None) == 429
    )

    # No-op rate budget to avoid fixture file creation
    class _NoopBudget:
        def acquire(self, **kw): pass
        async def aacquire(self, **kw): pass
    monkeypatch.setattr("think.rate_limiter.get_rate_budget", lambda: _NoopBudget())

    class DummyModels:
        def __init__(self):
            self.calls = 0

        async def generate_content(self, **kwargs):
            self.calls += 1
            raise FakeClientError(429, {"error": "rate limit"})

    client = SimpleNamespace(aio=SimpleNamespace(models=DummyModels()))

    for _ in range(5):
        with pytest.raises(FakeClientError):
            asyncio.run(google_provider.run_agenerate("hello", client=client))

    with pytest.raises(CircuitOpenError):
        asyncio.run(google_provider.run_agenerate("hello", client=client))

    assert client.aio.models.calls == 5


def test_health_file_write_on_open(tmp_path):
    """AC6: Circuit state visible in journal/health/agents.json."""
    health_path = tmp_path / "health" / "agents.json"
    health_path.parent.mkdir(parents=True)
    health_path.write_text(
        json.dumps(
            {
                "results": [{"provider": "google", "ok": True}],
                "checked_at": "2026-04-02T00:00:00+00:00",
            }
        )
    )

    cb = CircuitBreaker("google", failure_threshold=5)
    cb._health_path = health_path

    for _ in range(5):
        cb.record_failure(FakeClientError(429))

    data = json.loads(health_path.read_text())
    assert "circuit_breakers" in data
    assert data["circuit_breakers"]["google"]["state"] == "open"
    assert data["circuit_breakers"]["google"]["failure_count"] == 5
    assert data["results"] == [{"provider": "google", "ok": True}]
    assert data["checked_at"] == "2026-04-02T00:00:00+00:00"


def test_callosum_events_on_state_transitions(monkeypatch):
    """AC3: Circuit emits provider.unhealthy/provider.healthy via Callosum."""
    events = []

    def fake_callosum_send(tract, event, **fields):
        events.append({"tract": tract, "event": event, **fields})
        return True

    monkeypatch.setattr("think.callosum.callosum_send", fake_callosum_send)

    cb = CircuitBreaker("google", failure_threshold=2)
    cb.record_failure(FakeClientError(429))
    cb.record_failure(FakeClientError(429))

    assert len(events) == 1
    assert events[0]["tract"] == "provider"
    assert events[0]["event"] == "unhealthy"
    assert events[0]["provider"] == "google"
    assert "cooldown_s" in events[0]

    cb._opened_at = time.time() - cb._current_cooldown - 1
    assert cb.state == cb.HALF_OPEN
    cb.record_success()

    assert len(events) == 2
    assert events[1]["tract"] == "provider"
    assert events[1]["event"] == "healthy"
    assert events[1]["provider"] == "google"
