# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities and types for AI providers.

This module contains:
- Event TypedDicts emitted by providers during agent execution
- GenerateResult TypedDict returned by run_generate/run_agenerate
- JSONEventCallback for event emission
- Utility functions for common provider operations
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from think.utils import now_ms

# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------


class ToolStartEvent(TypedDict, total=False):
    """Event emitted when a tool starts."""

    event: Literal["tool_start"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    call_id: Optional[str]  # Unique ID to pair with tool_end event
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class ToolEndEvent(TypedDict, total=False):
    """Event emitted when a tool finishes."""

    event: Literal["tool_end"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    result: Any
    call_id: Optional[str]  # Matches the call_id from tool_start
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class StartEvent(TypedDict, total=False):
    """Event emitted when an agent run begins."""

    event: Required[Literal["start"]]
    ts: Required[int]
    prompt: Required[str]
    name: Required[str]
    model: Required[str]
    provider: Required[str]
    session_id: Optional[str]  # CLI session ID for continuation
    chat_id: Optional[str]  # Chat ID for reverse lookup
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class FinishEvent(TypedDict, total=False):
    """Event emitted when an agent run finishes successfully."""

    event: Required[Literal["finish"]]
    ts: Required[int]
    result: Required[str]
    usage: Optional[dict[str, Any]]
    cli_session_id: Optional[str]  # Provider CLI session/thread ID for resume
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class ErrorEvent(TypedDict, total=False):
    """Event emitted when an error occurs."""

    event: Literal["error"]
    ts: int
    error: str
    trace: Optional[str]
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class AgentUpdatedEvent(TypedDict, total=False):
    """Event emitted when the agent context changes."""

    event: Required[Literal["agent_updated"]]
    ts: Required[int]
    agent: Required[str]
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class ThinkingEvent(TypedDict, total=False):
    """Event emitted when thinking/reasoning summaries are available.

    For Anthropic models, may include a signature for verification when
    passing thinking blocks back during tool use continuations.
    For redacted thinking, summary will contain "[redacted]" and
    redacted_data will contain the encrypted content.
    """

    event: Required[Literal["thinking"]]
    ts: Required[int]
    summary: Required[str]
    model: Optional[str]
    signature: Optional[str]  # Anthropic thinking block signature
    redacted_data: Optional[str]  # Encrypted data for redacted thinking
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class FallbackEvent(TypedDict, total=False):
    """Event emitted when provider fallback occurs."""

    event: Required[Literal["fallback"]]
    ts: Required[int]
    original_provider: Required[str]
    backup_provider: Required[str]
    reason: Required[str]  # "preflight" or "on_failure"
    error: Optional[str]  # Error message for on_failure case


Event = Union[
    ToolStartEvent,
    ToolEndEvent,
    StartEvent,
    FinishEvent,
    ErrorEvent,
    ThinkingEvent,
    AgentUpdatedEvent,
    FallbackEvent,
]


# ---------------------------------------------------------------------------
# Usage Schema
# ---------------------------------------------------------------------------

# Canonical keys for the normalized usage dict returned by all providers.
# log_token_usage() passes through exactly these keys (when present and non-zero).
USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cache_creation_tokens",
        "requests",
    }
)

# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------


class GenerateResult(TypedDict, total=False):
    """Result from provider run_generate/run_agenerate functions.

    Structured result that allows the wrapper to handle cross-cutting concerns
    like token logging and JSON validation centrally.

    The thinking field contains dicts with: summary (str), signature (optional str),
    redacted_data (optional str for Anthropic redacted thinking).
    """

    text: Required[str]  # Response text
    usage: Optional[dict]  # Normalized usage dict (input_tokens, output_tokens, etc.)
    finish_reason: Optional[str]  # Normalized: "stop", "max_tokens", "safety", etc.
    thinking: Optional[list]  # List of thinking block dicts


# ---------------------------------------------------------------------------
# JSONEventCallback
# ---------------------------------------------------------------------------


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if "ts" not in data:
            data = {**data, "ts": now_ms()}
        if self.callback:
            self.callback(data)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Raw Event Trimming
# ---------------------------------------------------------------------------

# Structural keys preserved when trimming oversized raw events.
_RAW_STRUCTURAL_KEYS = frozenset(
    {
        "type",
        "id",
        "tool_id",
        "tool_name",
        "role",
        "event_type",
        "timestamp",
    }
)

_RAW_BYTE_LIMIT = 16_384  # 16 KB


def safe_raw(
    events: list[dict[str, Any]],
    limit: int = _RAW_BYTE_LIMIT,
) -> list[dict[str, Any]]:
    """Return *events* as-is if small enough, otherwise a trimmed version.

    When the JSON-serialized size exceeds *limit* bytes, each event is reduced
    to its structural keys and a ``_raw_trimmed`` dict is appended with the
    original byte count and the limit that was applied.
    """
    serialized = json.dumps(events, ensure_ascii=False)
    if len(serialized.encode("utf-8")) <= limit:
        return events

    trimmed = [
        {k: v for k, v in e.items() if k in _RAW_STRUCTURAL_KEYS} for e in events
    ]
    trimmed.append(
        {"_raw_trimmed": {"original_bytes": len(serialized), "limit": limit}}
    )
    return trimmed


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and rejecting requests."""

    def __init__(self, provider: str, cooldown_remaining: float):
        self.provider = provider
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker open for {provider} ({cooldown_remaining:.0f}s remaining)"
        )


class CircuitBreaker:
    """Per-process circuit breaker for API providers.

    States: closed (normal), open (rejecting), half_open (probing after cooldown).
    On failure_threshold consecutive quota errors, opens the circuit.
    Cooldown doubles on each failed probe, capped at max_cooldown_s.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        provider,
        failure_threshold=5,
        cooldown_s=60,
        max_cooldown_s=600,
    ):
        self.provider = provider
        self.failure_threshold = failure_threshold
        self._initial_cooldown = cooldown_s
        self.max_cooldown_s = max_cooldown_s
        self._state = self.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._current_cooldown = cooldown_s
        self._health_path = None

    @property
    def state(self):
        if self._state == self.OPEN and self._opened_at is not None:
            if time.time() - self._opened_at >= self._current_cooldown:
                self._state = self.HALF_OPEN
        return self._state

    def check(self):
        """Raise CircuitOpenError if circuit is open. Call before each request."""
        s = self.state
        if s == self.OPEN:
            remaining = self._current_cooldown - (time.time() - self._opened_at)
            raise CircuitOpenError(self.provider, max(0, remaining))

    def record_success(self):
        """Record a successful API call. Closes circuit if half-open."""
        if self._state != self.CLOSED:
            self._state = self.CLOSED
            self._failure_count = 0
            self._current_cooldown = self._initial_cooldown
            self._opened_at = None
            self._emit("provider", "healthy", provider=self.provider)
            self._write_health()
        else:
            self._failure_count = 0

    def record_failure(self, error):
        """Record a quota/429 failure. May open the circuit."""
        self._failure_count += 1
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            self._opened_at = time.time()
            self._current_cooldown = min(
                self._current_cooldown * 2, self.max_cooldown_s
            )
            self._emit(
                "provider",
                "unhealthy",
                provider=self.provider,
                cooldown_s=self._current_cooldown,
            )
            self._write_health()
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            self._opened_at = time.time()
            self._emit(
                "provider",
                "unhealthy",
                provider=self.provider,
                cooldown_s=self._current_cooldown,
            )
            self._write_health()

    def _emit(self, tract, event, **fields):
        """Emit callosum event. Best-effort, never raises."""
        try:
            from think.callosum import callosum_send

            callosum_send(tract, event, **fields)
        except Exception:
            pass

    def _write_health(self):
        """Write circuit breaker state to agents.json. Best-effort."""
        if self._health_path is None:
            return
        try:
            import fcntl
            from datetime import datetime, timezone

            health_dir = self._health_path.parent
            health_dir.mkdir(parents=True, exist_ok=True)
            lock_path = health_dir / "agents.json.lock"
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    data = {}
                    if self._health_path.exists():
                        data = json.loads(self._health_path.read_text())
                    cb_data = data.setdefault("circuit_breakers", {})
                    cb_data[self.provider] = {
                        "state": self._state,
                        "failure_count": self._failure_count,
                        "cooldown_s": self._current_cooldown,
                    }
                    if self._opened_at is not None:
                        cb_data[self.provider]["opened_at"] = datetime.fromtimestamp(
                            self._opened_at, tz=timezone.utc
                        ).isoformat()
                    self._health_path.write_text(json.dumps(data, indent=2))
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception:
            pass


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "Event",
    "GenerateResult",
    "JSONEventCallback",
    "ThinkingEvent",
    "USAGE_KEYS",
    "safe_raw",
]
