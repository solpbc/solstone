# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""In-memory pairing token store."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, replace

from think.pairing.config import (
    MAX_TOKEN_TTL_SECONDS,
    MIN_TOKEN_TTL_SECONDS,
    get_token_ttl_seconds,
)


@dataclass(frozen=True)
class PairingToken:
    token: str
    issued_at: int
    expires_at: int
    ttl_seconds: int
    consumed_at: int | None


_TOKENS: dict[str, PairingToken] = {}
_TOKENS_LOCK = threading.Lock()


def _now(now: int | None) -> int:
    return int(now if now is not None else time.time())


def _clamp_ttl(ttl_seconds: int) -> int:
    return max(MIN_TOKEN_TTL_SECONDS, min(MAX_TOKEN_TTL_SECONDS, ttl_seconds))


def _purge_expired_locked(now: int, *, exclude: str | None = None) -> int:
    expired = [
        token
        for token, entry in _TOKENS.items()
        if token != exclude and entry.expires_at <= now
    ]
    for token in expired:
        del _TOKENS[token]
    return len(expired)


def create_token(
    *, ttl_seconds: int | None = None, now: int | None = None
) -> PairingToken:
    ts = _now(now)
    effective_ttl = _clamp_ttl(
        get_token_ttl_seconds() if ttl_seconds is None else int(ttl_seconds)
    )
    entry = PairingToken(
        token=f"ptk_{secrets.token_urlsafe(32)}",
        issued_at=ts,
        expires_at=ts + effective_ttl,
        ttl_seconds=effective_ttl,
        consumed_at=None,
    )
    with _TOKENS_LOCK:
        _purge_expired_locked(ts)
        _TOKENS[entry.token] = entry
    return entry


def consume_token(token: str, *, now: int | None = None) -> PairingToken | None:
    ts = _now(now)
    with _TOKENS_LOCK:
        _purge_expired_locked(ts, exclude=token)
        entry = _TOKENS.get(token)
        if entry is None:
            return None
        if entry.expires_at <= ts or entry.consumed_at is not None:
            return None
        consumed = replace(entry, consumed_at=ts)
        _TOKENS[token] = consumed
        return consumed


def peek_token(token: str, *, now: int | None = None) -> PairingToken | None:
    ts = _now(now)
    with _TOKENS_LOCK:
        _purge_expired_locked(ts, exclude=token)
        return _TOKENS.get(token)


def purge_expired_tokens(*, now: int | None = None) -> int:
    ts = _now(now)
    with _TOKENS_LOCK:
        return _purge_expired_locked(ts)


__all__ = [
    "PairingToken",
    "consume_token",
    "create_token",
    "peek_token",
    "purge_expired_tokens",
]
