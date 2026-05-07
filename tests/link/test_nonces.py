# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from pathlib import Path

from solstone.think.link.nonces import NONCE_TTL_SECONDS, Nonce, NonceStore


def test_add_and_consume(tmp_path: Path) -> None:
    store = NonceStore(tmp_path / "nonces.json")

    added = store.add("abc123", "phone", now=1000)
    consumed = store.consume("abc123", now=1001)

    assert added == Nonce(
        value="abc123",
        device_label="phone",
        issued_at=1000,
        expires_at=1000 + NONCE_TTL_SECONDS,
        used=False,
    )
    assert consumed == Nonce(
        value="abc123",
        device_label="phone",
        issued_at=1000,
        expires_at=1000 + NONCE_TTL_SECONDS,
        used=True,
    )


def test_consume_is_single_use(tmp_path: Path) -> None:
    store = NonceStore(tmp_path / "nonces.json")

    store.add("abc123", "phone", now=1000)

    assert store.consume("abc123", now=1001) is not None
    assert store.consume("abc123", now=1002) is None


def test_expired_nonce_rejected(tmp_path: Path) -> None:
    store = NonceStore(tmp_path / "nonces.json")

    store.add("abc123", "phone", now=1000)

    assert store.consume("abc123", now=1000 + NONCE_TTL_SECONDS + 1) is None


def test_unknown_nonce_returns_none(tmp_path: Path) -> None:
    store = NonceStore(tmp_path / "nonces.json")

    assert store.consume("never-added") is None


def test_gc_removes_expired_and_used(tmp_path: Path) -> None:
    store = NonceStore(tmp_path / "nonces.json")

    store.add("live", "device", now=1000)
    store.add("used", "device", now=1000)
    store.consume("used", now=1001)

    removed = store.gc(now=1001)

    assert removed == 1
    assert [entry.value for entry in store.snapshot()] == ["live"]

    store.add("fresh", "device", now=2000)
    store.add("also_expired", "device", now=2000 - NONCE_TTL_SECONDS - 10)
    store.gc(now=2000)

    assert {entry.value for entry in store.snapshot()} == {"fresh"}


def test_persistence_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "nonces.json"
    first = NonceStore(path)
    first.add("shared", "device", now=1000)

    second = NonceStore(path)
    entry = second.consume("shared", now=1001)

    assert entry is not None
    assert entry.value == "shared"
