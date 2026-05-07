# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from solstone.think.pairing import tokens


def setup_function() -> None:
    tokens._TOKENS.clear()


def test_create_token_uses_expected_shape_and_metadata():
    token = tokens.create_token(ttl_seconds=600, now=1000)

    assert token.token.startswith("ptk_")
    assert token.issued_at == 1000
    assert token.expires_at == 1600
    assert token.ttl_seconds == 600
    assert token.consumed_at is None


def test_create_token_clamps_ttl():
    low = tokens.create_token(ttl_seconds=1, now=1000)
    high = tokens.create_token(ttl_seconds=9999, now=1000)

    assert low.ttl_seconds == 60
    assert low.expires_at == 1060
    assert high.ttl_seconds == 3600
    assert high.expires_at == 4600


def test_consume_token_marks_token_used_once():
    created = tokens.create_token(ttl_seconds=600, now=1000)

    first = tokens.consume_token(created.token, now=1100)
    second = tokens.consume_token(created.token, now=1101)
    peeked = tokens.peek_token(created.token, now=1101)

    assert first is not None
    assert first.consumed_at == 1100
    assert second is None
    assert peeked is not None
    assert peeked.consumed_at == 1100


def test_consume_token_rejects_expired_token():
    created = tokens.create_token(ttl_seconds=60, now=1000)

    assert tokens.consume_token(created.token, now=1060) is None
    expired = tokens.peek_token(created.token, now=1060)
    assert expired is not None
    assert expired.expires_at == 1060
    assert expired.consumed_at is None


def test_purge_expired_tokens_removes_only_expired_entries():
    keep = tokens.create_token(ttl_seconds=600, now=1000)
    drop = tokens.create_token(ttl_seconds=60, now=1000)

    purged = tokens.purge_expired_tokens(now=1060)

    assert purged == 1
    assert tokens.peek_token(drop.token, now=1060) is None
    assert tokens.peek_token(keep.token, now=1060) is not None
