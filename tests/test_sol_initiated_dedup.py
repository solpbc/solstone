# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import pytest

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)
from solstone.convey.sol_initiated.dedup import (
    _is_live_for_dedup,
    _is_unresolved_for_supersede,
    parse_dedupe_window,
)


def _request(request_id: str, *, ts: int = 1_000, dedupe: str = "k") -> dict:
    return {
        "kind": KIND_SOL_CHAT_REQUEST,
        "ts": ts,
        "request_id": request_id,
        "dedupe": dedupe,
    }


def test_parse_dedupe_window_units() -> None:
    assert parse_dedupe_window("1s") == 1_000
    assert parse_dedupe_window("2m") == 120_000
    assert parse_dedupe_window("3h") == 10_800_000
    assert parse_dedupe_window("4d") == 345_600_000


@pytest.mark.parametrize("value", ["", "24", "h", "0h", "1w"])
def test_parse_dedupe_window_rejects_malformed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_dedupe_window(value)


def test_live_for_dedup_expires_and_matches_key() -> None:
    events = [_request("r1", ts=1_000, dedupe="a")]

    assert _is_live_for_dedup(events, "a", 1_000, 1_999) is True
    assert _is_live_for_dedup(events, "a", 1_000, 2_000) is False
    assert _is_live_for_dedup(events, "b", 1_000, 1_500) is False


def test_dismissal_releases_dedup() -> None:
    events = [
        _request("r1", ts=1_000, dedupe="a"),
        {
            "kind": KIND_OWNER_CHAT_DISMISSED,
            "ts": 1_500,
            "request_id": "r1",
        },
    ]

    assert _is_live_for_dedup(events, "a", 10_000, 2_000) is False


def test_unresolved_for_supersede_walks_back() -> None:
    events = [
        _request("old"),
        {
            "kind": KIND_SOL_CHAT_REQUEST_SUPERSEDED,
            "ts": 2_000,
            "request_id": "old",
            "replaced_by": "new",
        },
        _request("new", ts=2_001),
    ]

    assert _is_unresolved_for_supersede(events) == "new"
    assert _is_unresolved_for_supersede([*events, {"kind": "sol_message"}]) is None
