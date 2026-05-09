# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)
from solstone.convey.sol_initiated.state import latest_unresolved_sol_chat_request


def _request(request_id: str, summary: str = "Notice this", ts: int = 1000) -> dict:
    return {
        "kind": KIND_SOL_CHAT_REQUEST,
        "request_id": request_id,
        "summary": summary,
        "ts": ts,
    }


def test_empty_list_returns_none() -> None:
    assert latest_unresolved_sol_chat_request([]) is None


def test_single_request_returns_it_with_event_index() -> None:
    assert latest_unresolved_sol_chat_request([_request("req")]) == {
        "request_id": "req",
        "summary": "Notice this",
        "ts": 1000,
        "event_index": 0,
    }


def test_request_followed_by_open_is_resolved() -> None:
    assert (
        latest_unresolved_sol_chat_request(
            [
                _request("req"),
                {"kind": KIND_OWNER_CHAT_OPEN, "request_id": "req", "surface": "test"},
            ]
        )
        is None
    )


def test_request_followed_by_dismissed_is_resolved() -> None:
    assert (
        latest_unresolved_sol_chat_request(
            [
                _request("req"),
                {
                    "kind": KIND_OWNER_CHAT_DISMISSED,
                    "request_id": "req",
                    "surface": "test",
                    "reason": None,
                },
            ]
        )
        is None
    )


def test_request_followed_by_superseded_is_resolved() -> None:
    assert (
        latest_unresolved_sol_chat_request(
            [
                _request("req"),
                {
                    "kind": KIND_SOL_CHAT_REQUEST_SUPERSEDED,
                    "request_id": "req",
                    "replaced_by": "next",
                },
            ]
        )
        is None
    )


def test_two_unresolved_requests_returns_latter() -> None:
    assert latest_unresolved_sol_chat_request(
        [_request("first", "First", 1000), _request("second", "Second", 2000)]
    ) == {
        "request_id": "second",
        "summary": "Second",
        "ts": 2000,
        "event_index": 1,
    }


def test_first_dismissed_second_unresolved_returns_second() -> None:
    assert latest_unresolved_sol_chat_request(
        [
            _request("first", "First", 1000),
            {
                "kind": KIND_OWNER_CHAT_DISMISSED,
                "request_id": "first",
                "surface": "test",
                "reason": None,
            },
            _request("second", "Second", 2000),
        ]
    ) == {
        "request_id": "second",
        "summary": "Second",
        "ts": 2000,
        "event_index": 2,
    }


def test_unrelated_chat_events_are_ignored() -> None:
    assert latest_unresolved_sol_chat_request(
        [
            {"kind": "owner_message", "text": "hello"},
            _request("req", "Summary", 1000),
            {"kind": "sol_message", "text": "reply"},
        ]
    ) == {
        "request_id": "req",
        "summary": "Summary",
        "ts": 1000,
        "event_index": 1,
    }


def test_missing_summary_falls_back_to_empty_string() -> None:
    assert latest_unresolved_sol_chat_request(
        [{"kind": KIND_SOL_CHAT_REQUEST, "request_id": "req", "ts": 1000}]
    ) == {
        "request_id": "req",
        "summary": "",
        "ts": 1000,
        "event_index": 0,
    }
