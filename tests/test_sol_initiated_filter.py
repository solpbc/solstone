# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
    SURFACE_CONVEY,
)
from solstone.observe.sol_chat_filter import filter_sol_chat_event


def _fields_by_kind() -> dict[str, dict]:
    return {
        KIND_SOL_CHAT_REQUEST: {
            "request_id": "req",
            "summary": "Notice this",
            "message": "details",
            "category": "briefing",
            "dedupe": "key",
            "dedupe_window": "24h",
            "since_ts": 1,
            "trigger_talent": "reflection",
        },
        KIND_SOL_CHAT_REQUEST_SUPERSEDED: {
            "request_id": "old",
            "replaced_by": "new",
        },
        KIND_OWNER_CHAT_OPEN: {
            "request_id": "req",
            "surface": SURFACE_CONVEY,
        },
        KIND_OWNER_CHAT_DISMISSED: {
            "request_id": "req",
            "surface": SURFACE_CONVEY,
            "reason": None,
        },
    }


def test_filter_normalizes_callosum_frames_for_all_kinds() -> None:
    for kind, fields in _fields_by_kind().items():
        frame = {"tract": "chat", "event": kind, "ts": 1234, **fields}

        assert filter_sol_chat_event(frame) == {"kind": kind, "ts": 1234, **fields}


def test_filter_normalizes_chronicle_events_for_all_kinds() -> None:
    for kind, fields in _fields_by_kind().items():
        frame = {"kind": kind, "ts": 1234, **fields}

        assert filter_sol_chat_event(frame) == {"kind": kind, "ts": 1234, **fields}


def test_filter_rejects_non_chat_tract() -> None:
    assert (
        filter_sol_chat_event(
            {"tract": "supervisor", "event": KIND_SOL_CHAT_REQUEST, "ts": 1234}
        )
        is None
    )


def test_filter_rejects_missing_kind_or_event() -> None:
    assert filter_sol_chat_event({"tract": "chat", "ts": 1234}) is None


def test_filter_rejects_unrelated_event() -> None:
    assert filter_sol_chat_event({"tract": "chat", "event": "owner_message"}) is None


def test_filter_rejects_non_dict_input() -> None:
    assert filter_sol_chat_event(None) is None


def test_filter_does_not_import_private_chat_stream_internals() -> None:
    text = Path("solstone/observe/sol_chat_filter.py").read_text(encoding="utf-8")

    assert "chat_stream" not in text
