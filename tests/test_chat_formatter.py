# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc


from think.chat_formatter import format_chat
from think.formatters import get_formatter


def test_format_chat_produces_markdown_for_each_kind():
    entries = [
        {
            "ts": 1,
            "kind": "owner_message",
            "text": "Need a diff",
        },
        {
            "ts": 2,
            "kind": "sol_message",
            "text": "I can do that",
        },
        {
            "ts": 3,
            "kind": "talent_spawned",
            "name": "exec",
            "task": "compare drafts",
        },
        {
            "ts": 4,
            "kind": "talent_finished",
            "name": "exec",
            "summary": "summarized the differences",
        },
        {
            "ts": 5,
            "kind": "talent_errored",
            "name": "exec",
            "reason": "repo unavailable",
        },
        {
            "ts": 6,
            "kind": "chat_error",
            "reason": "chat had trouble — try again",
        },
    ]

    chunks, meta = format_chat(
        entries,
        {"owner_name": "Alice", "agent_name": "Sol-agent"},
    )

    assert meta == {"indexer": {"agent": "chat"}}
    assert [chunk["markdown"] for chunk in chunks] == [
        "**Alice** Need a diff",
        "**Sol-agent** I can do that",
        "*[exec spawned: compare drafts]*",
        "*[exec finished: summarized the differences]*",
        "*[exec errored: repo unavailable]*",
        "*[chat trouble: chat had trouble — try again]*",
    ]


def test_format_chat_uses_identity_owner_and_agent_names(monkeypatch):
    monkeypatch.setattr(
        "think.chat_formatter.get_config",
        lambda: {
            "identity": {"name": "Alice Smith", "preferred": "Alice"},
            "agent": {"name": "Sol-agent"},
        },
    )

    chunks, _meta = format_chat(
        [
            {"ts": 1, "kind": "owner_message", "text": "hello"},
            {"ts": 2, "kind": "sol_message", "text": "hi"},
        ]
    )

    assert [chunk["markdown"] for chunk in chunks] == [
        "**Alice** hello",
        "**Sol-agent** hi",
    ]


def test_format_chat_fallback_labels_when_identity_missing(monkeypatch):
    monkeypatch.setattr("think.chat_formatter.get_config", lambda: {})

    chunks, _meta = format_chat(
        [
            {"ts": 1, "kind": "owner_message", "text": "hello"},
            {"ts": 2, "kind": "sol_message", "text": "hi"},
        ]
    )

    assert [chunk["markdown"] for chunk in chunks] == [
        "**Owner** hello",
        "**Sol** hi",
    ]


def test_get_formatter_chat_jsonl_wins_over_talents_fallback():
    formatter = get_formatter("20260420/chat/120000_300/chat.jsonl")

    assert formatter is not None
    assert formatter.__module__ == "think.chat_formatter"
    assert formatter.__name__ == "format_chat"
