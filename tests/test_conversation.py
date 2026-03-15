# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.conversation module — conversation memory service."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def journal_dir(tmp_path):
    """Create a temporary journal directory."""
    journal = tmp_path / "journal"
    journal.mkdir()
    with mock.patch("think.conversation.get_journal", return_value=str(journal)):
        yield journal


# ---------------------------------------------------------------------------
# record_exchange
# ---------------------------------------------------------------------------


def test_record_exchange_writes_jsonl(journal_dir):
    """Exchange is appended to conversation/exchanges.jsonl."""
    from think.conversation import record_exchange

    record_exchange(
        ts=1710000000000,
        facet="work",
        app="entities",
        path="/app/entities/adrian",
        user_message="what's our history with adrian?",
        agent_response="You met Adrian at betaworks.",
        muse="unified",
        agent_id="12345",
    )

    jsonl_path = journal_dir / "conversation" / "exchanges.jsonl"
    assert jsonl_path.exists()

    with open(jsonl_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    ex = lines[0]
    assert ex["ts"] == 1710000000000
    assert ex["facet"] == "work"
    assert ex["app"] == "entities"
    assert ex["user_message"] == "what's our history with adrian?"
    assert ex["agent_response"] == "You met Adrian at betaworks."
    assert ex["muse"] == "unified"
    assert ex["agent_id"] == "12345"


def test_record_exchange_writes_journal_segment(journal_dir):
    """Exchange creates a journal segment markdown file for search indexing."""
    from think.conversation import record_exchange

    # Use a known timestamp: 2026-03-15 14:30:00 UTC
    ts = int(datetime(2026, 3, 15, 14, 30, 0).timestamp() * 1000)

    record_exchange(
        ts=ts,
        facet="work",
        app="calendar",
        path="/app/calendar",
        user_message="move my 3pm to 4pm",
        agent_response="Done — moved 'DVD sync' to 4pm.",
        muse="unified",
        agent_id="67890",
    )

    # Check journal segment directory: YYYYMMDD/conversation/HHMMSS_1/agents/
    day = datetime.fromtimestamp(ts / 1000).strftime("%Y%m%d")
    time_key = datetime.fromtimestamp(ts / 1000).strftime("%H%M%S")
    md_path = journal_dir / day / "conversation" / f"{time_key}_1" / "agents" / "conversation.md"

    assert md_path.exists()
    content = md_path.read_text()
    assert "move my 3pm to 4pm" in content
    assert "Done — moved 'DVD sync' to 4pm." in content
    assert "**Facet:** work" in content
    assert "calendar" in content


def test_record_exchange_appends_multiple(journal_dir):
    """Multiple exchanges append to the same JSONL file."""
    from think.conversation import record_exchange

    record_exchange(
        ts=1710000001000,
        user_message="hello",
        agent_response="hi there",
        muse="triage",
    )
    record_exchange(
        ts=1710000002000,
        user_message="what time is it?",
        agent_response="It's 2pm.",
        muse="triage",
    )

    jsonl_path = journal_dir / "conversation" / "exchanges.jsonl"
    with open(jsonl_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0]["user_message"] == "hello"
    assert lines[1]["user_message"] == "what time is it?"


def test_record_exchange_skips_empty(journal_dir):
    """Empty user_message or agent_response is silently skipped."""
    from think.conversation import record_exchange

    record_exchange(user_message="", agent_response="response", muse="triage")
    record_exchange(user_message="hello", agent_response="", muse="triage")

    jsonl_path = journal_dir / "conversation" / "exchanges.jsonl"
    assert not jsonl_path.exists()


# ---------------------------------------------------------------------------
# get_recent_exchanges
# ---------------------------------------------------------------------------


def test_get_recent_exchanges_empty(journal_dir):
    """Returns empty list when no exchanges exist."""
    from think.conversation import get_recent_exchanges

    assert get_recent_exchanges() == []


def test_get_recent_exchanges_returns_last_n(journal_dir):
    """Returns the last N exchanges."""
    from think.conversation import get_recent_exchanges, record_exchange

    for i in range(15):
        record_exchange(
            ts=1710000000000 + i * 1000,
            user_message=f"msg {i}",
            agent_response=f"resp {i}",
            muse="triage",
        )

    recent = get_recent_exchanges(limit=5)
    assert len(recent) == 5
    assert recent[0]["user_message"] == "msg 10"
    assert recent[-1]["user_message"] == "msg 14"


def test_get_recent_exchanges_filters_by_facet(journal_dir):
    """Facet filter returns only matching exchanges."""
    from think.conversation import get_recent_exchanges, record_exchange

    record_exchange(
        ts=1710000001000,
        facet="work",
        user_message="work question",
        agent_response="work answer",
        muse="triage",
    )
    record_exchange(
        ts=1710000002000,
        facet="personal",
        user_message="personal question",
        agent_response="personal answer",
        muse="triage",
    )

    work = get_recent_exchanges(facet="work")
    assert len(work) == 1
    assert work[0]["facet"] == "work"

    personal = get_recent_exchanges(facet="personal")
    assert len(personal) == 1
    assert personal[0]["facet"] == "personal"


# ---------------------------------------------------------------------------
# get_today_exchanges
# ---------------------------------------------------------------------------


def test_get_today_exchanges_filters_by_day(journal_dir):
    """Only returns exchanges from today."""
    from think.conversation import get_today_exchanges, record_exchange
    from think.utils import now_ms

    # Record an exchange with current timestamp (today)
    record_exchange(
        ts=now_ms(),
        user_message="today question",
        agent_response="today answer",
        muse="triage",
    )

    # Record an exchange with old timestamp (not today)
    record_exchange(
        ts=1000000000000,  # 2001-09-08
        user_message="old question",
        agent_response="old answer",
        muse="triage",
    )

    today = get_today_exchanges()
    assert len(today) == 1
    assert today[0]["user_message"] == "today question"


# ---------------------------------------------------------------------------
# build_memory_context
# ---------------------------------------------------------------------------


def test_build_memory_context_empty(journal_dir):
    """Returns empty string when no exchanges exist."""
    from think.conversation import build_memory_context

    assert build_memory_context() == ""


def test_build_memory_context_includes_recent(journal_dir):
    """Context includes recent exchanges."""
    from think.conversation import build_memory_context, record_exchange
    from think.utils import now_ms

    ts = now_ms()
    record_exchange(
        ts=ts,
        facet="work",
        app="entities",
        user_message="who is adrian?",
        agent_response="Adrian is the CTO of Own Company.",
        muse="unified",
    )

    context = build_memory_context()
    assert "who is adrian?" in context
    assert "Adrian is the CTO" in context
    assert "Recent Conversations" in context


def test_build_memory_context_truncates_long_responses(journal_dir):
    """Long agent responses are truncated in context output."""
    from think.conversation import (
        MAX_RESPONSE_CHARS,
        build_memory_context,
        record_exchange,
    )
    from think.utils import now_ms

    long_response = "x" * 500
    record_exchange(
        ts=now_ms(),
        user_message="tell me a story",
        agent_response=long_response,
        muse="unified",
    )

    context = build_memory_context()
    # Response should be truncated
    assert "..." in context
    assert long_response not in context


def test_build_memory_context_earlier_today(journal_dir):
    """When more exchanges exist today than the recent limit, earlier ones are compact."""
    from think.conversation import build_memory_context, record_exchange
    from think.utils import now_ms

    ts = now_ms()
    # Record 15 exchanges "today"
    for i in range(15):
        record_exchange(
            ts=ts + i * 1000,
            user_message=f"question {i}",
            agent_response=f"answer {i}",
            muse="unified",
        )

    context = build_memory_context(recent_limit=10)
    assert "Earlier Today" in context
    assert "Recent Conversations" in context


# ---------------------------------------------------------------------------
# inject_memory
# ---------------------------------------------------------------------------


def test_inject_memory_replaces_marker():
    """Injection point is replaced with memory context."""
    from think.conversation import inject_memory

    instruction = """## Before

## Conversation Memory

<!-- CONVERSATION_MEMORY_INJECTION_POINT
This section is populated by the conversation memory service.
Until then, each exchange is independent.
-->

## After"""

    result = inject_memory(instruction, "### Recent\nHello world")
    assert "CONVERSATION_MEMORY_INJECTION_POINT" not in result
    assert "### Recent\nHello world" in result
    assert "## Before" in result
    assert "## After" in result


def test_inject_memory_no_marker():
    """If no marker present, instruction is returned unchanged."""
    from think.conversation import inject_memory

    instruction = "No marker here."
    result = inject_memory(instruction, "some context")
    assert result == instruction


def test_inject_memory_empty_context():
    """Empty context gets a placeholder message."""
    from think.conversation import inject_memory

    instruction = "<!-- CONVERSATION_MEMORY_INJECTION_POINT -->"
    result = inject_memory(instruction, "")
    assert "No conversation history yet." in result


# ---------------------------------------------------------------------------
# _format_exchange
# ---------------------------------------------------------------------------


def test_format_exchange_full():
    """Full format includes user message and truncated response."""
    from think.conversation import _format_exchange

    ex = {
        "ts": 1710000000000,
        "app": "entities",
        "facet": "work",
        "user_message": "who is adrian?",
        "agent_response": "Adrian is the CTO.",
    }

    result = _format_exchange(ex, compact=False)
    assert "User: who is adrian?" in result
    assert "Sol: Adrian is the CTO." in result
    assert "entities" in result
    assert "work" in result


def test_format_exchange_compact():
    """Compact format is a one-liner."""
    from think.conversation import _format_exchange

    ex = {
        "ts": 1710000000000,
        "app": "calendar",
        "facet": "work",
        "user_message": "what's on my schedule today?",
        "agent_response": "You have 3 meetings.",
    }

    result = _format_exchange(ex, compact=True)
    assert result.startswith("- [")
    assert "what's on my schedule today?" in result
    assert "You have 3 meetings" not in result  # Compact omits response


# ---------------------------------------------------------------------------
# Pre-hook integration
# ---------------------------------------------------------------------------


def test_conversation_memory_pre_hook(journal_dir):
    """Pre-hook injects memory into user instruction."""
    from muse.conversation_memory import pre_process
    from think.conversation import record_exchange
    from think.utils import now_ms

    # Record an exchange first
    record_exchange(
        ts=now_ms(),
        facet="work",
        user_message="hello",
        agent_response="hi there!",
        muse="unified",
    )

    context = {
        "user_instruction": """Some instructions.

## Conversation Memory

<!-- CONVERSATION_MEMORY_INJECTION_POINT
Populated by conversation memory service.
-->

## Other section""",
        "facet": "work",
    }

    result = pre_process(context)
    assert result is not None
    assert "user_instruction" in result
    assert "CONVERSATION_MEMORY_INJECTION_POINT" not in result["user_instruction"]
    assert "hello" in result["user_instruction"]
    assert "hi there!" in result["user_instruction"]


def test_conversation_memory_pre_hook_no_marker():
    """Pre-hook returns None when no injection marker present."""
    from muse.conversation_memory import pre_process

    context = {"user_instruction": "No marker here."}
    result = pre_process(context)
    assert result is None
