# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Claude CLI translator in think.providers.anthropic."""

import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from think.providers.anthropic import _translate_claude
from think.providers.cli import ThinkingAggregator
from think.providers.shared import JSONEventCallback


def _anthropic_provider():
    return importlib.import_module("think.providers.anthropic")


@pytest.fixture
def callback_events():
    """Collect emitted events."""
    events = []
    return events, JSONEventCallback(lambda e: events.append(e))


@pytest.fixture
def aggregator(callback_events):
    """Create a ThinkingAggregator."""
    _, cb = callback_events
    return ThinkingAggregator(cb, model="claude-sonnet-4-20250514")


@pytest.fixture
def state():
    """Create mutable state dicts for translator."""
    return {"pending_tools": {}, "result_meta": {}}


class TestTranslateClaudeSystemEvent:
    def test_init_returns_session_id(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-123",
            "model": "claude-sonnet-4-20250514",
        }
        result = _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert result == "sess-123"
        assert len(events) == 0  # No events emitted for system init

    def test_non_init_system_event(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {"type": "system", "subtype": "other"}
        result = _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert result is None


class TestTranslateClaudeAssistantEvent:
    def test_text_accumulates(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {
            "type": "assistant",
            "message": {
                "id": "msg_01",
                "content": [{"type": "text", "text": "Hello there"}],
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert aggregator.flush_as_result() == "Hello there"
        assert len(events) == 0  # No events emitted for plain text

    def test_thinking_emits_event(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {
            "type": "assistant",
            "message": {
                "id": "msg_01",
                "content": [{"type": "thinking", "thinking": "Let me consider..."}],
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert len(events) == 1
        assert events[0]["event"] == "thinking"
        assert events[0]["summary"] == "Let me consider..."
        assert events[0]["model"] == "claude-sonnet-4-20250514"
        assert events[0]["raw"] == [event]

    def test_tool_use_emits_tool_start(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {
            "type": "assistant",
            "message": {
                "id": "msg_01",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"file_path": "/tmp/test.txt"},
                    }
                ],
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert len(events) == 1
        assert events[0]["event"] == "tool_start"
        assert events[0]["tool"] == "Read"
        assert events[0]["args"] == {"file_path": "/tmp/test.txt"}
        assert events[0]["call_id"] == "toolu_01"
        assert events[0]["raw"] == [event]
        # Verify pending_tools tracking
        assert "toolu_01" in state["pending_tools"]

    def test_text_before_tool_use_flushed_as_thinking(
        self, aggregator, callback_events, state
    ):
        """Text followed by tool_use should flush text as thinking."""
        events, cb = callback_events
        event = {
            "type": "assistant",
            "message": {
                "id": "msg_01",
                "content": [
                    {"type": "text", "text": "I'll read that file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"file_path": "/tmp/test.txt"},
                    },
                ],
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        # Should emit: thinking (flushed text), then tool_start
        assert len(events) == 2
        assert events[0]["event"] == "thinking"
        assert events[0]["summary"] == "I'll read that file."
        assert events[1]["event"] == "tool_start"
        assert events[1]["tool"] == "Read"

    def test_multiple_tool_uses(self, aggregator, callback_events, state):
        """Multiple tool_use blocks should each emit tool_start."""
        events, cb = callback_events
        event = {
            "type": "assistant",
            "message": {
                "id": "msg_01",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"file_path": "/a.txt"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "Read",
                        "input": {"file_path": "/b.txt"},
                    },
                ],
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert len(events) == 2
        assert events[0]["event"] == "tool_start"
        assert events[0]["call_id"] == "toolu_01"
        assert events[1]["event"] == "tool_start"
        assert events[1]["call_id"] == "toolu_02"


class TestTranslateClaudeUserEvent:
    def test_tool_result_emits_tool_end(self, aggregator, callback_events, state):
        events, cb = callback_events
        # First, register a pending tool
        state["pending_tools"]["toolu_01"] = {
            "tool": "Read",
            "args": {"file_path": "/tmp/test.txt"},
        }
        event = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "file contents here",
                    }
                ]
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert len(events) == 1
        assert events[0]["event"] == "tool_end"
        assert events[0]["tool"] == "Read"
        assert events[0]["args"] == {"file_path": "/tmp/test.txt"}
        assert events[0]["result"] == "file contents here"
        assert events[0]["call_id"] == "toolu_01"
        assert events[0]["raw"] == [event]
        # Verify pending tool was consumed
        assert "toolu_01" not in state["pending_tools"]

    def test_tool_result_unknown_id(self, aggregator, callback_events, state):
        """tool_result with unknown ID should still emit tool_end."""
        events, cb = callback_events
        event = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_unknown",
                        "content": "result",
                    }
                ]
            },
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert len(events) == 1
        assert events[0]["event"] == "tool_end"
        assert events[0]["tool"] == ""
        assert events[0]["call_id"] == "toolu_unknown"


class TestTranslateClaudeResultEvent:
    def test_stores_usage_and_cost(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {
            "type": "result",
            "total_cost_usd": 0.0042,
            "usage": {"input_tokens": 150, "output_tokens": 30},
            "session_id": "sess-123",
        }
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert state["result_meta"]["cost_usd"] == 0.0042
        assert state["result_meta"]["usage"]["input_tokens"] == 150
        assert state["result_meta"]["usage"]["output_tokens"] == 30
        assert state["result_meta"]["usage"]["total_tokens"] == 180
        assert len(events) == 0  # No events emitted for result

    def test_result_without_usage(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {"type": "result", "total_cost_usd": 0.001}
        _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert state["result_meta"]["cost_usd"] == 0.001
        assert "usage" not in state["result_meta"]


class TestTranslateClaudeUnknownEvent:
    def test_unknown_type_ignored(self, aggregator, callback_events, state):
        events, cb = callback_events
        event = {"type": "unknown_event", "data": "whatever"}
        result = _translate_claude(
            event, aggregator, cb, state["pending_tools"], state["result_meta"]
        )
        assert result is None
        assert len(events) == 0


class TestTranslateClaudeFixtureSequence:
    """Test processing a full sequence of events from fixture file."""

    def test_full_tool_cycle(self, callback_events):
        """Process a complete tool-use cycle and verify all events."""
        events, cb = callback_events
        aggregator = ThinkingAggregator(cb, model="claude-sonnet-4-20250514")
        pending_tools: dict = {}
        result_meta: dict = {}

        fixture_path = Path(__file__).parent / "fixtures" / "claude_cli_events.jsonl"
        with open(fixture_path) as f:
            raw_events = [json.loads(line) for line in f if line.strip()]

        session_id = None
        for raw_event in raw_events:
            sid = _translate_claude(
                raw_event, aggregator, cb, pending_tools, result_meta
            )
            if sid:
                session_id = sid

        # Verify session ID captured from init event
        assert session_id == "test-session-abc123"

        # Verify events emitted
        event_types = [e["event"] for e in events]
        assert "tool_start" in event_types
        assert "tool_end" in event_types

        # Verify tool pairing
        tool_starts = [e for e in events if e["event"] == "tool_start"]
        tool_ends = [e for e in events if e["event"] == "tool_end"]
        assert len(tool_starts) == 1
        assert len(tool_ends) == 1
        assert tool_starts[0]["call_id"] == tool_ends[0]["call_id"]
        assert tool_ends[0]["result"] == "Hello, world!"

        # Verify usage captured
        assert result_meta["cost_usd"] == 0.0042
        assert result_meta["usage"]["input_tokens"] == 150
        assert result_meta["usage"]["output_tokens"] == 30

        # Verify final text in aggregator (last assistant text)
        result = aggregator.flush_as_result()
        assert "Hello, world!" in result

        # Verify all pending tools consumed
        assert len(pending_tools) == 0


class TestRunCogitateCommand:
    """Tests for run_cogitate command construction."""

    def _mock_runner(self):
        """Create a MockCLIRunner that captures the command."""

        class MockCLIRunner:
            last_instance = None

            def __init__(self, **kwargs):
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session"
                self.run = AsyncMock(return_value="result")
                MockCLIRunner.last_instance = self

        return MockCLIRunner

    def test_plan_mode_with_sol_call_allowed(self):
        provider = _anthropic_provider()
        MockCLIRunner = self._mock_runner()
        with (
            patch("think.providers.anthropic.CLIRunner", MockCLIRunner),
            patch("think.providers.anthropic.check_cli_binary"),
        ):
            asyncio.run(
                provider.run_cogitate(
                    {"prompt": "hello", "model": "claude-sonnet-4"}, lambda e: None
                )
            )
        cmd = MockCLIRunner.last_instance.cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "plan"
        assert cmd[cmd.index("--allowedTools") + 1] == "Bash(sol call *)"
