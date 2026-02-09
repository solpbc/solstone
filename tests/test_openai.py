# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import functools
import importlib
from unittest.mock import AsyncMock, patch

from think.models import GPT_5
from think.providers.cli import ThinkingAggregator
from think.providers.shared import JSONEventCallback


def _openai_provider():
    return importlib.import_module("think.providers.openai")


def _make_test_harness():
    """Create a callback/aggregator pair for testing _translate_codex."""
    events = []
    cb = JSONEventCallback(lambda e: events.append(e))
    aggregator = ThinkingAggregator(cb, GPT_5)
    return events, cb, aggregator


class TestTranslateCodex:
    def test_thread_started_returns_id(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {"type": "thread.started", "thread_id": "abc-123"}

        result = provider._translate_codex(event, aggregator, cb)

        assert result == "abc-123"
        assert events == []

    def test_turn_started_ignored(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {"type": "turn.started"}

        result = provider._translate_codex(event, aggregator, cb)

        assert result is None
        assert events == []

    def test_reasoning_emits_thinking(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {
            "type": "item.completed",
            "item": {"id": "r1", "type": "reasoning", "text": "Let me think..."},
        }

        result = provider._translate_codex(event, aggregator, cb)

        assert result is None
        assert len(events) == 1
        assert events[0]["event"] == "thinking"
        assert events[0]["summary"] == "Let me think..."
        assert events[0]["raw"] == [event]

    def test_agent_message_accumulates(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event1 = {
            "type": "item.completed",
            "item": {"id": "m1", "type": "agent_message", "text": "Hello "},
        }
        event2 = {
            "type": "item.completed",
            "item": {"id": "m2", "type": "agent_message", "text": "world"},
        }

        provider._translate_codex(event1, aggregator, cb)
        provider._translate_codex(event2, aggregator, cb)

        assert events == []
        assert aggregator.has_content

    def test_command_started_emits_tool_start(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {
            "type": "item.started",
            "item": {
                "id": "cmd1",
                "type": "command_execution",
                "command": "echo hi",
            },
        }

        result = provider._translate_codex(event, aggregator, cb)

        assert result is None
        assert len(events) == 1
        assert events[0]["event"] == "tool_start"
        assert events[0]["tool"] == "bash"
        assert events[0]["args"] == {"command": "echo hi"}
        assert events[0]["call_id"] == "cmd1"
        assert events[0]["raw"] == [event]

    def test_command_started_flushes_thinking(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        aggregator.accumulate("I should run a command now")
        event = {
            "type": "item.started",
            "item": {
                "id": "cmd1",
                "type": "command_execution",
                "command": "echo hi",
            },
        }

        provider._translate_codex(event, aggregator, cb)

        assert len(events) == 2
        assert events[0]["event"] == "thinking"
        assert events[1]["event"] == "tool_start"
        assert events[0]["raw"] == [event]
        assert events[1]["raw"] == [event]

    def test_command_completed_emits_tool_end(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {
            "type": "item.completed",
            "item": {
                "id": "cmd1",
                "type": "command_execution",
                "command": "echo hi",
                "aggregated_output": "hi\n",
            },
        }

        result = provider._translate_codex(event, aggregator, cb)

        assert result is None
        assert len(events) == 1
        assert events[0]["event"] == "tool_end"
        assert events[0]["tool"] == "bash"
        assert events[0]["args"] == {"command": "echo hi"}
        assert events[0]["result"] == "hi\n"
        assert events[0]["call_id"] == "cmd1"
        assert events[0]["raw"] == [event]

    def test_turn_completed_captures_usage(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        usage_holder = [{}]
        translate = functools.partial(
            provider._translate_codex, usage_holder=usage_holder
        )
        event = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 5,
                "output_tokens": 2,
            },
        }

        result = translate(event, aggregator, cb)

        assert result is None
        assert usage_holder[0] == {
            "input_tokens": 10,
            "cached_input_tokens": 5,
            "output_tokens": 2,
        }
        assert events == []

    def test_turn_completed_without_holder(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 5,
                "output_tokens": 2,
            },
        }

        result = provider._translate_codex(event, aggregator, cb)

        assert result is None
        assert events == []

    def test_unknown_event_ignored(self):
        provider = _openai_provider()
        events, cb, aggregator = _make_test_harness()
        event = {"type": "something.unknown", "data": "whatever"}

        result = provider._translate_codex(event, aggregator, cb)

        assert result is None
        assert events == []


class TestRunCogitate:
    def test_basic_command_construction(self):
        provider = _openai_provider()
        events = []

        class MockCLIRunner:
            last_instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session-id"
                self.run = AsyncMock(return_value="test result")
                MockCLIRunner.last_instance = self

        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            result = asyncio.run(
                provider.run_cogitate(
                    {"prompt": "hello", "model": GPT_5}, events.append
                )
            )

        assert result == "test result"
        assert MockCLIRunner.last_instance is not None
        assert MockCLIRunner.last_instance.cmd[:6] == [
            "codex",
            "exec",
            "--json",
            "-s",
            "read-only",
            "-m",
        ]
        assert MockCLIRunner.last_instance.cmd[-1] == "-"

    def test_resume_command(self):
        provider = _openai_provider()
        events = []

        class MockCLIRunner:
            last_instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session-id"
                self.run = AsyncMock(return_value="test result")
                MockCLIRunner.last_instance = self

        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            asyncio.run(
                provider.run_cogitate(
                    {
                        "prompt": "hello",
                        "model": GPT_5,
                        "session_id": "thread-abc",
                    },
                    events.append,
                )
            )

        assert MockCLIRunner.last_instance is not None
        assert "resume" in MockCLIRunner.last_instance.cmd
        assert "thread-abc" in MockCLIRunner.last_instance.cmd

    def test_system_instruction_prepended(self):
        provider = _openai_provider()
        events = []

        class MockCLIRunner:
            last_instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session-id"
                self.run = AsyncMock(return_value="test result")
                MockCLIRunner.last_instance = self

        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            asyncio.run(
                provider.run_cogitate(
                    {
                        "prompt": "hello",
                        "model": GPT_5,
                        "system_instruction": "You are a system",
                    },
                    events.append,
                )
            )

        assert MockCLIRunner.last_instance is not None
        assert MockCLIRunner.last_instance.prompt_text.startswith(
            "You are a system\n\n"
        )

    def test_finish_event_emitted(self):
        provider = _openai_provider()
        events = []

        class MockCLIRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session-id"
                self.run = AsyncMock(return_value="result text")

        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            result = asyncio.run(
                provider.run_cogitate(
                    {"prompt": "hello", "model": GPT_5}, events.append
                )
            )

        assert result == "result text"
        assert events[-1]["event"] == "finish"
        assert events[-1]["result"] == "result text"
        assert events[-1]["cli_session_id"] == "test-session-id"

    def test_finish_event_without_session_id(self):
        provider = _openai_provider()
        events = []

        class MockCLIRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = None  # no session ID
                self.run = AsyncMock(return_value="result text")

        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            asyncio.run(
                provider.run_cogitate(
                    {"prompt": "hello", "model": GPT_5}, events.append
                )
            )

        finish = events[-1]
        assert finish["event"] == "finish"
        assert finish["result"] == "result text"
        assert "cli_session_id" not in finish

    def test_finish_event_includes_usage(self):
        provider = _openai_provider()
        events = []
        expected_usage = {"input_tokens": 100, "output_tokens": 50}

        class MockCLIRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "sid"
                # Simulate translate populating usage_holder via the translate callback
                translate_fn = kwargs["translate"]
                agg = kwargs["aggregator"]
                cb = kwargs["callback"]
                turn_event = {"type": "turn.completed", "usage": expected_usage}
                translate_fn(turn_event, agg, cb)
                self.run = AsyncMock(return_value="done")

        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            asyncio.run(
                provider.run_cogitate(
                    {"prompt": "hello", "model": GPT_5}, events.append
                )
            )

        finish = events[-1]
        assert finish["event"] == "finish"
        assert finish["usage"] == expected_usage

    def test_error_emits_event_and_reraises(self):
        provider = _openai_provider()
        events = []

        class MockCLIRunner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session-id"
                self.run = AsyncMock(side_effect=RuntimeError("boom"))

        exc = None
        with patch("think.providers.openai.CLIRunner", MockCLIRunner):
            try:
                asyncio.run(
                    provider.run_cogitate(
                        {"prompt": "hello", "model": GPT_5}, events.append
                    )
                )
            except RuntimeError as caught:
                exc = caught

        assert exc is not None
        assert str(exc) == "boom"
        assert getattr(exc, "_evented", False) is True
        assert events[-1]["event"] == "error"
        assert events[-1]["error"] == "boom"
