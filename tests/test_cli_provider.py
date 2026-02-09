# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.providers.cli — CLI subprocess runner infrastructure."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from think.providers.cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
)
from think.providers.shared import JSONEventCallback

# ---------------------------------------------------------------------------
# assemble_prompt
# ---------------------------------------------------------------------------


class TestAssemblePrompt:
    def test_all_fields(self):
        config = {
            "transcript": "Speaker A: hello",
            "extra_context": "Today is Monday",
            "user_instruction": "Summarize the transcript",
            "prompt": "What happened?",
            "system_instruction": "You are a helpful assistant",
        }
        body, system = assemble_prompt(config)
        assert "Speaker A: hello" in body
        assert "Today is Monday" in body
        assert "Summarize the transcript" in body
        assert "What happened?" in body
        assert system == "You are a helpful assistant"
        # Parts joined with double newlines
        assert body.count("\n\n") == 3

    def test_prompt_only(self):
        config = {"prompt": "hello"}
        body, system = assemble_prompt(config)
        assert body == "hello"
        assert system is None

    def test_empty_config(self):
        body, system = assemble_prompt({})
        assert body == ""
        assert system is None

    def test_skips_empty_values(self):
        config = {
            "transcript": "",
            "extra_context": None,
            "user_instruction": "Do something",
            "prompt": "Go",
        }
        body, system = assemble_prompt(config)
        assert body == "Do something\n\nGo"
        assert system is None

    def test_system_instruction_empty_string(self):
        config = {"prompt": "test", "system_instruction": ""}
        _, system = assemble_prompt(config)
        assert system is None


# ---------------------------------------------------------------------------
# ThinkingAggregator
# ---------------------------------------------------------------------------


class TestThinkingAggregator:
    def _make_aggregator(self):
        """Create aggregator with event capture."""
        events = []
        cb = JSONEventCallback(events.append)
        agg = ThinkingAggregator(cb, model="test-model")
        return agg, events

    def test_accumulate_and_flush_as_thinking(self):
        agg, events = self._make_aggregator()
        agg.accumulate("hello ")
        agg.accumulate("world")
        agg.flush_as_thinking(raw_events=[{"type": "message"}])

        assert len(events) == 1
        assert events[0]["event"] == "thinking"
        assert events[0]["summary"] == "hello world"
        assert events[0]["model"] == "test-model"
        assert events[0]["raw"] == [{"type": "message"}]

    def test_flush_thinking_empty_buffer_is_noop(self):
        agg, events = self._make_aggregator()
        agg.flush_as_thinking()
        assert len(events) == 0

    def test_flush_thinking_whitespace_only_is_noop(self):
        agg, events = self._make_aggregator()
        agg.accumulate("   ")
        agg.flush_as_thinking()
        assert len(events) == 0

    def test_flush_as_result(self):
        agg, events = self._make_aggregator()
        agg.accumulate("final answer")
        result = agg.flush_as_result()
        assert result == "final answer"
        # No events emitted for result flush
        assert len(events) == 0
        # Buffer is cleared
        assert agg.flush_as_result() == ""

    def test_multiple_thinking_flushes(self):
        """Simulate text -> tool -> text -> tool -> text pattern."""
        agg, events = self._make_aggregator()

        # First text chunk (before first tool call)
        agg.accumulate("Let me check...")
        agg.flush_as_thinking()

        # Second text chunk (between tool calls)
        agg.accumulate("Now let me verify...")
        agg.flush_as_thinking()

        # Final text (the result)
        agg.accumulate("The answer is 42")
        result = agg.flush_as_result()

        assert len(events) == 2
        assert events[0]["summary"] == "Let me check..."
        assert events[1]["summary"] == "Now let me verify..."
        assert result == "The answer is 42"

    def test_has_content(self):
        agg, _ = self._make_aggregator()
        assert not agg.has_content
        agg.accumulate("x")
        assert agg.has_content
        agg.flush_as_result()
        assert not agg.has_content

    def test_no_raw_events(self):
        agg, events = self._make_aggregator()
        agg.accumulate("thinking")
        agg.flush_as_thinking()
        assert "raw" not in events[0]

    def test_strips_whitespace(self):
        agg, events = self._make_aggregator()
        agg.accumulate("  padded  ")
        agg.flush_as_thinking()
        assert events[0]["summary"] == "padded"


class TestCLIRunnerFirstEventTimeout:
    def test_first_event_timeout_includes_stderr(self):
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        class HangingStdout:
            async def readline(self):
                future = asyncio.get_running_loop().create_future()
                return await future

        class MockStderr:
            def __init__(self):
                self._lines = [b"Please authenticate first\n"]
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._lines):
                    raise StopAsyncIteration
                line = self._lines[self._index]
                self._index += 1
                return line

        process = AsyncMock()
        process.stdout = HangingStdout()
        process.stderr = MockStderr()
        process.stdin = AsyncMock()
        process.stdin.write = lambda _data: None
        process.stdin.close = lambda: None
        process.kill = lambda: None
        process.wait = AsyncMock(return_value=0)

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test prompt",
            translate=lambda _event, _agg, _cb: None,
            callback=callback,
            aggregator=aggregator,
            timeout=5,
            first_event_timeout=0.1,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
            pytest.raises(RuntimeError) as exc_info,
        ):
            asyncio.run(runner.run())

        message = str(exc_info.value)
        assert "authenticate" in message.lower()
        assert "Check that the CLI tool is installed and authenticated." in message

        error_events = [event for event in events if event.get("event") == "error"]
        assert len(error_events) == 1
        assert "Please authenticate first" in error_events[0]["error"]
