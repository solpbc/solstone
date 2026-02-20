# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.providers.cli — CLI subprocess runner infrastructure."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from think.providers.cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
    build_cogitate_env,
)
from think.providers.shared import JSONEventCallback, safe_raw

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


class _MockStderr:
    """Async iterator yielding pre-set stderr lines."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class _MockStdout:
    """Async iterator yielding pre-set stdout lines, with readline support."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._index = 0

    async def readline(self):
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def _make_process(stdout_lines, stderr_lines, return_code):
    """Create a mock process with given stdout/stderr/exit code."""
    process = AsyncMock()
    process.stdout = _MockStdout(stdout_lines)
    process.stderr = _MockStderr(stderr_lines)
    process.stdin = AsyncMock()
    process.stdin.write = lambda _data: None
    process.stdin.close = lambda: None
    process.kill = lambda: None
    process.wait = AsyncMock(return_value=return_code)
    return process


class TestCLIRunnerExitCode:
    """Tests for CLIRunner handling of non-zero exit codes."""

    def test_nonzero_exit_no_output_raises(self):
        """CLI exits with error and no result → RuntimeError with stderr."""
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        process = _make_process(
            stdout_lines=[],
            stderr_lines=[b"TerminalQuotaError: quota exhausted\n"],
            return_code=1,
        )

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=lambda _e, _a, _c: None,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
            pytest.raises(RuntimeError, match="quota exhausted"),
        ):
            asyncio.run(runner.run())

        # CLIRunner should NOT emit error events — that's the caller's job
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 0

    def test_nonzero_exit_with_output_returns_result(self):
        """CLI exits with error but produced output → return result + warning."""
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        # translate accumulates text from stdout events
        def translate(event, agg, cb):
            if event.get("type") == "text":
                agg.accumulate(event["content"])
            return None

        process = _make_process(
            stdout_lines=[b'{"type": "text", "content": "The answer is 42"}\n'],
            stderr_lines=[b"Warning: something went wrong\n"],
            return_code=1,
        )

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=translate,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
        ):
            result = asyncio.run(runner.run())

        assert result == "The answer is 42"
        warning_events = [e for e in events if e.get("event") == "warning"]
        assert len(warning_events) == 1
        assert "code 1" in warning_events[0]["message"]
        assert "something went wrong" in warning_events[0]["stderr"]

    def test_zero_exit_empty_result_ok(self):
        """CLI exits 0 with no output → return empty string, no error."""
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        process = _make_process(
            stdout_lines=[],
            stderr_lines=[],
            return_code=0,
        )

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=lambda _e, _a, _c: None,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
        ):
            result = asyncio.run(runner.run())

        assert result == ""
        assert not [e for e in events if e.get("event") in ("error", "warning")]


class TestCLIRunnerFirstEventTimeout:
    def test_first_event_timeout_includes_stderr(self):
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        class HangingStdout:
            async def readline(self):
                future = asyncio.get_running_loop().create_future()
                return await future

        process = _make_process([], [b"Please authenticate first\n"], 0)
        process.stdout = HangingStdout()  # Override with hanging version

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


# ---------------------------------------------------------------------------
# safe_raw
# ---------------------------------------------------------------------------


class TestSafeRaw:
    def test_small_event_returned_unchanged(self):
        events = [{"type": "tool_use", "tool_name": "read_file", "tool_id": "t1"}]
        assert safe_raw(events) is events

    def test_large_event_trimmed(self):
        big_output = "x" * 20_000
        events = [
            {
                "type": "tool_result",
                "tool_id": "t1",
                "output": big_output,
                "extra_field": "value",
            }
        ]
        result = safe_raw(events)
        assert result is not events
        # Should keep only structural keys
        assert result[0] == {"type": "tool_result", "tool_id": "t1"}
        # Last element is the trimmed metadata
        meta = result[-1]["_raw_trimmed"]
        assert meta["limit"] == 16_384
        assert meta["original_bytes"] > 16_384

    def test_custom_limit(self):
        events = [{"type": "message", "content": "a" * 200}]
        # Under custom limit
        assert safe_raw(events, limit=1024) is events
        # Over custom limit
        result = safe_raw(events, limit=50)
        assert result is not events
        assert result[-1]["_raw_trimmed"]["limit"] == 50

    def test_structural_keys_preserved(self):
        events = [
            {
                "type": "tool_use",
                "id": "abc",
                "tool_id": "t1",
                "tool_name": "search",
                "role": "assistant",
                "event_type": "message",
                "timestamp": 12345,
                "big_content": "z" * 20_000,
            }
        ]
        result = safe_raw(events)
        kept = result[0]
        assert kept == {
            "type": "tool_use",
            "id": "abc",
            "tool_id": "t1",
            "tool_name": "search",
            "role": "assistant",
            "event_type": "message",
            "timestamp": 12345,
        }

    def test_multiple_events(self):
        events = [
            {"type": "msg", "data": "a" * 10_000},
            {"type": "msg", "data": "b" * 10_000},
        ]
        result = safe_raw(events)
        assert len(result) == 3  # 2 trimmed events + 1 metadata
        assert result[0] == {"type": "msg"}
        assert result[1] == {"type": "msg"}
        assert "_raw_trimmed" in result[2]


# ---------------------------------------------------------------------------
# build_cogitate_env
# ---------------------------------------------------------------------------


class TestBuildCogitateEnv:
    """Tests for build_cogitate_env — API key stripping for CLI subprocesses."""

    def test_default_strips_key(self):
        """No auth config → default platform mode → key removed."""
        config = {"providers": {}}
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY" not in env

    def test_explicit_platform_strips_key(self):
        """auth.anthropic = "platform" → key removed."""
        config = {"providers": {"auth": {"anthropic": "platform"}}}
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY" not in env

    def test_api_key_mode_preserves_key(self):
        """auth.anthropic = "api_key" → key preserved."""
        config = {"providers": {"auth": {"anthropic": "api_key"}}}
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert env["ANTHROPIC_API_KEY"] == "sk-secret"

    def test_missing_auth_section_strips_key(self):
        """No providers section at all → safe default, key removed."""
        config = {}
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("OPENAI_API_KEY")
        assert "OPENAI_API_KEY" not in env

    def test_other_env_vars_preserved(self):
        """Non-API-key vars are never stripped."""
        config = {"providers": {}}
        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "sk-secret", "HOME": "/home/test"},
                clear=False,
            ),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert env["HOME"] == "/home/test"

    def test_key_not_in_env_is_harmless(self):
        """Stripping a key that doesn't exist doesn't error."""
        config = {"providers": {}}
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_API_KEY" not in env

    def test_per_provider_independence(self):
        """Each provider's auth mode is independent."""
        config = {
            "providers": {
                "auth": {
                    "anthropic": "api_key",
                    "openai": "platform",
                }
            }
        }
        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "sk-ant", "OPENAI_API_KEY": "sk-oai"},
                clear=False,
            ),
            patch("think.utils.get_config", return_value=config),
        ):
            ant_env = build_cogitate_env("ANTHROPIC_API_KEY")
            oai_env = build_cogitate_env("OPENAI_API_KEY")
        assert ant_env["ANTHROPIC_API_KEY"] == "sk-ant"
        assert "OPENAI_API_KEY" not in oai_env
